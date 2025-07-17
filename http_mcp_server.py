from mcp.server.fastmcp import FastMCP
import os
import sys
import json
import re
import torch
import logging
import xml.etree.ElementTree as ET
import pandas as pd
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login, hf_hub_download
from MediTab.meditab.bert import BertTabClassifier, BertTabTokenizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

PREDICT_VARIANT = "2b-predict"
quant_config = BitsAndBytesConfig(load_in_4bit=True)

tdc_prompts_filepath = hf_hub_download(
    repo_id=f"google/txgemma-{PREDICT_VARIANT}",
    filename="tdc_prompts.json"
)
with open(tdc_prompts_filepath, "r") as f:
    tdc_prompts_json = json.load(f)

predict_tokenizer = AutoTokenizer.from_pretrained(f"google/txgemma-{PREDICT_VARIANT}", token=HF_TOKEN)
predict_model = AutoModelForCausalLM.from_pretrained(
    f"google/txgemma-{PREDICT_VARIANT}",
    device_map="auto",
    token=HF_TOKEN,
    quantization_config=quant_config,
)

meditab_model = BertTabClassifier.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
meditab_tokenizer = BertTabTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")

PANACEA_MODEL   = "linjc16/Panacea-7B-Chat"
PANACEA_CACHE   = os.getenv("PANACEA_CACHE", "/tmp/panacea_cache")

panacea_tokenizer = AutoTokenizer.from_pretrained(
    PANACEA_MODEL, cache_dir=PANACEA_CACHE, padding_side="left"
)
if panacea_tokenizer.pad_token is None:
    panacea_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
panacea_tokenizer.model_max_length = 1000000000000000019884624838656

panacea_model = AutoModelForCausalLM.from_pretrained(
    PANACEA_MODEL,
    cache_dir=PANACEA_CACHE,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
)
panacea_model.eval()
result_cache = {}

def txgemma_predict(prompt: str, max_new_tokens=16) -> str:
    input_ids = predict_tokenizer(prompt, return_tensors="pt").to(predict_model.device)
    outputs = predict_model.generate(**input_ids, max_new_tokens=max_new_tokens)
    return predict_tokenizer.decode(outputs[0], skip_special_tokens=True)

def parse_xml_patients(xml_file, num_patients=3):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    patients = []
    for topic in root.findall('.//topic'):
        topic_number = topic.get('number')
        text_version = topic.find('text_version').text
        patients.append({'topic_number': topic_number, 'text_version': text_version})
    return patients

def build_panacea_prompt(patient_note: str, trial_summary: str) -> str:
    return (
        f"Hello. You are a helpful assistant for clinical trial recruitment. Your task is to compare a given patient note and the inclusion criteria of a clinical trial to determine the patient's eligibility. "
        f"The factors that allow someone to participate in a clinical study are called inclusion criteria. They are based on characteristics such as age, gender, the type and stage of a disease, previous treatment history, and other medical conditions.\n\n"
        f"The assessment of eligibility has a three-point scale: 0) Excluded (patient meets inclusion criteria, but is excluded on the grounds of the trial's exclusion criteria); "
        f"1) Not relevant (patient does not have sufficient information to qualify for the trial); 2) Eligible (patient meets inclusion criteria and exclusion criteria do not apply). "
        f"You should make a trial-level eligibility on each patient for the clinical trial, i.e., output the scale for the assessment of eligibility.\n\n"
        f"Here is the patient note:\n{patient_note}\n\n"
        f"Here is the clinical trial:\n{trial_summary}\n\n"
        f"Let's think step by step.\n"
        f"Finally, you should always repeat Trial-level eligibility in the last line by `Trial-level eligibility: `, e.g., `Trial-level eligibility: 2) Eligible.`."
    )

def build_trial_csv(trial_text: str) -> pd.DataFrame:
    return pd.DataFrame({
        "nct_id": ["T001"],
        "sentence": [trial_text],
        "phase": ["Phase 1"],
        "label": [0]
    })

mcp = FastMCP("CombinedServer")

@mcp.tool(name="predict_properties")
def predict_properties(smiles: str) -> dict:
    logger.info(f"predict_properties for SMILES: {smiles}")
    if not smiles:
        raise ValueError("Missing SMILES input")

    properties = {}
    fields = [
        ("BBB Permeability", "BBB_Martins"),
        ("Toxicity", "ClinTox"),
        ("Lipophilicity", "Lipophilicity_AstraZeneca"),
        ("Solubility", "Solubility_AqSolDB"),
        ("PPBR", "PPBR_AZ"),
        ("Half-Life", "Half_Life_Obach")
    ]

    for name, key in fields:
        prompt = tdc_prompts_json[key].replace("{Drug SMILES}", smiles)
        result = txgemma_predict(prompt).strip()
        logger.debug(f"[{name}] Raw output: {result}")

        response = result.lower()

        if name == "Toxicity":
            if "not toxic" in response or "(a)" in response:
                properties["Toxicity"] = "Not toxic"
            elif "toxic" in response or "(b)" in response:
                properties["Toxicity"] = "Toxic"
            else:
                properties["Toxicity"] = "Unknown"

        elif name == "BBB Permeability":
            if "crosses" in response or "(b)" in response:
                properties["BBB_Permeability"] = "Crosses BBB"
            elif "does not cross" in response or "(a)" in response:
                properties["BBB_Permeability"] = "Does not cross BBB"
            else:
                properties["BBB_Permeability"] = "Unknown"

        else:
            number_match = re.search(r"Answer:(\d+)", result)
            if number_match:
                try:
                    properties[name] = int(number_match.group(1))
                except ValueError:
                    properties[name] = 0
            else:
                properties[name] = 0

    return properties

@mcp.tool(name="toxicity_screening")
def toxicity_screening(smiles_list: List[str]) -> dict:
    logger.info(f"toxicity_screening for SMILES list: {smiles_list}")
    non_toxic = []
    for smiles in smiles_list:
        try:
            props = predict_properties(smiles)
            if props.get("Toxicity", "Unknown") == "Not toxic":
                non_toxic.append(smiles)
        except Exception as e:
            logger.error(f"Error processing {smiles}: {e}")
            continue

    if not non_toxic:
        return {"error": "All compounds toxic or invalid", "retry": True}
    return {"non_toxic_smiles": non_toxic}

@mcp.tool(name="match_patient_trial")
async def match_patient_trial(xml_path: str, trial_summary: str, max_new_tokens: int = 1024,
    outfile: str = "matched_patients.json") -> Dict:
    logger.info(f"[match_patient_trial] xml={xml_path}")

    cache_key = hashlib.sha256(f"{xml_path}:{trial_summary}".encode()).hexdigest()
    if cache_key in result_cache:
        logger.info("Returning cached result")
        return result_cache[cache_key]

    if not os.path.exists(xml_path):
        return {"error": f"XML not found: {xml_path}", "retry": True}

    patients = parse_xml_patients(xml_path)
    patients_df = pd.DataFrame([
        {"pid": p["topic_number"], "sentence": p["text_version"]}
        for p in patients
    ])

    if patients_df.empty:
        return {"error": "No patients found in XML", "retry": True}

    matches = []

    for idx, row in patients_df.iterrows():
        prompt = build_panacea_prompt(row.sentence, trial_summary)
        try:
            inputs = panacea_tokenizer(
                prompt,
                truncation=True,
                max_length=2048,
                return_tensors="pt"
            ).to(panacea_model.device)
            with torch.no_grad():
                outputs = panacea_model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    pad_token_id=panacea_tokenizer.pad_token_id,
                )
            answer = panacea_tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "Trial-level eligibility" in answer:
                elig_line = answer.split("Trial-level eligibility:")[-1].strip()
                if elig_line.startswith("2"):
                    matches.append({"pid": row.pid, "eligibility": elig_line})
        except Exception as e:
            logger.error(f"Error for patient {row.pid}: {e}")
            continue

    abs_path = os.path.abspath(outfile)
    with open(abs_path, "w") as f:
        json.dump(matches, f, indent=2)

    result = {
        "matched_patients_file": abs_path,
        "total_patients_parsed": len(patients_df),
        "matched_patients_count": len(matches),
        "matches": matches
    }
    result_cache[cache_key] = result
    return result

@mcp.tool(name="predict_trial_success")
async def predict_trial_success(trial_text: str) -> str:
    logger.info(f"predict_trial_success for trial text length: {len(trial_text)}")
    try:
        trial_df = build_trial_csv(trial_text)
        inputs = meditab_tokenizer(
            trial_df["sentence"].tolist(),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        with torch.no_grad():
            meditab_model.eval()
            outputs = meditab_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            probs = torch.sigmoid(outputs.logits).squeeze().numpy()
        trial_df["success_probability"] = probs.tolist()
        return json.dumps({
            "nct_id": trial_df["nct_id"].iloc[0],
            "success_probability": trial_df["success_probability"].iloc[0]
        })
    except Exception as e:
        return json.dumps({"error": str(e)})

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
