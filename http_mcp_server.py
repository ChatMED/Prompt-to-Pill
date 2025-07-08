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

def txgemma_predict(prompt: str, max_new_tokens=16) -> str:
    input_ids = predict_tokenizer(prompt, return_tensors="pt").to(predict_model.device)
    outputs = predict_model.generate(**input_ids, max_new_tokens=max_new_tokens)
    return predict_tokenizer.decode(outputs[0], skip_special_tokens=True)

def load_patients_from_xml(xml_path: str) -> pd.DataFrame:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    rows = []
    for topic in root.findall("topic"):
        pid = topic.attrib["number"]
        text = topic.findtext("text_version") or topic.findtext("expanded")
        rows.append({
            "pid": pid,
            "sentence": text.strip() if text else "",
            "label": 0
        })
    return pd.DataFrame(rows)

def build_patient_trial_pairs(patients_df: pd.DataFrame, trial_text: str) -> pd.DataFrame:
    return pd.DataFrame({
        "pid": patients_df["pid"],
        "sentence_patient": patients_df["sentence"],
        "sentence_trial": [trial_text]*len(patients_df),
        "label": 0
    })

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
async def match_patient_trial(xml_path: str, trial_text: str) -> str:
    logger.info(f"match_patient_trial with xml_path={xml_path}")
    try:
        if not os.path.exists(xml_path):
            return json.dumps({"error": f"XML not found: {xml_path}", "retry": True})

        patients_df = load_patients_from_xml(xml_path)
        pairs_df = build_patient_trial_pairs(patients_df, trial_text)

        pairs_df["text"] = pairs_df["sentence_patient"] + " [SEP] " + pairs_df["sentence_trial"]
        inputs = meditab_tokenizer(
            pairs_df["text"].tolist(),
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
            probs = torch.sigmoid(outputs.logits).squeeze().detach().numpy()
        pairs_df["match_probability"] = probs.tolist()
        result = pairs_df[["pid", "match_probability"]].to_dict(orient="records")
        return json.dumps({"match_probability": result})
    except Exception as e:
        return json.dumps({"error": str(e), "retry": True})

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
    print("🚀 Combined FastMCP server starting on port 8000", flush=True)
    mcp.run(transport="streamable-http")
