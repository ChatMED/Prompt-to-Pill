from mcp.server.fastmcp import FastMCP
import os, json, time, hashlib, logging
from typing import List, Dict, Any, Tuple
import pandas as pd
import xml.etree.ElementTree as ET

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

logger = logging.getLogger("PatientMatchingMCP")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

# ---------- Minimal PANACEA loader ----------
_panacea = {"tok": None, "model": None}

def get_panacea():
    if _panacea["tok"] is None or _panacea["model"] is None:
        PANACEA_MODEL = os.getenv("PANACEA_MODEL", "path/to/panacea")
        _panacea["tok"] = AutoTokenizer.from_pretrained(PANACEA_MODEL, padding_side="left")
        if _panacea["tok"].pad_token is None:
            _panacea["tok"].add_special_tokens({"pad_token": "[PAD]"})
        _panacea["tok"].model_max_length = 2048
        _panacea["model"] = AutoModelForCausalLM.from_pretrained(
            PANACEA_MODEL,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
            low_cpu_mem_usage=True,
        )
        _panacea["model"].eval()
    return _panacea["tok"], _panacea["model"]

def parse_xml_patients(xml_file: str, num_patients: int = 3):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    patients = []
    for topic in root.findall('.//topic'):
        topic_number = topic.get('number')
        text_version = topic.find('text_version').text
        patients.append({'topic_number': topic_number, 'text_version': text_version})
    return patients

def build_patient_matching_messages(patient_note: str, trial_summary: str) -> Tuple[str, str]:

    system_msg = (
        "Hello. You are a helpful assistant for clinical trial recruitment. "
        "Your task is to compare a given patient note and the inclusion criteria of a clinical trial to determine the patient's eligibility. "
        "The factors that allow someone to participate in a clinical study are called inclusion criteria. "
        "They are based on characteristics such as age, gender, the type and stage of a disease, previous treatment history, and other medical conditions.\n\n"
        "The assessment of eligibility has a three-point scale:\n"
        "0) Excluded (patient meets inclusion criteria, but is excluded on the grounds of the trial's exclusion criteria)\n"
        "1) Not relevant (patient does not have sufficient information to qualify for the trial)\n"
        "2) Eligible (patient meets inclusion criteria and exclusion criteria do not apply)\n\n"
        "You should make a trial-level eligibility on each patient for the clinical trial, i.e., output the scale for the assessment of eligibility."
    )
    user_msg = (
        f"Here is the patient note:\n{patient_note}\n\n"
        f"Here is the clinical trial:\n{trial_summary}\n\n"
        f"Let's think step by step.\n"
        f"Finally, you should always repeat Trial-level eligibility in the last line by "
        f"`Trial-level eligibility: `, e.g., `Trial-level eligibility: 2) Eligible.`."
    )
    return system_msg, user_msg

def panacea_chat_generate(
    system_msg: str,
    user_msg: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> str:
    tok, model = get_panacea()

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token or "[PAD]"
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id
    eos_id = tok.eos_token_id or tok.pad_token_id

    try:
        enc = tok.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        input_ids = enc.to(model.device)
        attention_mask = (input_ids != tok.pad_token_id).long()
    except Exception as e:
        formatted = f"System: {system_msg}\nUser: {user_msg}\nAssistant:"
        enc = tok(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True,
        )
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)

    gen = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tok.pad_token_id,
        eos_token_id=eos_id,
    )

    generated_only = gen[0, input_ids.shape[-1]:]
    text = tok.decode(generated_only, skip_special_tokens=True)
    return text.strip()

mcp = FastMCP("PatientMatchingServer")

@mcp.tool(name="match_patient_trial")
async def match_patient_trial(
    xml_path: str,
    trial_text: str,
    start: int = 0,
    end: int = 30,
    max_new_tokens: int = 512,
    outfile: str = "matched_patients.json",
) -> Dict[str, Any]:
    logger.info(f"[match_patient_trial] xml={xml_path}, start={start}, end={end}")

    cache_key = hashlib.sha256(
        f"{xml_path}:{trial_text}:{start}:{end}".encode()
    ).hexdigest()
    if cache_key in result_cache:
        logger.info("Returning cached result")
        return result_cache[cache_key]

    if not os.path.exists(xml_path):
        return {"error": f"XML not found: {xml_path}", "retry": True}

    try:
        patients = parse_xml_patients(xml_path)  # you provide this
    except Exception as e:
        return {"error": f"Failed to parse XML: {e}", "retry": True}

    patients_df = pd.DataFrame([
        {"pid": p["topic_number"], "sentence": p["text_version"]}
        for p in patients[start:end]
    ])
    if patients_df.empty:
        return {"error": "No patients in this chunk", "retry": True}

    matches: List[Dict[str, Any]] = []
    last_keep_alive = time.time()
    keep_alive_interval = 30

    for _, row in patients_df.iterrows():
        now = time.time()
        if now - last_keep_alive >= keep_alive_interval:
            logger.info(f"Keep-alive: processing pid={row.pid}, progress {len(matches)}/{len(patients_df)}")
            last_keep_alive = now

        system_msg, user_msg = build_patient_matching_messages(  # you provide this
            patient_note=row.sentence,
            trial_summary=trial_text
        )

        try:
            answer = panacea_chat_generate(
                system_msg=system_msg,
                user_msg=user_msg,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9
            )
        except Exception as e:
            logger.error(f"Generation error for pid={row.pid}: {e}")
            continue

        elig_line = None
        if "Trial-level eligibility" in answer:
            elig_line = answer.split("Trial-level eligibility:", 1)[-1].strip()

        if elig_line and elig_line.startswith("2"):
            matches.append({"pid": row.pid, "eligibility": elig_line})

        logger.info(f"Progress: {len(matches)}/{len(patients_df)} eligible so far")

    abs_path = os.path.abspath(outfile)
    with open(abs_path, "w") as f:
        json.dump(matches, f, indent=2)

    result = {
        "matched_patients_file": abs_path,
        "total_patients_parsed": int(len(patients_df)),
        "matched_patients_count": int(len(matches)),
        "matches": matches
    }
    result_cache[cache_key] = result
    return result

if __name__ == "__main__":
    mcp.run(transport="stdio")
