# trialgen_mcp_server.py
from mcp.server.fastmcp import FastMCP
import os, logging
from typing import Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

logger = logging.getLogger("TrialGenMCP")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

# ---------- PANACEA loader ----------
_panacea = {"tok": None, "model": None}

def get_panacea():
    if _panacea["tok"] is None or _panacea["model"] is None:
        PANACEA_MODEL = os.getenv("PANACEA_MODEL", "/path/to/panacea")
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

def panacea_chat_generate(system_msg: str, user_msg: str, max_new_tokens: int = 512,
                          temperature: float = 0.7, top_p: float = 0.9) -> str:
    tok, model = get_panacea()
    messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token or "[PAD]"
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id
    eos_id = tok.eos_token_id or tok.pad_token_id

    try:
        enc = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        input_ids = enc.to(model.device)
        attention_mask = (input_ids != tok.pad_token_id).long()
    except Exception:
        formatted = f"System: {system_msg}\nUser: {user_msg}\nAssistant:"
        enc = tok(formatted, return_tensors="pt", truncation=True, max_length=2048, padding=True)
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
    return tok.decode(gen[0, input_ids.shape[-1]:], skip_special_tokens=True).strip()

INCLUSION_CRITERIA_INSTRUCTIONS = r"""
You are a helpful assistant for clinical trial criteria design.
Your task is to generate inclusion criteria for a given trial text. Use only the trial text.  
"""
EXCLUSION_CRITERIA_INSTRUCTIONS = r"""
You are a helpful assistant for clinical trial criteria design.
Your task is to generate exclusion criteria for a given trial text. Use only the trial text.
"""

OUTCOMES_INSTRUCTIONS = r"""
You are a helpful assistant for clinical trial outcomes design.
Your task is to generate primary and secondary outcomes for a given trial text.
"""

ARMS_INSTRUCTIONS = r"""
You are a helpful assistant for clinical trial study arms design.
Your task is to generate study arms for a given trial text.
"""

def build_component_messages(instruction: str, trial_text: str) -> tuple:
    # Make the instruction the system role message
    system_msg = instruction.strip()
    # Keep the trial details as user input only
    user_msg = f"Here is TRIAL TEXT:\n{trial_text}"
    return system_msg, user_msg

# ---------- Tool ----------
mcp = FastMCP("TrialGenerationServer")

@mcp.tool(name="panacea_extract_components")
async def panacea_extract_components(trial_text: str, max_new_tokens: int = 1024) -> Dict[str, Any]:
    try:
        criteria_system, criteria_user = build_component_messages(INCLUSION_CRITERIA_INSTRUCTIONS, trial_text)
        raw_inclusion_criteria = panacea_chat_generate(criteria_system, criteria_user, max_new_tokens=768)

        criteria_system, criteria_user = build_component_messages(EXCLUSION_CRITERIA_INSTRUCTIONS, trial_text)
        raw_exclusion_criteria = panacea_chat_generate(criteria_system, criteria_user, max_new_tokens=768)

        outcomes_system, outcomes_user = build_component_messages(OUTCOMES_INSTRUCTIONS, trial_text)
        raw_outcomes = panacea_chat_generate(outcomes_system, outcomes_user, max_new_tokens=768)

        arms_system, arms_user = build_component_messages(ARMS_INSTRUCTIONS, trial_text)
        raw_arms = panacea_chat_generate(arms_system, arms_user, max_new_tokens=768)

        return {
            "_raw": {
                "inclusion_criteria": raw_inclusion_criteria,
                "exclusion_criteria": raw_exclusion_criteria,
                "outcomes": raw_outcomes,
                "arms": raw_arms
            }
        }
    except Exception as e:
        return {
            "criteria": {"inclusion": [], "exclusion": []},
            "outcomes": {"primary": [], "secondary": [], "primary_flat": [], "secondary_flat": []},
            "arms": {"arms": [], "simplified": []},
            "error": str(e)
        }

if __name__ == "__main__":
    mcp.run(transport="stdio")
