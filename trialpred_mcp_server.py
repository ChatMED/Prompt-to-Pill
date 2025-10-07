from mcp.server.fastmcp import FastMCP
import logging, json
from typing import Tuple
import pandas as pd
import torch

from MediTab.meditab.bert import BertTabClassifier, BertTabTokenizer

logger = logging.getLogger("TrialPredictionMCP")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

_meditab = {"tok": None, "model": None}

def get_meditab():
    if _meditab["tok"] is None or _meditab["model"] is None:
        _meditab["model"] = BertTabClassifier.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
        _meditab["tok"] = BertTabTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
        _meditab["model"].eval()
    return _meditab["tok"], _meditab["model"]

def build_trial_csv(trial_text: str) -> pd.DataFrame:
    return pd.DataFrame({
        "nct_id": ["T001"],
        "sentence": [trial_text],
        "phase": ["Phase 1"],
        "label": [0]
    })

mcp = FastMCP("TrialPredictionServer")

@mcp.tool(name="predict_trial_success")
async def predict_trial_success(trial_text: str) -> str:
    logger.info(f"predict_trial_success for trial text length: {len(trial_text)}")
    tok, model = get_meditab()
    try:
        trial_df = build_trial_csv(trial_text)
        inputs = tok(trial_df["sentence"].tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            probs = torch.sigmoid(outputs.logits).squeeze().numpy()
        trial_df["success_probability"] = probs.tolist()
        return json.dumps({
            "nct_id": trial_df["nct_id"].iloc[0],
            "success_probability": float(trial_df["success_probability"].iloc[0])
        })
    except Exception as e:
        return json.dumps({"error": str(e)})

if __name__ == "__main__":
    mcp.run(transport="stdio")
