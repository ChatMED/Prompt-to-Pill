from mcp.server.fastmcp import FastMCP
from gradio_client import Client
import os, sys
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import logging
logging.getLogger().setLevel(logging.WARNING)
for name in ("gradio_client", "httpx", "urllib3"):
    logging.getLogger(name).setLevel(logging.ERROR)


SPACE_ID = "ChemFM/molecular_property_prediction_zero_gpu"



# Canonical property list from the Space (you can also fetch via /get_description)
PROPERTIES = [
    'Drug Oral Bioavailability', 'Blood-Brain Barrier Permeability',
    'Drug Half-Life Duration','Drug Mutagenicity','Drug Clearance from Hepatocyte Experiments',
    'Drug Clearance from Microsome Experiments','Drug-Induced Liver Injury','hERG Channel Blockage',
    'Drug Acute Toxicity','Plasma Protein Binding Rate','P-glycoprotein Inhibition',
    'Drug Aqueous Solubility','Volume of Distribution at Steady State','CYP2C9 Inhibition',
    'CYP3A4 Inhibition','CYP2C9 Substrate','CYP2D6 Inhibition','CYP2D6 Substrate',
    'Drug Human Intestinal Absorption','CYP3A4 Substrate','Drug Permeability'
]

mcp = FastMCP("ChemFMProxy")
def _as_smiles_list(smiles):
    """Normalize input to a list[str]."""
    if smiles is None:
        return []
    if isinstance(smiles, str):
        s = smiles.strip()
        return [s] if s else []
    if isinstance(smiles, (list, tuple)):
        return [str(x).strip() for x in smiles if str(x).strip()]
    # fallback: force to str
    return [str(smiles).strip()]
def _client():
    return Client(SPACE_ID, hf_token= 'your_HF_token') #put your HF token here

@mcp.tool(
    name="chemfm_list_properties",
    description="List all property names supported by the ChemFM Space."
)
def chemfm_list_properties() -> dict:
    return {"ok": True, "properties": PROPERTIES}

@mcp.tool(
    name="chemfm_get_description",
    description="Get the ChemFM description for a given property."
)
def chemfm_get_description(property_name: str = "Drug Oral Bioavailability") -> dict:
    if property_name not in PROPERTIES:
        return {"ok": False, "error": f"Unknown property '{property_name}'. Use chemfm_list_properties first."}
    try:
        desc = _client().predict(property_name, api_name="/get_description")
        return {"ok": True, "property": property_name, "description": desc}
    except Exception as e:
        return {"ok": False, "error": f"get_description failed: {e}"}

@mcp.tool(
    name="chemfm_predict_single",
    description="Predict ONE property for ONE or MANY SMILES via /predict_single_label. Input 'smiles' can be a string or list of strings."
)
def chemfm_predict_single(smiles: str | list[str], property_name: str) -> dict:
    if property_name not in PROPERTIES:
        return {"ok": False, "error": f"Unknown property '{property_name}'. Use chemfm_list_properties first."}

    smiles_list = _as_smiles_list(smiles)
    if not smiles_list:
        return {"ok": False, "error": "No valid SMILES provided."}

    # Single input: preserve original response shape for backward compatibility
    if len(smiles_list) == 1:
        s = smiles_list[0]
        try:
            value_dict, status = _client().predict(s, property_name, api_name="/predict_single_label")
            return {
                "ok": True,
                "property": property_name,
                "smiles": s,
                "prediction": value_dict.get("label"),
                "confidences": value_dict.get("confidences"),
                "status": status,
                "raw": value_dict,
            }
        except Exception as e:
            return {"ok": False, "error": f"predict_single_label failed: {e}", "smiles": s, "property": property_name}

    # Batch input: return a list of per-SMILES results
    results = []
    for s in smiles_list:
        try:
            value_dict, status = _client().predict(s, property_name, api_name="/predict_single_label")
            results.append({
                "smiles": s,
                "prediction": value_dict.get("label"),
                "confidences": value_dict.get("confidences"),
                "status": status,
                "raw": value_dict,
            })
        except Exception as e:
            results.append({"smiles": s, "error": str(e)})
    return {"ok": True, "property": property_name, "results": results}
@mcp.tool(
    name="chemfm_predict_many",
    description="Predict MANY properties for ONE or MANY SMILES (loops /predict_single_label). 'smiles' can be a string or list of strings."
)
def chemfm_predict_many(smiles: str | list[str], properties: list[str]) -> dict:
    if not isinstance(properties, (list, tuple)) or not properties:
        return {"ok": False, "error": "Provide a non-empty list of property names."}
    unknown = [p for p in properties if p not in PROPERTIES]
    if unknown:
        return {"ok": False, "error": f"Unknown properties: {unknown}"}

    smiles_list = _as_smiles_list(smiles)
    if not smiles_list:
        return {"ok": False, "error": "No valid SMILES provided."}

    if len(smiles_list) == 1:
        s = smiles_list[0]
        out = {}
        for p in properties:
            try:
                value_dict, status = _client().predict(s, p, api_name="/predict_single_label")
                out[p] = {
                    "prediction": value_dict.get("label"),
                    "confidences": value_dict.get("confidences"),
                    "status": status,
                    "raw": value_dict,
                }
            except Exception as e:
                out[p] = {"error": str(e)}
        return {"ok": True, "smiles": s, "results": out}

    batch_out = {}
    for s in smiles_list:
        prop_out = {}
        for p in properties:
            try:
                value_dict, status = _client().predict(s, p, api_name="/predict_single_label")
                prop_out[p] = {
                    "prediction": value_dict.get("label"),
                    "confidences": value_dict.get("confidences"),
                    "status": status,
                    "raw": value_dict,
                }
            except Exception as e:
                prop_out[p] = {"error": str(e)}
        batch_out[s] = prop_out

    return {"ok": True, "results": batch_out}

if __name__ == "__main__":
    mcp.run('stdio')
