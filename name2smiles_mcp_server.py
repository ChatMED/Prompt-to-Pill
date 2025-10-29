from mcp.server.fastmcp import FastMCP
import requests
import json
from urllib.parse import quote
from time import sleep
import re
from typing import Union, List

mcp = FastMCP("SmilesServer")
PUBCHEM_API_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

def _retry_get(url, timeout=10, retries=3, backoff_s=1.0):
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt < retries - 1 and ("429" in str(e) or "timed out" in str(e).lower()):
                sleep(backoff_s)
                continue
            raise

def _pick_preferred_name(synonyms):
    # Prefer short, human-friendly names (heuristic)
    if not synonyms:
        return None
    def score(s):
        bad = any(ch in s for ch in [",",";","=","[","]"])
        length_pen = 0 if 3 <= len(s) <= 30 else 2
        title_like = bool(re.match(r"^[A-Z][A-Za-z0-9\- ]+$", s))
        word_count = len(s.split())
        return (
            0 if title_like else 1,            # prefer TitleCase/alnum
            0 if not bad else 1,               # avoid punctuation typical of formulas
            0 if word_count <= 3 else 1,       # fewer words
            length_pen,                        # reasonable length
            len(s)                             # shortest wins
        )
    return sorted(synonyms, key=score)[0]

@mcp.tool(name="get_smiles_from_drug_name")
def get_smiles_from_drug_name(drug_name: str) -> str:

    retries = 3
    for attempt in range(retries):
        try:
            # Normalize drug name (capitalize to match PubChem convention)
            drug_name_normalized = drug_name.capitalize()
            url = f"{PUBCHEM_API_BASE}/compound/name/{quote(drug_name_normalized)}/property/IsomericSMILES/JSON"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            if 'PropertyTable' in data and 'Properties' in data['PropertyTable']:
                properties = data['PropertyTable']['Properties']
                if properties and 'SMILES' in properties[0]:
                    return json.dumps({"smiles": properties[0]['SMILES']})

            return json.dumps({"smiles": None, "error": f"No SMILES found for '{drug_name}' in PubChem."})
        except requests.RequestException as e:
            if attempt < retries - 1 and "429" in str(e):
                sleep(1)
                continue
            return json.dumps({"smiles": None, "error": f"Error querying PubChem: {str(e)}"})

@mcp.tool(name="get_drug_name_from_smiles")
def get_drug_name_from_smiles(smiles_input: Union[str, List[str]]) -> str:
    results = []
    smiles_list = [smiles_input] if isinstance(smiles_input, str) else smiles_input

    for smiles in smiles_list:
        try:
            smiles_q = quote(smiles.strip())
            cid_url = f"{PUBCHEM_API_BASE}/compound/smiles/{smiles_q}/cids/JSON"
            cid_data = _retry_get(cid_url)
            cids = cid_data.get("IdentifierList", {}).get("CID", [])
            if not cids:
                results.append({
                    "cid": None,
                    "synonyms": [],
                    "iupac_name": None,
                    "unii": None,
                    "brand_names": [],
                    "error": f"No CID found for SMILES: {smiles}"
                })
                continue
            cid = int(cids[0])

            syn_url = f"{PUBCHEM_API_BASE}/compound/cid/{cid}/synonyms/JSON"
            syn_data = _retry_get(syn_url)
            synonyms = syn_data.get("InformationList", {}).get("Information", [{}])[0].get("Synonym", []) or []

            unii = None
            for synonym in synonyms:
                if not synonym:
                    continue
                m = re.search(r'\bUNII[-\s:]*([A-Z0-9]{10})\b', synonym.upper())
                if m:
                    unii = m.group(1)
                    break

            brand_names = []
            if unii:
                try:
                    # NDC (brand_name, brand_name_base)
                    ndc = _retry_get(f"https://api.fda.gov/drug/ndc.json?search=openfda.unii:\"{unii}\"&limit=100")
                    for item in ndc.get("results", []):
                        if item.get("brand_name"):
                            brand_names.append(item["brand_name"].strip())
                        if item.get("brand_name_base"):
                            brand_names.append(item["brand_name_base"].strip())
                except Exception:
                    pass
            results.append({
                "cid": cid,
                "synonyms": synonyms[:5],
                "iupac_name": None,
                "unii": unii,
                "brand_names": sorted(set(brand_names)),
                "error": None
            })

        except Exception as e:
            results.append({
                "cid": None,
                "iupac_name": None,
                "unii": None,
                "brand_names": [],
                "error": f"{type(e).__name__}: {e}"
            })

    return json.dumps(results)


if __name__ == "__main__":
    mcp.run(transport="stdio")
