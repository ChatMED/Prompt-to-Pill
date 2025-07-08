from mcp.server.fastmcp import FastMCP
import json
from typing import List, Union
from docking_module import dock_with_vina

mcp = FastMCP("Docking")

@mcp.tool(name="run_docking")
async def run_docking(uniprot_id: str, smiles_list: Union[List[str], str]) -> str:
    if isinstance(smiles_list, str):
        smiles_list = json.loads(smiles_list)
    smiles = []
    scores = []
    for smi in smiles_list:
        score = dock_with_vina(uniprot_id, smi)
        smiles.append(smi)
        scores.append(score)
    return json.dumps({"smiles": smiles, "scores": scores})

if __name__ == "__main__":
    mcp.run(transport="stdio")