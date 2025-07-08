from mcp.server.fastmcp import FastMCP
import pandas as pd
import ast
from DrugGen.drugGen_generator import run_inference
import json

mcp = FastMCP("DrugGen")

@mcp.tool(name="run_druggen")
def run_druggen(uniprot_id: str, num_generated: int = 5) -> str:
    output_file = "output.txt"
    run_inference(uniprot_ids=[uniprot_id], num_generated=num_generated, output_file=output_file)
    df = pd.read_csv(output_file, sep='\t')
    smiles_raw_list = df['SMILES'].tolist()
    smiles_list = []
    for smiles_str in smiles_raw_list:
        parsed = ast.literal_eval(smiles_str)
        smiles_list.extend(parsed)
    return json.dumps({"smiles": smiles_list[:num_generated]})

if __name__ == "__main__":
    mcp.run(transport="stdio")