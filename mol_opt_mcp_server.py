from mcp.server.fastmcp import FastMCP
import os, json, logging
from typing import Dict
from huggingface_hub import login, hf_hub_download
from llama_cpp import Llama

def get_drugassist():
    user_secrets = UserSecretsClient()
    hf_token = ["HF_TOKEN"]
    if not hf_token:
        raise RuntimeError("HF_TOKEN not set in Kaggle secrets.")
    gguf_path = hf_hub_download(
        repo_id="blazerye/DrugAssist-7B",
        filename="DrugAssist-7B-4bit.gguf",
        token=hf_token,
    )
    llm = Llama(model_path=gguf_path)
    return llm

mcp = FastMCP("MolOptServer")

@mcp.tool(name="molecule_optimizer")
async def molecule_optimizer(smiles: str, properties: str, action: str):
    prompt = (
        f"I have a molecule with the SMILES notation {smiles}. "
        f"Suggest modifications to {action} its {properties} value while maintaining its core structure."
    )
    llm = get_drugassist()
    respond = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}]
    )
    dic = respond['choices'][0]
    message = dic['message']
    return json.dumps({'message': message})
        return {"error": str(e)}

if __name__ == "__main__":
    mcp.run(transport="stdio")
