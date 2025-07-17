import os, sys, subprocess, requests
from typing import Tuple, Optional
from rdkit import Chem
from rdkit.Chem import AllChem

CHAIN = "A"
BOX_PAD = 4.0
BOX_MIN = 18.0
EXH = 4

PDBe_BEST = "https://www.ebi.ac.uk/pdbe/graph-api/mappings/best_structures/{}"
RCSB_PDB = "https://files.rcsb.org/download/{}.pdb"
AF_URL = "https://alphafold.ebi.ac.uk/files/AF-{}-F1-model_v4.pdb"


def get_structure(uniprot: str) -> Tuple[str, str, Optional[str]]:
    uni = uniprot.upper()
    try:
        best = requests.get(PDBe_BEST.format(uni), timeout=10).json()[uni][0]
        pdb_id = best["pdb_id"].upper()
        pdb_path = f"{pdb_id}.pdb"
        if not os.path.exists(pdb_path):
            open(pdb_path, "w").write(requests.get(RCSB_PDB.format(pdb_id), timeout=10).text)
        lig_resn = None
        with open(pdb_path) as fh:
            for ln in fh:
                if ln.startswith("HETATM") and ln[21]==CHAIN and ln[17:20].strip() not in {"HOH", "HOH", "SO4", "NAG", "NDG", "NA"}:
                    lig_resn = ln[17:20].strip(); break
        return pdb_path, pdb_id, lig_resn
    except Exception:
        pdb_path = f"{uni}_AF.pdb"
        if not os.path.exists(pdb_path):
            open(pdb_path, "w").write(requests.get(AF_URL.format(uni), timeout=10).text)
        print(f"✓ AlphaFold model for {uni}")
        return pdb_path, f"AF_{uni}", None


def split_receptor_ligand(pdb_in: str, lig_resn: str | None) -> Tuple[str, Optional[str]]:
    rec, lig = "receptor_raw.pdb", "ligand_ref.pdb"
    ligand_found = False
    if lig_resn:
        with open(lig, "w") as L, open(pdb_in) as src:
            for ln in src:
                if ln.startswith("HETATM") and ln[17:20].strip()==lig_resn and ln[21]==CHAIN:
                    L.write(ln); ligand_found = True
            L.write("END\n")
    if not ligand_found:
        lig = None
    with open(rec, "w") as R, open(pdb_in) as src:
        for ln in src:
            if ln.startswith("ATOM") and ln[21]==CHAIN:
                R.write(ln)
        R.write("END\n")
    return rec, lig

def grid_from_file(pdb: str):
    xs=ys=zs=[]
    xs,ys,zs=[],[],[]
    for ln in open(pdb):
        skip_resns = {"HOH", "SO4", "NAG", "NDG", "NA"}
        resn = ln[17:20].strip()
        if ln.startswith("HETATM") and ln[21] == CHAIN and resn not in skip_resns:
            xs.append(float(ln[30:38])); ys.append(float(ln[38:46])); zs.append(float(ln[46:54]))
    cx,cy,cz = sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs)
    s = max(BOX_MIN, max(xs)-min(xs)+2*BOX_PAD, max(ys)-min(ys)+2*BOX_PAD, max(zs)-min(zs)+2*BOX_PAD)
    return cx,cy,cz,s

def obabel_pdbqt(pdb: str, out: str, is_lig: bool):
    cmd=["obabel","-ipdb",pdb,"-opdbqt","-O",out,"--partialcharge","gasteiger"]
    if not is_lig: cmd.extend(["-xh","-xr"])
    subprocess.run(cmd, check=True, capture_output=True)

def smiles_to_pdbqt(smiles: str) -> str:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=42); AllChem.MMFFOptimizeMolecule(mol)
    Chem.SDWriter("ligand_tmp.sdf").write(mol)
    subprocess.run(["obabel","-isdf","ligand_tmp.sdf","-opdbqt","-O","ligand_gen.pdbqt","--partialcharge","gasteiger"], check=True)
    return "ligand_gen.pdbqt"

def run_vina(rec_pqt: str, lig_pqt: str, cx: float, cy: float, cz: float, box: float) -> float:
    cmd=["vina",
         "--receptor",rec_pqt,
         "--ligand",lig_pqt,
         "--center_x",str(cx),
         "--center_y",str(cy),
         "--center_z",str(cz),
         "--size_x",str(box),
         "--size_y",str(box),
         "--size_z",str(box),
         "--exhaustiveness",str(EXH),
         "--cpu","1"]
    out=subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    for ln in out.splitlines():
        if ln.strip().startswith("1 "):
            return float(ln.split()[1])
    return 0.0

def dock_with_vina(uniprot: str, smiles: str="") -> float:
    pdb_path, pdb_id, lig_resn = get_structure(uniprot)
    rec_pdb, lig_pdb = split_receptor_ligand(pdb_path, lig_resn)

    if lig_pdb is None and smiles == "":
        sys.exit("No ligand on chain A; please supply SMILES.")

    grid_ref = lig_pdb if lig_pdb else pdb_path
    cx,cy,cz,box = grid_from_file(grid_ref)

    obabel_pdbqt(rec_pdb, "receptor.pdbqt", is_lig=False)
    lig_pqt = smiles_to_pdbqt(smiles) if smiles else "ligand_ref.pdbqt"
    if smiles == "":
        obabel_pdbqt(lig_pdb, lig_pqt, is_lig=True)

    score = run_vina("receptor.pdbqt", lig_pqt, cx,cy,cz, box)
    return score
