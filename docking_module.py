import os
import subprocess
from rdkit import Chem
from rdkit.Chem import AllChem
import requests
from io import StringIO

def dock_with_vina(uniprot_id: str, drug_smiles: str) -> float:
    protein_pdb = get_alphafold_structure(uniprot_id)
    if not protein_pdb:
        print(f"Could not retrieve structure for {uniprot_id}")
        return 0.0

    ligand_pdbqt = prepare_ligand_from_smiles(drug_smiles)
    if not ligand_pdbqt:
        print("Could not prepare ligand")
        return 0.0

    receptor_pdbqt = prepare_receptor(protein_pdb)

    binding_affinity = run_vina_docking(receptor_pdbqt, ligand_pdbqt)

    return binding_affinity

def get_alphafold_structure(uniprot_id: str) -> str:
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(f"{uniprot_id}.pdb", "w") as f:
                f.write(response.text)
            return f"{uniprot_id}.pdb"
        else:
            print(f"AlphaFold structure not available for {uniprot_id}")
            return None
    except Exception as e:
        print(f"Error downloading structure: {e}")
        return None
def prepare_ligand_from_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)

    writer = Chem.SDWriter("ligand.sdf")
    writer.write(mol)
    writer.close()

    try:
        cmd = ["obabel", "-isdf", "ligand.sdf", "-opdbqt", "-O", "ligand.pdbqt", "--partialcharge", "gasteiger"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0 and os.path.exists("ligand.pdbqt"):
            return "ligand.pdbqt"
        else:
            print(f"OpenBabel error: {result.stderr}")
            return None

    except Exception as e:
        print(f"Error preparing ligand: {e}")
        return None

def prepare_receptor(pdb_file: str) -> str:
    try:
        cmd = ["obabel", "-ipdb", pdb_file, "-opdbqt", "-O", "receptor.pdbqt",
               "-xr", "--partialcharge", "gasteiger"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0 and os.path.exists("receptor.pdbqt"):
            return "receptor.pdbqt"
        else:
            print(f"OpenBabel receptor error: {result.stderr}")
            return None

    except Exception as e:
        print(f"Error preparing receptor: {e}")
        return None

def convert_pdb_to_pdbqt(pdb_content: str) -> str:
    lines = pdb_content.split('\n')
    pdbqt_lines = []

    for line in lines:
        if line.startswith('ATOM') or line.startswith('HETATM'):
            if len(line) >= 78:
                atom_type = line[76:78].strip()
                charge = "0.000"
                pdbqt_line = line[:70] + f"{charge:>6}" + f"{atom_type:>2}"
                pdbqt_lines.append(pdbqt_line)

    return '\n'.join(pdbqt_lines)


def run_vina_docking(receptor_pdbqt: str, ligand_pdbqt: str) -> float:
    center_x, center_y, center_z = 31.724, -22.0063, -17.132
    size_x, size_y, size_z = 30, 30, 30

    vina_cmd = [
        "vina",
        "--receptor", receptor_pdbqt,
        "--ligand", ligand_pdbqt,
        "--center_x", str(center_x),
        "--center_y", str(center_y),
        "--center_z", str(center_z),
        "--size_x", str(size_x),
        "--size_y", str(size_y),
        "--size_z", str(size_z),
        "--out", "docked.pdbqt"
    ]

    try:
        result = subprocess.run(vina_cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            output = result.stdout

            for line in output.split('\n'):
                if 'REMARK VINA RESULT:' in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'RESULT:' and i + 1 < len(parts):
                            try:
                                return float(parts[i + 1])
                            except ValueError:
                                continue

            if os.path.exists("docked.pdbqt"):
                with open("docked.pdbqt", "r") as f:
                    for line in f:
                        if line.startswith("REMARK VINA RESULT:"):
                            parts = line.split()
                            if len(parts) >= 4:
                                try:
                                    return float(parts[3])
                                except ValueError:
                                    continue

            print("Could not parse binding affinity from output")
            return 0.0
        else:
            print(f"Vina failed: {result.stderr}")
            return 0.0

    except Exception as e:
        print(f"Error running Vina: {e}")
        return 0.0

