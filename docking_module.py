import subprocess
import requests
import gzip
from typing import Tuple, Optional
from rdkit import Chem
from rdkit.Chem import AllChem
import csv
import shutil
import json
from pathlib import Path
import os
import platform

BOX_PAD = 4.0
BOX_MIN = 20
EXH = 4
MAX_BOX_SIZE = 30

PDBe_BEST = "https://www.ebi.ac.uk/pdbe/api/mappings/best_structures/{}"
RCSB_PDB = "https://files.rcsb.org/download/{id}.pdb"
PDBe_PDB = "https://www.ebi.ac.uk/pdbe/entry-files/download/pdb{id}.ent"
RCSB_CIF = "https://files.rcsb.org/download/{id}.cif"
AF_URL = "https://alphafold.ebi.ac.uk/files/AF-{}-F1-model_v4.pdb"

SKIP_RESNS = {"HOH", "SO4", "NAG", "NDG", "NA"}

def _nonempty(path: str, min_bytes: int = 200) -> bool:
    try:
        return Path(path).exists() and Path(path).stat().st_size >= min_bytes
    except Exception:
        return False

def fix_pdb_heme_elements(input_pdb: str, output_pdb: str) -> bool:
    heme_atom_elements = {
        "CHA": "C", "CHB": "C", "CHC": "C", "CHD": "C",
        "C1A": "C", "C2A": "C", "C3A": "C", "C4A": "C",
        "C1B": "C", "C2B": "C", "C3B": "C", "C4B": "C",
        "C1C": "C", "C2C": "C", "C3C": "C", "C4C": "C",
        "C1D": "C", "C2D": "C", "C3D": "C", "C4D": "C",
        "CMA": "C", "CMB": "C", "CMC": "C", "CMD": "C",
        "CAA": "C", "CAB": "C", "CAC": "C", "CAD": "C",
        "CBA": "C", "CBB": "C", "CBC": "C", "CBD": "C",
        "CGA": "C", "CGB": "C", "CGC": "C", "CGD": "C",
        "NA": "N", "NB": "N", "NC": "N", "ND": "N",
        "O1A": "O", "O2A": "O", "O1B": "O", "O2B": "O",
        "O1C": "O", "O2C": "O", "O1D": "O", "O2D": "O",
        "FE": "FE"
    }
    fixed = False
    with open(input_pdb, "r", errors="ignore") as f_in, open(output_pdb, "w") as f_out:
        for line in f_in:
            if line.startswith(("ATOM", "HETATM")) and len(line) >= 78:
                resn = line[17:20].strip().upper()
                if resn == "HEM":
                    atom_name = line[12:16].strip().upper()
                    if atom_name in heme_atom_elements:
                        elem = heme_atom_elements[atom_name]
                        if line[76:78].strip().upper() != elem:
                            line = line[:76] + f"{elem:>2}" + line[78:]
                            fixed = True
            f_out.write(line)
    return fixed

def fix_pdb_fad_elements(input_pdb: str, output_pdb: str) -> bool:
    fixed = False
    def infer_elem_from_name(name: str) -> str:
        name = name.strip().upper()
        for c in name:
            if c in ("C", "N", "O", "S", "P", "H"):
                return c
        return "C"
    with open(input_pdb, "r", errors="ignore") as fin, open(output_pdb, "w") as fout:
        for line in fin:
            if line.startswith(("ATOM", "HETATM")) and len(line) >= 80:
                resn = line[17:20].strip().upper()
                if resn == "FAD":
                    atom_name = line[12:16].strip()
                    elem = infer_elem_from_name(atom_name)
                    if line[76:78].strip().upper() != elem:
                        line = line[:76] + f"{elem:>2}" + line[78:]
                        fixed = True
            fout.write(line)
    return fixed

def _run(cmd: list, check=True):
    return subprocess.run(cmd, check=check, capture_output=True, text=True)

def vina_parse_top_score(log_path: str) -> Optional[float]:
    try:
        with open(log_path, "r", errors="ignore") as fh:
            for ln in fh:
                s = ln.strip()
                if s.startswith("1 "):
                    parts = s.split()
                    if len(parts) >= 2:
                        return float(parts[1])
    except Exception:
        pass
    return None

def _count_heavy_atoms_pdb(pdb_path: str) -> int:
    heavy = 0
    try:
        with open(pdb_path, "r", errors="ignore") as fh:
            for ln in fh:
                if ln.startswith(("ATOM", "HETATM")):
                    elem = (ln[76:78].strip() if len(ln) >= 78 else ln[12:16].strip()[0]).upper()
                    if elem != "H":
                        heavy += 1
    except Exception:
        pass
    return heavy

def _count_heavy_atoms_pdbqt(pdbqt_path: str) -> int:
    heavy = 0
    try:
        with open(pdbqt_path, "r", errors="ignore") as fh:
            for ln in fh:
                if ln.startswith(("ATOM","HETATM")):
                    elem = (ln[76:78].strip() if len(ln) >= 78 else "").upper()
                    if elem != "H":
                        heavy += 1
    except Exception:
        pass
    return heavy

def tool_versions() -> dict:
    vers = {}
    try:
        out = _run(["vina", "--version"]).stdout.strip().splitlines()
        vers["vina"] = out[0] if out else "unknown"
    except Exception:
        vers["vina"] = "unavailable"
    try:
        out = _run(["obabel", "-V"]).stdout.strip().splitlines()
        vers["obabel"] = out[0] if out else "unknown"
    except Exception:
        vers["obabel"] = "unavailable"
    try:
        import rdkit
        vers["rdkit"] = getattr(rdkit, "__version__", "unknown")
    except Exception:
        vers["rdkit"] = "unavailable"
    vers["p2rank"] = os.environ.get("P2RANK_PATH", "")
    return vers

def write_manifest(path: str, payload: dict):
    Path(path).write_text(json.dumps(payload, indent=2))

def _is_alphafold_id(pdb_id: str) -> bool:
    return pdb_id.upper().startswith("AF_")

def _p2rank_base_dir() -> Path:
    env = os.environ.get("P2RANK_PATH", "").strip()
    if env:
        p = Path(env)
        if p.exists():
            return p
    fallbacks = [
        r"C:\Users\ivana\PycharmProjects\Prompt_to_pill_Simulator\p2rank_2.5.1",
        r"C:\p2rank_2.5.1",
        "/opt/p2rank_2.5.1",
        "/usr/local/p2rank_2.5.1",
    ]
    for fb in fallbacks:
        p = Path(fb)
        if p.exists():
            return p
    raise RuntimeError(
        "P2Rank base directory not found. Set P2RANK_PATH to the P2Rank folder "
        "(the folder that contains pranker/prank/pranker.bat)."
    )

def available_p2rank() -> bool:
    try:
        base = _p2rank_base_dir()
    except RuntimeError:
        return False
    candidates = (["pranker.bat", "prank.bat"] if platform.system().lower().startswith("win")
                  else ["pranker", "prank"])
    return any((base / c).exists() for c in candidates)

def _p2rank_exe() -> str:
    base = _p2rank_base_dir()
    if platform.system().lower().startswith("win"):
        for c in ["pranker.bat", "prank.bat"]:
            p = base / c
            if p.exists():
                return str(p)
    else:
        for c in ["pranker", "prank"]:
            p = base / c
            if p.exists():
                return str(p)
    raise RuntimeError(
        f"P2Rank executable not found under {base}. Expected one of "
        f"{'pranker.bat/prank.bat' if platform.system().lower().startswith('win') else 'pranker/prank'}."
    )

def _find_p2rank_predictions_csv(folder: Path) -> Optional[Path]:
    patterns = ("*_predictions.csv", "*.predictions.csv", "*_predictions.csv.gz", "*.predictions.csv.gz")
    for pat in patterns:
        hits = sorted(folder.glob(pat))
        if hits:
            return hits[0]
    for pat in patterns:
        hits = sorted(folder.rglob(pat))
        if hits:
            return hits[0]
    return None

def _open_text_any(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")

def run_p2rank_list(pdb_path: str, pdb_id: str, threads: int = 4,
                    out_root: str = "p2rank_out", visuals: bool = False):
    exe = _p2rank_exe()
    out_root = Path(out_root)
    out_root.mkdir(exist_ok=True)
    stem = Path(pdb_path).stem
    job_dir = out_root / stem
    if job_dir.exists():
        shutil.rmtree(job_dir, ignore_errors=True)
    job_dir.mkdir(parents=True, exist_ok=True)
    cmd = [exe, "predict", "-f", pdb_path, "-o", str(job_dir), "-threads", str(max(1, int(threads)))]
    if _is_alphafold_id(pdb_id):
        cmd += ["-c", "alphafold"]
    if not visuals:
        cmd += ["-visualizations", "0"]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        return []
    pred_csv = _find_p2rank_predictions_csv(job_dir)
    if not pred_csv:
        return []
    pockets = []
    with _open_text_any(pred_csv) as fh:
        reader = csv.DictReader(fh, skipinitialspace=True)
        if reader.fieldnames:
            reader.fieldnames = [h.strip().lower() for h in reader.fieldnames]
        for row in reader:
            try:
                prob = float(row.get("probability") or row.get("probability_score") or 0.0)
                cx = float(row["center_x"])
                cy = float(row["center_y"])
                cz = float(row["center_z"])
                residue_ids = row.get("residue_ids")
            except Exception:
                continue
            base = max(BOX_MIN, 22.0)
            size = min(MAX_BOX_SIZE, max(base, BOX_MIN + prob * 6.0))
            pockets.append({"prob": prob, "cx": cx, "cy": cy, "cz": cz, "size": size, "residue_ids": residue_ids})
    pockets.sort(key=lambda d: d["prob"], reverse=True)
    for i, p in enumerate(pockets):
        p["rank"] = i
    return pockets

def predict_docking_box(pdb_path: str, pdb_id: str,
                        chain: Optional[str], ligand_pdb: Optional[str],
                        p2rank_only: bool = True, pocket_index: int = 0,
                        threads: int = 4) -> Tuple[float, float, float, float]:
    if available_p2rank():
        pockets = run_p2rank_list(pdb_path, pdb_id, threads=threads, out_root="p2rank_out", visuals=False)
        if pockets:
            idx = max(0, min(pocket_index, len(pockets)-1))
            sel = pockets[idx]
            cx, cy, cz, box = sel["cx"], sel["cy"], sel["cz"], sel["size"]
            box = max(BOX_MIN, min(MAX_BOX_SIZE, float(box)))
            return cx, cy, cz, box
        if p2rank_only:
            raise RuntimeError("P2Rank returned no pockets.")
    else:
        if p2rank_only:
            raise RuntimeError("P2Rank is not available. Set $P2RANK_PATH to the P2Rank folder (containing pranker/prank).")
    if ligand_pdb and not p2rank_only:
        xs, ys, zs = [], [], []
        with open(ligand_pdb, "r", errors="ignore") as fh:
            for ln in fh:
                if ln.startswith("HETATM"):
                    xs.append(float(ln[30:38]))
                    ys.append(float(ln[38:46]))
                    zs.append(float(ln[46:54]))
        if xs:
            cx = sum(xs)/len(xs)
            cy = sum(ys)/len(ys)
            cz = sum(zs)/len(zs)
            span_x = max(xs) - min(xs)
            span_y = max(ys) - min(ys)
            span_z = max(zs) - min(zs)
            box = min(MAX_BOX_SIZE, max(BOX_MIN, span_x + 2*BOX_PAD, span_y + 2*BOX_PAD, span_z + 2*BOX_PAD))
            return cx, cy, cz, box
    return grid_from_file(pdb_path, chain)

def _fetch_text(url: str, timeout=15) -> str:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "curl/8"})
    r.raise_for_status()
    return r.text

def _looks_like_pdb(txt: str) -> bool:
    return ("ATOM  " in txt) or ("HETATM" in txt) or txt.startswith(("HEADER", "TITLE "))

def _ensure_structure_pdb(pdb_id: str) -> str:
    pid = pdb_id.upper()
    out_pdb = f"{pid}.pdb"
    if os.path.exists(out_pdb):
        with open(out_pdb, "r", errors="ignore") as fh:
            if _looks_like_pdb(fh.read()):
                return out_pdb
    try:
        txt = _fetch_text(RCSB_PDB.format(id=pid))
        if _looks_like_pdb(txt):
            open(out_pdb, "w").write(txt)
            return out_pdb
    except Exception:
        pass
    try:
        txt = _fetch_text(PDBe_PDB.format(id=pid.lower()))
        if _looks_like_pdb(txt):
            open(out_pdb, "w").write(txt)
            return out_pdb
    except Exception:
        pass
    cif_path = f"{pid}.cif"
    try:
        txt = _fetch_text(RCSB_CIF.format(id=pid))
        open(cif_path, "w").write(txt)
        cp = subprocess.run(
            ["obabel", "-imcif", cif_path, "-opdb", "-O", out_pdb],
            check=True, capture_output=True, text=True
        )
        with open(out_pdb, "r", errors="ignore") as fh:
            if not _looks_like_pdb(fh.read()):
                raise RuntimeError("Converted PDB does not look valid.")
        return out_pdb
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Open Babel conversion failed:\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to obtain a usable structure for {pid}: {e}") from e

def _detect_first_chain(pdb_path: str) -> Optional[str]:
    with open(pdb_path, "r", errors="ignore") as fh:
        for ln in fh:
            if ln.startswith("ATOM"):
                return ln[21]
    return None

def _detect_ligand_any_chain(pdb_path: str) -> Optional[Tuple[str, str]]:
    with open(pdb_path, "r", errors="ignore") as fh:
        for ln in fh:
            if ln.startswith("HETATM"):
                resn = ln[17:20].strip()
                ch = ln[21]
                if resn not in SKIP_RESNS:
                    return resn, ch
    return None

def get_structure(uniprot: str, prefer_holo: bool = True) -> Tuple[str, str, Optional[str], Optional[str]]:
    token = uniprot.strip()
    if len(token) == 4 and token.isalnum():
        pdb_id = token.upper()
        pdb_path = _ensure_structure_pdb(pdb_id)
        lig = _detect_ligand_any_chain(pdb_path)
        if lig:
            lig_resn, chain = lig
        else:
            chain = _detect_first_chain(pdb_path)
            lig_resn = None
        return pdb_path, pdb_id, lig_resn, chain
    uni = token.upper()
    try:
        best_list = requests.get(PDBe_BEST.format(uni), timeout=12).json().get(uni) or []
        if not best_list:
            raise KeyError(f"No PDBe best structure mapping for {uni}")
        candidates = [entry["pdb_id"].upper() for entry in best_list]
        picked = None
        for pid in candidates if prefer_holo else candidates[:1]:
            try:
                test_pdb = _ensure_structure_pdb(pid)
                lig = _detect_ligand_any_chain(test_pdb)
                if lig:
                    lig_resn, chain = lig
                    picked = (test_pdb, pid, lig_resn, chain)
                    break
            except Exception:
                continue
        if picked is None:
            pdb_id = candidates[0]
            pdb_path = _ensure_structure_pdb(pdb_id)
            lig = _detect_ligand_any_chain(pdb_path)
            if lig:
                lig_resn, chain = lig
            else:
                chain = _detect_first_chain(pdb_path)
                lig_resn = None
            picked = (pdb_path, pdb_id, lig_resn, chain)
        return picked
    except Exception:
        pdb_path = f"{uni}_AF.pdb"
        if not os.path.exists(pdb_path):
            txt = _fetch_text(AF_URL.format(uni))
            if not _looks_like_pdb(txt):
                raise RuntimeError(f"AlphaFold PDB for {uni} looks invalid.")
            open(pdb_path, "w").write(txt)
        chain = _detect_first_chain(pdb_path)
        return pdb_path, f"AF_{uni}", None, chain

def split_receptor_ligand(pdb_in: str, lig_resn: Optional[str], chain: Optional[str]) -> Tuple[str, Optional[str]]:
    rec, lig = "receptor_raw.pdb", "ligand_ref.pdb"
    ligand_found = False
    if lig_resn:
        with open(lig, "w") as L, open(pdb_in, "r", errors="ignore") as src:
            for ln in src:
                if ln.startswith("HETATM") and ln[17:20].strip() == lig_resn and (chain is None or ln[21] == chain):
                    L.write(ln)
                    ligand_found = True
        if ligand_found:
            with open(lig, "a") as L:
                L.write("END\n")
    if not ligand_found:
        lig = None
    KEEP_HET = {
        "HEM", "HEC", "HEA", "FAD", "FMN", "NAD", "NAP", "NDP", "COA", "SAM", "SAH", "PLP", "TPP",
        "ZN", "FE", "MG", "MN", "CA", "NA", "CU", "CL"
    }
    SKIP = {
        "HOH", "WAT", "DOD", "SO4", "PO4", "PEG", "MPD", "GOL", "ETO", "NAG", "NDG",
        "VO4", "VOH", "VIO", "V", "K", "CR", "CO", "NI", "AL", "TI", "SI"
    }
    with open(rec, "w") as R, open(pdb_in, "r", errors="ignore") as src:
        for ln in src:
            if ln.startswith("ATOM"):
                if (chain is None) or (ln[21] == chain):
                    R.write(ln)
            elif ln.startswith("HETATM"):
                resn = ln[17:20].strip().upper()
                ch = ln[21]
                is_ligand = lig_resn and (resn == lig_resn) and ((chain is None) or (ch == chain))
                if is_ligand:
                    continue
                if resn in SKIP:
                    continue
                if resn in KEEP_HET:
                    if len(ln) < 80:
                        ln = ln.rstrip("\n") + " " * (80 - len(ln)) + "\n"
                    elem = ln[76:78].strip().upper()
                    if elem not in KEEP_HET and elem not in {"C","N","O","S","P","H"}:
                        ln = ln[:76] + " M" + ln[78:]
                    R.write(ln)
        R.write("END\n")
    if not os.path.getsize(rec) > 100:
        raise RuntimeError("receptor_raw.pdb is empty after split (check chain filtering).")
    return rec, lig

def grid_from_file(pdb: str, chain: Optional[str]) -> Tuple[float, float, float, float]:
    xs, ys, zs = [], [], []
    with open(pdb, "r", errors="ignore") as fh:
        for ln in fh:
            if ln.startswith("HETATM"):
                resn = ln[17:20].strip()
                if resn in SKIP_RESNS: continue
                if chain is not None and ln[21] != chain: continue
                xs.append(float(ln[30:38]))
                ys.append(float(ln[38:46]))
                zs.append(float(ln[46:54]))
    if not xs:
        with open(pdb, "r", errors="ignore") as fh:
            for ln in fh:
                if ln.startswith("ATOM") and (chain is None or ln[21] == chain):
                    xs.append(float(ln[30:38]))
                    ys.append(float(ln[38:46]))
                    zs.append(float(ln[46:54]))
    if not xs:
        with open(pdb, "r", errors="ignore") as fh:
            for ln in fh:
                if ln.startswith("ATOM"):
                    xs.append(float(ln[30:38]))
                    ys.append(float(ln[38:46]))
                    zs.append(float(ln[46:54]))
    if not xs:
        raise ValueError(f"No valid ATOM or HETATM records found in {pdb}")
    cx, cy, cz = sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs)
    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)
    span_z = max(zs) - min(zs)
    s = min(MAX_BOX_SIZE, max(BOX_MIN, span_x + 2 * BOX_PAD, span_y + 2 * BOX_PAD, span_z + 2 * BOX_PAD))
    return cx, cy, cz, s

def obabel_pdbqt(pdb: str, out: str, is_lig: bool):
    if is_lig:
        cmd = ["obabel","-ipdb",pdb,"-opdbqt","-O",out,"--partialcharge","gasteiger"]
        cp = subprocess.run(cmd, capture_output=True, text=True)
        if cp.returncode != 0 or not _nonempty(out):
            raise RuntimeError(f"Ligand PDBQT failed\nSTDOUT:\n{cp.stdout}\nSTDERR:\n{cp.stderr}")
        return
    in_heavy = _count_heavy_atoms_pdb(pdb)
    min_accept = max(10, int(0.6 * in_heavy)) if in_heavy > 0 else 10
    recipes = [
        (["obabel","-ipdb",pdb,"-opdbqt","-O",out,"-p","7.4","--partialcharge","gasteiger","-xh","-xr"], "gasteiger_pH7.4"),
        (["obabel","-ipdb",pdb,"-opdbqt","-O",out,"--partialcharge","gasteiger","-xh","-xr"],                 "gasteiger"),
        (["obabel","-ipdb",pdb,"-opdbqt","-O",out,"-xh","-xr"],                                               "no_charges"),
        (["obabel","-ipdb",pdb,"-opdbqt","-O",out],                                                            "minimal"),
    ]
    last = None
    for cmd, tag in recipes:
        cp = subprocess.run(cmd, capture_output=True, text=True)
        last = cp
        out_heavy = _count_heavy_atoms_pdbqt(out)
        if _nonempty(out) and out_heavy >= min_accept:
            return
    raise RuntimeError(
        f"Receptor PDBQT failed for {pdb}\n"
        f"Expected >= {min_accept} heavy atoms (input had ~{in_heavy}); "
        f"got {_count_heavy_atoms_pdbqt(out)}.\n"
        f"STDOUT:\n{(last.stdout if last else '')}\nSTDERR:\n{(last.stderr if last else '')}"
    )

def replace_boron_with_carbon(mol: Chem.Mol) -> Tuple[Chem.Mol, bool]:
    replaced = False
    rw = Chem.RWMol(mol)
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() == 5:
            atom.SetAtomicNum(6)
            replaced = True
    mol2 = rw.GetMol()
    try:
        Chem.SanitizeMol(
            mol2,
            Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
        )
        Chem.AssignStereochemistry(mol2, cleanIt=True, force=True)
    except Exception:
        Chem.SanitizeMol(mol2, Chem.SanitizeFlags.SANITIZE_NONE)
    return mol2, replaced

def build_3d_minimize(mol: Chem.Mol, seed: int = 42) -> Chem.Mol:
    molH = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    AllChem.EmbedMolecule(molH, params)
    try:
        AllChem.MMFFOptimizeMolecule(molH)
    except Exception:
        AllChem.UFFOptimizeMolecule(molH)
    return molH

def smiles_to_pdbqt(smiles: str, note_path: str = "ligand_prep.json") -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES.")
    mol, boron_replaced = replace_boron_with_carbon(mol)
    mol3d = build_3d_minimize(mol, seed=42)
    sdf_tmp = "ligand_tmp.sdf"
    with Chem.SDWriter(sdf_tmp) as w:
        if mol3d.GetNumConformers() == 0:
            raise RuntimeError("3D embedding failed (no conformer).")
        mol3d.SetProp("_Name", "ligand_after_B_to_C" if boron_replaced else "ligand_original")
        w.write(mol3d)
    try:
        cp = subprocess.run(
            ["obabel", "-isdf", sdf_tmp, "-opdbqt", "-O", "ligand_gen.pdbqt", "--partialcharge", "gasteiger"],
            check=True, capture_output=True, text=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Open Babel failed converting SDF to PDBQT\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}"
        ) from e
    if not _nonempty("ligand_gen.pdbqt", min_bytes=200):
        raise RuntimeError("Generated PDBQT is empty; check ligand preparation.")
    try:
        note = {
            "input_smiles": smiles,
            "boron_replaced_with_carbon": bool(boron_replaced),
            "rdkit_version": tool_versions().get("rdkit", "unknown"),
            "obabel_version": tool_versions().get("obabel", "unknown")
        }
        write_manifest(note_path, note)
    except Exception:
        pass
    return "ligand_gen.pdbqt"

def run_vina(rec_pqt, lig_pqt, cx, cy, cz, box) -> float:
    cmd = [
        "vina",
        "--receptor", rec_pqt,
        "--ligand",   lig_pqt,
        "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
        "--size_x",   str(box), "--size_y",   str(box), "--size_z",   str(box),
        "--exhaustiveness", str(EXH),
        "--num_modes", "20",
        "--energy_range", "4",
        "--seed", "42",
        "--cpu", "1",
        "--out", "pose_best.pdbqt",
        "--log", "vina.log",
    ]
    try:
        cp = subprocess.run(cmd, check=True, capture_output=True, text=True)
        out_all = (cp.stdout or "") + "\n" + (cp.stderr or "")
        for ln in out_all.splitlines():
            if ln.strip().startswith("1 "):
                parts = ln.split()
                if len(parts) >= 2:
                    return float(parts[1])
        score = vina_parse_top_score("vina.log")
        if score is not None:
            return score
        raise RuntimeError("Could not parse Vina output or log for top score.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Vina failed.\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}") from e

def save_vina_box(center: Tuple[float, float, float], size: float,
                  cfg_path: str = "vina_box.txt", json_path: str = "vina_box.json"):
    cx, cy, cz = center
    with open(cfg_path, "w") as f:
        f.write(
            f"center_x = {cx:.3f}\n"
            f"center_y = {cy:.3f}\n"
            f"center_z = {cz:.3f}\n"
            f"size_x   = {size:.3f}\n"
            f"size_y   = {size:.3f}\n"
            f"size_z   = {size:.3f}\n"
        )
    try:
        write_manifest(json_path, {"center": [cx, cy, cz], "size": size})
    except Exception:
        pass

def dock_with_vina(uniprot: str,
                   smiles: str = "",
                   k: int = 3,
                   p2rank_only: bool = True,
                   threads: int = 4):
    pdb_path, pdb_id, lig_resn, chain = get_structure(uniprot)
    rec_pdb, lig_pdb = split_receptor_ligand(pdb_path, lig_resn, chain)
    tmp = rec_pdb
    out1 = "receptor_fix_fad.pdb"
    if fix_pdb_fad_elements(tmp, out1):
        tmp = out1
    out2 = "receptor_fix_hem.pdb"
    if fix_pdb_heme_elements(tmp, out2):
        tmp = out2
    rec_pdb = tmp
    if lig_pdb is None and smiles == "":
        raise SystemExit("No ligand detected; please supply a SMILES string.")
    if not available_p2rank():
        if p2rank_only:
            raise RuntimeError("P2Rank is not available. Set $P2RANK_PATH.")
    pockets = run_p2rank_list(pdb_path, pdb_id, threads=threads, out_root="p2rank_out", visuals=False)
    if not pockets:
        if p2rank_only:
            raise RuntimeError("P2Rank returned no pockets.")
    obabel_pdbqt(rec_pdb, "receptor.pdbqt", is_lig=False)
    if smiles:
        lig_pqt = smiles_to_pdbqt(smiles)
    else:
        obabel_pdbqt(lig_pdb, "ligand_ref.pdbqt", is_lig=True)
        lig_pqt = "ligand_ref.pdbqt"
    best = None
    limit = min(k, len(pockets))
    for i in range(limit):
        p = pockets[i]
        cx, cy, cz = p["cx"], p["cy"], p["cz"]
        box = max(BOX_MIN, min(MAX_BOX_SIZE, float(p["size"])))
        score = run_vina("receptor.pdbqt", lig_pqt, cx, cy, cz, box)
        pose_out = f"pose_pocket{p['rank']}.pdbqt"
        log_out  = f"vina_pocket{p['rank']}.log"
        try:
            if os.path.exists("pose_best.pdbqt"):
                os.replace("pose_best.pdbqt", pose_out)
            if os.path.exists("vina.log"):
                os.replace("vina.log", log_out)
        except Exception:
            pass
        if (best is None) or (score <= best["score"]):
            best = {"score": score, "center": (cx, cy, cz), "box": box,
                    "pose": pose_out, "log": log_out}
    if best is None:
        raise RuntimeError("No successful docking among tried pockets.")
    shutil.copyfile(best["pose"], "pose_best.pdbqt")
    shutil.copyfile(best["log"], "vina.log")
    save_vina_box(best["center"], best["box"])
    return best["score"], best["center"], best["box"]
