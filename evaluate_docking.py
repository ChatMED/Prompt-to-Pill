import os
import json
import math
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Set
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolAlign, AdjustQueryParameters, AdjustQueryProperties
import docking_module as dm

def _relaxed_full_mapping(ref: Chem.Mol, prb: Chem.Mol):
    if ref.GetNumAtoms() != prb.GetNumAtoms():
        return None
    params = AdjustQueryParameters()
    params.makeBondsGeneric = True
    params.aromatizeIfPossible = False
    params.makeAtomsGeneric = False
    ref_q = AdjustQueryProperties(ref, params)
    match = prb.GetSubstructMatch(ref_q, useChirality=False, useQueryQueryMatches=True)
    if match and len(match) == ref.GetNumAtoms():
        return list(zip(match, range(ref.GetNumAtoms())))
    return None

def _nonempty(path: str, min_bytes: int = 200) -> bool:
    try:
        return Path(path).exists() and Path(path).stat().st_size >= min_bytes
    except Exception:
        return False

def _mol2_to_pdb(mol2_path: str, pdb_path: str):
    cmd = ["obabel", "-imol2", mol2_path, "-opdb", "-O", pdb_path]
    cp = subprocess.run(cmd, capture_output=True, text=True)
    if cp.returncode != 0 or not _nonempty(pdb_path):
        raise RuntimeError(f"MOL2 to PDB conversion failed\nSTDOUT:\n{cp.stdout}\nSTDERR:\n{cp.stderr}")

def _pdbqt_to_sdf(pdbqt_path: str, sdf_path: str):
    cmd = ["obabel", "-ipdbqt", pdbqt_path, "-osdf", "-O", sdf_path, "-d"]
    cp = subprocess.run(cmd, capture_output=True, text=True)
    if cp.returncode != 0 or not _nonempty(sdf_path):
        raise RuntimeError(f"PDBQT to SDF conversion failed\nSTDOUT:\n{cp.stdout}\nSTDERR:\n{cp.stderr}")

def _rmsd_heavy_atoms(ref_mol_path: str, docked_mol_path: str) -> Optional[float]:
    ref_mol = _load_mol_via_sdf(ref_mol_path)
    docked_mol = _load_mol_via_sdf(docked_mol_path)
    if not ref_mol or not docked_mol:
        return None
    if ref_mol.GetNumAtoms() != docked_mol.GetNumAtoms():
        return None
    try:
        return rdMolAlign.GetBestRMS(docked_mol, ref_mol)
    except Exception:
        amap = _relaxed_full_mapping(ref_mol, docked_mol)
        if not amap:
            return None
        try:
            return float(rdMolAlign.AlignMol(docked_mol, ref_mol, atomMap=amap))
        except Exception:
            return None

def _mol_to_sdf_via_obabel(mol_path: str, sdf_path: str):
    first = ""
    with open(mol_path, "r", encoding="utf-8", errors="ignore") as fh:
        for _ in range(5):
            line = fh.readline()
            if not line:
                break
            first += line
    in_fmt = "mol2" if "@<TRIPOS>MOLECULE" in first or first.strip().startswith("@<TRIPOS>") else "mol"
    cmd = ["obabel", f"-i{in_fmt}", mol_path, "-osdf", "-O", sdf_path, "-d"]
    cp = subprocess.run(cmd, capture_output=True, text=True)
    if cp.returncode != 0 or not _nonempty(sdf_path, min_bytes=10):
        raise RuntimeError(f"{in_fmt.upper()} to SDF conversion failed\nSTDOUT:\n{cp.stdout}\nSTDERR:\n{cp.stderr}")

def _load_mol_via_sdf(mol_path: str):
    tmp_sdf = Path(mol_path).with_suffix(".tmp.sdf")
    try:
        first = ""
        with open(mol_path, "r", encoding="utf-8", errors="ignore") as fh:
            for _ in range(5):
                ln = fh.readline()
                if not ln:
                    break
                first += ln
        in_fmt = "mol2" if "@<TRIPOS>MOLECULE" in first or first.strip().startswith("@<TRIPOS>") else "mol"
        cmd = ["obabel", f"-i{in_fmt}", str(mol_path), "-osdf", "-O", str(tmp_sdf)]
        cp = subprocess.run(cmd, capture_output=True, text=True)
        if cp.returncode != 0 or not _nonempty(tmp_sdf, min_bytes=10):
            print(f"[lig-load] OBabel conversion FAILED for {mol_path}")
            print(f"[lig-load] cmd: {' '.join(cmd)}")
            print(f"[lig-load] stdout:\n{cp.stdout}")
            print(f"[lig-load] stderr:\n{cp.stderr}")
        else:
            mol = None
            try:
                suppl = Chem.SDMolSupplier(str(tmp_sdf), removeHs=True, sanitize=False)
                for m in suppl:
                    if m is not None:
                        mol = m
                        break
            except Exception as e:
                print(f"[lig-load] RDKit SDMolSupplier error for {mol_path}: {e}")
                mol = None
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol)
                except Exception:
                    try:
                        Chem.SanitizeMol(
                            mol,
                            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
                        )
                    except Exception:
                        pass
                return mol
            else:
                print(f"[lig-load] RDKit got None from OBabel SDF for {mol_path}. "
                      f"SDF size={tmp_sdf.stat().st_size if tmp_sdf.exists() else 0} bytes")
    finally:
        try:
            tmp_sdf.unlink(missing_ok=True)
        except Exception:
            pass
    try:
        with open(mol_path, "r", encoding="utf-8", errors="ignore") as fh:
            lines = fh.readlines()
        while lines and not lines[-1].strip():
            lines.pop()
        if len(lines) >= 4 and ("V2000" not in lines[3] and "V3000" not in lines[3]):
            lines[3] = lines[3].rstrip("\n").rstrip() + " V2000\n"
        if not any(l.strip() == "M  END" for l in lines):
            lines.append("M  END\n")
        block = "".join(lines)
        mol = Chem.MolFromMolBlock(block, sanitize=False, removeHs=True, strictParsing=False)
        if mol is None:
            print(f"[lig-load] Repair fallback still returned None for {mol_path}")
            return None
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            try:
                Chem.SanitizeMol(
                    mol,
                    sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
                )
            except Exception:
                pass
        print(f"[lig-load] Used repair fallback for {mol_path}")
        return mol
    except Exception as e:
        print(f"[lig-load] Repair fallback error for {mol_path}: {e}")
        return None

def _mol_to_pdb_via_obabel(mol_path: str, pdb_path: str):
    first = ""
    with open(mol_path, "r", encoding="utf-8", errors="ignore") as fh:
        for _ in range(5):
            line = fh.readline()
            if not line:
                break
            first += line
    in_fmt = "mol2" if "@<TRIPOS>MOLECULE" in first or first.strip().startswith("@<TRIPOS>") else "mol"
    cmd = ["obabel", f"-i{in_fmt}", mol_path, "-opdb", "-O", pdb_path]
    cp = subprocess.run(cmd, capture_output=True, text=True)
    if cp.returncode != 0 or not _nonempty(pdb_path, min_bytes=10):
        raise RuntimeError(f"{in_fmt.upper()} to PDB conversion failed\nSTDOUT:\n{cp.stdout}\nSTDERR:\n{cp.stderr}")

def _get_ligand_proximal_residues(receptor_pdb: str, ligand_mol: str, distance_cutoff: float = 5.0) -> Set[str]:
    RDLogger.DisableLog('rdApp.*')
    try:
        ligand = _load_mol_via_sdf(ligand_mol)
        if not ligand or ligand.GetNumAtoms() == 0:
            print(f"Warning: Failed to parse ligand via OBabel/RDKit: {ligand_mol}")
            return set()
    except Exception as e:
        print(f"Error parsing ligand MOL file {ligand_mol}: {e}")
        return set()
    conf = ligand.GetConformer()
    lig_coords = [(conf.GetAtomPosition(i).x,
                   conf.GetAtomPosition(i).y,
                   conf.GetAtomPosition(i).z) for i in range(ligand.GetNumAtoms())]
    residues = set()
    with open(receptor_pdb, "r", encoding="utf-8", errors="ignore") as fh:
        for ln in fh:
            if ln.startswith("ATOM"):
                try:
                    x, y, z = float(ln[30:38]), float(ln[38:46]), float(ln[46:54])
                    elem = ln[76:78].strip() if len(ln) >= 78 else ln[12:16].strip()[0]
                    if elem == 'H':
                        continue
                    res_id = f"{ln[21]}_{ln[22:26].strip()}"
                    for lx, ly, lz in lig_coords:
                        if math.dist((x, y, z), (lx, ly, lz)) <= distance_cutoff:
                            residues.add(res_id)
                            break
                except (ValueError, IndexError):
                    continue
    RDLogger.EnableLog('rdApp.*')
    return residues

def _calculate_pocket_precision(predicted_resids: Set[str], true_resids: Set[str]) -> float:
    if not predicted_resids:
        return 0.0
    intersection = len(predicted_resids & true_resids)
    return intersection / len(predicted_resids)

def _is_protease_like(pdb_path: str) -> bool:
    key_residues = {631, 656, 659, 662, 666, 711}
    found = set()
    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as fh:
        for ln in fh:
            if ln.startswith("ATOM"):
                try:
                    resnum = int(ln[22:26])
                    if resnum in key_residues:
                        found.add(resnum)
                except:
                    continue
    return len(found) >= 3

def sdf_to_pdbqt(sdf_path: str, pdbqt_out: str):
    cp = dm._run(["obabel", "-isdf", sdf_path, "-opdbqt", "-O", pdbqt_out, "--partialcharge", "gasteiger"])
    if not Path(pdbqt_out).exists() or Path(pdbqt_out).stat().st_size == 0:
        raise RuntimeError(
            f"Open Babel failed converting {sdf_path} -> {pdbqt_out}\nSTDOUT:\n{cp.stdout}\nSTDERR:\n{cp.stderr}"
        )

def evaluate_redocking_case(case_name: str, receptor_in: str, ligand_in: str, out_dir: str,
                            use_p2rank: bool = True, exhaustiveness: int = 4) -> Dict:
    start_time = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(out_dir)
    receptor_pdb = "receptor.pdb"
    _mol2_to_pdb(receptor_in, receptor_pdb)
    pdb_id = case_name
    receptor_fixed_pdb = "receptor_fixed.pdb"
    dm.fix_pdb_heme_elements(receptor_pdb, receptor_fixed_pdb)
    receptor_fixed2_pdb = "receptor_fixed2.pdb"
    dm.fix_pdb_fad_elements(receptor_fixed_pdb, receptor_fixed2_pdb)
    receptor_pdb = receptor_fixed2_pdb
    lig_sdf = "ligand_ref.sdf"
    _mol_to_sdf_via_obabel(ligand_in, lig_sdf)
    dm.obabel_pdbqt(receptor_pdb, "receptor.pdbqt", is_lig=False)
    sdf_to_pdbqt(lig_sdf, "ligand_ref.pdbqt")
    cx, cy, cz, box = None, None, None, None
    pocket_resids = set()
    if use_p2rank:
        pockets = dm.run_p2rank_list(pdb_path=receptor_pdb, pdb_id=pdb_id, threads=4, out_root="p2rank")
        if not pockets:
            raise RuntimeError("P2Rank returned no pockets")
        top_pocket = pockets[0]
        cx, cy, cz, box = top_pocket["cx"], top_pocket["cy"], top_pocket["cz"], top_pocket["size"]
        pocket_resids = set(top_pocket["residue_ids"].split())
    else:
        cx, cy, cz, box = dm.predict_docking_box(
            receptor_pdb, pdb_id, chain=None, ligand_pdb=None, p2rank_only=False
        )
        pocket_resids = set()
    box = max(dm.BOX_MIN, min(dm.MAX_BOX_SIZE, float(box)))
    old_exh = dm.EXH
    dm.EXH = exhaustiveness
    score = dm.run_vina("receptor.pdbqt", "ligand_ref.pdbqt", cx, cy, cz, box)
    dm.EXH = old_exh
    docked_sdf = "pose_best.sdf"
    _pdbqt_to_sdf("pose_best.pdbqt", docked_sdf)
    rmsd = _rmsd_heavy_atoms(lig_sdf, docked_sdf)
    true_resids = _get_ligand_proximal_residues(receptor_pdb, ligand_in)
    pocket_precision = _calculate_pocket_precision(pocket_resids, true_resids)
    tools = dm.tool_versions()
    runtime = round(time.time() - start_time, 3)
    result = {
        "case": case_name,
        "score": score,
        "rmsd": rmsd,
        "center": (round(cx, 3), round(cy, 3), round(cz, 3)),
        "size": round(box, 3),
        "runtime_sec": runtime,
        "pocket_precision": pocket_precision,
        "tools": tools
    }
    dm.write_manifest(str(out_dir / "run_manifest.json"), result)
    return result

def evaluate_astex_dataset(dataset_path: str, out_dir: str, exhaustiveness: int = 4) -> Dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {"complexes": [], "summary": {}}
    total_start = time.time()
    failures = []
    success_count = 0
    rmsd_values = []
    runtimes = []
    pocket_accuracies = []
    dataset_path = Path(dataset_path)
    complex_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    if len(complex_dirs) == 0:
        raise ValueError(f"No directories found in {dataset_path}")
    for idx, complex_dir in enumerate(complex_dirs, 1):
        case_name = complex_dir.name
        print(f"Processing {case_name} ({idx}/{len(complex_dirs)})...")
        complex_out = out_dir / case_name
        complex_out.mkdir(exist_ok=True)
        ligand_in = complex_dir / "ligand.mol"
        receptor_in = complex_dir / "protein.mol2"
        if not (ligand_in.exists() and receptor_in.exists()):
            failures.append({"case": case_name, "error": "Missing ligand.mol or protein.mol2"})
            continue
        try:
            result = evaluate_redocking_case(
                case_name=case_name,
                receptor_in=str(receptor_in),
                ligand_in=str(ligand_in),
                out_dir=str(complex_out),
                use_p2rank=True,
                exhaustiveness=exhaustiveness
            )
            complex_result = {
                "case": case_name,
                "score": result["score"],
                "rmsd": result["rmsd"],
                "runtime_sec": result["runtime_sec"],
                "pocket_precision": result["pocket_precision"],
                "center": result["center"],
                "box_size": result["size"],
                "tools": result["tools"]
            }
            results["complexes"].append(complex_result)
            if result["rmsd"] is not None and result["rmsd"] < 2.0:
                success_count += 1
            if result["rmsd"] is not None:
                rmsd_values.append(result["rmsd"])
            runtimes.append(result["runtime_sec"])
            pocket_accuracies.append(result["pocket_precision"])
        except Exception as e:
            failures.append({"case": case_name, "error": str(e)})
            continue
    results["summary"] = {
        "total_complexes": len(complex_dirs),
        "successful": len(results["complexes"]),
        "failures": len(failures),
        "failure_details": failures,
        "success_rate_rmsd_2A": (success_count / len(results["complexes"]) * 100) if results["complexes"] else 0.0,
        "mean_rmsd": (sum(rmsd_values) / len(rmsd_values)) if rmsd_values else None,
        "mean_runtime_sec": (sum(runtimes) / len(runtimes)) if runtimes else None,
        "mean_pocket_precision": (sum(pocket_accuracies) / len(pocket_accuracies)) if pocket_accuracies else None,
        "total_time_sec": round(time.time() - total_start, 3)
    }
    Path(out_dir / "astex_evaluation.json").write_text(json.dumps(results, indent=2))
    print(f"Evaluation complete. Results saved to {out_dir}/astex_evaluation.json")
    print(f"Summary: Success Rate (RMSD < 2Å): {results['summary']['success_rate_rmsd_2A']:.2f}%")
    print(f"Mean RMSD: {results['summary']['mean_rmsd']:.2f} Å" if results['summary']['mean_rmsd'] else "N/A")
    print(f"Mean Pocket Precision: {results['summary']['mean_pocket_precision']:.2f}" if results['summary'][
        'mean_pocket_precision'] else "N/A")
    print(f"Total Time: {results['summary']['total_time_sec']:.2f} sec")
    print(f"Failures: {len(failures)}/{len(complex_dirs)}")
    return results

if __name__ == "__main__":
    dataset_path = r"path\to\astex_diverse_set"
    out_dir = r"path\to\out_dir"
    results = evaluate_astex_dataset(dataset_path, out_dir, exhaustiveness=4)
