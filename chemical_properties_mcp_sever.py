# chemical_properties_mcp_server.py
# MCP server exposing pKa / logD / physchem tools with SMILES-list inputs

from typing import List, Dict, Any, Optional
from functools import lru_cache
import numpy as np
import math

from mcp.server.fastmcp import FastMCP

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt

from pkapredict import predict_pKa , load_model
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
DESCRIPTOR_NAMES: List[str] = [
    'MaxAbsEStateIndex', 'MaxEStateIndex', 'MinAbsEStateIndex', 'MinEStateIndex', 'SPS',
    'NumValenceElectrons', 'MaxPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge', 'MinAbsPartialCharge',
    'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3', 'BCUT2D_MWHI', 'BCUT2D_MWLOW',
    'BCUT2D_CHGLO', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi1',
    'Chi1n', 'Chi3v', 'Chi4v', 'HallKierAlpha', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 'PEOE_VSA1',
    'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6',
    'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4',
    'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11',
    'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7',
    'SlogP_VSA8', 'TPSA', 'EState_VSA1', 'EState_VSA10', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4',
    'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'VSA_EState1',
    'VSA_EState10', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState6', 'VSA_EState7',
    'VSA_EState8', 'VSA_EState9', 'FractionCSP3', 'NHOHCount', 'NOCount', 'NumAliphaticHeterocycles',
    'NumAliphaticRings', 'NumAromaticHeterocycles', 'NumAromaticRings', 'NumBridgeheadAtoms',
    'NumHAcceptors', 'NumHeteroatoms', 'NumHeterocycles', 'NumRotatableBonds', 'NumSaturatedHeterocycles',
    'NumSaturatedRings', 'Phi', 'MolMR', 'fr_Al_COO', 'fr_ArN', 'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_OH',
    'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0',
    'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_alkyl_halide',
    'fr_allylic_oxid', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl', 'fr_azide', 'fr_azo', 'fr_ester',
    'fr_ether', 'fr_guanido', 'fr_halogen', 'fr_imidazole', 'fr_lactam', 'fr_methoxy', 'fr_nitrile',
    'fr_nitroso', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_piperdine', 'fr_pyridine', 'fr_quatN',
    'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_tetrazole', 'fr_thiazole'
]


def _mol_from_smiles(smiles: str) -> Chem.Mol:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError("Invalid SMILES")
    return m


def _rdkit_physchem(m: Chem.Mol) -> Dict[str, Any]:
    cano = Chem.MolToSmiles(m, isomericSmiles=True, canonical=True)
    logP = Crippen.MolLogP(m)
    MR = Crippen.MolMR(m)
    TPSA = rdMolDescriptors.CalcTPSA(m)
    HBD = rdMolDescriptors.CalcNumHBD(m)
    HBA = rdMolDescriptors.CalcNumHBA(m)
    RotB = rdMolDescriptors.CalcNumRotatableBonds(m)
    FractionCSP3 = rdMolDescriptors.CalcFractionCSP3(m)
    MW = Descriptors.MolWt(m)
    ExactMW = CalcExactMolWt(m)
    BertzCT = Descriptors.BertzCT(m)
    NumArom = rdMolDescriptors.CalcNumAromaticRings(m)
    NumAliph = rdMolDescriptors.CalcNumAliphaticRings(m)
    try:
        from rdkit.Chem import QED
        qed = float(QED.qed(m))
    except Exception:
        qed = float("nan")

    return {
        "canonical_smiles": cano,
        "MW": float(MW),
        "ExactMW": float(ExactMW),
        "logP": float(logP),
        "MR": float(MR),
        "TPSA": float(TPSA),
        "HBD": int(HBD),
        "HBA": int(HBA),
        "RotB": int(RotB),
        "FractionCSP3": float(FractionCSP3),
        "BertzCT": float(BertzCT),
        "NumAromaticRings": int(NumArom),
        "NumAliphaticRings": int(NumAliph),
        "QED": qed,
    }


def _logD_acid(logP: float, pKa: float, pH: float = 7.4) -> float:
    P = 10 ** logP
    denom = 1 + 10 ** (pH - pKa)
    D = P / denom
    return float(math.log10(D))


def _logD_base(logP: float, pKa: float, pH: float = 7.4) -> float:
    P = 10 ** logP
    denom = 1 + 10 ** (pKa - pH)
    D = P / denom
    return float(math.log10(D))


@lru_cache(maxsize=1)
def _cached_model():
    m = load_model()
    return m


mcp = FastMCP("chem-properties")


@mcp.tool()
def predict_pka_batch(smiles_list: List[str]) -> Dict[str, Any]:
    """
    Predict pKa for a list of SMILES using the trained LightGBM model (pkapredict).
    Returns {"preds": List[Optional[float]], "errors": Dict[index, message]}
    """
    model = _cached_model()
    preds: List[Optional[float]] = [None] * len(smiles_list)
    errors: Dict[int, str] = {}

    for i, s in enumerate(smiles_list):
        try:
            # Your library builds descriptors internally from descriptor_names
            y = predict_pKa(smiles=s, model=model, descriptor_names=DESCRIPTOR_NAMES)
            val = float(np.asarray(y).ravel()[0])
            preds[i] = val
        except Exception as e:
            errors[i] = f"{s}: {e}"
    return {"preds": preds, "errors": errors}


@mcp.tool()
def logd_acid_batch(smiles_list: List[str], pH: float = 7.4) -> Dict[str, Any]:
    model = _cached_model()
    out: List[Optional[float]] = [None] * len(smiles_list)
    errors: Dict[int, str] = {}

    for i, s in enumerate(smiles_list):
        try:
            m = _mol_from_smiles(s)
            logP = float(Crippen.MolLogP(m))
            y = predict_pKa(smiles=s, model=model, descriptor_names=DESCRIPTOR_NAMES)
            pKa = float(np.asarray(y).ravel()[0])
            out[i] = _logD_acid(logP, pKa, pH)
        except Exception as e:
            errors[i] = f"{s}: {e}"
    return {"logD": out, "errors": errors, "pH": pH}


@mcp.tool()
def logd_base_batch(smiles_list: List[str], pH: float = 7.4) -> Dict[str, Any]:
    model = _cached_model()
    out: List[Optional[float]] = [None] * len(smiles_list)
    errors: Dict[int, str] = {}

    for i, s in enumerate(smiles_list):
        try:
            m = _mol_from_smiles(s)
            logP = float(Crippen.MolLogP(m))
            y = predict_pKa(smiles=s, model=model, descriptor_names=DESCRIPTOR_NAMES)
            pKa = float(np.asarray(y).ravel()[0])
            out[i] = _logD_base(logP, pKa, pH)
        except Exception as e:
            errors[i] = f"{s}: {e}"
    return {"logD": out, "errors": errors, "pH": pH}


@mcp.tool()
def rdkit_physchem_batch(smiles_list: List[str], pH: float = 7.4) -> Dict[str, Any]:
    model = _cached_model()
    results: List[Optional[Dict[str, Any]]] = [None] * len(smiles_list)
    errors: Dict[int, str] = {}

    for i, s in enumerate(smiles_list):
        try:
            m = _mol_from_smiles(s)
            props = _rdkit_physchem(m)
            y = predict_pKa(smiles=s, model=model, descriptor_names=DESCRIPTOR_NAMES)
            pKa = float(np.asarray(y).ravel()[0])
            props["pKa"] = pKa
            # add both logDs; caller can choose which to use
            props["logD_acid"] = _logD_acid(props["logP"], pKa, pH)
            props["logD_base"] = _logD_base(props["logP"], pKa, pH)
            results[i] = props
        except Exception as e:
            errors[i] = f"{s}: {e}"
    return {"results": results, "errors": errors, "pH": pH}


@mcp.tool()
def predict_all_batch(
        smiles_list: List[str],
        is_acid: Optional[bool] = None,
        is_base: Optional[bool] = None,
        pH: float = 7.4
) -> Dict[str, Any]:
    if is_acid and is_base:
        # If both true, we still compute both logDs but label 'logD_selected' as None
        pass

    model = _cached_model()
    results: List[Optional[Dict[str, Any]]] = [None] * len(smiles_list)
    errors: Dict[int, str] = {}

    for i, s in enumerate(smiles_list):
        try:
            m = _mol_from_smiles(s)
            props = _rdkit_physchem(m)

            y = predict_pKa(smiles=s, model=model, descriptor_names=DESCRIPTOR_NAMES)
            pKa = float(np.asarray(y).ravel()[0])
            props["pKa"] = pKa

            lda = _logD_acid(props["logP"], pKa, pH)
            ldb = _logD_base(props["logP"], pKa, pH)
            props["logD_acid"] = lda
            props["logD_base"] = ldb

            selected = None
            if is_acid is True and is_base is not True:
                selected = lda
            elif is_base is True and is_acid is not True:
                selected = ldb
            elif (is_acid is False) and (is_base is False):
                # neutral assumption
                selected = props["logP"]

            props["logD_selected"] = selected
            props["pH"] = pH
            props["is_acid_hint"] = is_acid
            props["is_base_hint"] = is_base

            results[i] = props
        except Exception as e:
            errors[i] = f"{s}: {e}"
    return {"results": results, "errors": errors}


def _lipinski_violations(c: Dict[str, Any]) -> List[str]:
    v = []
    if c.get("MW", float("inf")) > 500: v.append("MW>500")
    if c.get("logP", float("inf")) > 5: v.append("logP>5")
    if c.get("HBD", float("inf")) > 5: v.append("HBD>5")
    if c.get("HBA", float("inf")) > 10: v.append("HBA>10")
    return v


def _veber_pass(c: Dict[str, Any]) -> bool:
    return (c.get("RotB", float("inf")) <= 10) and (c.get("TPSA", float("inf")) <= 140)


def _lead_rank_key(c: Dict[str, Any]):
    qed = c.get("QED", 0.0)
    logp = c.get("logP", 2.5)
    rotb = c.get("RotB", 100)
    tpsa = c.get("TPSA", 999)
    return (-qed, abs(logp - 2.5), rotb, tpsa)


@mcp.tool()
def select_leads_by_rules(results: List[Dict[str, Any]], n: int = 4) -> Dict[str, Any]:
    passed, rejected = [], []

    for c in results:
        lviol = _lipinski_violations(c)
        vpass = _veber_pass(c)

        if (len(lviol) == 0) and vpass:
            passed.append(c)
        else:
            reasons = []
            if lviol:
                reasons.append("Lipinski: " + ", ".join(lviol))
            if not vpass:
                sub = []
                if c.get("RotB", 0) > 10: sub.append("RotB>10")
                if c.get("TPSA", 0) > 140: sub.append("TPSA>140")
                if sub:
                    reasons.append("Veber: " + ", ".join(sub))
            rejected.append({"compound": c, "reason": "; ".join(reasons) or "Rule fail"})

    leads = sorted(passed, key=_lead_rank_key)[:n]

    return {
        "leads": leads,
        "rejected": rejected,
        "criteria": {
            "lipinski": "MW<=500, logP<=5, HBD<=5, HBA<=10",
            "veber": "RotB<=10, TPSA<=140"
        }
    }


@mcp.tool()
def select_leads_from_smiles(smiles_list: List[str], n: int = 5, pH: float = 7.4) -> Dict[str, Any]:
    computed = rdkit_physchem_batch(smiles_list=smiles_list, pH=pH)
    results = computed.get("results", []) if isinstance(computed, dict) else []
    selection = select_leads_by_rules(results=results, n=n)
    return {
        "pH": pH,
        "results_count": len(results),
        **selection
    }


if __name__ == "__main__":
    mcp.run()
