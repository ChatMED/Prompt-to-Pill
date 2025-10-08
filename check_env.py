from __future__ import annotations
import argparse, json, os, platform, shutil, sys
from pathlib import Path

REQUIRED_PYTHON_MAJOR = 3
REQUIRED_PYTHON_MINOR = 11

REQUIRED_BINARIES = [
    ("vina", "AutoDock Vina executor"),
    ("obabel", "Open Babel converter"),
    ("java", "Java runtime (needed by P2Rank)"),
]

P2RANK_LAUNCHERS = ["pranker", "pranker.bat", "prank", "prank.bat"]

CORE_PKGS = ["torch", "transformers", "rdkit", "autogen", "autogen_agentchat", "autogen_core", "autogen_ext",
             "openai", "llama_cpp"]

def ok(x): return "\033[92mOK\033[0m"
def fail(x): return "\033[91mFAIL\033[0m"
def warn(x): return "\033[93mWARN\033[0m"

def which(name): return shutil.which(name) or ""

def check_python():
    v = sys.version_info
    good = (v.major == REQUIRED_PYTHON_MAJOR and v.minor == REQUIRED_PYTHON_MINOR)
    return {"name":"python_version","ok":good,"required":True,"found":f"{v.major}.{v.minor}.{v.micro}",
            "expected":f"{REQUIRED_PYTHON_MAJOR}.{REQUIRED_PYTHON_MINOR}.x"}

def check_os():
    return {"name":"os","ok":True,"required":False,"found":platform.platform()}

def check_bins():
    out=[]
    for b,desc in REQUIRED_BINARIES:
        p=which(b)
        out.append({"name":f"bin:{b}","ok":bool(p),"required":True,"desc":desc,"found":p or "NOT FOUND"})
    return out

def check_p2rank():
    res=[]
    path=os.environ.get("P2RANK_PATH","").strip()
    if not path:
        res.append({"name":"env:P2RANK_PATH","ok":False,"required":True,"found":"NOT SET"})
        return res
    p=Path(path)
    res.append({"name":"env:P2RANK_PATH","ok":p.exists(),"required":True,"found":str(p.resolve())})
    launcher_ok=any((p/x).exists() for x in P2RANK_LAUNCHERS)
    res.append({"name":"p2rank_launcher","ok":launcher_ok,"required":True,
                "found":", ".join(str(p/x) for x in P2RANK_LAUNCHERS)})
    # models dir optional; many bundles differ—skip strict check
    return res

def check_gpu():
    info={"name":"gpu","ok":True,"required":False,"found":"N/A"}
    try:
        import torch
        info["found"]="CUDA available: "+torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA (CPU/MPS mode)"
    except Exception as e:
        info["ok"]=False; info["found"]=f"PyTorch not importable ({e})"
    return info

def check_core_pkgs():
    res=[]
    for name in CORE_PKGS:
        try:
            mod = __import__(name)
            ver = getattr(mod,"__version__","unknown")
            res.append({"name":f"py:{name}","ok":True,"required":True,"found":ver})
        except Exception as e:
            res.append({"name":f"py:{name}","ok":False,"required":True,"found":f"IMPORT ERROR: {e}"})
    return res

def summarize(items):
    all_req_ok=True
    for x in items:
        if x.get("required") and not x.get("ok"):
            all_req_ok=False
    return all_req_ok

def pretty(items):
    print("\n=== Prompt-to-Pill Environment Check ===\n")
    for c in items:
        status = ok("") if c["ok"] else (fail("") if c.get("required") else warn(""))
        req = "(required)" if c.get("required") else "(optional)"
        desc = f" — {c['desc']}" if c.get("desc") else ""
        print(f"[{status}] {c['name']} {req}{desc}")
        if c.get("found"): print(f"        found: {c['found']}")
    print("")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--json",action="store_true")
    args=ap.parse_args()

    checks=[]
    checks.append(check_python())
    checks.append(check_os())
    checks.extend(check_bins())
    checks.extend(check_p2rank())
    checks.extend(check_core_pkgs())
    checks.append(check_gpu())

    ok_all = summarize(checks)
    if args.json:
        print(json.dumps({"ok":ok_all,"checks":checks},indent=2))
    else:
        pretty(checks)
        print("Required checks passed." if ok_all else "One or more required checks failed.")
    sys.exit(0 if ok_all else 1)

if __name__=="__main__":
    main()
