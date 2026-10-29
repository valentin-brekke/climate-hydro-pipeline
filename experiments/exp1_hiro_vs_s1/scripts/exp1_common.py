"""Config loading and provenance for Experiment 1 scripts.

Every output gets the config hash and git commit in its attributes
(plan §10), so a zarr can always be traced back to the settings and code
that produced it.
"""
import hashlib
import subprocess
import sys
from pathlib import Path

import yaml

EXP_DIR = Path(__file__).resolve().parents[1]
REPO = EXP_DIR.parents[1]
DEFAULT_CONFIG = EXP_DIR / "config.yaml"

# Reused processing libraries (same import style as their own run_*.py scripts).
sys.path.insert(0, str(REPO / "processing" / "catchment_weighting" / "scripts"))
sys.path.insert(0, str(REPO / "processing" / "temporal_binning" / "scripts"))


def load_config(path=DEFAULT_CONFIG):
    path = Path(path)
    raw = path.read_bytes()
    cfg = yaml.safe_load(raw)
    p = cfg["paths"]
    fill = {"data_root": p["data_root"]}
    for k, v in list(p.items()):
        if isinstance(v, str):
            p[k] = v.replace("{data_root}", fill["data_root"])
        elif isinstance(v, dict):
            p[k] = {kk: vv.replace("{data_root}", fill["data_root"]) for kk, vv in v.items()}
    cfg["_hash"] = hashlib.sha256(raw).hexdigest()[:12]
    cfg["_path"] = str(path)
    return cfg


def git_commit():
    try:
        sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", str(EXP_DIR)],
                               capture_output=True, text=True, check=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def provenance(cfg, script):
    return {"exp1_config_hash": cfg["_hash"], "exp1_git_commit": git_commit(),
            "exp1_script": Path(script).name}


def out_path(cfg, *parts):
    p = Path(cfg["paths"]["out_root"]).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
