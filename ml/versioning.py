"""
ml/versioning.py  —  Model version registry for PPP-Risk
"""
import json, shutil
from datetime import datetime, timezone
from pathlib import Path

BASE      = Path(__file__).parent
MODEL_DIR = BASE / "model"
VER_DIR   = MODEL_DIR / "versions"
REGISTRY  = MODEL_DIR / "model_registry.json"
PROD_PKL  = MODEL_DIR / "risk_model.pkl"
PROD_METR = MODEL_DIR / "model_metrics.json"

AUC_GATE_DELTA = -0.005   # new model AUC must be >= (prev_auc + this) to be promoted


def _load() -> list:
    if REGISTRY.exists():
        return json.loads(REGISTRY.read_text(encoding="utf-8"))
    return []


def _dump(reg: list):
    VER_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(reg, indent=2, default=str), encoding="utf-8")


def _f(metrics: dict, key: str) -> float:
    v = metrics.get(key)
    return round(float(v), 4) if v is not None else 0.0


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def save_version(metrics: dict, pkl_path, source: str = "synthetic",
                 n_real_rows: int = 0) -> dict:
    """
    Archive a trained model and register it. Returns the new registry entry.
    The model is NOT promoted to production here — caller decides.
    """
    VER_DIR.mkdir(parents=True, exist_ok=True)
    reg = _load()
    ver = max((e["version"] for e in reg), default=0) + 1
    ts  = datetime.now(timezone.utc)
    tag = f"v{ver:03d}_{ts.strftime('%Y%m%d_%H%M%S')}"

    dst_pkl  = VER_DIR / f"{tag}_risk_model.pkl"
    dst_metr = VER_DIR / f"{tag}_metrics.json"
    shutil.copy2(pkl_path, dst_pkl)
    dst_metr.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    entry = {
        "version":          ver,
        "tag":              tag,
        "timestamp":        ts.isoformat(),
        "model_file":       f"model/versions/{dst_pkl.name}",
        "metrics_file":     f"model/versions/{dst_metr.name}",
        "source":           source,
        "n_real_rows":      n_real_rows,
        "holdout_auc":      _f(metrics, "holdout_auc"),
        "holdout_f1":       _f(metrics, "holdout_f1"),
        "holdout_prec":     _f(metrics, "holdout_prec"),
        "holdout_rec":      _f(metrics, "holdout_rec"),
        "cv_auc":           _f(metrics, "cv_auc"),
        "cv_std":           _f(metrics, "cv_std"),
        "holdout_bs":       _f(metrics, "holdout_bs"),
        "holdout_mcc":      _f(metrics, "holdout_mcc"),
        "n_train":          metrics.get("n_train"),
        "promoted":         False,
        "rejected":         False,
        "rejection_reason": None,
    }
    reg.append(entry)
    _dump(reg)
    return entry


def mark_promoted(version: int):
    """Update the registry to flag a version as current production (no file copy)."""
    reg = _load()
    for e in reg:
        e["promoted"] = (e["version"] == version)
    _dump(reg)


def promote(version: int):
    """Restore a versioned model to the production files and mark it as current."""
    reg   = _load()
    entry = next((e for e in reg if e["version"] == version), None)
    if not entry:
        raise ValueError(f"Version {version} not in registry")
    shutil.copy2(BASE / entry["model_file"],  PROD_PKL)
    shutil.copy2(BASE / entry["metrics_file"], PROD_METR)
    for e in reg:
        e["promoted"] = (e["version"] == version)
    _dump(reg)


def reject(version: int, reason: str):
    """Mark a version as rejected — saved in history but never served in production."""
    reg   = _load()
    entry = next((e for e in reg if e["version"] == version), None)
    if entry:
        entry["rejected"]         = True
        entry["rejection_reason"] = reason
        _dump(reg)


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def current_production_auc() -> float | None:
    if not PROD_METR.exists():
        return None
    try:
        return float(json.loads(PROD_METR.read_text(encoding="utf-8")).get("holdout_auc") or 0)
    except Exception:
        return None


def current_promoted_version() -> int | None:
    for e in reversed(_load()):
        if e.get("promoted"):
            return e["version"]
    return None


def latest_entry() -> dict | None:
    reg = _load()
    return reg[-1] if reg else None


def get_all() -> list:
    return _load()
