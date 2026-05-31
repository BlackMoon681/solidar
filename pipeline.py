"""
pipeline.py  —  PPP-Risk Platform Refresh Pipeline
====================================================
Orchestrates the four stages that need to run whenever new data
arrives in the database (new Joget project entries):

  Stage 1 — ETL
      Detects new rows in the raw Joget tables, re-runs the ETL
      and updates the cleaned tables used by Superset and the model.

  Stage 2 — Chatbot index
      Rebuilds the FAISS vector index if the document store changed
      (new files added to chatbot/documents.jsonl).

  Stage 3 — Model retraining  [threshold-gated]
      The current model is trained on synthetic data and does NOT
      need retraining on every ETL run.  This stage activates only
      when >= MIN_REAL_ROWS new labeled rows accumulate in
      app_fd_ppp_clean AND enough positive cases exist for CV.
      Until then it is skipped — the synthetic model remains in use.

  Stage 4 — Figure regeneration  [optional]
      Regenerates the 9 thesis figures (img/chapter04/).
      Off by default; enable with REGENERATE_FIGURES=True.

Usage
-----
Standalone:
    python pipeline.py               # run all stages once
    python pipeline.py --stages etl chatbot   # run specific stages

Inside FastAPI (imported by api.py):
    from pipeline import run_pipeline, get_status
"""

import os, sys, time, json, logging, hashlib, argparse
from datetime import datetime, timezone
from pathlib import Path

# Force UTF-8 in all child processes (avoids cp1252 errors on Windows)
_UTF8_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).parent
STATE_FILE   = ROOT / "pipeline_state.json"
LOG_FILE     = ROOT / "pipeline.log"
ETL_MAIN     = ROOT / "etl" / "main.py"
BUILD_INDEX  = ROOT / "chatbot" / "build_index.py"
TRAIN_PY     = ROOT / "ml" / "train.py"
GEN_FIG_PY   = ROOT / "generate_figures.py"
DOCS_JSONL   = ROOT / "chatbot" / "data" / "documents.jsonl"

sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MIN_REAL_ROWS      = 150   # minimum rows in app_fd_ppp_clean before real-data retrain
MIN_POS_FOR_TRAIN  = 30    # minimum positive labels (risque_depassement=1)
REGENERATE_FIGURES = False  # set True to rebuild thesis figures each run

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("pipeline")

# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------
def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def _save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------
def _get_engine():
    from core.config import Config
    from sqlalchemy import create_engine
    return create_engine(Config.SQLALCHEMY_DATABASE_URI)

def _db_row_count(table: str) -> int:
    import sqlalchemy as sa
    engine = _get_engine()
    with engine.connect() as conn:
        result = conn.execute(sa.text(f"SELECT COUNT(*) FROM {table}"))
        return result.scalar()

def _db_positive_count(table: str, col: str = "risque_depassement") -> int:
    import sqlalchemy as sa
    engine = _get_engine()
    with engine.connect() as conn:
        result = conn.execute(
            sa.text(f"SELECT COUNT(*) FROM {table} WHERE {col} = 1")
        )
        return result.scalar()

# ---------------------------------------------------------------------------
# Stage 1 — ETL
# ---------------------------------------------------------------------------
RAW_TABLES = ["app_fd_base_ppp", "app_fd_base_pgd", "app_fd_base_pgd_etude"]

def stage_etl(state: dict, force: bool = False) -> dict:
    log.info("--- Stage 1 : ETL ---")
    result = {"ran": False, "new_rows": {}, "error": None}

    try:
        counts = {t: _db_row_count(t) for t in RAW_TABLES}
    except Exception as e:
        result["error"] = f"DB error: {e}"
        log.warning(f"  ETL skipped — cannot reach DB: {e}")
        return result

    prev_counts = state.get("raw_row_counts", {})
    new_rows    = {t: counts[t] - prev_counts.get(t, 0) for t in RAW_TABLES}
    result["new_rows"] = new_rows
    total_new = sum(new_rows.values())

    for t in RAW_TABLES:
        log.info(f"  {t}: {counts[t]} rows (prev={prev_counts.get(t, 0)}, new={new_rows[t]})")

    if not force and total_new <= 0:
        log.info("  No new rows in any source table. ETL skipped.")
        return result

    log.info(f"  {total_new} new row(s) detected across source tables. Running ETL...")

    import subprocess
    proc = subprocess.run(
        [sys.executable, "-m", "etl.main"],
        capture_output=True, text=True, timeout=300,
        cwd=str(ROOT), env=_UTF8_ENV,
    )
    if proc.returncode == 0:
        log.info("  ETL completed successfully.")
        state["raw_row_counts"] = counts
        state["last_etl"] = datetime.now(timezone.utc).isoformat()
        result["ran"] = True
    else:
        err = proc.stderr[-500:] if proc.stderr else "unknown error"
        result["error"] = err
        log.error(f"  ETL FAILED:\n{err}")

    return result

# ---------------------------------------------------------------------------
# Stage 2 — Chatbot FAISS index rebuild
# ---------------------------------------------------------------------------
def _docs_hash() -> str:
    if not DOCS_JSONL.exists():
        return ""
    return hashlib.md5(DOCS_JSONL.read_bytes()).hexdigest()

def stage_chatbot(state: dict, force: bool = False) -> dict:
    log.info("--- Stage 2 : Chatbot index ---")
    result = {"ran": False, "error": None}

    current_hash = _docs_hash()
    prev_hash    = state.get("docs_hash", "")

    if not force and current_hash == prev_hash:
        log.info("  Document store unchanged. Index rebuild skipped.")
        return result

    if not BUILD_INDEX.exists():
        log.warning(f"  build_index.py not found at {BUILD_INDEX}. Skipped.")
        return result

    log.info("  Document store changed — rebuilding FAISS index...")
    import subprocess
    proc = subprocess.run(
        [sys.executable, str(BUILD_INDEX)],
        capture_output=True, text=True, timeout=180,
        cwd=str(ROOT), env=_UTF8_ENV,
    )
    if proc.returncode == 0:
        log.info("  FAISS index rebuilt successfully.")
        state["docs_hash"]    = current_hash
        state["last_chatbot"] = datetime.now(timezone.utc).isoformat()
        result["ran"] = True
    else:
        err = proc.stderr[-500:] if proc.stderr else "unknown error"
        result["error"] = err
        log.error(f"  Index rebuild FAILED:\n{err}")

    return result

# ---------------------------------------------------------------------------
# Stage 3 — ML model retraining (threshold-gated)
# ---------------------------------------------------------------------------
def stage_model(state: dict, force: bool = False) -> dict:
    log.info("--- Stage 3 : Model retraining ---")
    result = {"ran": False, "skipped_reason": "", "error": None}

    try:
        clean_count = _db_row_count("app_fd_ppp_clean")
        pos_count   = _db_positive_count("app_fd_ppp_clean")
    except Exception as e:
        result["skipped_reason"] = f"DB unreachable: {e}"
        log.warning(f"  Model check skipped — {e}")
        return result

    log.info(f"  app_fd_ppp_clean: {clean_count} rows, {pos_count} positives")

    if not force:
        if clean_count < MIN_REAL_ROWS:
            msg = (f"  Only {clean_count} real rows "
                   f"(need >= {MIN_REAL_ROWS}). Synthetic model kept.")
            result["skipped_reason"] = msg
            log.info(msg)
            return result

        if pos_count < MIN_POS_FOR_TRAIN:
            msg = (f"  Only {pos_count} positive labels "
                   f"(need >= {MIN_POS_FOR_TRAIN}). Synthetic model kept.")
            result["skipped_reason"] = msg
            log.info(msg)
            return result

        prev_clean = state.get("clean_row_count_at_last_train", 0)
        if clean_count - prev_clean < 20:
            msg = f"  Only {clean_count - prev_clean} new clean rows since last train. Skipped."
            result["skipped_reason"] = msg
            log.info(msg)
            return result

    log.info(f"  Threshold met ({clean_count} rows, {pos_count} positives). "
             "Re-running ml/train.py on real data...")

    from ml.versioning import (
        current_production_auc, current_promoted_version,
        latest_entry, mark_promoted, promote, reject, AUC_GATE_DELTA,
    )

    prev_auc = current_production_auc()
    prev_ver = current_promoted_version()
    log.info(f"  Current production AUC: {prev_auc} (version {prev_ver})")

    import subprocess
    proc = subprocess.run(
        [sys.executable, str(TRAIN_PY), "--source", "real"],
        capture_output=True, text=True, timeout=600,
        cwd=str(ROOT), env=_UTF8_ENV,
    )

    if proc.returncode != 0:
        err = proc.stderr[-500:] if proc.stderr else "unknown error"
        result["error"] = err
        log.error(f"  Training FAILED:\n{err}")
        return result

    new_entry = latest_entry()
    new_auc   = new_entry["holdout_auc"] if new_entry else None
    new_ver   = new_entry["version"]     if new_entry else None
    log.info(f"  New model AUC: {new_auc} (version {new_ver})")

    # AUC gate: reject if new model degrades beyond tolerance
    if prev_auc is not None and new_auc is not None:
        threshold = prev_auc + AUC_GATE_DELTA
        if new_auc < threshold:
            reason = (f"AUC gate failed: {new_auc:.4f} < {prev_auc:.4f}"
                      f" (min allowed: {threshold:.4f})")
            log.warning(f"  {reason}")
            log.warning("  Rolling back to previous production version...")

            if prev_ver is not None:
                promote(prev_ver)
                log.info(f"  Rolled back to version {prev_ver}")
            if new_ver is not None:
                reject(new_ver, reason)

            result["skipped_reason"] = reason
            result["new_version"]    = new_ver
            return result

    log.info(f"  AUC gate passed ({prev_auc} -> {new_auc}). Promoting version {new_ver}.")
    if new_ver is not None:
        mark_promoted(new_ver)

    state["clean_row_count_at_last_train"] = clean_count
    state["last_model_train"] = datetime.now(timezone.utc).isoformat()
    result["ran"]         = True
    result["new_version"] = new_ver
    result["prev_auc"]    = prev_auc
    result["new_auc"]     = new_auc
    return result

# ---------------------------------------------------------------------------
# Stage 4 — Figure regeneration (optional)
# ---------------------------------------------------------------------------
def stage_figures(state: dict) -> dict:
    log.info("--- Stage 4 : Figure regeneration ---")
    result = {"ran": False, "error": None}

    if not REGENERATE_FIGURES:
        log.info("  REGENERATE_FIGURES=False. Skipped.")
        return result

    import subprocess
    proc = subprocess.run(
        [sys.executable, str(GEN_FIG_PY)],
        capture_output=True, text=True, timeout=300
    )
    if proc.returncode == 0:
        log.info("  Figures regenerated.")
        state["last_figures"] = datetime.now(timezone.utc).isoformat()
        result["ran"] = True
    else:
        err = proc.stderr[-500:] if proc.stderr else "unknown error"
        result["error"] = err
        log.error(f"  Figure generation FAILED:\n{err}")

    return result

# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
def run_pipeline(stages: list = None, force: bool = False) -> dict:
    """
    Run the pipeline. stages: list of stage names to run, or None for all.
    Returns a summary dict.
    """
    all_stages = ["etl", "chatbot", "model", "figures"]
    run_stages = stages or all_stages

    log.info("=" * 60)
    log.info(f"Pipeline started  (stages={run_stages}, force={force})")
    log.info("=" * 60)

    state   = _load_state()
    started = datetime.now(timezone.utc)
    summary = {"started": started.isoformat(), "stages": {}}

    if "etl" in run_stages:
        summary["stages"]["etl"] = stage_etl(state, force=force)

    if "chatbot" in run_stages:
        summary["stages"]["chatbot"] = stage_chatbot(state, force=force)

    if "model" in run_stages:
        summary["stages"]["model"] = stage_model(state, force=force)

    if "figures" in run_stages:
        summary["stages"]["figures"] = stage_figures(state)

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    summary["elapsed_seconds"] = round(elapsed, 1)
    summary["finished"] = datetime.now(timezone.utc).isoformat()

    state["last_run"]      = summary["finished"]
    state["last_run_summary"] = {
        k: {"ran": v.get("ran"), "error": v.get("error")}
        for k, v in summary["stages"].items()
    }
    _save_state(state)

    errors = [k for k, v in summary["stages"].items() if v.get("error")]
    if errors:
        log.warning(f"Pipeline finished with errors in: {errors}")
    else:
        log.info(f"Pipeline finished OK in {elapsed:.1f}s")

    return summary

# ---------------------------------------------------------------------------
# Status (for API endpoint)
# ---------------------------------------------------------------------------
def get_status() -> dict:
    state = _load_state()
    try:
        raw_counts  = {t: _db_row_count(t) for t in RAW_TABLES}
        clean_count = _db_row_count("app_fd_ppp_clean")
        pos_count   = _db_positive_count("app_fd_ppp_clean")
        db_ok = True
    except Exception as e:
        raw_counts  = {}
        clean_count = pos_count = 0
        db_ok = False

    model_path = ROOT / "ml" / "model" / "risk_model.pkl"
    metrics_path = ROOT / "ml" / "model" / "model_metrics.json"

    model_auc = None
    if metrics_path.exists():
        try:
            m = json.loads(metrics_path.read_text())
            model_auc = m.get("holdout_auc")
        except Exception:
            pass

    return {
        "db_ok":          db_ok,
        "raw_rows":       raw_counts,
        "clean_rows":     clean_count,
        "positive_labels":pos_count,
        "model_ready":    model_path.exists(),
        "model_auc":      model_auc,
        "retrain_threshold": MIN_REAL_ROWS,
        "retrain_eligible": clean_count >= MIN_REAL_ROWS and pos_count >= MIN_POS_FOR_TRAIN,
        "last_run":       state.get("last_run"),
        "last_etl":       state.get("last_etl"),
        "last_chatbot":   state.get("last_chatbot"),
        "last_model_train":state.get("last_model_train"),
        "last_run_summary":state.get("last_run_summary", {}),
        "log_tail":       _tail_log(20),
    }

def _tail_log(n: int = 20) -> list[str]:
    if not LOG_FILE.exists():
        return []
    lines = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-n:]

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPP-Risk Pipeline Runner")
    parser.add_argument(
        "--stages", nargs="+",
        choices=["etl", "chatbot", "model", "figures"],
        help="Stages to run (default: all)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Skip change-detection checks and run all stages unconditionally"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Print current pipeline status and exit"
    )
    args = parser.parse_args()

    if args.status:
        import pprint
        status = get_status()
        status.pop("log_tail", None)  # exclude log from print
        pprint.pprint(status)
    else:
        run_pipeline(stages=args.stages, force=args.force)
