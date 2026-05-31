"""
tests/test_performance.py
--------------------------
Benchmarks de performance : inférence ML, recherche FAISS, ETL, pipeline.

Run:  python tests/test_performance.py
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).parent.parent

THRESHOLDS = {
    "inference_ms":     50,     # inférence modèle < 50 ms
    "faiss_ms":          5,     # recherche FAISS  < 5 ms
    "feature_eng_ms":   20,     # feature engineering < 20 ms
    "pipeline_status_ms": 500,  # lecture état pipeline < 500 ms
}

_results = []

def bench(name, fn, n=100, threshold_ms=None):
    # warmup
    fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)

    avg = np.mean(times)
    p95 = np.percentile(times, 95)
    p99 = np.percentile(times, 99)
    mi  = np.min(times)
    ma  = np.max(times)
    ok  = threshold_ms is None or avg < threshold_ms

    status = "\033[92mOK \033[0m" if ok else "\033[91mSLOW\033[0m"
    print(f"  [{status}]  {name:<40}  avg={avg:7.3f}ms  p95={p95:7.3f}ms  p99={p99:7.3f}ms  "
          f"(n={n}, min={mi:.3f}, max={ma:.3f})")
    _results.append({"name": name, "avg_ms": round(avg, 3), "p95_ms": round(p95, 3),
                     "p99_ms": round(p99, 3), "passed": ok})
    return {"avg": avg, "p95": p95, "p99": p99}


# ---------------------------------------------------------------------------
# Load shared resources
# ---------------------------------------------------------------------------

def _load():
    for name in ("risk_model.pkl", "budget_model_v13.pkl", "budget_model_v12.pkl"):
        p = ROOT / "ml" / "model" / name
        if p.exists():
            saved = joblib.load(p)
            return saved["model"], saved["features"], saved.get("calibrator")
    raise FileNotFoundError

model, FEATURES, CALIBRATOR = _load()
faiss_index = faiss.read_index(str(ROOT / "chatbot" / "index" / "faiss_index.bin"))
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Precompute test inputs
_BASE_ROW = {f: 0.0 for f in FEATURES}
_BASE_ROW.update({
    "current_progress": 0.50, "remaining_ratio": 0.50,
    "budget_gap": 500_000, "nb_marches": 4,
    "progress_pressure": 0.50 / 0.51,
    "cost_pressure": 500_000 / 1.50,
    "delay": 1, "cost_overruns": 1,
    "execution_index": 0.25, "budget_index": 0.25,
    "total_risk_density": 2 / 21,
})
_X = pd.DataFrame([_BASE_ROW])[FEATURES].fillna(0)

_QUERY_VEC = np.array(embed_model.encode(["projet PPP retard budget"])).astype("float32")

# ---------------------------------------------------------------------------
# BM-P01 — Inférence XGBoost
# ---------------------------------------------------------------------------

def _inference():
    raw = float(model.predict_proba(_X)[0][1])
    if CALIBRATOR:
        raw = float(CALIBRATOR.transform([raw])[0])
    return raw

# ---------------------------------------------------------------------------
# BM-P02 — Feature engineering (simulation de prepare_features)
# ---------------------------------------------------------------------------

def _feature_eng():
    row = {f: 0.0 for f in FEATURES}
    cp, rr = 0.50, 0.50
    row.update({
        "current_progress": cp, "remaining_ratio": rr,
        "budget_gap": 500_000, "nb_marches": 4,
        "progress_pressure": cp / (rr + 0.01),
        "cost_pressure": 500_000 / (rr + 1.0),
        "total_risk_density": 2 / 21,
    })
    return pd.DataFrame([row])[FEATURES].fillna(0)

# ---------------------------------------------------------------------------
# BM-P03 — Recherche FAISS
# ---------------------------------------------------------------------------

def _faiss_search():
    _, _ = faiss_index.search(_QUERY_VEC, 5)

# ---------------------------------------------------------------------------
# BM-P04 — Lecture de l'état du pipeline
# ---------------------------------------------------------------------------

def _pipeline_state():
    state_file = ROOT / "pipeline_state.json"
    return json.loads(state_file.read_text("utf-8"))

# ---------------------------------------------------------------------------
# BM-P05 — Inférence complète (feature eng + modèle + calibration)
# ---------------------------------------------------------------------------

def _full_inference():
    X   = _feature_eng()
    raw = float(model.predict_proba(X)[0][1])
    if CALIBRATOR:
        raw = float(CALIBRATOR.transform([raw])[0])
    return raw

# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("  Benchmarks de Performance — PPP-Risk Platform")
    print("=" * 80 + "\n")

    print(f"  {'Test':<44}  {'avg':>10}  {'p95':>10}  {'p99':>10}")
    print(f"  {'-' * 80}")

    r1 = bench("BM-P01 | Inférence XGBoost seule",
               _inference, n=1000,
               threshold_ms=THRESHOLDS["inference_ms"])

    r2 = bench("BM-P02 | Feature engineering",
               _feature_eng, n=1000,
               threshold_ms=THRESHOLDS["feature_eng_ms"])

    r3 = bench("BM-P03 | Recherche FAISS (top-5, 829 vecteurs)",
               _faiss_search, n=1000,
               threshold_ms=THRESHOLDS["faiss_ms"])

    r4 = bench("BM-P04 | Lecture état pipeline (JSON)",
               _pipeline_state, n=500,
               threshold_ms=THRESHOLDS["pipeline_status_ms"])

    r5 = bench("BM-P05 | Pipeline complet (eng+inférence+calib)",
               _full_inference, n=1000,
               threshold_ms=THRESHOLDS["inference_ms"] + THRESHOLDS["feature_eng_ms"])

    # Summary
    print(f"\n  Modèle : {ROOT / 'ml' / 'model'}")
    print(f"  Features : {len(FEATURES)}")
    print(f"  Index FAISS : {faiss_index.ntotal} vecteurs × {faiss_index.d} dimensions")

    passed = sum(1 for r in _results if r["passed"])
    total  = len(_results)
    print(f"\n  Seuils respectés : {passed}/{total}")
    print("=" * 80)

    # Export JSON for thesis
    report = {
        "model_features": int(len(FEATURES)),
        "faiss_vectors":  int(faiss_index.ntotal),
        "faiss_dim":      int(faiss_index.d),
        "benchmarks":     [
            {k: bool(v) if isinstance(v, (bool,)) else v for k, v in r.items()}
            for r in _results
        ],
    }
    out = ROOT / "tests" / "perf_results.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\n  Résultats exportés → {out}")
    sys.exit(0 if passed == total else 1)
