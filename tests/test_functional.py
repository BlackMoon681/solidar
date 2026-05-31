"""
tests/test_functional.py
------------------------
Tests fonctionnels du Modèle PPP-Risk et du chatbot RAG.
Teste directement les fonctions métier sans serveur HTTP.

Run:  python tests/test_functional.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"

_results = []

def check(name, condition, got=None, expected=None):
    status = PASS if condition else FAIL
    msg = f"[{status} ]  {name}"
    if not condition:
        msg += f"\n         got={got!r}  expected={expected!r}"
    print(msg)
    _results.append((name, condition))
    return condition


# ---------------------------------------------------------------------------
# Load model once
# ---------------------------------------------------------------------------

def _load_model():
    for name in ("risk_model.pkl", "budget_model_v13.pkl", "budget_model_v12.pkl"):
        p = ROOT / "ml" / "model" / name
        if p.exists():
            saved = joblib.load(p)
            return saved["model"], saved["features"], saved.get("calibrator")
    raise FileNotFoundError("No model found in ml/model/")

model, FEATURES, CALIBRATOR = _load_model()


def _predict(overrides: dict) -> float:
    row = {f: 0.0 for f in FEATURES}
    cp  = overrides.get("current_progress", 0.5)
    rr  = overrides.get("remaining_ratio",  1.0 - cp)
    row.update({
        "current_progress":  cp,
        "remaining_ratio":   rr,
        "budget_gap":        overrides.get("budget_gap", 1_000_000),
        "nb_marches":        overrides.get("nb_marches", 3),
        "progress_pressure": cp / (rr + 0.01),
        "cost_pressure":     overrides.get("budget_gap", 1_000_000) / (rr + 1.0),
        "budget_risk":       overrides.get("budget_risk", 0.0),
    })
    for k, v in overrides.items():
        if k in row:
            row[k] = v

    # Compute indices from binary flags
    flags = {
        "execution_index":   ["delay", "schedule_slip", "pacing", "pressure"],
        "budget_index":      ["cost_overruns", "budget_revisions", "funding_risk", "margin_pressure"],
        "operational_index": ["errors", "contractor_issues", "resource_shortage", "coordination"],
        "external_index":    ["client_changes", "supplier_delays", "regulatory", "external_risk", "dependency"],
        "governance_index":  ["reporting", "decision_delay", "risk_tracking", "escalation"],
    }
    for idx, cols in flags.items():
        row[idx] = sum(float(overrides.get(c, 0)) for c in cols) / len(cols)

    all_bins = [c for cols in flags.values() for c in cols]
    row["total_risk_density"] = sum(float(overrides.get(c, 0)) for c in all_bins) / 21.0

    X   = pd.DataFrame([row])[FEATURES].fillna(0)
    raw = float(model.predict_proba(X)[0][1])
    if CALIBRATOR:
        raw = float(CALIBRATOR.transform([raw])[0])
    return max(0.0, min(1.0, raw))


def _level(p):
    if p < 0.15: return "Low"
    if p < 0.40: return "Medium"
    if p < 0.70: return "High"
    return "Critical"


# ---------------------------------------------------------------------------
# TC-F01 — Projet sans risque (tous signaux à 0, avancement 90 %)
# ---------------------------------------------------------------------------

def test_f01_low_risk():
    p = _predict({"current_progress": 0.90, "budget_gap": 5_000_000})
    check("TC-F01 | Projet sain → niveau Faible",
          _level(p) == "Low",
          got=f"prob={p:.3f} niveau={_level(p)}", expected="Low")


# ---------------------------------------------------------------------------
# TC-F02 — Projet à risque critique (6 signaux actifs, budget serré)
# ---------------------------------------------------------------------------

def test_f02_critical_risk():
    p = _predict({
        "current_progress": 0.20, "budget_gap": 30_000,
        "delay": 1, "schedule_slip": 1, "cost_overruns": 1,
        "funding_risk": 1, "errors": 1, "contractor_issues": 1,
        "budget_risk": 0.5,
    })
    check("TC-F02 | Projet critique → niveau High ou Critical",
          _level(p) in ("High", "Critical"),
          got=f"prob={p:.3f} niveau={_level(p)}", expected="High|Critical")


# ---------------------------------------------------------------------------
# TC-F03 — Projet risque modéré (2 signaux, avancement 50 %)
# ---------------------------------------------------------------------------

def test_f03_medium_risk():
    p = _predict({
        "current_progress": 0.50, "budget_gap": 500_000,
        "delay": 1, "schedule_slip": 1,
    })
    check("TC-F03 | Projet modéré → niveau Low, Medium ou High",
          _level(p) in ("Low", "Medium", "High"),
          got=f"prob={p:.3f} niveau={_level(p)}", expected="Low|Medium|High")


# ---------------------------------------------------------------------------
# TC-F04 — Monotonie du score : plus de signaux = score plus élevé
#          Utilise budget serré (50 000 TND) pour activer la sensibilité
# ---------------------------------------------------------------------------

def test_f04_monotonicity():
    base = {"current_progress": 0.30, "budget_gap": 50_000}
    p0 = _predict(base)
    p1 = _predict({**base, "delay": 1, "cost_overruns": 1, "funding_risk": 1})
    p2 = _predict({**base,
                   "delay": 1, "schedule_slip": 1, "pacing": 1,
                   "cost_overruns": 1, "budget_revisions": 1, "funding_risk": 1,
                   "errors": 1, "contractor_issues": 1, "resource_shortage": 1,
                   "reporting": 1, "decision_delay": 1})
    check("TC-F04 | Monotonie : p(0 signal) <= p(3 signaux) <= p(11 signaux)",
          p0 <= p1 and p1 <= p2,
          got=f"p0={p0:.3f} p1={p1:.3f} p2={p2:.3f}",
          expected="p0 <= p1 <= p2")


# ---------------------------------------------------------------------------
# TC-F05 — Budget très serré = score plus élevé qu'avec budget confortable
# ---------------------------------------------------------------------------

def test_f05_budget_sensitivity():
    p_large = _predict({"current_progress": 0.50, "budget_gap": 10_000_000})
    p_small = _predict({"current_progress": 0.50, "budget_gap": 10_000})
    check("TC-F05 | Budget serré → score > budget confortable",
          p_small >= p_large,
          got=f"p_small={p_small:.3f} p_large={p_large:.3f}",
          expected="p_small >= p_large")


# ---------------------------------------------------------------------------
# TC-F06 — Cohérence avancement tardif : risque plus élevé en phase précoce
#          (si budget_gap est très faible)
# ---------------------------------------------------------------------------

def test_f06_late_stage_low_budget():
    p_early = _predict({
        "current_progress": 0.10, "budget_gap": 5_000,
        "funding_risk": 1, "cost_overruns": 1, "errors": 1,
        "delay": 1, "schedule_slip": 1,
    })
    p_late = _predict({"current_progress": 0.95, "budget_gap": 8_000_000})
    check("TC-F06 | Phase précoce + budget serré + risques → score >= phase tardive saine",
          p_early >= p_late,
          got=f"p_early={p_early:.3f} p_late={p_late:.3f}",
          expected="p_early >= p_late")


# ---------------------------------------------------------------------------
# TC-F07 — FAISS index accessible et retourne top-k documents
# ---------------------------------------------------------------------------

def test_f07_faiss_index():
    try:
        import faiss, json
        from sentence_transformers import SentenceTransformer

        idx  = faiss.read_index(str(ROOT / "chatbot" / "index" / "faiss_index.bin"))
        meta = json.loads((ROOT / "chatbot" / "data" / "documents_with_meta.json").read_text("utf-8"))
        emb  = SentenceTransformer("all-MiniLM-L6-v2")

        q  = emb.encode(["projet environnemental retard budget"])
        q  = np.array(q).astype("float32")
        D, I = idx.search(q, 5)

        ok = (
            len(I[0]) == 5
            and all(0 <= i < len(meta) for i in I[0])
            and all(d >= 0 for d in D[0])
        )
        check("TC-F07 | FAISS retourne 5 documents valides",
              ok, got=f"indices={I[0].tolist()}", expected="5 indices valides")
    except Exception as e:
        check("TC-F07 | FAISS retourne 5 documents valides", False, got=str(e))


# ---------------------------------------------------------------------------
# TC-F08 — Pipeline state : fichier d'état lisible
# ---------------------------------------------------------------------------

def test_f08_pipeline_state():
    import json
    state_file = ROOT / "pipeline_state.json"
    try:
        state = json.loads(state_file.read_text("utf-8"))
        ok = (
            "raw_row_counts" in state
            and "last_run" in state
            and isinstance(state["raw_row_counts"], dict)
        )
        check("TC-F08 | pipeline_state.json valide et cohérent",
              ok, got=list(state.keys()), expected="raw_row_counts + last_run")
    except Exception as e:
        check("TC-F08 | pipeline_state.json valide et cohérent", False, got=str(e))


# ---------------------------------------------------------------------------
# TC-F09 — Model registry si disponible
# ---------------------------------------------------------------------------

def test_f09_model_registry():
    registry_file = ROOT / "ml" / "model" / "model_registry.json"
    if not registry_file.exists():
        print(f"[ SKIP ]  TC-F09 | Registre des versions (aucun run de re-entraînement encore)")
        _results.append(("TC-F09 (skipped)", True))
        return
    import json
    reg = json.loads(registry_file.read_text("utf-8"))
    ok = isinstance(reg, list) and all("holdout_auc" in e for e in reg)
    check("TC-F09 | Registre des versions valide",
          ok, got=f"{len(reg)} versions", expected="liste avec holdout_auc")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Tests Fonctionnels — PPP-Risk Platform")
    print("=" * 60 + "\n")

    test_f01_low_risk()
    test_f02_critical_risk()
    test_f03_medium_risk()
    test_f04_monotonicity()
    test_f05_budget_sensitivity()
    test_f06_late_stage_low_budget()
    test_f07_faiss_index()
    test_f08_pipeline_state()
    test_f09_model_registry()

    passed = sum(1 for _, ok in _results if ok)
    total  = len(_results)
    print(f"\n{'=' * 60}")
    print(f"  Résultat : {passed}/{total} tests passés")
    print("=" * 60)
    sys.exit(0 if passed == total else 1)
