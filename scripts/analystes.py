"""
scripts/analystes.py
--------------------
Generates statistics for chapter 4 of the thesis:
  - Target class distributions (DB, V3, V4)
  - V12 model: feature importances, performance metrics
  - V4 dataset feature analysis

Run from project root:
    python scripts/analystes.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine
from core.config import Config

engine = create_engine(Config.SQLALCHEMY_DW_URI)   # cleaned facts in cleaned_dw

_root      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V3_PATH    = os.path.join(_root, "ml", "data", "ppp_noise_augmented_v3.csv")
V4_PATH    = os.path.join(_root, "ml", "data", "ppp_synthetic_balanced_v4.csv")
MODEL_PATH = os.path.join(_root, "ml", "model", "budget_model_v12.pkl")

TARGETS = [
    "risque_retard",
    "risque_gros_retard",
    "risque_depassement",
    "risque_fort_depassement",
]

SEP = "=" * 70


def pct(val, total):
    return round(val / total * 100, 1) if total > 0 else 0.0


def dist(df, target):
    if target not in df.columns:
        return None
    vc    = df[target].value_counts()
    total = vc.sum()
    return {0: pct(vc.get(0, 0), total), 1: pct(vc.get(1, 0), total),
            "n0": int(vc.get(0, 0)), "n1": int(vc.get(1, 0)), "total": int(total)}


# ── 1. DB distributions ──────────────────────────────────────────────────────
print(f"\n{'DB (app_fd_ppp_clean) — ORIGINAL'}")
print(SEP)
df_db  = pd.read_sql(f"SELECT * FROM {Config.TABLE_PPP_CLEAN}", engine)
db_dist = {}
for t in TARGETS:
    d = dist(df_db, t)
    if d:
        db_dist[t] = d
        print(f"  {t:<32}  n={d['total']:>4}  |  0: {d[0]:>5.1f}%  1: {d[1]:>5.1f}%"
              f"  (n0={d['n0']}, n1={d['n1']})")

# ── 2. V3 distributions ──────────────────────────────────────────────────────
print(f"\nV3 AUGMENTED (ppp_noise_augmented_v3)")
print(SEP)
v3_dist = {}
if os.path.exists(V3_PATH):
    df_v3  = pd.read_csv(V3_PATH)
    n_orig = int((df_v3["is_augmented"] == 0).sum()) if "is_augmented" in df_v3.columns else len(df_v3)
    n_aug  = int((df_v3["is_augmented"] == 1).sum()) if "is_augmented" in df_v3.columns else 0
    print(f"  Total rows: {len(df_v3)}  (originals: {n_orig}, augmented: {n_aug})")
    for t in TARGETS:
        d = dist(df_v3, t)
        if d:
            v3_dist[t] = d
            print(f"  {t:<32}  n={d['total']:>4}  |  0: {d[0]:>5.1f}%  1: {d[1]:>5.1f}%"
                  f"  (n0={d['n0']}, n1={d['n1']})")
else:
    print(f"  V3 file not found: {V3_PATH}")

# ── 3. V4 distributions ──────────────────────────────────────────────────────
print(f"\nV4 BALANCED SYNTHETIC (ppp_synthetic_balanced_v4)")
print(SEP)
if os.path.exists(V4_PATH):
    df_v4 = pd.read_csv(V4_PATH)
    d     = dist(df_v4, "risque_depassement")
    if d:
        print(f"  Total rows: {d['total']}")
        print(f"  risque_depassement  ->  0: {d[0]}%  (n={d['n0']})  |  1: {d[1]}%  (n={d['n1']})")
    print(f"\n  Feature stats (describe):")
    print(df_v4.describe().round(4).to_string())
    print(f"\n  Mean per class (risque_depassement):")
    print(df_v4.groupby("risque_depassement").mean().round(4).to_string())
else:
    print(f"  V4 file not found: {V4_PATH}")

# ── 4. V12 model — feature importances ───────────────────────────────────────
print(f"\nV12 MODEL — FEATURE IMPORTANCES")
print(SEP)
if os.path.exists(MODEL_PATH):
    saved     = joblib.load(MODEL_PATH)
    mdl       = saved["model"]
    features  = saved["features"]
    freq_maps = saved.get("freq_maps", {})

    importances        = pd.Series(mdl.feature_importances_, index=features)
    importances_sorted = importances.sort_values(ascending=False)

    print(f"  {'Feature':<30}  {'Importance':>10}  {'%':>7}")
    print(f"  {'-'*50}")
    for feat, imp in importances_sorted.items():
        print(f"  {feat:<30}  {imp:>10.4f}  {imp*100:>6.2f}%")

    top_feat = importances_sorted.index[0]
    top_imp  = importances_sorted.iloc[0]
    print(f"\n  Top feature: '{top_feat}' = {top_imp:.4f} ({top_imp*100:.1f}%)")
    print(f"  Freq map keys available: {list(freq_maps.keys())}")
else:
    print(f"  Model not found: {MODEL_PATH}")

# ── 5. Summary comparison table ──────────────────────────────────────────────
print(f"\nSUMMARY — CLASS DISTRIBUTION COMPARISON")
print(SEP)
print(f"  {'Target':<32}  {'DB (class 1)':>12}  {'V3 (class 1)':>12}")
print(f"  {'-'*60}")
for t in TARGETS:
    db_c1 = f"{db_dist[t][1]:.1f}%  (n={db_dist[t]['n1']})" if t in db_dist else "N/A"
    v3_c1 = f"{v3_dist[t][1]:.1f}%  (n={v3_dist[t]['n1']})" if t in v3_dist else "N/A"
    print(f"  {t:<32}  {db_c1:>22}  {v3_c1:>22}")

print(f"\n  V4 (risque_depassement only): ~50/50  — 10,000 synthetic rows")

# ── 6. Real model performance ─────────────────────────────────────────────────
print(f"\nV12 REAL PERFORMANCE METRICS (from training log)")
print(SEP)
print("  Evaluation strategy : RepeatedStratifiedKFold (3 repeats x 5 folds = 15 scores)")
print("  Data                : 703 raw DB rows (25 positives, 3.6%) - no augmentation in CV")
print("  SMOTE               : applied inside each training fold only")
print("  Near-duplicate rate : 5.0% (EXCELLENT - direct DB load)")
print()
print("  CV AUC-ROC         : 0.9736  +/-  0.0139")
print("  CV Avg Precision   : 0.6418  +/-  0.1788  (baseline = 0.0356, 18x lift)")
print()
print("  Holdout (141 rows, 5 positives):")
print("    Accuracy          : 97.2%")
print("    Class-1 Precision : 57.1%")
print("    Class-1 Recall    : 80.0%")
print("    Class-1 F1        : 66.7%")
print("    AUC-ROC           : 0.9853")
print("    Brier Score       : 0.0244")
print("    Confusion matrix  : TN=133  FP=3  FN=1  TP=4")
print()
print("  Training set after SMOTE: 1,084 rows (542 per class)")
print("  Model : XGBClassifier  n_estimators=400, max_depth=4, lr=0.03")
