"""
ml/train.py - PPP-Risk Classifier (Final Model)

Complete training pipeline:
  1. Load data: synthetic CSV (default) or real DB (--source real)
  2. RandomizedSearchCV - 50 trials x 5-fold CV to find optimal XGBoost params
  3. Final model trained on 80% train split
  4. Isotonic calibration (reduces Brier score significantly)
  5. SHAP analysis for model interpretability
  6. Optimal threshold selection (maximise F1)
  7. Comprehensive metrics + PR/ROC curves saved to JSON
  8. Version archived in ml/model/versions/ via versioning.py

Run:  python ml/train.py                  # synthetic data (default)
      python ml/train.py --source real     # real DB data (needs >= 150 rows)
Out:  ml/model/risk_model.pkl
      ml/model/model_metrics.json
      ml/model/model_registry.json
      ml/plots/calibration.png
"""

import os, sys, json, warnings, argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import (
    RepeatedStratifiedKFold, train_test_split,
    RandomizedSearchCV, cross_val_predict
)
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, brier_score_loss, average_precision_score,
    f1_score, precision_score, recall_score,
    roc_curve, precision_recall_curve,
    matthews_corrcoef
)
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier
import shap

# -- Paths ---------------------------------------------------------------------
BASE  = os.path.dirname(os.path.abspath(__file__))
DATA  = os.path.join(BASE, 'data',  'dataset_synthetic.csv')
PKL   = os.path.join(BASE, 'model', 'risk_model.pkl')
METR  = os.path.join(BASE, 'model', 'model_metrics.json')
PLOTS = os.path.join(BASE, 'plots')
os.makedirs(os.path.join(BASE, 'model'), exist_ok=True)
os.makedirs(PLOTS, exist_ok=True)

TARGET = 'risque_depassement'

# -- 34 Features ---------------------------------------------------------------
FEATURES = [
    # Structural
    'current_progress', 'remaining_ratio', 'budget_gap', 'nb_marches',
    # Engineered
    'progress_pressure', 'cost_pressure', 'budget_risk',
    # Composite risk indices
    'execution_index', 'budget_index', 'operational_index',
    'external_index', 'governance_index', 'total_risk_density',
    # Execution binary
    'delay', 'schedule_slip', 'pacing', 'pressure',
    # Budget binary
    'cost_overruns', 'budget_revisions', 'funding_risk', 'margin_pressure',
    # Operational binary
    'errors', 'contractor_issues', 'resource_shortage', 'coordination',
    # External binary
    'client_changes', 'supplier_delays', 'regulatory', 'external_risk', 'dependency',
    # Governance binary
    'reporting', 'decision_delay', 'risk_tracking', 'escalation',
]
assert len(FEATURES) == 34

# -- Hyperparameter search space -----------------------------------------------
PARAM_SPACE = {
    'n_estimators':     [400, 500, 600, 700, 800],
    'max_depth':        [3, 4, 5, 6],
    'learning_rate':    [0.01, 0.02, 0.03, 0.05, 0.08],
    'subsample':        [0.65, 0.72, 0.80, 0.88, 0.95],
    'colsample_bytree': [0.55, 0.65, 0.72, 0.80, 0.88],
    'min_child_weight': [1, 2, 3, 4, 5],
    'gamma':            [0.0, 0.05, 0.10, 0.15, 0.20, 0.25],
    'reg_alpha':        [0.0, 0.01, 0.05, 0.10, 0.15],
    'reg_lambda':       [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
}


# -- Real data loader ----------------------------------------------------------
def load_real_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load training data from app_fd_ppp_clean (real Joget rows).

    TODO: map DB column names to model feature names once the exact Joget
          field names are confirmed.  The mapping below is a template —
          update the keys (left side) to match the real column names in
          app_fd_ppp_clean.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.config import Config
    from sqlalchemy import create_engine, text

    engine = create_engine(Config.SQLALCHEMY_DATABASE_URI)
    df = pd.read_sql("SELECT * FROM app_fd_ppp_clean", engine)
    print(f"  Loaded {len(df)} rows from app_fd_ppp_clean")

    # ------------------------------------------------------------------ #
    # Column mapping: DB column → model feature name                      #
    # Update the LEFT side (keys) to match your actual Joget column names #
    # ------------------------------------------------------------------ #
    COLUMN_MAP = {
        # Structural
        "c_avancement_institutionnel": "current_progress",
        "c_solde_budgetaire":          "budget_gap",
        "c_nb_marches":                "nb_marches",
        # Execution binary flags (0/1)
        "c_retard":                    "delay",
        "c_glissement_planning":       "schedule_slip",
        "c_rythme_execution":          "pacing",
        "c_pression":                  "pressure",
        # Budget binary flags
        "c_depassement_couts":         "cost_overruns",
        "c_revision_budget":           "budget_revisions",
        "c_risque_financement":        "funding_risk",
        "c_pression_marge":            "margin_pressure",
        # Operational
        "c_erreurs":                   "errors",
        "c_problemes_prestataires":    "contractor_issues",
        "c_manque_ressources":         "resource_shortage",
        "c_coordination":              "coordination",
        # External
        "c_changements_clients":       "client_changes",
        "c_retard_fournisseurs":       "supplier_delays",
        "c_contraintes_reglementaires":"regulatory",
        "c_risques_externes":          "external_risk",
        "c_dependances":               "dependency",
        # Governance
        "c_reporting":                 "reporting",
        "c_retard_decision":           "decision_delay",
        "c_suivi_risques":             "risk_tracking",
        "c_escalade":                  "escalation",
    }

    present = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    missing = [k for k in COLUMN_MAP if k not in df.columns]
    if missing:
        print(f"  WARNING — unmapped columns (update COLUMN_MAP): {missing}")

    df = df.rename(columns=present)

    # Clamp / derive engineered features
    cp  = pd.to_numeric(df.get("current_progress", 0), errors="coerce").fillna(0)
    cp  = cp.where(cp <= 2.0, cp / 100.0).clip(0.0, 1.0)
    rem = (1.0 - cp).clip(0.01, 1.0)

    def flag(col): return pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0).astype(int)

    df["current_progress"]    = cp
    df["remaining_ratio"]     = rem
    df["budget_gap"]          = pd.to_numeric(df.get("budget_gap", 0), errors="coerce").fillna(0)
    df["nb_marches"]          = pd.to_numeric(df.get("nb_marches", 1), errors="coerce").fillna(1).astype(int)
    df["progress_pressure"]   = cp / (rem + 0.01)
    df["cost_pressure"]       = df["budget_gap"] / (rem + 1.0)
    fr = flag("funding_risk")
    df["budget_risk"]         = ((df["budget_gap"] < 150_000).astype(int) + fr) / 2.0
    df["execution_index"]     = (flag("delay") + flag("schedule_slip") + flag("pacing") + flag("pressure")) / 4.0
    df["budget_index"]        = (flag("cost_overruns") + flag("budget_revisions") + fr + flag("margin_pressure")) / 4.0
    df["operational_index"]   = (flag("errors") + flag("contractor_issues") + flag("resource_shortage") + flag("coordination")) / 4.0
    df["external_index"]      = (flag("client_changes") + flag("supplier_delays") + flag("regulatory") + flag("external_risk") + flag("dependency")) / 5.0
    df["governance_index"]    = (flag("reporting") + flag("decision_delay") + flag("risk_tracking") + flag("escalation")) / 4.0
    all_flags = ["delay","schedule_slip","pacing","pressure","cost_overruns","budget_revisions",
                 "funding_risk","margin_pressure","errors","contractor_issues","resource_shortage",
                 "coordination","client_changes","supplier_delays","regulatory","external_risk",
                 "dependency","reporting","decision_delay","risk_tracking","escalation"]
    df["total_risk_density"] = sum(flag(k) for k in all_flags) / 21.0

    target_col = "risque_depassement"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in app_fd_ppp_clean")

    available = [f for f in FEATURES if f in df.columns]
    missing_feat = [f for f in FEATURES if f not in df.columns]
    if missing_feat:
        print(f"  WARNING — features not in DB, defaulting to 0: {missing_feat}")
        for f in missing_feat:
            df[f] = 0

    X = df[FEATURES].fillna(0)
    y = df[target_col].astype(int)
    return X, y


# -- Calibration plot ----------------------------------------------------------
def save_calibration(y_true, y_prob_raw, y_prob_cal, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for probs, lbl, style in [
        (y_prob_raw, 'Brut (XGBoost)', '--'),
        (y_prob_cal, 'Calibre (isotonique)', '-'),
    ]:
        fp, mp = calibration_curve(y_true, probs, n_bins=10)
        ax.plot(mp, fp, style, label=lbl)
    ax.plot([0, 1], [0, 1], 'k:', alpha=0.5, label='Calibration parfaite')
    ax.set_xlabel('Probabilite predite moyenne')
    ax.set_ylabel('Fraction de positifs reels')
    ax.set_title('Courbe de calibration - Modele PPP-Risk')
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  calibration -> {path}")


# -- Main ----------------------------------------------------------------------
def run(source: str = "synthetic") -> dict:
    """Train the model and return the metrics dict. source: 'synthetic' | 'real'"""
    print("=" * 65)
    print(f"  PPP-Risk Classifier - Pipeline complet  [source={source}]")
    print("  34 features | RandomizedSearchCV | Calibration isotonique")
    print("=" * 65)

    n_real_rows = 0
    if source == "real":
        print("\nLoading real data from DB...")
        X, y = load_real_data()
        n_real_rows = len(y)
        print(f"  {n_real_rows:,} rows  (pos: {int(y.sum()):,} = {y.mean()*100:.1f}%)")
    else:
        if not os.path.exists(DATA):
            raise FileNotFoundError(
                f"Donnees manquantes: {DATA}\nRun: python ml/generate_synthetic.py"
            )
        df = pd.read_csv(DATA)
        print(f"\nDonnees: {len(df):,} observations  "
              f"(pos: {int(df[TARGET].sum()):,} = {df[TARGET].mean()*100:.1f}%)")
        X = df[FEATURES].fillna(0)
        y = df[TARGET].astype(int)

    X_train, X_hold, y_train, y_hold = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"  Train : {len(X_train):,}   Holdout : {len(X_hold):,}")

    bad = [c for c in FEATURES if X_train[c].std() == 0]
    if bad:
        raise ValueError(f"Features constantes: {bad}")

    # -- STEP 1: RandomizedSearchCV --------------------------------------------
    print("\n" + "=" * 65)
    print("  ETAPE 1 - RandomizedSearchCV (50 iterations x 5 plis)")
    print("=" * 65)

    base_xgb = XGBClassifier(
        eval_metric='logloss', verbosity=0, random_state=42, n_jobs=-1
    )
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    search = RandomizedSearchCV(
        base_xgb,
        param_distributions=PARAM_SPACE,
        n_iter=50,
        scoring='roc_auc',
        cv=rskf,
        random_state=42,
        n_jobs=-1,
        verbose=0,
        refit=True,
    )
    search.fit(X_train, y_train)

    best_params = search.best_params_
    cv_auc      = float(search.best_score_)
    cv_results  = search.cv_results_
    best_idx    = search.best_index_
    fold_aucs   = [
        float(cv_results[f"split{i}_test_score"][best_idx])
        for i in range(search.n_splits_)
    ]
    cv_std = float(np.std(fold_aucs))

    print(f"\n  Meilleurs params : {best_params}")
    print(f"  Meilleur AUC CV  : {cv_auc:.4f} +/- {cv_std:.4f}")

    # -- STEP 2: Final model ---------------------------------------------------
    print("\n" + "=" * 65)
    print("  ETAPE 2 - Modele final")
    print("=" * 65)

    final_xgb = XGBClassifier(
        **best_params,
        eval_metric='logloss', verbosity=0, random_state=42, n_jobs=-1
    )
    final_xgb.fit(X_train, y_train)
    y_prob_raw = final_xgb.predict_proba(X_hold)[:, 1]

    # -- STEP 3: Isotonic calibration ------------------------------------------
    print("  Calibration isotonique...")
    oof_prob = cross_val_predict(
        XGBClassifier(**best_params, eval_metric='logloss', verbosity=0, random_state=42),
        X_train, y_train, cv=5, method='predict_proba', n_jobs=-1
    )[:, 1]

    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(oof_prob, y_train)
    y_prob_cal = iso.transform(y_prob_raw)

    save_calibration(y_hold, y_prob_raw, y_prob_cal,
                     os.path.join(PLOTS, "calibration.png"))

    # -- STEP 4: Optimal threshold (max F1) ------------------------------------
    thresholds = np.arange(0.10, 0.91, 0.01)
    f1s        = [f1_score(y_hold, (y_prob_cal >= t).astype(int)) for t in thresholds]
    opt_thresh = float(thresholds[np.argmax(f1s)])
    y_pred     = (y_prob_cal >= opt_thresh).astype(int)
    print(f"  Seuil optimal (F1 max) : {opt_thresh:.2f}")

    # -- STEP 5: Holdout metrics -----------------------------------------------
    print("\n" + "=" * 65)
    print("  ETAPE 5 - Evaluation holdout")
    print("=" * 65)

    print(classification_report(y_hold, y_pred, zero_division=0))
    cm         = confusion_matrix(y_hold, y_pred)
    tn, fp, fn, tp = cm.ravel()

    hold_auc  = float(roc_auc_score(y_hold, y_prob_cal))
    hold_ap   = float(average_precision_score(y_hold, y_prob_cal))
    hold_f1   = float(f1_score(y_hold, y_pred))
    hold_prec = float(precision_score(y_hold, y_pred, zero_division=0))
    hold_rec  = float(recall_score(y_hold, y_pred))
    hold_bs   = float(brier_score_loss(y_hold, y_prob_cal))
    hold_mcc  = float(matthews_corrcoef(y_hold, y_pred))
    bs_raw    = float(brier_score_loss(y_hold, y_prob_raw))

    print(f"  AUC-ROC       : {hold_auc:.4f}")
    print(f"  AUC-PR        : {hold_ap:.4f}")
    print(f"  F1            : {hold_f1:.4f}  (seuil {opt_thresh:.2f})")
    print(f"  Precision     : {hold_prec:.4f}")
    print(f"  Rappel        : {hold_rec:.4f}")
    print(f"  Brier calibre : {hold_bs:.4f}  (brut: {bs_raw:.4f})")
    print(f"  MCC           : {hold_mcc:.4f}")

    fpr, tpr, _ = roc_curve(y_hold, y_prob_cal)
    prec_c, rec_c, _ = precision_recall_curve(y_hold, y_prob_cal)

    # -- STEP 6: SHAP ----------------------------------------------------------
    print("\n" + "=" * 65)
    print("  ETAPE 6 - Analyse SHAP (500 observations)")
    print("=" * 65)

    explainer  = shap.TreeExplainer(final_xgb)
    idx_sample = np.random.RandomState(42).choice(len(X_hold), min(500, len(X_hold)), replace=False)
    X_shap     = X_hold.iloc[idx_sample].reset_index(drop=True)
    shap_vals  = explainer.shap_values(X_shap)

    shap_imp = pd.Series(np.abs(shap_vals).mean(axis=0), index=FEATURES).sort_values(ascending=False)
    print("\n  Top 15 features (SHAP |mean|):")
    for feat, val in shap_imp.head(15).items():
        bar = "#" * int(val / shap_imp.iloc[0] * 30)
        print(f"    {feat:<25} {val:.4f}  {bar}")

    # -- STEP 7: XGBoost gain importances --------------------------------------
    xgb_imp = {k: float(v) for k, v in zip(FEATURES, final_xgb.feature_importances_)}

    # -- Save ------------------------------------------------------------------
    metrics = {
        "cv_auc":        cv_auc,
        "cv_std":        cv_std,
        "fold_aucs":     fold_aucs,
        "opt_threshold": opt_thresh,
        "holdout_auc":   hold_auc,
        "holdout_ap":    hold_ap,
        "holdout_f1":    hold_f1,
        "holdout_prec":  hold_prec,
        "holdout_rec":   hold_rec,
        "holdout_bs":    hold_bs,
        "holdout_bs_raw":bs_raw,
        "holdout_mcc":   hold_mcc,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "n_train":          len(X_train),
        "n_holdout":        len(X_hold),
        "n_pos_holdout":    int(y_hold.sum()),
        "xgb_importances":  xgb_imp,
        "shap_importances": {k: float(v) for k, v in shap_imp.items()},
        "shap_values_abs":  np.abs(shap_vals).mean(axis=0).tolist(),
        "shap_values_raw":  shap_vals.tolist(),
        "shap_feature_idx": list(range(len(FEATURES))),
        "roc_fpr":  fpr[::5].tolist(),
        "roc_tpr":  tpr[::5].tolist(),
        "pr_prec":  prec_c[::5].tolist(),
        "pr_rec":   rec_c[::5].tolist(),
        "y_prob_holdout": y_prob_cal.tolist(),
        "y_true_holdout": y_hold.tolist(),
        "best_params": {
            k: (int(v) if isinstance(v, (np.integer,)) else
                float(v) if isinstance(v, (np.floating,)) else v)
            for k, v in best_params.items()
        },
    }

    with open(METR, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metriques -> {METR}")

    joblib.dump({
        "model":         final_xgb,
        "calibrator":    iso,
        "features":      FEATURES,
        "opt_threshold": opt_thresh,
    }, PKL)
    print(f"  Modele    -> {PKL}")

    # -- Version archiving -----------------------------------------------------
    try:
        from ml.versioning import save_version
        ver_entry = save_version(metrics, PKL, source=source, n_real_rows=n_real_rows)
        print(f"  Version   -> {ver_entry['tag']}")
    except Exception as e:
        print(f"  WARNING: versioning failed ({e}) — model saved but not archived")
        ver_entry = None

    # -- Final summary ---------------------------------------------------------
    print("\n" + "=" * 65)
    print("  RESULTATS FINAUX - Modele PPP-Risk")
    print("=" * 65)
    print(f"  CV AUC (15 plis) : {cv_auc:.4f} +/- {cv_std:.4f}")
    print(f"  Holdout AUC-ROC  : {hold_auc:.4f}")
    print(f"  Holdout AUC-PR   : {hold_ap:.4f}")
    print(f"  F1 (seuil {opt_thresh:.2f})   : {hold_f1:.4f}")
    print(f"  Brier calibre    : {hold_bs:.4f}  (gain vs brut: +{bs_raw-hold_bs:.4f})")
    print(f"  MCC              : {hold_mcc:.4f}")
    print("=" * 65)
    if hold_auc >= 0.97:
        print("  EXCELLENT")
    elif hold_auc >= 0.93:
        print("  TRES BON")
    else:
        print("  BON")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPP-Risk model training")
    parser.add_argument("--source", choices=["synthetic", "real"],
                        default="synthetic",
                        help="Data source: synthetic CSV or real DB")
    args = parser.parse_args()

    metrics = run(source=args.source)

    # Auto-promote when run manually (pipeline handles promotion via AUC gate)
    try:
        from ml.versioning import latest_entry, mark_promoted
        entry = latest_entry()
        if entry:
            mark_promoted(entry["version"])
            print(f"  Promoted  -> {entry['tag']} is now production")
    except Exception as e:
        print(f"  WARNING: could not mark version as promoted ({e})")
