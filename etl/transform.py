"""
etl/transform.py
----------------
Cleans the three raw Joget fact sources and resolves PPP foreign keys to
human-readable labels (denormalised columns) using the dimension label maps.

Produces star-schema-ready facts for cleaned_dw:
  fact_ppp        — risk indicators + resolved *_label columns + kept FK keys
  fact_pgd        — budget normalised to TND
  fact_pgd_etude  — cost normalised to TND
"""
import re
import numpy as np
import pandas as pd

UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-", re.I)


# ---------------------------------------------------------------------------
# Currency
# ---------------------------------------------------------------------------
def convert_to_tnd(value, currency_text: str = None):
    if pd.isna(value):
        return np.nan
    try:
        val = float(str(value).replace(",", "").strip())
        if isinstance(currency_text, str):
            t = currency_text.lower()
            if "€" in currency_text or "euro" in t:
                val *= 3.35
            elif "$" in currency_text or "dollar" in t:
                val *= 3.15
        return round(val, 2)
    except Exception:
        return np.nan


def _resolve(series: pd.Series, label_map: dict) -> pd.Series:
    """
    Map a FK column to its label. Columns may hold a UUID (resolve via map),
    an integer code (resolve via map), or already-free text (keep as-is).
    """
    def one(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        key = str(v).strip()
        if key in label_map:                       # exact key hit (UUID or int code)
            return label_map[key]
        if not UUID_RE.match(key):                 # already a readable label
            return key
        return key                                 # unresolved UUID — keep raw
    return series.apply(one)


# ---------------------------------------------------------------------------
# PPP
# ---------------------------------------------------------------------------
def clean_ppp(df: pd.DataFrame, label_maps: dict | None = None) -> pd.DataFrame:
    df = df.copy()
    label_maps = label_maps or {}

    budget_cols = ["c_budget_global_planifie", "c_budget_global_consomme",
                   "c_budget_realisation_planifie", "c_budget_realisation_consomme"]
    for col in budget_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "c_retard_avance_en_jours" in df.columns:
        df["c_retard_avance_en_jours"] = pd.to_numeric(df["c_retard_avance_en_jours"], errors="coerce")

    # === RISK TARGETS ===
    df["risque_retard"]      = (df["c_retard_avance_en_jours"] > 365).astype(int)
    df["risque_gros_retard"] = (df["c_retard_avance_en_jours"] > 730).astype(int)
    df["taux_depasse_budget"] = np.where(
        df["c_budget_global_planifie"] > 0,
        (df["c_budget_global_consomme"] - df["c_budget_global_planifie"]) / df["c_budget_global_planifie"],
        np.nan,
    )
    df["risque_depassement"]       = (df["taux_depasse_budget"] > 0).astype(int)
    df["risque_fort_depassement"]  = (df["taux_depasse_budget"] > 0.03).astype(int)
    df["taux_depasse_budget"]      = df["taux_depasse_budget"].fillna(0)
    df["c_retard_avance_en_jours"] = df["c_retard_avance_en_jours"].fillna(0)

    # === DENORMALISED LABELS (FK -> readable) ===
    LABEL_OUT = {
        "c_axe":                       "axe_label",
        "c_sous_axe":                  "sous_axe_label",
        "c_classification":            "classification_label",
        "c_sous_classification":       "sous_classification_label",
        "c_acteur_dimplementation":    "acteur_implementation_label",
        "c_delimination":              "delimitation_label",
        "c_initiateur":                "initiateur_label",
        "c_owner":                     "owner_label",
        "c_mode_financement":          "mode_financement_label",
        "c_mode_passation_marche":     "mode_passation_label",
        "c_tutelle_sectorielle":       "tutelle_sectorielle_label",
        "c_zonage":                    "zonage_label",
        "c_donateur":                  "donateur_label",
        "c_situation_actuelle_projet": "situation_label",
        "c_gouvernorat":               "gouvernorat_label",
        "c_delegation":                "delegation_label",
        "c_commune":                   "secteur_label",
    }
    resolved = 0
    for fk_col, out_col in LABEL_OUT.items():
        if fk_col in df.columns:
            df[out_col] = _resolve(df[fk_col], label_maps.get(fk_col, {}))
            resolved += 1

    print(f"PPP cleaned: {df.shape[0]} rows  ({resolved} FK columns resolved to labels)")
    print(f"   Risque Retard (>1 an)         : {df['risque_retard'].sum()} ({df['risque_retard'].mean()*100:.1f}%)")
    print(f"   Risque Gros Retard (>2 ans)   : {df['risque_gros_retard'].sum()}")
    print(f"   Risque Depassement            : {df['risque_depassement'].sum()}")
    print(f"   Risque Fort Depassement (>3%) : {df['risque_fort_depassement'].sum()}")
    return df


# ---------------------------------------------------------------------------
# PGD
# ---------------------------------------------------------------------------
def clean_pgd(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["c_annee"] = pd.to_numeric(df["c_annee"], errors="coerce")
    raw_budget = df["c_budget"].astype(str).str.replace(r"[^0-9.,]", "", regex=True)
    df["c_budget_tnd"] = [
        convert_to_tnd(b, cur) for b, cur in zip(raw_budget, df.get("c_bailleur", pd.Series([None]*len(df))))
    ]
    print(f"PGD cleaned: {df.shape[0]} rows")
    return df


# ---------------------------------------------------------------------------
# PGD Etude
# ---------------------------------------------------------------------------
def clean_pgd_etude(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["c_cout"]     = pd.to_numeric(df["c_cout"], errors="coerce")
    df["c_annee"]    = pd.to_numeric(df["c_annee"], errors="coerce")
    df["c_cout_tnd"] = df["c_cout"].apply(convert_to_tnd)
    print(f"PGD Etude cleaned: {df.shape[0]} rows")
    return df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_transformation(label_maps: dict | None = None):
    from etl.extract import extract_from_db
    from core.config import Config

    print("Starting data cleaning & transformation...\n")
    ppp   = extract_from_db(Config.TABLE_PPP)
    pgd   = extract_from_db(Config.TABLE_PGD)
    etude = extract_from_db(Config.TABLE_PGD_ETUDE)

    ppp_clean   = clean_ppp(ppp, label_maps)
    pgd_clean   = clean_pgd(pgd)
    etude_clean = clean_pgd_etude(etude)
    return ppp_clean, pgd_clean, etude_clean
