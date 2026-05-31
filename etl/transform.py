import pandas as pd
import numpy as np
from datetime import datetime


def convert_to_tnd(value, currency_text: str = None):
    """Simple currency normalization to TND"""
    if pd.isna(value):
        return np.nan
    try:
        val = float(str(value).replace(',', '').strip())
        # Very basic conversion (you can improve later)
        if isinstance(currency_text, str):
            if '€' in currency_text or 'euro' in currency_text.lower():
                val *= 3.35  # approx 1€ = 3.35 TND
            elif '$' in currency_text or 'dollar' in currency_text.lower():
                val *= 3.15  # approx 1$ = 3.15 TND
        return round(val, 2)
    except:
        return np.nan


import numpy as np
import pandas as pd


def clean_ppp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Budget to numeric
    budget_cols = ['c_budget_global_planifie', 'c_budget_global_consomme',
                   'c_budget_realisation_planifie', 'c_budget_realisation_consomme']
    for col in budget_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Delay column to numeric
    if 'c_retard_avance_en_jours' in df.columns:
        df['c_retard_avance_en_jours'] = pd.to_numeric(df['c_retard_avance_en_jours'], errors='coerce')

    # === TARGETS ===
    df['risque_retard'] = (df['c_retard_avance_en_jours'] > 365).astype(int)
    df['risque_gros_retard'] = (df['c_retard_avance_en_jours'] > 730).astype(int)

    df['taux_depasse_budget'] = np.where(
        df['c_budget_global_planifie'] > 0,
        (df['c_budget_global_consomme'] - df['c_budget_global_planifie']) / df['c_budget_global_planifie'],
        np.nan
    )
    df['risque_depassement'] = (df['taux_depasse_budget'] > 0).astype(int)
    df['risque_fort_depassement'] = (df['taux_depasse_budget'] > 0.03).astype(int)

    # Fill NaNs
    df['taux_depasse_budget'] = df['taux_depasse_budget'].fillna(0)
    df['c_retard_avance_en_jours'] = df['c_retard_avance_en_jours'].fillna(0)

    print(f"✅ PPP cleaned: {df.shape[0]} rows")
    print(
        f"   Risque Retard (>1 an)          : {df['risque_retard'].sum()} projects ({df['risque_retard'].mean() * 100:.1f}%)")
    print(f"   Risque Gros Retard (>2 ans)    : {df['risque_gros_retard'].sum()} projects")
    print(f"   Risque Dépassement             : {df['risque_depassement'].sum()} projects")
    print(f"   Risque Fort Dépassement (>3%)  : {df['risque_fort_depassement'].sum()} projects")

    return df


def clean_pgd(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['c_annee'] = pd.to_numeric(df['c_annee'], errors='coerce')

    # Budget cleaning + TND conversion
    df['c_budget'] = df['c_budget'].astype(str).str.replace(r'[^0-9.,]', '', regex=True)
    df['c_budget_tnd'] = df['c_budget'].apply(lambda x: convert_to_tnd(x, df.get('c_bailleur')))

    print(f"✅ PGD cleaned: {df.shape[0]} rows")
    return df


def clean_pgd_etude(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['c_cout'] = pd.to_numeric(df['c_cout'], errors='coerce')
    df['c_annee'] = pd.to_numeric(df['c_annee'], errors='coerce')
    df['c_cout_tnd'] = df['c_cout'].apply(convert_to_tnd)

    print(f"✅ PGD Etude cleaned: {df.shape[0]} rows")
    return df


def run_transformation():
    from etl.extract import extract_from_db
    from core.config import Config

    print("Starting data cleaning & transformation...\n")

    ppp = extract_from_db(Config.TABLE_PPP)
    pgd = extract_from_db(Config.TABLE_PGD)
    etude = extract_from_db(Config.TABLE_PGD_ETUDE)

    ppp_clean = clean_ppp(ppp)
    pgd_clean = clean_pgd(pgd)
    etude_clean = clean_pgd_etude(etude)

    return ppp_clean, pgd_clean, etude_clean