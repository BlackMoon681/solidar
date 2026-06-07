"""
chatbot/data_prep.py
--------------------
Reads PPP, PGD, and Etude tables from the DB and writes documents.jsonl.
Run this first, then build_index.py.
"""

import os
import sys
import json
import pandas as pd
from sqlalchemy import create_engine

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.config import Config

_dir = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(_dir, "data", "documents.jsonl")
os.makedirs(os.path.join(_dir, "data"), exist_ok=True)

engine = create_engine(Config.SQLALCHEMY_DW_URI)   # clean facts live in cleaned_dw

ppp       = pd.read_sql(f"SELECT * FROM {Config.TABLE_PPP_CLEAN}",      engine)
pgd       = pd.read_sql(f"SELECT * FROM {Config.TABLE_PGD_CLEAN}",      engine)
pgd_etude = pd.read_sql(f"SELECT * FROM {Config.TABLE_PGD_ETUDE_CLEAN}", engine)


def s(x):
    if pd.isna(x) if not isinstance(x, str) else False:
        return ""
    v = str(x).strip()
    # MySQL returns latin-1 bytes decoded as latin-1; re-decode as cp1252
    try:
        v = v.encode("latin-1").decode("cp1252")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return v


documents = []

for _, row in ppp.iterrows():
    text = (
        f"Projet PPP: {s(row.get('c_nom_projet'))}\n\n"
        f"Gouvernorat: {s(row.get('gouvernorat_label'))}\n"
        f"Délégation: {s(row.get('delegation_label'))}\n"
        f"Secteur: {s(row.get('secteur_label'))}\n"
        f"Statut: {s(row.get('situation_label'))}\n"
        f"Phase actuelle: {s(row.get('c_phase_en_cours'))}\n\n"
        f"Budget prévu: {s(row.get('c_budget_global_planifie'))} TND\n"
        f"Budget consommé: {s(row.get('c_budget_global_consomme'))} TND\n"
        f"Solde restant: {s(row.get('c_solde_budget_global_restant'))}\n\n"
        f"Avancement: {s(row.get('c_taux_avancement_institutionnel'))}\n"
        f"Bénéficiaires: {s(row.get('c_nombre_beneficiaires'))}\n"
        f"ODD: {s(row.get('c_odd'))}\n"
        f"Acteur: {s(row.get('acteur_implementation_label'))}\n"
        f"Bailleur: {s(row.get('c_bailleur_fonds'))}\n"
        f"Axe: {s(row.get('axe_label'))}\n"
        f"Classification: {s(row.get('classification_label'))}"
    )
    documents.append({
        "source": "PPP",
        "text": text,
        "metadata": {
            "id":      s(row.get("id")),
            "project": s(row.get("c_nom_projet")),
            "region":  s(row.get("gouvernorat_label")),
            "status":  s(row.get("situation_label")),
        }
    })

for _, row in pgd.iterrows():
    text = (
        f"Projet: {s(row.get('c_projet'))}\n"
        f"Promoteur: {s(row.get('c_promoteur'))}\n"
        f"Région: {s(row.get('c_region'))}\n"
        f"Année: {s(row.get('c_annee'))}\n\n"
        f"Activités:\n{s(row.get('c_activite'))}\n\n"
        f"Contexte:\n{s(row.get('c_context'))}\n\n"
        f"Objectifs:\n{s(row.get('c_objectifs'))}\n\n"
        f"Budget: {s(row.get('c_budget_tnd'))} TND\n"
        f"Bailleur: {s(row.get('c_bailleur'))}"
    )
    documents.append({
        "source": "PGD",
        "text": text,
        "metadata": {
            "id":      s(row.get("id")),
            "project": s(row.get("c_projet")),
            "region":  s(row.get("c_region")),
            "year":    s(row.get("c_annee")),
        }
    })

for _, row in pgd_etude.iterrows():
    text = (
        f"Etude: {s(row.get('c_intitule'))}\n\n"
        f"Domaine: {s(row.get('c_domaine'))}\n"
        f"Thème: {s(row.get('c_theme'))}\n"
        f"Année: {s(row.get('c_annee'))}\n\n"
        f"Catégorie: {s(row.get('c_categorie'))}\n"
        f"Expert(s): {s(row.get('c_expert'))}\n\n"
        f"Bénéficiaires: {s(row.get('c_beneficiaires'))}\n"
        f"Institution: {s(row.get('c_acteur_institutionnel'))}\n\n"
        f"Bailleur: {s(row.get('c_bailleur'))}\n"
        f"Cadre: {s(row.get('c_cadre'))}\n\n"
        f"Lien: {s(row.get('c_lien_pdf'))}\n"
        f"Coût: {s(row.get('c_cout_tnd'))} TND"
    )
    documents.append({
        "source": "ETUDE",
        "text": text,
        "metadata": {
            "id":     s(row.get("id")),
            "title":  s(row.get("c_intitule")),
            "domain": s(row.get("c_domaine")),
            "year":   s(row.get("c_annee")),
        }
    })

with open(OUT, "w", encoding="utf-8") as f:
    for doc in documents:
        f.write(json.dumps(doc, ensure_ascii=False) + "\n")

print(f"Generated {len(documents)} documents -> {OUT}")
