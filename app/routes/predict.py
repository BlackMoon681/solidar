import pandas as pd
from fastapi import APIRouter

from app.schemas import ProjectInput
from app.dependencies import model, FEATURES, CALIBRATOR, FREQ_MAPS, get_groq_client, GROQ_MODEL

router = APIRouter()

FIELD_LABELS_FR = {
    "current_progress":   "Avancement institutionnel",
    "budget_gap":         "Solde budgétaire restant (TND)",
    "nb_marches":         "Nombre de marchés",
    "delay":              "Projet en retard",
    "schedule_slip":      "Glissement du planning",
    "pacing":             "Rythme d'exécution insuffisant",
    "pressure":           "Pression élevée",
    "cost_overruns":      "Dépassement des coûts",
    "budget_revisions":   "Révision du budget",
    "funding_risk":       "Risque de financement",
    "margin_pressure":    "Pression sur la marge",
    "errors":             "Présence d'erreurs",
    "contractor_issues":  "Problèmes avec prestataires",
    "resource_shortage":  "Manque de ressources",
    "coordination":       "Problèmes de coordination",
    "client_changes":     "Changements clients",
    "supplier_delays":    "Retard fournisseurs",
    "regulatory":         "Contraintes réglementaires",
    "external_risk":      "Risques externes",
    "dependency":         "Problèmes de dépendance",
    "reporting":          "Problèmes de reporting",
    "decision_delay":     "Retard de décision",
    "risk_tracking":      "Suivi des risques insuffisant",
    "escalation":         "Escalade déclenchée",
}

BINARY_FLAGS = [
    "delay", "schedule_slip", "pacing", "pressure",
    "cost_overruns", "budget_revisions", "funding_risk", "margin_pressure",
    "errors", "contractor_issues", "resource_shortage", "coordination",
    "client_changes", "supplier_delays", "regulatory", "external_risk", "dependency",
    "reporting", "decision_delay", "risk_tracking", "escalation",
]


def get_risk_category(prob: float) -> str:
    if prob < 0.15: return "Low"
    if prob < 0.40: return "Medium"
    if prob < 0.70: return "High"
    return "Critical"


def prepare_features(data: dict) -> pd.DataFrame:
    cp = float(data["current_progress"])
    if cp > 2.0:
        cp /= 100.0
    cp = max(0.0, min(1.0, cp))

    remaining = max(0.01, 1.0 - cp)
    rem_ratio = data.get("remaining_ratio")
    if rem_ratio is None:
        rem_ratio = remaining
    else:
        rem_ratio = float(rem_ratio)
        if rem_ratio > 2.0:
            rem_ratio /= 100.0
        rem_ratio = max(0.01, min(1.0, rem_ratio))

    budget_gap = float(data.get("budget_gap", 0.0))
    nb_marches = int(data.get("nb_marches", 1))

    def flag(key): return int(data.get(key, 0))

    fr = flag("funding_risk")
    progress_pressure    = cp / (rem_ratio + 0.01)
    cost_pressure        = budget_gap / (rem_ratio + 1.0)
    budget_risk          = ((1 if budget_gap < 150_000 else 0) + fr) / 2.0

    phase_raw     = str(data.get("phase_raw", ""))
    situation_raw = str(data.get("situation_raw", ""))
    phase_encoded     = FREQ_MAPS.get("phase_raw",     {}).get(phase_raw,     0.0)
    situation_encoded = FREQ_MAPS.get("situation_raw", {}).get(situation_raw, 0.0)

    execution_index   = (flag("delay") + flag("schedule_slip") + flag("pacing") + flag("pressure")) / 4.0
    budget_index_val  = (flag("cost_overruns") + flag("budget_revisions") + fr + flag("margin_pressure")) / 4.0
    operational_index = (flag("errors") + flag("contractor_issues") + flag("resource_shortage") + flag("coordination")) / 4.0
    external_index    = (flag("client_changes") + flag("supplier_delays") + flag("regulatory") + flag("external_risk") + flag("dependency")) / 5.0
    governance_index  = (flag("reporting") + flag("decision_delay") + flag("risk_tracking") + flag("escalation")) / 4.0
    total_risk_density = sum(flag(k) for k in BINARY_FLAGS) / 21.0

    row = {
        "current_progress":          cp,
        "remaining_ratio":           rem_ratio,
        "budget_gap":                budget_gap,
        "nb_marches":                nb_marches,
        "progress_pressure":         progress_pressure,
        "cost_pressure":             cost_pressure,
        "budget_risk":               budget_risk,
        "execution_index":           execution_index,
        "budget_index":              budget_index_val,
        "operational_index":         operational_index,
        "external_index":            external_index,
        "governance_index":          governance_index,
        "total_risk_density":        total_risk_density,
        "delay":                     flag("delay"),
        "schedule_slip":             flag("schedule_slip"),
        "pacing":                    flag("pacing"),
        "pressure":                  flag("pressure"),
        "cost_overruns":             flag("cost_overruns"),
        "budget_revisions":          flag("budget_revisions"),
        "funding_risk":              fr,
        "margin_pressure":           flag("margin_pressure"),
        "errors":                    flag("errors"),
        "contractor_issues":         flag("contractor_issues"),
        "resource_shortage":         flag("resource_shortage"),
        "coordination":              flag("coordination"),
        "client_changes":            flag("client_changes"),
        "supplier_delays":           flag("supplier_delays"),
        "regulatory":                flag("regulatory"),
        "external_risk":             flag("external_risk"),
        "dependency":                flag("dependency"),
        "reporting":                 flag("reporting"),
        "decision_delay":            flag("decision_delay"),
        "risk_tracking":             flag("risk_tracking"),
        "escalation":                flag("escalation"),
        "procurement_progress":      cp,
        "progress_procurement_gap":  0.0,
        "funding_risk_bin":          fr,
        "phase_encoded":             phase_encoded,
        "situation_encoded":         situation_encoded,
    }
    return pd.DataFrame([row])[FEATURES].fillna(0)


def format_data_for_prompt(data: dict) -> str:
    cp = data.get("current_progress", 0)
    lines = [
        f"- Avancement institutionnel : {round(cp * 100, 1)} %",
        f"- Solde budgétaire restant   : {data.get('budget_gap', 0):,.0f} TND",
        f"- Nombre de marchés          : {data.get('nb_marches', 0)}",
    ]
    active = [FIELD_LABELS_FR[k] for k in BINARY_FLAGS if data.get(k, 0) == 1 and k in FIELD_LABELS_FR]
    if active:
        lines.append(f"- Signaux de risque actifs ({len(active)}) :")
        for lbl in active:
            lines.append(f"    • {lbl}")
    else:
        lines.append("- Aucun signal de risque déclaré")
    return "\n".join(lines)


def generate_explanation(data: dict, proba: float, risk_level: str) -> str:
    import os
    if not os.getenv("GROQ_API_KEY"):
        return "Clé GROQ manquante."

    pct       = round(proba * 100, 1)
    n_active  = sum(1 for k in BINARY_FLAGS if data.get(k, 0) == 1)
    niveau_fr = {"Low": "Faible", "Medium": "Modéré", "High": "Élevé", "Critical": "Critique"}.get(risk_level, risk_level)

    prompt = f"""Tu es directeur de projet dans une réunion d'avancement.

Contexte projet :
{format_data_for_prompt(data)}

Score de risque global du projet : {pct} %
Niveau de risque global : {niveau_fr}
Nombre de signaux de risque déclarés : {n_active}/21

IMPORTANT — ce score représente le risque global du projet (retards, budget, exécution, coordination, facteurs externes, gouvernance), PAS uniquement le risque de dépassement budgétaire. Ne mentionne jamais "probabilité de dépassement budgétaire" dans ta réponse.

Règles strictes :
- Parle naturellement, pas comme un rapport
- Pas de listes, pas de titres, pas de structure formelle
- Max 4–5 phrases courtes
- Utilise "niveau de risque global", "score de risque", ou "évaluation globale" — jamais "dépassement budgétaire"
- Ton adapté : Faible=rassurant, Modéré=vigilant, Élevé=sérieux, Critique=urgent
- Si plusieurs signaux actifs, mentionne les domaines les plus préoccupants (délais, budget, opérationnel, externe, gouvernance)

Réponds maintenant."""

    response = get_groq_client().chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "Tu es expert en gestion des risques de projets PPP publics tunisiens. Le score affiché est un indicateur de risque global couvrant les délais, le budget, l'exécution, la coordination, les facteurs externes et la gouvernance."},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.2,
        max_tokens=400,
    )
    return response.choices[0].message.content


@router.post("/predict")
def predict(input_data: ProjectInput):
    data  = input_data.dict()
    X     = prepare_features(data)
    raw   = float(model.predict_proba(X)[0][1])
    proba = float(CALIBRATOR.transform([raw])[0]) if CALIBRATOR is not None else raw
    proba = max(0.0, min(1.0, proba))
    risk_level  = get_risk_category(proba)
    explanation = generate_explanation(data, proba, risk_level)
    return {
        "probability": round(proba, 3),
        "risk_level":  risk_level,
        "explanation": explanation,
    }
