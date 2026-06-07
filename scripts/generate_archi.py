"""
scripts/generate_archi.py
-------------------------
Regenerates img/archi.png — the 5-layer platform architecture — fixing the
AI module: the chatbot RAG uses Groq · LLaMA 3.3 70B + all-MiniLM-L6-v2 + FAISS
(the previous diagram wrongly showed "OpenAI · GPT-OSS-120B").
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUTS = [ROOT / "_rapport" / "img" / "archi.png",
        ROOT / "assets" / "img" / "archi.png"]

plt.rcParams.update({"font.family": "DejaVu Sans"})

# layer colors (band, inner border/text accent)
L = {
    "pres": ("#10396b", "#9db8de"),
    "trait": ("#0b5d49", "#7fd3bd"),
    "stock": ("#6e5210", "#e0b765"),
    "viz":   ("#3a2f73", "#b3a7e8"),
    "ia":    ("#5e261d", "#e3a99c"),
}

fig, ax = plt.subplots(figsize=(9.2, 11))
ax.set_xlim(0, 10); ax.set_ylim(0, 12.6); ax.axis("off")
fig.patch.set_facecolor("white")

def band(y, h, title, key):
    bg, _ = L[key]
    ax.add_patch(FancyBboxPatch((0.3, y), 9.4, h, boxstyle="round,pad=0.05",
                 facecolor=bg, edgecolor="none", zorder=1))
    ax.text(0.7, y + h - 0.32, title, fontsize=11.5, fontweight="bold",
            color="white", zorder=4)

def card(cx, cy, w, h, title, subs, key):
    bg, accent = L[key]
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h, boxstyle="round,pad=0.05",
                 facecolor="none", edgecolor=accent, linewidth=1.6, zorder=3))
    ax.text(cx, cy + h/2 - 0.34, title, ha="center", fontsize=10.5,
            fontweight="bold", color="white", zorder=4)
    for i, s in enumerate(subs):
        ax.text(cx, cy + h/2 - 0.72 - i*0.32, s, ha="center", fontsize=8.2,
                color=accent, zorder=4)

def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="#9aa3b2", lw=1.6,
                                mutation_scale=14), zorder=2)

# ── Layer 1 : présentation ──
band(11.0, 1.4, "Couche présentation", "pres")
card(5.0, 11.55, 4.6, 0.95, "Joget",
     ["Saisie des données · Paramétrage · Interface utilisateur"], "pres")

# ── Layer 2 : traitement ──
band(8.9, 1.7, "Couche traitement", "trait")
card(2.9, 9.5, 3.9, 1.1, "FastAPI",
     ["Orchestration", "Backend chatbot · Prédictions"], "trait")
card(7.1, 9.5, 4.0, 1.1, "Python · Pandas",
     ["ETL · Nettoyage · Transformation", "Enrichissement des données"], "trait")
arrow(4.95, 9.5, 5.05, 9.5)

# ── Layer 3 : stockage ──
band(6.6, 1.7, "Couche stockage", "stock")
card(2.9, 7.2, 3.9, 1.1, "MySQL — Base Joget",
     ["Données applicatives", "Formulaires · Users"], "stock")
card(7.1, 7.2, 4.0, 1.1, "MySQL — cleaned_dw",
     ["Données traitées", "Tables de faits · Dimensions"], "stock")
arrow(4.95, 7.2, 5.05, 7.2)

# ── Layer 4 : visualisation ──
band(4.3, 1.7, "Couche visualisation", "viz")
card(2.9, 4.9, 3.9, 1.1, "Apache Superset",
     ["Tableaux de bord dynamiques · KPI"], "viz")
card(7.1, 4.9, 4.0, 1.1, "Chart.js",
     ["Visualisations web interactives"], "viz")

# ── Layer 5 : IA ──
band(1.7, 1.9, "Modules intelligence artificielle", "ia")
card(2.9, 2.45, 3.9, 1.25, "Groq · LLaMA 3.3 70B",
     ["Chatbot intelligent · RAG", "FAISS + all-MiniLM-L6-v2"], "ia")
card(7.1, 2.45, 4.0, 1.25, "XGBClassifier",
     ["Prédiction des risques projet", "Modélisation · Scoring"], "ia")

# vertical connectors between layers (center)
for y1, y2 in [(11.0, 10.6), (8.9, 8.3), (6.6, 6.0), (4.3, 3.6)]:
    arrow(5.0, y1, 5.0, y2)

plt.tight_layout()
for out in OUTS:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out.relative_to(ROOT)}")
plt.close(fig)
