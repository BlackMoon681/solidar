"""
scripts/generate_dw_schema.py
-----------------------------
Generates the cleaned_dw galaxy-schema ERD with table boxes, column names,
PK/FK markers and relationship lines — replacing constellation_schema_dw.png.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUTS = [ROOT / "_rapport" / "img" / "constellation_schema_dw.png",
        ROOT / "assets" / "img" / "constellation_schema_dw.png"]

plt.rcParams.update({"font.family": "DejaVu Sans"})

C_FACT = "#1d4ed8"
C_PPP  = "#0e7a5f"
C_GEO  = "#7c3aed"
C_PGD  = "#b45309"
C_LINE = "#b9c0cc"
ROW_H  = 0.32
HEAD_H = 0.42

fig, ax = plt.subplots(figsize=(26, 16))
ax.set_xlim(0, 26); ax.set_ylim(0, 16); ax.axis("off")
fig.patch.set_facecolor("white")

def table(x, y_top, w, title, rows, color):
    """Draw a table; return dict of anchors. y_top = top edge."""
    n = len(rows)
    total_h = HEAD_H + n * ROW_H
    # header
    ax.add_patch(FancyBboxPatch((x, y_top - HEAD_H), w, HEAD_H,
        boxstyle="round,pad=0.01,rounding_size=0.06", facecolor=color,
        edgecolor=color, linewidth=1, zorder=4))
    ax.text(x + w/2, y_top - HEAD_H/2, title, ha="center", va="center",
            fontsize=8.6, fontweight="bold", color="white", zorder=5)
    # body
    ax.add_patch(Rectangle((x, y_top - total_h), w, n * ROW_H,
        facecolor="white", edgecolor=color, linewidth=1.2, zorder=3))
    for i, (txt, kind) in enumerate(rows):
        ry = y_top - HEAD_H - i * ROW_H - ROW_H/2
        if i:
            ax.plot([x, x + w], [y_top - HEAD_H - i*ROW_H]*2, color="#eef0f4", lw=0.5, zorder=4)
        tag = {"pk": "PK ", "fk": "FK ", "": ""}[kind]
        fw = "bold" if kind == "pk" else "normal"
        col = "#b91c1c" if kind == "fk" else ("#111827" if kind == "pk" else "#374151")
        ax.text(x + 0.12, ry, tag + txt, ha="left", va="center",
                fontsize=7.0, color=col, fontweight=fw, zorder=5)
    return {"l": (x, y_top - total_h/2), "r": (x + w, y_top - total_h/2),
            "t": (x + w/2, y_top), "b": (x + w/2, y_top - total_h)}

def link(a, b, color=C_LINE):
    ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=0.9, zorder=1, alpha=0.55)

# ─────────────────────────────────────────────────────────────
# FACTS (center column)
# ─────────────────────────────────────────────────────────────
fact_pgd = table(11.0, 15.6, 3.6, "fact_pgd", [
    ("id", "pk"), ("c_projet", ""), ("c_promoteur", ""),
    ("c_bailleur", "fk"), ("c_region", ""), ("c_annee", ""),
    ("c_budget_tnd", ""),
], C_FACT)

fact_ppp = table(10.7, 11.2, 4.2, "fact_ppp", [
    ("id", "pk"), ("c_nom_projet", ""),
    ("c_axe", "fk"), ("c_sous_axe", "fk"), ("c_classification", "fk"),
    ("c_mode_financement", "fk"), ("c_gouvernorat", "fk"),
    ("c_delegation", "fk"), ("c_commune", "fk"),
    ("c_budget_global_planifie", ""), ("c_part_budget_etat", ""),
    ("risque_retard", ""), ("risque_depassement", ""),
    ("+ 13 colonnes *_label", ""),
], C_FACT)

fact_etude = table(11.0, 4.3, 3.6, "fact_pgd_etude", [
    ("id", "pk"), ("c_intitule", ""), ("c_domaine", "fk"),
    ("c_theme", "fk"), ("c_categorie", "fk"), ("c_bailleur", "fk"),
    ("c_annee", ""), ("c_cout_tnd", ""),
], C_FACT)

# ─────────────────────────────────────────────────────────────
# PPP dimensions — two left columns
# ─────────────────────────────────────────────────────────────
def dim(x, y_top, name, extra=None, color=C_PPP, w=3.3):
    rows = [("key", "pk"), ("label", "")]
    if extra:
        rows.append((extra, "fk"))
    return table(x, y_top, w, name, rows, color)

colA_x, colB_x = 0.4, 4.1
ppp_colA = [
    ("dim_axe", None), ("dim_classification", None),
    ("dim_mode_financement", None), ("dim_mode_passation", None),
    ("dim_tutelle_sectorielle", None), ("dim_zonage", None),
    ("dim_situation", None),
]
ppp_colB = [
    ("dim_sous_axe", "axe_key"), ("dim_sous_classification", "classification_key"),
    ("dim_acteur_implementation", None), ("dim_delimitation", None),
    ("dim_initiateur", None), ("dim_owner", None), ("dim_donateur", None),
]

def stack(col_x, items, y0=15.6, gap=0.55):
    y = y0
    anchors = []
    for name, extra in items:
        a = dim(col_x, y, name, extra)
        anchors.append(a)
        h = HEAD_H + (3 if extra else 2) * ROW_H
        y -= h + gap
    return anchors

aA = stack(colA_x, ppp_colA)
aB = stack(colB_x, ppp_colB)
for a in aA:   # far-left → route to fact_ppp left
    link(a["r"], fact_ppp["l"])
for a in aB:
    link(a["r"], fact_ppp["l"])

# ─────────────────────────────────────────────────────────────
# Right column — geo dims (linked to PPP) + PGD dims
# ─────────────────────────────────────────────────────────────
geo_x, pgd_x = 17.0, 21.4
geo = [
    dim(geo_x, 15.6, "dim_gouvernorat", None, C_GEO),
    dim(geo_x, 13.6, "dim_delegation", "gouvernorat_key", C_GEO),
    dim(geo_x, 11.2, "dim_secteur", "delegation_key", C_GEO),
]
# geo hierarchy chain + link to fact_ppp
link(geo[0]["b"], geo[1]["t"], C_GEO)
link(geo[1]["b"], geo[2]["t"], C_GEO)
for g in geo:
    link(g["l"], fact_ppp["r"], color="#d9c9f5")

pgd_dims = [
    dim(pgd_x, 8.6, "dim_bailleur_pgd", None, C_PGD),
    dim(pgd_x, 6.4, "dim_domaine_pgd", None, C_PGD),
    dim(pgd_x, 4.4, "dim_theme_pgd", None, C_PGD),
    dim(pgd_x, 2.4, "dim_categorie_pgd", None, C_PGD),
]
link(pgd_dims[0]["l"], fact_pgd["r"], color="#f0ddbf")
for d in pgd_dims:
    link(d["l"], fact_etude["r"], color="#f0ddbf")

# ─────────────────────────────────────────────────────────────
# Title + legend
# ─────────────────────────────────────────────────────────────
ax.text(13, 15.85, "Entrepôt cleaned_dw — schéma en constellation (galaxy schema)",
        ha="center", fontsize=15, fontweight="bold", color="#111827")
ax.text(13, 15.5, "3 tables de faits · 21 dimensions · PK = clé primaire, FK = clé étrangère",
        ha="center", fontsize=10, color="#6b7280")

from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(facecolor=C_FACT, label="Tables de faits"),
    Patch(facecolor=C_PPP,  label="Dimensions PPP"),
    Patch(facecolor=C_GEO,  label="Dimensions géographiques"),
    Patch(facecolor=C_PGD,  label="Dimensions PGD / Études"),
], loc="lower center", ncol=4, fontsize=10, framealpha=0.95, bbox_to_anchor=(0.5, 0.0))

plt.tight_layout()
for out in OUTS:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out.relative_to(ROOT)}")
plt.close(fig)
