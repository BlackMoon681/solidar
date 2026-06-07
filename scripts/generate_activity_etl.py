"""
scripts/generate_activity_etl.py
---------------------------------
Generates activity_etl.png — updated ETL activity diagram reflecting the
galaxy-schema pipeline: dimensions + 3 facts loaded into cleaned_dw.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUTS = [ROOT / "activity_etl.png",
        ROOT / "_rapport" / "img" / "activity_etl.png"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})

# ─── professional palette ────────────────────────────────────────────────────
C_BG    = "#F7F8FA"          # off-white background
C_EX_BG = "#EBF2FA"          # Extract lane fill
C_TR_BG = "#EAF4EF"          # Transform lane fill
C_LD_BG = "#FDF3E7"          # Load lane fill

C_EX_BOX = "#FFFFFF"         # box fill
C_TR_BOX = "#FFFFFF"
C_LD_BOX = "#FFFFFF"

C_EX_ACC = "#2C6FAC"         # Extract accent (border + header)
C_TR_ACC = "#1E8A5E"         # Transform accent
C_LD_ACC = "#C06000"         # Load accent

C_ERR_BG  = "#FEF2F2"
C_ERR_ACC = "#C0392B"

C_DIAM_BG  = "#FFFBEB"
C_DIAM_ACC = "#B7950B"

C_ARROW = "#4A4A5A"
C_HDR   = "#1C2833"          # swimlane header text

BORDER  = "#CCCCCC"

fig, ax = plt.subplots(figsize=(11, 14))
ax.set_xlim(0, 11)
ax.set_ylim(0, 14)
ax.axis("off")
fig.patch.set_facecolor(C_BG)

# ─── swimlane bands ──────────────────────────────────────────────────────────
LANE_BOUNDS = [(0.1, 3.55), (3.6, 7.1), (7.15, 10.9)]
LANE_LABELS = ["Extract", "Transform", "Load"]
LANE_COLORS = [C_EX_BG, C_TR_BG, C_LD_BG]
LANE_ACCS   = [C_EX_ACC, C_TR_ACC, C_LD_ACC]

for (x0, x1), label, color, acc in zip(LANE_BOUNDS, LANE_LABELS, LANE_COLORS, LANE_ACCS):
    # main band
    ax.add_patch(FancyBboxPatch((x0, 0.2), x1-x0, 13.4,
                                boxstyle="round,pad=0.08",
                                facecolor=color, edgecolor="#BBBBBB", lw=0.9, zorder=0))
    # header bar
    ax.add_patch(FancyBboxPatch((x0, 13.0), x1-x0, 0.6,
                                boxstyle="round,pad=0.08",
                                facecolor=acc, edgecolor=acc, lw=0, zorder=1))
    ax.text((x0+x1)/2, 13.3, label, ha="center", va="center",
            fontsize=11, fontweight="bold", color="white", zorder=2)

# ─── helpers ─────────────────────────────────────────────────────────────────
def box(ax, x, y, w, h, text, sub=None, fc=C_EX_BOX, ec=C_EX_ACC, lw=1.2,
        fs=8.5, bold=False, sub_color="#666666"):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
                                boxstyle="round,pad=0.14",
                                facecolor=fc, edgecolor=ec, lw=lw, zorder=3))
    dy = 0.13 if sub else 0
    ax.text(x, y+dy, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal",
            color=C_HDR, zorder=4)
    if sub:
        ax.text(x, y-0.22, sub, ha="center", va="center",
                fontsize=6.8, color=sub_color, zorder=4)

def diamond(ax, x, y, w, h, text):
    pts = [(x, y+h/2), (x+w/2, y), (x, y-h/2), (x-w/2, y)]
    ax.add_patch(plt.Polygon(pts, closed=True,
                             facecolor=C_DIAM_BG, edgecolor=C_DIAM_ACC,
                             lw=1.2, zorder=3))
    ax.text(x, y, text, ha="center", va="center",
            fontsize=7.8, style="italic", color="#444", zorder=4)

def arr(ax, x0, y0, x1, y1, color=C_ARROW, lw=1.1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=11), zorder=5)

def label_txt(ax, x, y, text, color=C_ARROW):
    ax.text(x, y, text, fontsize=7.2, color=color, va="center", zorder=5)

def start_node(ax, x, y):
    ax.add_patch(plt.Circle((x, y), 0.20, fc=C_HDR, ec=C_HDR, zorder=6))

def end_node(ax, x, y):
    ax.add_patch(plt.Circle((x, y), 0.24, fc="#fff", ec=C_HDR, lw=1.5, zorder=6))
    ax.add_patch(plt.Circle((x, y), 0.14, fc=C_HDR, ec=C_HDR, zorder=7))

# ─── lane centres ─────────────────────────────────────────────────────────────
EX = (LANE_BOUNDS[0][0] + LANE_BOUNDS[0][1]) / 2   # ≈ 1.825
TR = (LANE_BOUNDS[1][0] + LANE_BOUNDS[1][1]) / 2   # ≈ 5.35
LD = (LANE_BOUNDS[2][0] + LANE_BOUNDS[2][1]) / 2   # ≈ 9.025

# ─── EXTRACT ─────────────────────────────────────────────────────────────────
start_node(ax, EX, 12.4)
arr(ax, EX, 12.20, EX, 11.73)

box(ax, EX, 11.43, 2.8, 0.52, "Connexion à jwdb",
    fc=C_EX_BOX, ec=C_EX_ACC, lw=1.6, bold=True)
arr(ax, EX, 11.17, EX, 10.60)

box(ax, EX, 10.30, 2.8, 0.58, "Extraire tables sources",
    sub="ppp / pgd / pgd_etude  +  21 tables param/géo",
    fc=C_EX_BOX, ec=C_EX_ACC)
arr(ax, EX, 10.01, EX, 9.38)

diamond(ax, EX, 9.05, 2.4, 0.60, "Données OK ?")

# ── Non → L-shaped path: right from diamond, up to error box (between Extraire and diamond) ──
ERR_X = EX + 1.55          # right of diamond centre, inside Extract lane
ERR_Y = 9.72               # above diamond (9.05) and below Extraire (10.30)
# L-shaped connector: horizontal right then up (no arrowhead on the line)
ax.plot([EX+1.2, ERR_X, ERR_X], [9.05, 9.05, ERR_Y-0.23],
        color=C_ERR_ACC, lw=1.1, zorder=5)
# arrowhead into the box top
arr(ax, ERR_X, ERR_Y-0.23, ERR_X, ERR_Y-0.21, color=C_ERR_ACC)
label_txt(ax, EX+1.22, 9.13, "Non", color=C_ERR_ACC)
box(ax, ERR_X, ERR_Y, 1.0, 0.42, "Erreur / Log",
    fc=C_ERR_BG, ec=C_ERR_ACC, lw=1.2, fs=7.8)

# ── Oui ↓ ──
label_txt(ax, EX+0.08, 8.75, "Oui", color="#1A7A40")
arr(ax, EX, 8.74, EX, 8.12)

box(ax, EX, 7.82, 2.8, 0.52, "Journaliser nb lignes",
    sub="ppp / pgd / pgd_etude",
    fc=C_EX_BOX, ec=C_EX_ACC)

# Extract → Transform
arr(ax, EX+1.4, 7.82, TR-1.75, 7.60, color=C_ARROW)

# ─── TRANSFORM ───────────────────────────────────────────────────────────────
box(ax, TR, 7.35, 2.8, 0.58, "build_dimensions()",
    sub="21 dim_*  (UUID keys + integer geo keys)",
    fc=C_TR_BOX, ec=C_TR_ACC)
arr(ax, TR, 7.06, TR, 6.48)

box(ax, TR, 6.17, 2.8, 0.54, "clean_ppp()",
    sub="risque_retard / gros_retard / depassement  +  FK → labels",
    fc=C_TR_BOX, ec=C_TR_ACC)
arr(ax, TR, 5.90, TR, 5.32)

box(ax, TR, 5.01, 2.8, 0.54, "clean_pgd()",
    sub="c_budget → c_budget_tnd  (TND / EUR / USD)",
    fc=C_TR_BOX, ec=C_TR_ACC)
arr(ax, TR, 4.74, TR, 4.17)

box(ax, TR, 3.86, 2.8, 0.54, "clean_pgd_etude()",
    sub="c_cout → c_cout_tnd  (TND)",
    fc=C_TR_BOX, ec=C_TR_ACC)
arr(ax, TR, 3.59, TR, 3.00)

box(ax, TR, 2.68, 2.8, 0.54, "Stats & journalisation",
    sub="distribution risques  par dataset",
    fc=C_TR_BOX, ec=C_TR_ACC)

# Transform → Load  (dims)
arr(ax, TR+1.4, 7.35, LD-1.45, 7.15, color=C_ARROW)
# Transform → Load  (facts)
arr(ax, TR+1.4, 2.68, LD-1.45, 4.55, color=C_ARROW)

# ─── LOAD ─────────────────────────────────────────────────────────────────────
box(ax, LD, 6.90, 2.8, 0.58, "ensure_warehouse()",
    sub="CREATE DATABASE IF NOT EXISTS cleaned_dw",
    fc=C_LD_BOX, ec=C_LD_ACC, lw=1.6, bold=True)
arr(ax, LD, 6.61, LD, 6.02)

box(ax, LD, 5.71, 2.8, 0.54, "load_dimensions()",
    sub="DROP + INSERT  ×21  dim_* tables",
    fc=C_LD_BOX, ec=C_LD_ACC)
arr(ax, LD, 5.44, LD, 4.84)

box(ax, LD, 4.53, 2.8, 0.54, "load_facts()",
    sub="fact_ppp / fact_pgd / fact_pgd_etude",
    fc=C_LD_BOX, ec=C_LD_ACC)
arr(ax, LD, 4.26, LD, 3.67)

box(ax, LD, 3.36, 2.8, 0.54, "to_sql()  chunksize=300",
    sub="utf8mb4  —  DROP + replace",
    fc=C_LD_BOX, ec=C_LD_ACC)
arr(ax, LD, 3.09, LD, 2.48)

end_node(ax, LD, 2.28)

# ─── caption ──────────────────────────────────────────────────────────────────
ax.text(5.5, 0.55,
        "Diagramme d'activité — Pipeline ETL  ·  jwdb  →  cleaned_dw  (schéma galaxie)",
        ha="center", va="center", fontsize=8, color="#888", style="italic")

plt.tight_layout(pad=0)
for out in OUTS:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=C_BG)
    print(f"Saved: {out}")
plt.close()
