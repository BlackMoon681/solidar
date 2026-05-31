"""
scripts/generate_chapter05_figures.py
--------------------------------------
Generates all 7 figures for Chapter 5 (Release 3 — Sprint 5).

Figures produced:
  assets/img/uc_sprint5.png               — Use case diagram
  assets/img/chapter05/01_pipeline_flow.png         — Pipeline automation flowchart
  assets/img/chapter05/02_architecture_comparison.png — Before/after refactoring
  assets/img/chapter05/03_test_results.png           — Test results overview
  assets/img/chapter05/04_benchmarks.png             — Performance benchmarks
  assets/img/chapter05/05_security_controls.png      — Security controls
  assets/img/chapter05/06_versioning_flow.png        — ML versioning flowchart

Run:
    python scripts/generate_chapter05_figures.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path

ROOT    = Path(__file__).parent.parent
OUT_DIR = ROOT / "assets" / "img" / "chapter05"
OUT_DIR.mkdir(parents=True, exist_ok=True)
UC_DIR  = ROOT / "assets" / "img"

DPI   = 180
C_BG  = "#FAFAFA"
C_BOX = "#2C3E50"     # dark blue – boxes/actors
C_GRN = "#27AE60"     # green  – pass / promote
C_RED = "#E74C3C"     # red    – fail / reject
C_ORG = "#F39C12"     # orange – warning / corrected
C_BLU = "#2980B9"     # blue   – neutral / info
C_LBL = "#FFFFFF"     # white label on coloured boxes
C_TXT = "#2C3E50"     # body text

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "figure.facecolor": C_BG,
    "axes.facecolor":   C_BG,
})

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def save(fig, path, tight=True):
    if tight:
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=C_BG)
    else:
        fig.savefig(path, dpi=DPI, facecolor=C_BG)
    plt.close(fig)
    print(f"  Saved -> {path.relative_to(ROOT)}")


def fbox(ax, x, y, w, h, text, facecolor=C_BLU, textcolor=C_LBL,
         fontsize=8.5, fontweight="normal", radius=0.04, alpha=1.0,
         wrap=False):
    box = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                         boxstyle=f"round,pad={radius}",
                         facecolor=facecolor, edgecolor="white",
                         linewidth=1.2, alpha=alpha, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight,
            color=textcolor, zorder=4,
            wrap=wrap, multialignment="center")


def diamond(ax, cx, cy, w, h, text, facecolor=C_ORG, textcolor="white", fontsize=8):
    pts = np.array([[cx, cy + h/2], [cx + w/2, cy],
                    [cx, cy - h/2], [cx - w/2, cy]])
    poly = plt.Polygon(pts, closed=True, facecolor=facecolor,
                       edgecolor="white", linewidth=1.2, zorder=3)
    ax.add_patch(poly)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, color=textcolor,
            fontweight="bold", zorder=4, multialignment="center")


def arrow(ax, x1, y1, x2, y2, color="#888", lw=1.5, label="", label_side="right"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=12),
                zorder=2)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dx = 0.12 if label_side == "right" else -0.12
        ax.text(mx + dx, my, label, ha="left" if label_side == "right" else "right",
                va="center", fontsize=7.5, color="#555")


def actor(ax, cx, cy, label, color=C_BOX):
    r = 0.22
    head = plt.Circle((cx, cy + r), r, fill=False, edgecolor=color, lw=1.8, zorder=3)
    ax.add_patch(head)
    ax.plot([cx, cx], [cy + r * 0.1, cy - r * 1.5], color=color, lw=1.8, zorder=3)
    ax.plot([cx - r * 1.5, cx + r * 1.5], [cy - r * 0.5, cy - r * 0.5],
            color=color, lw=1.8, zorder=3)
    ax.plot([cx, cx - r * 1.2], [cy - r * 1.5, cy - r * 3.2],
            color=color, lw=1.8, zorder=3)
    ax.plot([cx, cx + r * 1.2], [cy - r * 1.5, cy - r * 3.2],
            color=color, lw=1.8, zorder=3)
    ax.text(cx, cy - r * 3.8, label, ha="center", va="top",
            fontsize=9, fontweight="bold", color=color)


def ell(ax, cx, cy, rw, rh, text, facecolor="white", edgecolor=C_BOX,
        fontsize=8.2, textcolor=C_TXT):
    e = mpatches.Ellipse((cx, cy), rw, rh, facecolor=facecolor,
                         edgecolor=edgecolor, linewidth=1.4, zorder=3)
    ax.add_patch(e)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, color=textcolor, zorder=4, multialignment="center")


def uc_line(ax, ax1, ay1, ex, ey, style="-"):
    ax.plot([ax1, ex], [ay1, ey], color="#888", lw=1.2, zorder=2,
            linestyle=style)


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — Use Case Diagram  (uc_sprint5.png)
# ══════════════════════════════════════════════════════════════════════════════

def fig_use_case():
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")
    fig.patch.set_facecolor(C_BG)

    # System boundary
    sb = FancyBboxPatch((2.0, 0.4), 10, 8.1,
                        boxstyle="round,pad=0.1",
                        facecolor="#EBF5FB", edgecolor=C_BOX,
                        linewidth=2.0, zorder=1)
    ax.add_patch(sb)
    ax.text(7.0, 8.3, "Système PPP-Risk — Sprint 5", ha="center", va="center",
            fontsize=12, fontweight="bold", color=C_BOX, zorder=5)

    # Actors
    actor(ax, 0.8, 4.5, "Administrateur")
    actor(ax, 13.2, 5.5, "Agent métier")

    # Use case ellipses (x, y, label)
    ucs = [
        (5.5, 7.4, "Déclencher le\npipeline manuellement"),
        (5.5, 5.9, "Consulter l'état\ndu pipeline"),
        (5.5, 4.4, "Consulter les versions\ndu modèle ML"),
        (5.5, 2.9, "Rollback vers version\nprécédente"),
        (5.5, 1.4, "Exécuter la suite\nde tests"),
        (9.5, 7.4, "Consulter les\nbenchmarks"),
        (9.5, 5.9, "Consulter l'état\ndu pipeline"),
        (9.5, 4.4, "Consulter les\nrapports de sécurité"),
    ]
    for (cx, cy, txt) in ucs:
        ell(ax, cx, cy, 3.2, 0.9, txt)

    # Admin connections
    admin_x, admin_y_base = 1.5, 4.5
    for cx, cy, _ in ucs[:5]:
        ax.plot([admin_x, cx - 1.6], [admin_y_base, cy], color="#888", lw=1.1, zorder=2)

    # Agent connections
    agent_x, agent_y_base = 12.6, 5.5
    for cx, cy, _ in ucs[5:]:
        ax.plot([agent_x, cx + 1.6], [agent_y_base, cy], color="#888", lw=1.1, zorder=2)

    # <<include>> between overlapping use cases (same uc shared)
    ax.annotate("", xy=(8.04, 5.9), xytext=(7.12, 5.9),
                arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.0,
                                mutation_scale=10), zorder=2)
    ax.text(7.6, 6.1, "<<include>>", ha="center", fontsize=7, color="#666",
            style="italic")

    # Legend
    leg = [
        mpatches.Patch(facecolor="#EBF5FB", edgecolor=C_BOX, label="Frontière du système"),
        Line2D([0], [0], color="#888", lw=1.2, label="Association acteur / cas"),
    ]
    ax.legend(handles=leg, loc="lower left", fontsize=8, framealpha=0.8,
              bbox_to_anchor=(0.0, 0.0))

    save(fig, UC_DIR / "uc_sprint5.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — Pipeline Automation Flowchart
# ══════════════════════════════════════════════════════════════════════════════

def fig_pipeline_flow():
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8)
    ax.axis("off")
    fig.patch.set_facecolor(C_BG)

    ax.set_title("Architecture du pipeline d'automatisation", fontsize=12,
                 fontweight="bold", color=C_BOX, pad=8)

    # ── Trigger column (left) ──
    fbox(ax, 1.5, 7.0, 2.4, 0.6, "APScheduler\nMinuit (Africa/Tunis)",
         facecolor="#8E44AD", fontsize=8)
    fbox(ax, 1.5, 5.9, 2.4, 0.6, "POST /pipeline/run\n(déclenchement manuel)",
         facecolor="#8E44AD", fontsize=8)

    # Merge arrows
    arrow(ax, 1.5, 6.7, 1.5, 6.2)
    ax.plot([1.5, 3.5], [6.2, 6.2], color="#888", lw=1.5, zorder=2)
    ax.plot([1.5, 6.0], [5.6, 5.6], color="#888", lw=1.2, ls="--", zorder=2)

    # ── Stage 1 — ETL ──
    fbox(ax, 6.0, 7.0, 2.8, 0.65, "Étape 1 — ETL\nDétection multi-tables",
         facecolor=C_BLU)
    diamond(ax, 6.0, 6.0, 2.6, 0.75, "Nouvelles\nlignes ?", facecolor=C_ORG)
    fbox(ax, 8.6, 6.0, 1.6, 0.55, "Skip", facecolor="#95A5A6", fontsize=8)
    fbox(ax, 6.0, 5.0, 2.8, 0.65, "python -m etl.main\n+ mise à jour état",
         facecolor=C_GRN)

    arrow(ax, 6.0, 6.68, 6.0, 6.38)
    arrow(ax, 6.0, 5.62, 6.0, 5.32)
    ax.plot([7.3, 8.6], [6.0, 6.0], color="#888", lw=1.5, zorder=2)
    arrow(ax, 8.6, 6.0, 8.6, 5.85, color="#888")
    ax.text(7.9, 5.72, "NON", ha="center", fontsize=7.5, color="#888")
    ax.text(6.1, 5.75, "OUI", ha="left", fontsize=7.5, color=C_GRN)

    # ── Stage 2 — Chatbot ──
    fbox(ax, 6.0, 4.2, 2.8, 0.65, "Étape 2 — Chatbot\nVérification hash MD5",
         facecolor=C_BLU)
    diamond(ax, 6.0, 3.3, 2.6, 0.75, "Documents\nmodifiés ?", facecolor=C_ORG)
    fbox(ax, 8.6, 3.3, 1.6, 0.55, "Skip", facecolor="#95A5A6", fontsize=8)
    fbox(ax, 6.0, 2.4, 2.8, 0.65, "Rebuild FAISS index\nbuild_index.py",
         facecolor=C_GRN)

    arrow(ax, 6.0, 4.86, 6.0, 4.53)
    arrow(ax, 6.0, 3.93, 6.0, 3.68)
    arrow(ax, 6.0, 2.93, 6.0, 2.73)
    ax.plot([7.3, 8.6], [3.3, 3.3], color="#888", lw=1.5, zorder=2)
    arrow(ax, 8.6, 3.3, 8.6, 3.16, color="#888")
    ax.text(7.9, 3.02, "NON", ha="center", fontsize=7.5, color="#888")
    ax.text(6.1, 3.07, "OUI", ha="left", fontsize=7.5, color=C_GRN)

    # ── Stage 3 — ML ──
    fbox(ax, 6.0, 1.7, 2.8, 0.65, "Étape 3 — ML\nContrôle seuils",
         facecolor=C_BLU)
    diamond(ax, 6.0, 0.85, 2.6, 0.75, "Seuils atteints\n(150 lignes, 30+) ?",
            facecolor=C_ORG)
    fbox(ax, 8.6, 0.85, 1.6, 0.55, "Skip", facecolor="#95A5A6", fontsize=8)
    fbox(ax, 4.0, 0.85, 1.6, 0.55, "Retrain +\nAUC gate", facecolor=C_GRN, fontsize=7.5)

    arrow(ax, 6.0, 2.07, 6.0, 1.88)    # stage box down
    arrow(ax, 6.0, 1.32, 6.0, 1.23)    # stage to diamond
    ax.plot([7.3, 8.6], [0.85, 0.85], color="#888", lw=1.5, zorder=2)
    arrow(ax, 8.6, 0.85, 8.6, 0.7, color="#888")
    ax.text(7.9, 0.57, "NON", ha="center", fontsize=7.5, color="#888")
    ax.plot([4.7, 4.7], [0.85, 0.85], color="#888", lw=1.5, zorder=2)
    arrow(ax, 4.7, 0.85, 4.8, 0.85, color="#888")
    ax.text(5.0, 1.02, "OUI", ha="center", fontsize=7.5, color=C_GRN)

    # ── Connector between stages ──
    arrow(ax, 3.7, 6.2, 4.6, 6.2, color="#555")
    ax.plot([3.5, 4.6], [6.2, 6.2], color="#555", lw=1.5, zorder=2)
    ax.plot([4.6, 4.6], [6.2, 7.0], color="#555", lw=1.5, zorder=2)
    arrow(ax, 4.6, 7.0, 4.6, 7.0, color="#555")

    # Stage 1→2→3 vertical connectors on left side
    for y_from, y_to in [(4.86, 4.53), ]:
        pass  # already drawn above

    # Left vertical spine connecting stages
    ax.plot([3.5, 3.5], [0.85, 7.0], color="#555", lw=1.2, ls="--", zorder=1, alpha=0.4)

    # ── Legend ──
    handles = [
        mpatches.Patch(facecolor=C_BLU,  label="Étape principale"),
        mpatches.Patch(facecolor=C_ORG,  label="Décision (condition)"),
        mpatches.Patch(facecolor=C_GRN,  label="Action exécutée"),
        mpatches.Patch(facecolor="#95A5A6", label="Ignoré (skip)"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8,
              framealpha=0.85, bbox_to_anchor=(0.0, 0.0))

    save(fig, OUT_DIR / "01_pipeline_flow.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — Architecture Before / After
# ══════════════════════════════════════════════════════════════════════════════

def fig_architecture():
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 8))
    fig.patch.set_facecolor(C_BG)

    for ax in (ax_l, ax_r):
        ax.set_xlim(0, 7)
        ax.set_ylim(0, 9)
        ax.axis("off")
        ax.set_facecolor(C_BG)

    # ── LEFT: Before ──
    ax_l.set_title("Avant — api.py monolithique (500+ lignes)", fontsize=10.5,
                   fontweight="bold", color=C_RED, pad=6)

    mono = FancyBboxPatch((0.5, 0.5), 6, 7.8, boxstyle="round,pad=0.1",
                          facecolor="#FADBD8", edgecolor=C_RED, linewidth=2, zorder=1)
    ax_l.add_patch(mono)
    ax_l.text(3.5, 8.1, "api.py", ha="center", va="center",
              fontsize=13, fontweight="bold", color=C_RED)

    items_before = [
        "Imports & dotenv",
        "APScheduler lifespan",
        "CORS Middleware",
        "Model loading (joblib)",
        "FAISS + embed_model loading",
        "Groq client",
        "ProjectInput / Question (Pydantic)",
        "prepare_features()",
        "get_risk_category()",
        "format_data_for_prompt()",
        "generate_explanation()",
        "search() — FAISS",
        "generate_chat_answer()",
        "POST /predict",
        "POST /chat + OPTIONS",
        "GET /pipeline/status",
        "POST /pipeline/run",
        "GET /model/versions",
        "POST /model/rollback/{version}",
        "GET /",
    ]
    for i, txt in enumerate(items_before):
        y = 7.6 - i * 0.35
        color = C_RED if i % 2 == 0 else "#922B21"
        ax_l.text(1.0, y, f"• {txt}", va="center", fontsize=7.2, color=color)

    # ── RIGHT: After ──
    ax_r.set_title("Après — Architecture modulaire (7 modules)", fontsize=10.5,
                   fontweight="bold", color=C_GRN, pad=6)

    modules = [
        # (y_center, label, sublabels, color)
        (8.0, "app/", [
            "main.py — app factory, CORS, lifespan",
            "schemas.py — ProjectInput, Question",
            "dependencies.py — ML + chatbot loading",
            "routes/predict.py — /predict",
            "routes/chat.py — /chat",
            "routes/pipeline.py — /pipeline/*",
        ], "#1A5276"),
        (4.7, "ml/", [
            "train.py — entraînement + versioning",
            "versioning.py — registre + rollback",
            "report.py — rapport comparatif",
        ], "#1A5276"),
        (2.9, "etl/", [
            "extract.py  transform.py  load.py  main.py",
        ], "#1A5276"),
        (2.0, "chatbot/", [
            "data_prep.py  build_index.py",
            "data/  index/",
        ], "#1A5276"),
        (1.1, "core/config.py", [
            "Configuration DB centralisée",
        ], "#1A5276"),
    ]

    for (yc, label, subs, col) in modules:
        h = 0.4 + len(subs) * 0.38
        box = FancyBboxPatch((0.4, yc - h / 2 - 0.15), 6.2, h + 0.05,
                             boxstyle="round,pad=0.05",
                             facecolor="#EAF2FF", edgecolor=col, linewidth=1.5, zorder=2)
        ax_r.add_patch(box)
        ax_r.text(0.85, yc + h / 2 - 0.05, label, va="top",
                  fontsize=9, fontweight="bold", color=col)
        for j, s in enumerate(subs):
            ax_r.text(1.1, yc + h / 2 - 0.42 - j * 0.38, f"  {s}",
                      va="center", fontsize=7.5, color="#555")

    # pipeline.py at root
    fbox(ax_r, 3.5, 0.3, 5.8, 0.42, "pipeline.py  —  orchestrateur principal (racine du projet)",
         facecolor="#D5DBDB", textcolor=C_TXT, fontsize=8)

    # arrow between panels
    fig.text(0.505, 0.5, "→", ha="center", va="center", fontsize=28,
             color="#888", fontweight="bold")

    save(fig, OUT_DIR / "02_architecture_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 4 — Test Results Overview
# ══════════════════════════════════════════════════════════════════════════════

def fig_test_results():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5),
                             gridspec_kw={"width_ratios": [1.4, 1]})
    fig.patch.set_facecolor(C_BG)
    fig.suptitle("Résultats de la validation — Sprint 5", fontsize=12,
                 fontweight="bold", color=C_BOX, y=1.01)

    # ── Left: bar chart per test case ──
    ax = axes[0]
    ax.set_facecolor(C_BG)
    ax.spines[["top", "right"]].set_visible(False)

    categories = {
        "Fonctionnels": {
            "ids":    ["F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08", "F09"],
            "status": ["PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "SKIP"],
        },
        "Performance": {
            "ids":    ["P01", "P02", "P03", "P04", "P05"],
            "status": ["PASS", "PASS", "PASS", "PASS", "PASS"],
        },
        "Sécurité": {
            "ids":    ["S01", "S02", "S03", "S04", "S05", "S06", "S07"],
            "status": ["PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS"],
        },
    }

    color_map = {"PASS": C_GRN, "SKIP": C_ORG, "CORR": C_GRN, "FAIL": C_RED}
    label_map = {"PASS": "Passé", "SKIP": "Ignoré", "CORR": "Passé", "FAIL": "Échoué"}

    y_pos = 0
    yticks, ylabels = [], []
    for suite, data in categories.items():
        for tc_id, status in zip(data["ids"], data["status"]):
            col = color_map[status]
            ax.barh(y_pos, 1, color=col, height=0.72, left=0, zorder=3)
            ax.text(0.03, y_pos, f"TC-{tc_id}", va="center",
                    fontsize=8, color="white", fontweight="bold")
            yticks.append(y_pos)
            ylabels.append("")
            y_pos += 1
        # Suite label
        center = y_pos - len(data["ids"]) / 2.0 - 0.5
        ax.text(-0.05, center + len(data["ids"]) / 2.0 - 0.5,
                suite, va="center", ha="right",
                fontsize=9, fontweight="bold", color=C_BOX)
        # Separator
        ax.axhline(y_pos - 0.1, color="#ccc", lw=0.8, zorder=1)
        y_pos += 0.5

    ax.set_xlim(-0.01, 1.15)
    ax.set_ylim(-0.5, y_pos)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xticks([])
    ax.set_xlabel("")

    handles = [mpatches.Patch(facecolor=c, label=l)
               for s, (c, l) in {
                   "PASS": (C_GRN, "Passé"),
                   "SKIP": (C_ORG, "Ignoré (seuil non atteint)"),
                   "CORR": ("#1ABC9C", "Corrigé en sprint"),
               }.items()]
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)
    ax.set_title("Détail par cas de test", fontsize=10, color=C_BOX)

    # ── Right: donut charts ──
    ax2 = axes[1]
    ax2.set_facecolor(C_BG)
    ax2.axis("off")
    ax2.set_title("Vue synthétique par suite", fontsize=10, color=C_BOX)

    suites_summary = [
        ("Fonctionnels", 8, 1, 0),   # 8 pass, 1 skip (TC-F09, seuil non atteint)
        ("Performance",  5, 0, 0),
        ("Sécurité",     7, 0, 0),
    ]

    for i, (name, p, s, f) in enumerate(suites_summary):
        total = p + s + f
        vals  = [p, s, f + 0]
        cols  = [C_GRN, C_ORG, "#1ABC9C"]
        ys    = 0.75 - i * 0.30

        sub = fig.add_axes([0.67, ys - 0.06, 0.12, 0.20])
        sub.pie([p, s, max(f, 0.001)], colors=[C_GRN, C_ORG, "#1ABC9C"],
                startangle=90, wedgeprops={"linewidth": 1.5, "edgecolor": "white"})
        sub.set_facecolor(C_BG)

        fig.text(0.795, ys + 0.04, f"{name}", fontsize=9, fontweight="bold",
                 va="center", color=C_BOX)
        detail_txt = f"{p} passés" + (f"  |  {s} ignorés" if s else "")
        fig.text(0.795, ys - 0.01, detail_txt,
                 fontsize=7.5, va="center", color="#555")

    save(fig, OUT_DIR / "03_test_results.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 5 — Performance Benchmarks
# ══════════════════════════════════════════════════════════════════════════════

def fig_benchmarks():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5),
                             gridspec_kw={"width_ratios": [1.6, 1]})
    fig.patch.set_facecolor(C_BG)
    fig.suptitle("Benchmarks de performance — pipeline ML et chatbot RAG",
                 fontsize=12, fontweight="bold", color=C_BOX)

    # ── Left: bar chart ──
    ax = axes[0]
    ax.set_facecolor(C_BG)
    ax.spines[["top", "right"]].set_visible(False)

    labels = [
        "BM-01\nInférence XGBoost",
        "BM-02\nFeature engineering",
        "BM-03\nRecherche FAISS",
        "BM-04\nLecture état pipeline",
        "BM-05\nPipeline complet offline",
    ]
    avgs = [3.462, 1.074, 0.056, 0.120, 13.792]
    p95s = [4.273, 1.839, 0.084, 0.224, 30.017]
    p99s = [7.124, 2.602, 0.143, 0.336, 48.118]
    thresholds = [50, 20, 5, 500, 70]

    y = np.arange(len(labels))
    h = 0.28

    b1 = ax.barh(y + h, avgs,  height=h, color=C_GRN,  label="Moyenne (avg)", zorder=3)
    b2 = ax.barh(y,     p95s,  height=h, color=C_BLU,  label="95e percentile (p95)", zorder=3)
    b3 = ax.barh(y - h, p99s,  height=h, color=C_ORG,  label="99e percentile (p99)", zorder=3)

    # Threshold markers
    for i, thr in enumerate(thresholds):
        ax.plot([thr, thr], [i - h*1.5, i + h*1.5],
                color=C_RED, lw=1.5, ls="--", zorder=4, alpha=0.7)

    # Value labels
    for i, (a, p9, p99_v) in enumerate(zip(avgs, p95s, p99s)):
        ax.text(a + 0.3, i + h, f"{a:.3f} ms", va="center", fontsize=7.5, color=C_BOX)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Temps (ms)", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(0.01, 2000)
    ax.axvline(0.01, color="#ccc", lw=0.5)
    ax.grid(axis="x", alpha=0.3, zorder=0)

    handles = [b1, b2, b3,
               Line2D([0], [0], color=C_RED, lw=1.5, ls="--", label="Seuil maximal")]
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    ax.set_title("Composants ML/RAG (axe log)", fontsize=10, color=C_BOX)

    # ── Right: API breakdown ──
    ax2 = axes[1]
    ax2.set_facecolor(C_BG)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_title("Répartition de la latence\nAPI /predict (820 ms total)", fontsize=10, color=C_BOX)

    portions = [13.8, 6.2, 800]
    labels2  = ["Pipeline ML\n(13,8 ms)", "Réseau / overhead\n(6,2 ms)", "API Groq (LLM)\n(800 ms)"]
    colors2  = [C_GRN, C_BLU, C_ORG]
    explode  = (0.04, 0.02, 0.02)

    wedges, texts, autotexts = ax2.pie(
        portions, labels=None, colors=colors2,
        explode=explode, autopct="%1.1f%%",
        startangle=120, pctdistance=0.7,
        wedgeprops={"linewidth": 1.5, "edgecolor": "white"},
    )
    for at in autotexts:
        at.set_fontsize(8)
        at.set_color("white")
        at.set_fontweight("bold")

    ax2.legend(wedges, labels2, loc="lower center", fontsize=8,
               bbox_to_anchor=(0.5, -0.22), framealpha=0.85,
               ncol=1)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, OUT_DIR / "04_benchmarks.png", tight=False)


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 6 — Security Controls
# ══════════════════════════════════════════════════════════════════════════════

def fig_security():
    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Contrôles de sécurité — Sprint 5  (7/7 PASS)",
                 fontsize=11, fontweight="bold", color=C_BOX, pad=8)

    controls = [
        ("SC-01 — .env non commis dans Git",
         "PASS", ".gitignore couvrant .env, pipeline.log et artefacts ML"),
        ("SC-02 — Aucun credential en dur dans le code",
         "PASS", "100 % des fichiers .py scannés par regex"),
        ("SC-03 — Variables d'environnement critiques",
         "PASS", "6 variables présentes dans .env"),
        ("SC-04 — CORS sans wildcard",
         "PASS", "allow_origins restreint à localhost:8080"),
        ("SC-05 — Protection injection SQL",
         "PASS", "Aucune f-string dans execute() / read_sql()"),
        ("SC-06 — Intégrité du modèle pkl",
         "PASS", "SHA-256 : 409fb534...  |  AUC = 0,9733"),
        ("SC-07 — Fichiers sensibles dans .gitignore",
         "PASS", ".env, pipeline.log, pipeline_state.json, ml/model/versions/"),
    ]

    color_map  = {"PASS": C_GRN, "CORR": "#1ABC9C", "FAIL": C_RED}
    label_map  = {"PASS": "PASS",  "CORR": "CORRIGÉ", "FAIL": "FAIL"}

    y_positions = list(range(len(controls) - 1, -1, -1))

    for i, (name, _, detail) in enumerate(controls):
        yp    = y_positions[i]
        col   = C_GRN
        badge = "PASS"

        # Status badge
        badge_box = FancyBboxPatch((-0.05, yp - 0.32), 1.1, 0.64,
                                   boxstyle="round,pad=0.05",
                                   facecolor=col, edgecolor="white",
                                   linewidth=1.0, zorder=3)
        ax.add_patch(badge_box)
        ax.text(0.5, yp, badge, ha="center", va="center",
                fontsize=8, color="white", fontweight="bold", zorder=4)

        # Control name
        ax.text(1.25, yp + 0.1, name, va="center", fontsize=9,
                fontweight="bold", color=C_BOX)
        ax.text(1.25, yp - 0.18, detail, va="center", fontsize=7.8,
                color="#666", style="italic")

        # Separator
        if i < len(controls) - 1:
            ax.axhline(yp - 0.48, color="#e0e0e0", lw=0.8, xmin=0.02, xmax=0.98)

    ax.set_xlim(-0.2, 11)
    ax.set_ylim(-0.7, len(controls) - 0.3)
    ax.set_yticks([])
    ax.set_xticks([])

    handles = [
        mpatches.Patch(facecolor=C_GRN, label="PASS — conforme"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8.5, framealpha=0.9)

    save(fig, OUT_DIR / "05_security_controls.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 7 — Model Versioning Flow
# ══════════════════════════════════════════════════════════════════════════════

def fig_versioning_flow():
    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.5)
    ax.axis("off")
    fig.patch.set_facecolor(C_BG)
    ax.set_title("Cycle de vie du modèle ML — versionnement et porte AUC",
                 fontsize=12, fontweight="bold", color=C_BOX, pad=8)

    # ── Trigger ──
    fbox(ax, 2.0, 6.8, 3.0, 0.65,
         "Seuils atteints :\n≥ 150 lignes  |  ≥ 30 positifs  |  ≥ 20 nouvelles",
         facecolor="#8E44AD")
    arrow(ax, 2.0, 6.47, 2.0, 6.07)

    # ── Training ──
    fbox(ax, 2.0, 5.7, 3.0, 0.65,
         "Entraînement\nml/train.py --source real",
         facecolor=C_BLU)
    arrow(ax, 2.0, 5.37, 2.0, 4.97)

    # ── Archive ──
    fbox(ax, 2.0, 4.6, 3.0, 0.65,
         "Archivage versionné\nml/model/versions/vXXX_YYYYMMDD_risk_model.pkl",
         facecolor=C_BLU)
    arrow(ax, 2.0, 4.27, 2.0, 3.82)

    # ── AUC Gate ──
    diamond(ax, 2.0, 3.35, 3.8, 1.0,
            "AUC_new  ≥\nAUC_prod − 0,005 ?",
            facecolor=C_ORG)

    # ── YES branch ──
    arrow(ax, 3.9, 3.35, 6.5, 3.35, color=C_GRN)
    ax.text(5.1, 3.55, "OUI", fontsize=9, color=C_GRN, fontweight="bold")

    fbox(ax, 7.8, 3.35, 2.6, 0.70,
         "Promotion\nrisk_model.pkl ← nouvelle version",
         facecolor=C_GRN)

    fbox(ax, 7.8, 2.3, 2.6, 0.70,
         "Registre mis à jour\n\"promoted\": true",
         facecolor="#1ABC9C")
    arrow(ax, 7.8, 3.0, 7.8, 2.65)

    fbox(ax, 7.8, 1.3, 2.6, 0.70,
         "Modèle en production\nAPI FastAPI sert la nouvelle version",
         facecolor="#2ECC71")
    arrow(ax, 7.8, 1.95, 7.8, 1.65)

    # ── NO branch ──
    arrow(ax, 2.0, 2.85, 2.0, 2.1, color=C_RED)
    ax.text(2.1, 2.55, "NON", fontsize=9, color=C_RED, fontweight="bold")

    fbox(ax, 2.0, 1.7, 3.0, 0.65,
         "Rejet\n\"rejected\": true  +  raison archivée",
         facecolor=C_RED)
    arrow(ax, 2.0, 1.37, 2.0, 0.97)

    fbox(ax, 2.0, 0.65, 3.0, 0.65,
         "Rollback automatique\nversioning.promote(prev_ver)",
         facecolor="#E74C3C")

    # ── Rollback connects back to production ──
    ax.plot([3.5, 5.5], [0.65, 0.65], color="#aaa", lw=1.3, ls="--", zorder=2)
    ax.plot([5.5, 5.5], [0.65, 1.3], color="#aaa", lw=1.3, ls="--", zorder=2)
    ax.annotate("", xy=(6.5, 1.3), xytext=(5.5, 1.3),
                arrowprops=dict(arrowstyle="-|>", color="#aaa", lw=1.3, mutation_scale=10))
    ax.text(5.6, 0.82, "Restauration version précédente", fontsize=7.5,
            color="#888", style="italic")

    # ── Report ──
    fbox(ax, 10.5, 5.5, 2.2, 0.65,
         "ml/report.py\n--plot  |  --rollback N  |  --json",
         facecolor="#7F8C8D", fontsize=8)
    ax.plot([9.1, 10.5], [3.35, 3.35], color="#bbb", lw=1.2, ls=":", zorder=2)
    ax.plot([10.5, 10.5], [3.35, 5.17], color="#bbb", lw=1.2, ls=":", zorder=2)
    ax.text(10.5, 4.25, "Rapport\nversions", ha="center", fontsize=7.5,
            color="#999", style="italic")

    # ── Registry note ──
    fbox(ax, 10.5, 1.8, 2.2, 0.65,
         "model_registry.json\nHistorique complet",
         facecolor="#7F8C8D", fontsize=8)
    ax.plot([9.1, 10.5], [2.3, 2.3], color="#bbb", lw=1.2, ls=":", zorder=2)
    ax.plot([10.5, 10.5], [2.3, 1.92], color="#bbb", lw=1.2, ls=":", zorder=2)

    # ── Legend ──
    handles = [
        mpatches.Patch(facecolor="#8E44AD",  label="Déclencheur (seuils)"),
        mpatches.Patch(facecolor=C_BLU,      label="Traitement (entraînement / archivage)"),
        mpatches.Patch(facecolor=C_ORG,      label="Décision (porte AUC)"),
        mpatches.Patch(facecolor=C_GRN,      label="Promotion en production"),
        mpatches.Patch(facecolor=C_RED,      label="Rejet + rollback automatique"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9,
              bbox_to_anchor=(1.0, 0.0))

    save(fig, OUT_DIR / "06_versioning_flow.png")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\nGénération des figures — Chapitre 5\n")
    fig_use_case()
    fig_pipeline_flow()
    fig_architecture()
    fig_test_results()
    fig_benchmarks()
    fig_security()
    fig_versioning_flow()
    print(f"\n7 figures generees dans : {OUT_DIR.relative_to(ROOT)}")
