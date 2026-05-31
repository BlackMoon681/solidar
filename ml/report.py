"""
ml/model_report.py  —  PPP-Risk Model Version Report

Usage:
    python ml/model_report.py               # full history table
    python ml/model_report.py --plot        # also save AUC trend chart
    python ml/model_report.py --rollback 3  # restore version 3 to production
    python ml/model_report.py --json        # machine-readable output
"""

import sys, os, json, argparse
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.versioning import get_all, promote, current_promoted_version, AUC_GATE_DELTA

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _col(value, width, align="right"):
    s = str(value) if value is not None else "—"
    return s.ljust(width) if align == "left" else s.rjust(width)


def _delta(new, old, invert=False):
    """Return a signed delta string. invert=True means lower is better (Brier)."""
    if new is None or old is None:
        return ""
    d = new - old
    sign = "+" if d >= 0 else ""
    good = (d >= 0) if not invert else (d <= 0)
    marker = " ↑" if good else " ↓"
    return f"{sign}{d:+.4f}{marker}"


def _status(entry: dict, current_ver: int | None) -> str:
    if entry.get("promoted") or entry["version"] == current_ver:
        return "CURRENT"
    if entry.get("rejected"):
        short = (entry.get("rejection_reason") or "")[:50]
        return f"rejected  ({short})"
    return "archived"


# ---------------------------------------------------------------------------
# Print report
# ---------------------------------------------------------------------------

def print_report(versions: list, current_ver: int | None):
    if not versions:
        print("No versions in registry yet.")
        return

    HDR = f"{'#':>4}  {'Tag':<28}  {'Source':<9}  {'Train':>7}  " \
          f"{'AUC':>6}  {'F1':>6}  {'Prec':>6}  {'Rec':>6}  {'Brier':>6}  {'MCC':>6}  Status"
    SEP = "─" * len(HDR)

    print()
    print("PPP-Risk  —  Model Version History")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(SEP)
    print(HDR)
    print(SEP)

    for i, e in enumerate(versions):
        n_train = f"{e['n_train']:,}" if e.get("n_train") else "—"
        status  = _status(e, current_ver)
        marker  = ">>>" if status == "CURRENT" else "   "

        row = (
            f"{marker}{e['version']:>3}  "
            f"{e['tag']:<28}  "
            f"{e['source']:<9}  "
            f"{n_train:>7}  "
            f"{e['holdout_auc']:>6.4f}  "
            f"{e['holdout_f1']:>6.4f}  "
            f"{e['holdout_prec']:>6.4f}  "
            f"{e['holdout_rec']:>6.4f}  "
            f"{e['holdout_bs']:>6.4f}  "
            f"{e['holdout_mcc']:>6.4f}  "
            f"{status}"
        )
        print(row)

    print(SEP)

    # -- Delta table -----------------------------------------------------------
    if len(versions) > 1:
        print()
        print("Delta vs previous version:")
        print(SEP)
        for i in range(1, len(versions)):
            prev = versions[i - 1]
            curr = versions[i]
            d_auc  = _delta(curr["holdout_auc"],  prev["holdout_auc"])
            d_f1   = _delta(curr["holdout_f1"],   prev["holdout_f1"])
            d_bs   = _delta(curr["holdout_bs"],   prev["holdout_bs"],  invert=True)
            d_mcc  = _delta(curr["holdout_mcc"],  prev["holdout_mcc"])
            status = _status(curr, current_ver)
            gate   = f"  gate_delta={AUC_GATE_DELTA:+.3f}"
            print(f"  v{prev['version']:03d} → v{curr['version']:03d} "
                  f"({curr['source']})  "
                  f"AUC {d_auc},  F1 {d_f1},  Brier {d_bs},  MCC {d_mcc}"
                  f"  [{status}]{gate if status.startswith('rejected') else ''}")
        print(SEP)

    # -- Summary ---------------------------------------------------------------
    best = max(versions, key=lambda e: e["holdout_auc"])
    real = [e for e in versions if e["source"] == "real"]
    print()
    print(f"  Total versions  : {len(versions)}")
    print(f"  Real-data runs  : {len(real)}")
    print(f"  Best AUC ever  : {best['holdout_auc']:.4f}  ({best['tag']})")
    if current_ver:
        cur = next((e for e in versions if e["version"] == current_ver), None)
        if cur:
            print(f"  Current in prod : v{cur['version']:03d}  AUC={cur['holdout_auc']:.4f}"
                  f"  source={cur['source']}")
    print()


# ---------------------------------------------------------------------------
# AUC trend plot
# ---------------------------------------------------------------------------

def save_plot(versions: list, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tags  = [f"v{e['version']:03d}" for e in versions]
    aucs  = [e["holdout_auc"] for e in versions]
    cv_au = [e["cv_auc"]      for e in versions]
    f1s   = [e["holdout_f1"]  for e in versions]
    brier = [e["holdout_bs"]  for e in versions]

    colors = []
    for e in versions:
        if e.get("promoted"):
            colors.append("green")
        elif e.get("rejected"):
            colors.append("red")
        else:
            colors.append("steelblue")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("PPP-Risk  —  Model Version Performance", fontsize=13, fontweight="bold")

    def _bar(ax, values, title, ylabel, color_list=None, h_line=None):
        bars = ax.bar(tags, values, color=color_list or "steelblue", edgecolor="white", width=0.6)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_ylim(max(0, min(values) - 0.05), min(1.0, max(values) + 0.05))
        ax.tick_params(axis="x", rotation=45)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=7)
        if h_line is not None:
            ax.axhline(h_line, color="orange", linestyle="--", linewidth=1,
                       label=f"current prod ({h_line:.4f})")
            ax.legend(fontsize=7)

    # Find current prod AUC for reference line
    cur = next((e["holdout_auc"] for e in versions if e.get("promoted")), None)

    _bar(axes[0][0], aucs,  "Holdout AUC-ROC",    "AUC",   colors, h_line=cur)
    _bar(axes[0][1], cv_au, "CV AUC (5-fold)",     "AUC",   colors)
    _bar(axes[1][0], f1s,   "Holdout F1",          "F1",    colors)
    _bar(axes[1][1], brier, "Brier Score (lower=better)", "Brier",
         ["red" if c == "steelblue" else c for c in colors])

    # Legend
    from matplotlib.patches import Patch
    legend = [
        Patch(color="green",    label="Current production"),
        Patch(color="red",      label="Rejected by AUC gate"),
        Patch(color="steelblue",label="Archived"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved -> {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PPP-Risk model version report")
    parser.add_argument("--plot",     action="store_true",
                        help="Save AUC trend chart to ml/plots/version_history.png")
    parser.add_argument("--rollback", type=int, metavar="VERSION",
                        help="Roll back to VERSION (copies it to production)")
    parser.add_argument("--json",     action="store_true",
                        help="Output registry as JSON")
    args = parser.parse_args()

    versions    = get_all()
    current_ver = current_promoted_version()

    # -- Rollback --------------------------------------------------------------
    if args.rollback is not None:
        target = next((e for e in versions if e["version"] == args.rollback), None)
        if not target:
            print(f"ERROR: version {args.rollback} not found in registry.")
            sys.exit(1)
        confirm = input(
            f"Roll back to v{args.rollback:03d} ({target['tag']})? "
            f"AUC={target['holdout_auc']:.4f}  [y/N] "
        ).strip().lower()
        if confirm != "y":
            print("Aborted.")
            sys.exit(0)
        promote(args.rollback)
        print(f"  Rolled back to v{args.rollback:03d}. Production model updated.")
        versions    = get_all()
        current_ver = current_promoted_version()

    # -- JSON ------------------------------------------------------------------
    if args.json:
        print(json.dumps(versions, indent=2, default=str))
        return

    # -- Report ----------------------------------------------------------------
    print_report(versions, current_ver)

    # -- Plot ------------------------------------------------------------------
    if args.plot:
        if not versions:
            print("No versions to plot.")
        else:
            plots_dir = Path(__file__).parent / "plots"
            plots_dir.mkdir(exist_ok=True)
            save_plot(versions, plots_dir / "version_history.png")


if __name__ == "__main__":
    main()
