"""
plot_ablation.py - Ablation study comparison plot

Compares baseline vs ablation variants:
  - baseline:  full model (type + activity + observability + keypoint)
  - no_kpt:    remove keypoint head (no angle prediction)
  - no_obs:    remove observability head
  - no_both:   remove both keypoint + observability

Output: ablation_comparison.png (bar chart + learning curves)

Usage:
    python plot_ablation.py
    python plot_ablation.py --out results/ablation.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "structure"))

from config import REPO_ROOT

# ============================================================================
# Config: which runs to compare
# ============================================================================
# Value: run folder name (str) or path to a metrics.json file (Path)
CONFIGS = {
    "V9.1 Full\n(baseline)": "v9_v1_20260511_123100",
    "A: No hierarchical\nmasking": "v9_abl_A_20260512_073513",
    "B: No cell-aware\nkeypoints": "v9_abl_B_20260512_094654",
    "C: No flip\nidentity swap": "v9_abl_C_20260512_121713",
}

# Clip all runs to this number of epochs for a fair comparison
MAX_EPOCHS = 100

COLORS = ["#2196F3", "#FF9800", "#F44336", "#9C27B0"]

METRICS_DISPLAY = {
    "mAP_50": ("mAP@0.5", "higher is better", True),
    "cls_f1": ("Class F1", "higher is better", True),
    "obs_f1": ("Obs F1", "higher is better", True),
    "angle_mae": ("Angle MAE (°)", "lower is better", False),
    "composite": ("Composite Score", "higher is better", True),
}


# ============================================================================
# Load metrics
# ============================================================================
def load_run(run_name, max_epochs=None):
    # Accept either a file path (Path) or a run folder name (str)
    if isinstance(run_name, Path) and run_name.suffix == ".json":
        mpath = run_name
    else:
        mpath = REPO_ROOT / "train" / "model_1" / "checkpoint" / run_name / "metrics.json"
    if not mpath.exists():
        raise FileNotFoundError(f"No metrics file: {mpath}")
    with open(mpath) as f:
        d = json.load(f)
    if max_epochs:
        d = {k: (v[:max_epochs] if isinstance(v, list) else v) for k, v in d.items()}
    return d


def best_epoch_metrics(d):
    """Return metrics at best composite epoch."""
    comp = d["composite"]
    best_i = int(np.argmax(comp))
    return {
        k: (v[best_i] if isinstance(v, list) else v) for k, v in d.items()
    }, best_i + 1


# ============================================================================
# Plot
# ============================================================================
def make_plot(save_path):
    runs = {}
    for label, run_name in CONFIGS.items():
        try:
            d = load_run(run_name, max_epochs=MAX_EPOCHS)
            runs[label] = d
        except FileNotFoundError as e:
            print(f"WARN: {e}")

    labels = list(runs.keys())
    colors = COLORS[: len(labels)]

    # Get best-epoch metrics for each run
    best = {}
    best_ep = {}
    for label, d in runs.items():
        bm, ep = best_epoch_metrics(d)
        best[label] = bm
        best_ep[label] = ep

    # Print table
    print("=" * 80)
    print(
        f'{"Config":<28} {"Best ep":>7} {"mAP":>7} {"cls_f1":>7} '
        f'{"obs_f1":>7} {"angle":>8} {"composite":>10}'
    )
    print("-" * 80)
    for label in labels:
        b = best[label]
        name = label.replace("\n", " ")
        print(
            f"{name:<28} {best_ep[label]:>7} "
            f'{b.get("mAP_50", float("nan")):>7.3f} '
            f'{b.get("cls_f1", float("nan")):>7.3f} '
            f'{b.get("obs_f1", float("nan")):>7.3f} '
            f'{b.get("angle_mae", float("nan")):>8.2f} '
            f'{b.get("composite", float("nan")):>10.4f}'
        )
    print("=" * 80)

    # ── Layout: top = bar charts, bottom = learning curves ──
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.45, wspace=0.35)

    # ── Row 0: bar chart per metric ──
    metric_keys = list(METRICS_DISPLAY.keys())
    for col, (mkey, (title, hint, higher_better)) in enumerate(METRICS_DISPLAY.items()):
        ax = fig.add_subplot(gs[0, col])
        vals = [best[lbl].get(mkey, float("nan")) for lbl in labels]
        bars = ax.bar(range(len(labels)), vals, color=colors, width=0.6, alpha=0.85)

        # Value labels on bars
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    fontweight="bold",
                )

        ax.set_title(f"{title}\n({hint})", fontsize=9, fontweight="bold")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(
            [l.replace("\n", "\n") for l in labels], fontsize=7, rotation=0, ha="center"
        )
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max([v for v in vals if not np.isnan(v)] + [0.1]) * 1.18)

        # Highlight best bar
        if higher_better:
            best_idx = int(np.nanargmax(vals))
        else:
            best_idx = int(np.nanargmin(vals))
        bars[best_idx].set_edgecolor("black")
        bars[best_idx].set_linewidth(2)

    # ── Row 1: learning curves ──
    curve_metrics = [
        ("mAP_50", "mAP@0.5", "upper left"),
        ("cls_f1", "Class F1", "lower right"),
        ("obs_f1", "Obs F1", "lower right"),
        ("angle_mae", "Angle MAE (°)", "upper right"),
        ("composite", "Composite", "lower right"),
    ]

    for col, (mkey, title, loc) in enumerate(curve_metrics):
        ax = fig.add_subplot(gs[1, col])
        for (label, d), color in zip(runs.items(), colors):
            series = d.get(mkey, [])
            if isinstance(series, list) and series:
                epochs = range(1, len(series) + 1)
                ax.plot(
                    epochs,
                    series,
                    color=color,
                    linewidth=1.5,
                    label=label.replace("\n", " "),
                    alpha=0.85,
                )
                # Mark best epoch
                bm, ep = best_epoch_metrics(d)
                best_val = bm.get(mkey, float("nan"))
                if not np.isnan(best_val):
                    ax.scatter(ep, best_val, color=color, s=40, zorder=5, marker="*")

        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=8)
        ax.grid(alpha=0.3)
        if col == 0:
            ax.legend(fontsize=6.5, loc=loc)

    fig.suptitle(
        f"AFGNet V9.1 — Ablation Study ({MAX_EPOCHS} epochs)\n"
        "Comparing full model vs removing each of the 3 contributions\n"
        "★ = best epoch per run",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )

    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    out = Path(args.out) if args.out else REPO_ROOT / "train" / "model_1" / "checkpoint" / "ablation_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    make_plot(out)


if __name__ == "__main__":
    main()
