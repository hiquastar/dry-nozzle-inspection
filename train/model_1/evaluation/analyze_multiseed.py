"""
analyze_multiseed.py - Multi-seed robustness comparison

Compares seed=42 vs seed=123 across all 4 configurations:
  - Full model (V9.1 baseline)
  - Ablation A: no hierarchical masking
  - Ablation B: no cell-aware keypoints
  - Ablation C: no flip identity swap

Layout:
  Row 0: grouped bar charts (best epoch) per metric, seed42 vs seed123
  Row 1: composite learning curves, all (config, seed) combos overlaid

Usage:
    python analyze_multiseed.py
    python analyze_multiseed.py --out ../runs/multiseed_comparison.png
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
# Edit these after seed=123 training finishes
# ============================================================================
RUNS = {
    "Full\nModel": {
        "seed42": "v9_20260513_013233",
        "seed123": "v9_seed123_20260512_153529",
    },
    "A: No\nHier. Mask": {
        "seed42": "v9_abl_A_20260512_073513",
        "seed123": "v9_abl_A_seed123_20260512_174840",
    },
    "B: No\nCell KPT": {
        "seed42": "v9_abl_B_20260513_034047",
        "seed123": "v9_abl_B_seed123_20260512_200556",
    },
    "C: No\nFlip Swap": {
        "seed42": "v9_abl_C_20260512_121713",
        "seed123": "v9_abl_C_seed123_20260512_222732",
    },
}

MAX_EPOCHS = 100

METRICS_DISPLAY = {
    "mAP_50": ("mAP@0.5", True),
    "cls_f1": ("Class F1", True),
    "obs_f1": ("Obs F1", True),
    "angle_mae": ("Angle MAE (°)", False),
    "composite": ("Composite", True),
}

SEED_COLORS = {"seed42": "#2196F3", "seed123": "#FF9800"}
CONFIG_COLORS = ["#2196F3", "#F44336", "#4CAF50", "#9C27B0"]


# ============================================================================
def load_run(run_name, max_epochs=None):
    mpath = REPO_ROOT / "train" / "model_1" / "checkpoint" / run_name / "metrics.json"
    if not mpath.exists():
        return None
    with open(mpath) as f:
        d = json.load(f)
    if max_epochs:
        d = {k: (v[:max_epochs] if isinstance(v, list) else v) for k, v in d.items()}
    return d


def best_epoch_metrics(d):
    comp = d["composite"]
    best_i = int(np.argmax(comp))
    return {
        k: (v[best_i] if isinstance(v, list) else v) for k, v in d.items()
    }, best_i + 1


def is_placeholder(run_name):
    return "YYYYMMDD" in run_name


# ============================================================================
def make_plot(save_path):
    # Load all available data
    data = {}  # data[config][seed] = metrics dict
    for cfg_label, seeds in RUNS.items():
        data[cfg_label] = {}
        for seed_label, run_name in seeds.items():
            if is_placeholder(run_name):
                print(f"SKIP (placeholder): {cfg_label} / {seed_label}")
                continue
            d = load_run(run_name, max_epochs=MAX_EPOCHS)
            if d is None:
                print(f"WARN: not found: {run_name}")
                continue
            data[cfg_label][seed_label] = d

    cfg_labels = list(RUNS.keys())
    seed_labels = ["seed42", "seed123"]
    n_cfg = len(cfg_labels)
    n_metrics = len(METRICS_DISPLAY)

    # Print summary table
    print("=" * 90)
    header = f'{"Config":<18} {"Seed":<9}'
    for mkey, (title, _) in METRICS_DISPLAY.items():
        header += f" {title:>12}"
    print(header)
    print("-" * 90)
    for cfg in cfg_labels:
        for seed in seed_labels:
            if seed not in data[cfg]:
                continue
            bm, ep = best_epoch_metrics(data[cfg][seed])
            row = f'{cfg.replace(chr(10)," "):<18} {seed:<9}'
            for mkey in METRICS_DISPLAY:
                v = bm.get(mkey, float("nan"))
                row += f" {v:>12.4f}"
            print(row)
        print()
    print("=" * 90)

    # ── Layout ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, n_metrics, figure=fig, hspace=0.50, wspace=0.35)

    bar_width = 0.35
    x = np.arange(n_cfg)

    # ── Row 0: grouped bar charts ────────────────────────────────────────────
    for col, (mkey, (title, higher_better)) in enumerate(METRICS_DISPLAY.items()):
        ax = fig.add_subplot(gs[0, col])

        for s_idx, seed in enumerate(seed_labels):
            vals = []
            for cfg in cfg_labels:
                if seed in data[cfg]:
                    bm, _ = best_epoch_metrics(data[cfg][seed])
                    vals.append(bm.get(mkey, float("nan")))
                else:
                    vals.append(float("nan"))

            offset = (s_idx - 0.5) * bar_width
            bars = ax.bar(
                x + offset,
                vals,
                bar_width,
                label=seed,
                color=SEED_COLORS[seed],
                alpha=0.85,
            )

            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.003,
                        f"{v:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=6.5,
                        fontweight="bold",
                    )

        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([l.replace("\n", "\n") for l in cfg_labels], fontsize=7)
        ax.grid(axis="y", alpha=0.3)
        if col == 0:
            ax.legend(fontsize=7)

        # Determine y-limit from non-nan values
        all_vals = []
        for seed in seed_labels:
            for cfg in cfg_labels:
                if seed in data[cfg]:
                    bm, _ = best_epoch_metrics(data[cfg][seed])
                    v = bm.get(mkey, float("nan"))
                    if not np.isnan(v):
                        all_vals.append(v)
        if all_vals:
            ax.set_ylim(0, max(all_vals) * 1.20)

    # ── Row 1: composite learning curves ────────────────────────────────────
    for col, (mkey, (title, higher_better)) in enumerate(METRICS_DISPLAY.items()):
        ax = fig.add_subplot(gs[1, col])

        for c_idx, cfg in enumerate(cfg_labels):
            color = CONFIG_COLORS[c_idx]
            cfg_short = cfg.replace("\n", " ")
            for seed in seed_labels:
                if seed not in data[cfg]:
                    continue
                d = data[cfg][seed]
                series = d.get(mkey, [])
                if not (isinstance(series, list) and series):
                    continue
                linestyle = "-" if seed == "seed42" else "--"
                label = f"{cfg_short} ({seed})" if col == 0 else None
                ax.plot(
                    range(1, len(series) + 1),
                    series,
                    color=color,
                    linewidth=1.2,
                    linestyle=linestyle,
                    alpha=0.80,
                    label=label,
                )

                bm, ep = best_epoch_metrics(d)
                best_v = bm.get(mkey, float("nan"))
                if not np.isnan(best_v):
                    ax.scatter(ep, best_v, color=color, s=30, zorder=5, marker="*")

        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=8)
        ax.grid(alpha=0.3)
        if col == 0:
            ax.legend(fontsize=5.5, loc="lower right")

    fig.suptitle(
        f"AFGNet V9.1 — Multi-Seed Robustness (seed=42 vs seed=123)\n"
        f"solid = seed42 · dashed = seed123 · ★ = best epoch",
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

    out = (
        Path(args.out) if args.out else REPO_ROOT / "train" / "model_1" / "checkpoint" / "multiseed_comparison.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    make_plot(out)


if __name__ == "__main__":
    main()
