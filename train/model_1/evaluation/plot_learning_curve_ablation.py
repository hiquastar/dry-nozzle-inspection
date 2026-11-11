"""
plot_learning_curve_ablation.py

Learning-curve figure for the thesis
"Observability-Aware Multi-Task Learning for Spray Nozzle Inspection".

Uses REAL per-epoch train/val loss logged by train.py into each run's
runs/<run>/metrics.json (100 epochs). The 6th panel is the REAL instance-level
spray detection rate (post-NMS, IoU>0.5) from train/model_1/results/figures/confusion_matrices.json.

Produces:
  1. ONE large figure with all 5 configurations (train vs val per panel) + the
     detection bar chart in the 6th panel, and
  2. 5 separate per-config PNGs.

Each panel shows: validation loss (solid, bold) + train loss (dashed, faint) in
the config color, the train/val GAP shaded, the final value, and the BEST epoch
highlighted (lowest val loss: vertical line + star + label). Shared y-limits make
panel heights directly comparable.

Comparison is valid: all 5 runs use the same loss formula + same validation set
+ same seed 42.

Usage:
    python train/model_1/evaluation/plot_learning_curve_ablation.py
    python train/model_1/evaluation/plot_learning_curve_ablation.py --show
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Real per-epoch training logs (train.py saves these as METRICS_LOG each epoch).
RUN_PATHS = {
    "Full": "train/model_1/checkpoint/v9_fixes_reverse_seed42/metrics.json",
    "AblA": "train/model_1/checkpoint/v9_rabl_A_no_hier_seed42/metrics.json",
    "AblB": "train/model_1/checkpoint/v9_rabl_B_no_cellkpt_seed42/metrics.json",
    "AblC": "train/model_1/checkpoint/v9_rabl_C_no_flipswap_seed42/metrics.json",
    "AblD": "train/model_1/checkpoint/v9_abl_D_no_obs_head_seed42/metrics.json",
}

CONFIGS = {
    "Full":  {"title": "Full model",          "color": "#2a7fdb",
              "note": "Val cuối ~4.16 -> checkpoint được chọn"},
    "AblA":  {"title": "Abl A — no mask",      "color": "#e63946",
              "note": "Val CAO NHẤT (~11.5): học sai target nhiều nhất"},
    "AblB":  {"title": "Abl B — no cell kpt",  "color": "#f59e0b",
              "note": "Val THẤP NHẤT (~3.87) NHƯNG detection=0 -> shortcut"},
    "AblC":  {"title": "Abl C — no flip swap", "color": "#8b5cf6",
              "note": "Gần Full (~4.10)"},
    "AblD":  {"title": "Abl D — no obs head",  "color": "#22a559",
              "note": "Gần Full (~4.00)"},
}

Y_TOP = 14.0  # shared y-limit (covers all real val curves; Abl A ~11.5 is highest)

# Instance-level spray detection rate (recall) AFTER NMS, IoU>0.5, n=90 GT spray.
# Source: train/model_1/results/figures/confusion_matrices.json  (<run>.detection["0"].recall)
DETECTION_RATE = {"Full": 0.667, "AblA": 0.667, "AblB": 0.000,
                  "AblC": 0.667, "AblD": 0.667}


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def load_real_curves() -> dict:
    """Load REAL per-epoch train/val loss from each run's metrics.json."""
    import json
    root = _repo_root()
    out = {}
    for key, rel in RUN_PATHS.items():
        path = os.path.join(root, rel)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing real metrics log for {key}: {path}")
        with open(path, "r", encoding="utf-8") as f:
            h = json.load(f)
        val = np.asarray(h["val_loss"], dtype=float)
        train = np.asarray(h["train_loss"], dtype=float)
        out[key] = {
            "train": train,
            "val": val,
            "epochs": np.arange(1, len(val) + 1),  # 1-based for log x-axis
            "val_end": float(val[-1]),
        }
    return out


def draw_panel(ax, key: str, curves: dict) -> None:
    cfg = CONFIGS[key]
    c = cfg["color"]
    d = curves[key]
    val, train, epochs = d["val"], d["train"], d["epochs"]
    last_ep = int(epochs[-1])

    # Gap between train and val (visualises generalisation gap / overfitting).
    ax.fill_between(epochs, train, val, color=c, alpha=0.10, lw=0)
    ax.plot(epochs, train, color=c, lw=1.5, ls="--", alpha=0.5, label="train loss")
    ax.plot(epochs, val, color=c, lw=2.7, ls="-", label="val loss")
    ax.axhline(d["val_end"], color=c, lw=1.0, ls=":", alpha=0.6)
    ax.text(last_ep, d["val_end"], f" {d['val_end']:.2f}", color=c, va="center",
            ha="left", fontsize=9, clip_on=False)

    # BEST epoch = lowest validation loss (epochs are 1-based).
    best_idx = int(np.argmin(val))
    best_ep = int(epochs[best_idx])
    best_val = float(val[best_idx])
    ax.axvline(best_ep, color="#555555", lw=1.1, ls="--", alpha=0.7)
    ax.scatter([best_ep], [best_val], marker="*", s=260, color="#ffd000",
               edgecolor="black", linewidth=1.1, zorder=6)
    ax.annotate(f"best epoch = {best_ep}  (val = {best_val:.2f})",
                xy=(best_ep, best_val), xytext=(0.04, 0.98),
                textcoords="axes fraction",
                ha="left", va="top", fontsize=9, color="#1a1a1a",
                bbox=dict(boxstyle="round,pad=0.3", fc="#fff3bf", ec="#caa800", alpha=0.95),
                arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0))

    ax.set_title(cfg["title"], fontsize=13, fontweight="bold", color=c)
    # Per-panel y-range (each curve fills its own panel; NO shared scale, so the
    # ~4-loss runs aren't squashed by the Abl A ~11.5 outlier). Asymmetric pad
    # leaves room top (best-epoch note + legend) and bottom (config note).
    ymax_d = float(max(train.max(), val.max()))
    ymin_d = float(min(train.min(), val.min()))
    rng = max(1e-6, ymax_d - ymin_d)
    ax.set_ylim(max(0.0, ymin_d - 0.30 * rng), ymax_d + 0.40 * rng)
    # Log x-axis spreads the early epochs so the steep "plunge" is clearly visible.
    ax.set_xscale("log")
    ax.set_xlim(1, last_ep)
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_xlabel("Epoch (thang log — giãn các epoch đầu)", fontsize=10)
    ax.set_ylabel("Loss (sai số) — thấp = tốt", fontsize=10)
    ax.text(0.035, 0.04, cfg["note"], transform=ax.transAxes, fontsize=8.5,
            va="bottom", ha="left", color="#333333",
            bbox=dict(boxstyle="round,pad=0.35", fc="#f5f5f5", ec=c, alpha=0.9))


def draw_detection_bar(ax) -> None:
    """6th panel: instance-level spray detection rate bar chart (post-NMS)."""
    keys = list(CONFIGS.keys())
    labels = ["Full", "Abl A", "Abl B", "Abl C", "Abl D"]
    rates = [DETECTION_RATE[k] for k in keys]
    colors = [CONFIGS[k]["color"] for k in keys]
    xs = np.arange(len(keys))

    bars = ax.bar(xs, rates, width=0.66, color=colors, edgecolor="#333333",
                  linewidth=1.0, zorder=3)

    # Highlight Abl B (shortcut: detection = 0).
    bi = keys.index("AblB")
    bars[bi].set_edgecolor("#c1121f")
    bars[bi].set_linewidth(3.0)
    bars[bi].set_hatch("//")

    for x, r in zip(xs, rates):
        ax.text(x, r + 0.03, f"{r:.3f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold",
                color="#c1121f" if r == 0.0 else "#222222")

    # reference line at 0.667
    ax.axhline(0.667, color="#555555", ls="--", lw=1.3, zorder=2)
    ax.text(len(keys) - 0.45, 0.667, "0.667", va="bottom", ha="right",
            fontsize=9, color="#555555")

    ax.set_ylim(0, 1.0)
    ax.set_xlim(-0.6, len(keys) - 0.4)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Spray detection rate (recall)", fontsize=10)
    ax.set_title("Instance detection (sau NMS) - tang kiem chung cho val loss",
                 fontsize=11.5, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.text(0.02, 0.96, "n = 90 GT spray instances", transform=ax.transAxes,
            fontsize=8.5, va="top", color="#555555")

    note = ("Abl B: val loss THAP NHAT (~3.0) nhung detection = 0 -> shortcut.\n"
            "Chi nhin loss khong phat hien duoc; phai doi chieu instance-level.")
    ax.text(0.5, -0.30, note, transform=ax.transAxes, ha="center", va="top",
            fontsize=9, color="#7a1f1f",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fff0f0", ec="#c1121f"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Plot per-config ablation learning curves.")
    ap.add_argument("--out", default=os.path.join("Results", "learning_curve_ablation.png"),
                    help="Combined figure output path.")
    ap.add_argument("--panel-dir", default=os.path.join("Results", "lc_panels"),
                    help="Folder for the 5 individual panel PNGs.")
    ap.add_argument("--show", action="store_true", help="Open an interactive window too.")
    args = ap.parse_args(argv)

    if not args.show:
        matplotlib.use("Agg")

    curves = load_real_curves()
    keys = list(CONFIGS.keys())

    # --- Combined large figure: 5 loss panels + 1 detection bar chart ------
    # (sharey=False so the 6th panel can use a 0..1 scale; each loss panel
    #  already fixes its own y-limits in draw_panel, so they stay comparable.)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()
    for ax, key in zip(axes[:5], keys):
        draw_panel(ax, key, curves)

    # 6th panel (bottom-right): instance-level spray detection bar chart.
    draw_detection_bar(axes[5])

    fig.suptitle(
        "Đường cong huấn luyện THẬT theo từng cấu hình (Train vs Validation, 100 epoch)\n"
        "Observability-Aware Multi-Task Learning for Spray Nozzle Inspection",
        fontsize=15, fontweight="bold",
    )
    fig.text(0.5, 0.005,
             "Nguồn: runs/<run>/metrics.json (loss thật mỗi epoch) + "
             "train/model_1/results/figures/confusion_matrices.json (detection). Cùng val set + seed 42.",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved combined figure: {os.path.abspath(args.out)}")

    # --- 5 individual panel PNGs -------------------------------------------
    os.makedirs(args.panel_dir, exist_ok=True)
    for key in keys:
        f1, a1 = plt.subplots(figsize=(6.2, 4.6))
        draw_panel(a1, key, curves)
        f1.tight_layout()
        p = os.path.join(args.panel_dir, f"lc_{key}.png")
        f1.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(f1)
        print(f"Saved panel: {os.path.abspath(p)}")

    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
