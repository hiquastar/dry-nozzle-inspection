"""
plot_runs.py - Plot a comparison of v9.1a baseline vs v9.1b uncertainty.

Usage:
    python plot_runs.py                    # auto-find the 2 latest runs
    python plot_runs.py --live             # auto-refresh every 30s while training is ongoing
    python plot_runs.py --a <path> --b <path>  # specify explicit paths

Output: comparison_plot.png in the current directory
"""

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def find_latest_run(runs_dir, version_prefix):
    if not runs_dir.exists():
        return None
    candidates = sorted(
        [
            d
            for d in runs_dir.iterdir()
            if d.is_dir() and d.name.startswith(version_prefix)
        ],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_history(run_dir):
    if run_dir is None:
        return None
    f = run_dir / "metrics.json"
    if not f.exists():
        return None
    with open(f) as fp:
        return json.load(fp)


def safe_get(history, key):
    """Get the list, replace None with nan, return a numpy array."""
    if not history or key not in history:
        return None
    arr = [float("nan") if v is None else v for v in history[key]]
    return np.array(arr, dtype=float)


def plot_comparison(a_hist, b_hist, a_name, b_name, out_path, warmup_epoch=5):
    """Plot 6 panels: composite, mAP, cls_f1, obs_f1, angle_mae, uncertainty."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        f"Fair Compare: {a_name} (smooth_l1) vs {b_name} (gaussian_nll)",
        fontsize=13,
        fontweight="bold",
    )

    panels = [
        ("composite", "Composite Score", axes[0, 0], True),
        ("mAP_50", "mAP@.5", axes[0, 1], True),
        ("cls_f1", "Classification F1", axes[0, 2], True),
        ("obs_f1", "Observability F1", axes[1, 0], True),
        ("angle_mae", "Angle MAE (deg)", axes[1, 1], False),
    ]

    for key, title, ax, higher_better in panels:
        a_y = safe_get(a_hist, key)
        b_y = safe_get(b_hist, key)
        if a_y is not None:
            ax.plot(
                np.arange(1, len(a_y) + 1),
                a_y,
                "-o",
                label=a_name,
                color="steelblue",
                markersize=3,
                linewidth=1.5,
            )
        if b_y is not None:
            ax.plot(
                np.arange(1, len(b_y) + 1),
                b_y,
                "-s",
                label=b_name,
                color="crimson",
                markersize=3,
                linewidth=1.5,
            )
        # Warmup marker line
        ax.axvline(
            warmup_epoch + 0.5, color="gray", linestyle="--", alpha=0.4, linewidth=1
        )
        ax.text(
            warmup_epoch + 0.7,
            ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0,
            "NLL kicks in",
            fontsize=8,
            color="gray",
            va="top",
        )

        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(
            title + (" (higher better)" if higher_better else " (lower better)"),
            fontsize=10,
        )
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    # Uncertainty panel (only b has it)
    ax = axes[1, 2]
    b_lv_mean = safe_get(b_hist, "logvar_mean")
    b_lv_std = safe_get(b_hist, "logvar_std")
    b_r = safe_get(b_hist, "sigma_error_r")

    if b_lv_mean is not None:
        ep = np.arange(1, len(b_lv_mean) + 1)
        ax.plot(
            ep,
            b_lv_mean,
            "-o",
            label="logvar mean",
            color="darkorange",
            markersize=3,
            linewidth=1.5,
        )
        if b_lv_std is not None:
            ax.fill_between(
                ep,
                b_lv_mean - b_lv_std,
                b_lv_mean + b_lv_std,
                alpha=0.2,
                color="darkorange",
                label="±1 std",
            )
        if b_r is not None:
            ax2 = ax.twinx()
            ax2.plot(
                ep,
                b_r,
                "-^",
                label="σ-err corr r",
                color="green",
                markersize=3,
                linewidth=1.5,
            )
            ax2.axhline(0.4, color="green", linestyle=":", alpha=0.4)
            ax2.set_ylabel("Pearson r (σ vs |error|)", color="green", fontsize=9)
            ax2.tick_params(axis="y", labelcolor="green")
            ax2.set_ylim(-0.3, 1.0)
            ax2.legend(loc="lower right", fontsize=7)

    ax.axvline(warmup_epoch + 0.5, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("log_var", color="darkorange", fontsize=9)
    ax.tick_params(axis="y", labelcolor="darkorange")
    ax.set_title("v9.1b Uncertainty Diagnostics", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def print_best_summary(a_hist, b_hist, a_name, b_name):
    """Print the best-epoch comparison table."""

    def best(h, key):
        if not h or key not in h:
            return None
        arr = [float("nan") if v is None else v for v in h[key]]
        return arr

    if not a_hist or not b_hist:
        print("(Cannot compute summary - missing data)")
        return

    a_comp = best(a_hist, "composite")
    b_comp = best(b_hist, "composite")
    if a_comp is None or b_comp is None:
        return

    a_idx = int(np.nanargmax(a_comp))
    b_idx = int(np.nanargmax(b_comp))

    print()
    print("=" * 75)
    print("BEST EPOCH SUMMARY")
    print("=" * 75)
    print(f'{"Metric":<22} | {a_name:<22} | {b_name:<22}')
    print("-" * 75)
    metrics = [
        ("Best epoch", "composite_idx"),
        ("Composite", "composite"),
        ("mAP@.5", "mAP_50"),
        ("cls F1", "cls_f1"),
        ("obs F1", "obs_f1"),
        ("angle MAE", "angle_mae"),
        ("logvar mean", "logvar_mean"),
        ("logvar std", "logvar_std"),
        ("sigma-err r", "sigma_error_r"),
    ]
    for name, key in metrics:
        if key == "composite_idx":
            va, vb = a_idx + 1, b_idx + 1
            print(f"{name:<22} | {va:<22} | {vb:<22}")
        else:
            la = best(a_hist, key)
            lb = best(b_hist, key)
            va = la[a_idx] if la and a_idx < len(la) else None
            vb = lb[b_idx] if lb and b_idx < len(lb) else None
            fa = f"{va:.4f}" if isinstance(va, float) and not math.isnan(va) else "n/a"
            fb = f"{vb:.4f}" if isinstance(vb, float) and not math.isnan(vb) else "n/a"
            print(f"{name:<22} | {fa:<22} | {fb:<22}")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--a", type=str, default=None, help="Path to v9.1a run dir (default: auto-find)"
    )
    parser.add_argument(
        "--b", type=str, default=None, help="Path to v9.1b run dir (default: auto-find)"
    )
    parser.add_argument("--out", type=str, default="comparison_plot.png")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Refresh every 30s (for monitoring during training)",
    )
    parser.add_argument(
        "--warmup", type=int, default=5, help="Warmup epochs for kpt loss (default 5)"
    )
    args = parser.parse_args()

    # Repo root: this file lives in train/, repo/runs/ is one level up
    repo_root = Path(__file__).resolve().parents[3]  # evaluation -> model_1 -> train -> repo root
    runs_dir = repo_root / "train" / "model_1" / "checkpoint"

    while True:
        if args.a:
            a_run = Path(args.a)
        else:
            a_run = find_latest_run(runs_dir, "v9.1a_smooth_l1")
        if args.b:
            b_run = Path(args.b)
        else:
            b_run = find_latest_run(runs_dir, "v9.1b_gaussian_nll")

        a_name = a_run.name if a_run else "v9.1a (not found)"
        b_name = b_run.name if b_run else "v9.1b (not found)"

        a_hist = load_history(a_run)
        b_hist = load_history(b_run)

        a_n = len(a_hist["composite"]) if a_hist and "composite" in a_hist else 0
        b_n = len(b_hist["composite"]) if b_hist and "composite" in b_hist else 0
        print(f'\n[{time.strftime("%H:%M:%S")}] Found:')
        print(f"  A: {a_name}  ({a_n} epochs)")
        print(f"  B: {b_name}  ({b_n} epochs)")

        if a_hist or b_hist:
            plot_comparison(
                a_hist, b_hist, a_name, b_name, args.out, warmup_epoch=args.warmup
            )
            print_best_summary(a_hist, b_hist, a_name, b_name)
        else:
            print("No data yet.")

        if not args.live:
            break
        print("\nWaiting 30s for next update... (Ctrl+C to stop)")
        time.sleep(30)


if __name__ == "__main__":
    main()
