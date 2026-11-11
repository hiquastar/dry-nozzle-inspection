"""
plot_history.py - Visualize AFGNet v9 training history

Reads metrics.json saved by train.py and produces a comprehensive plot
covering all v9 metrics + thesis success criteria.

Usage:
    python plot_history.py                              # auto-find latest run
    python plot_history.py --run train/model_1/checkpoint/v9_20260505_xx    # specific run
    python plot_history.py --save plot.png              # save to file
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "structure"))

from config import REPO_ROOT, TARGETS


def find_latest_run(repo_root):
    """Find most recent v9 run directory."""
    runs_dir = Path(repo_root) / "train" / "model_1" / "checkpoint"
    if not runs_dir.exists():
        raise FileNotFoundError(f"No runs/ directory at {runs_dir}")

    v9_runs = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("v9_")]
    )
    if not v9_runs:
        raise FileNotFoundError(f"No v9_* runs found in {runs_dir}")

    return v9_runs[-1]


def load_metrics(run_dir):
    """Load metrics.json from run directory."""
    metrics_path = Path(run_dir) / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found at {metrics_path}")

    with open(metrics_path, "r") as f:
        return json.load(f)


def plot_history(history, run_name="", save_path=None):
    """Generate full training history plot."""
    epochs = list(range(1, len(history.get("train_loss", [])) + 1))

    if not epochs:
        print("No epochs in history yet")
        return

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle(f"AFGNet v9 Training History — {run_name}", fontsize=15, y=1.00)

    # ────────────────────────────────────────
    # Row 1: Loss + Detection + Classification + Hierarchy
    # ────────────────────────────────────────

    # [0,0] Loss
    ax = axes[0, 0]
    if "train_loss" in history:
        ax.plot(epochs, history["train_loss"], label="train_loss", color="C0")
    if "val_loss" in history:
        ax.plot(epochs, history["val_loss"], label="val_loss", color="C1")
    ax.set_title("Total Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    # [0,1] Detection mAP
    ax = axes[0, 1]
    if "mAP_50" in history:
        ax.plot(epochs, history["mAP_50"], label="mAP@.5", color="C2", linewidth=2)
        ax.axhline(
            y=TARGETS["mAP_50"],
            color="red",
            linestyle="--",
            alpha=0.5,
            label=f'Target ({TARGETS["mAP_50"]})',
        )
    ax.set_title("Detection mAP@0.5")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP")
    ax.set_ylim([-0.05, 1.05])
    ax.legend()
    ax.grid(alpha=0.3)

    # [0,2] Classification F1 (4-class softmax)
    ax = axes[0, 2]
    if "cls_f1" in history:
        ax.plot(
            epochs, history["cls_f1"], label="cls_f1 (4-class)", color="C0", linewidth=2
        )
        ax.axhline(
            y=TARGETS["cls_f1"],
            color="red",
            linestyle="--",
            alpha=0.5,
            label=f'Target ({TARGETS["cls_f1"]})',
        )
    ax.set_title("Classification F1 (4-class weighted)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1")
    ax.set_ylim([-0.05, 1.05])
    ax.legend()
    ax.grid(alpha=0.3)

    # [0,3] Hierarchical heads (type, activity, observ if present)
    ax = axes[0, 3]
    has_hierarchy = False
    if "type_f1" in history:
        ax.plot(epochs, history["type_f1"], label="type_f1", color="C3", linewidth=2)
        has_hierarchy = True
    if "activity_f1" in history:
        ax.plot(
            epochs, history["activity_f1"], label="activity_f1", color="C4", linewidth=2
        )
        has_hierarchy = True

    # observ_f1 (could be obs_f1 or observ_f1)
    obs_key = (
        "observ_f1"
        if "observ_f1" in history
        else ("obs_f1" if "obs_f1" in history else None)
    )
    if obs_key:
        ax.plot(
            epochs,
            history[obs_key],
            label="observ_f1 (CORE)",
            color="C5",
            linewidth=2.5,
        )

    ax.axhline(
        y=TARGETS["observability_f1"],
        color="red",
        linestyle="--",
        alpha=0.5,
        label=f'obs target ({TARGETS["observability_f1"]})',
    )

    if has_hierarchy:
        ax.set_title("Hierarchical Heads F1")
    else:
        ax.set_title("Observability F1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1")
    ax.set_ylim([-0.05, 1.05])
    ax.legend()
    ax.grid(alpha=0.3)

    # ────────────────────────────────────────
    # Row 2: Angle MAE + Composite + LR + Success criteria
    # ────────────────────────────────────────

    # [1,0] Angle MAE
    ax = axes[1, 0]
    if "angle_mae" in history:
        ax.plot(
            epochs, history["angle_mae"], color="C6", linewidth=2, label="angle_mae"
        )
        ax.axhline(
            y=TARGETS["angle_mae_deg"],
            color="red",
            linestyle="--",
            alpha=0.5,
            label=f'Target ({TARGETS["angle_mae_deg"]}°)',
        )
    ax.set_title("Spray Angle MAE (lower is better)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("degrees")
    ax.legend()
    ax.grid(alpha=0.3)

    # [1,1] Composite score
    ax = axes[1, 1]
    if "composite" in history:
        ax.plot(
            epochs, history["composite"], color="C7", linewidth=2.5, label="composite"
        )
        # Mark best
        best_ep = int(np.argmax(history["composite"]))
        best_val = history["composite"][best_ep]
        ax.scatter(
            [best_ep + 1],
            [best_val],
            color="red",
            s=100,
            zorder=5,
            label=f"Best: E{best_ep+1} = {best_val:.3f}",
        )
    ax.set_title("Composite Score (best epoch saved)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim([-0.05, 1.05])
    ax.legend()
    ax.grid(alpha=0.3)

    # [1,2] Learning rate
    ax = axes[1, 2]
    if "lr" in history:
        ax.plot(epochs, history["lr"], color="C8", linewidth=2)
    ax.set_title("Learning Rate")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR")
    ax.grid(alpha=0.3)

    # [1,3] Success criteria summary (text)
    ax = axes[1, 3]
    ax.axis("off")

    last_idx = len(epochs) - 1
    if last_idx >= 0:
        last_metrics = {
            "mAP@.5": history.get("mAP_50", [0])[last_idx],
            "cls_f1": history.get("cls_f1", [0])[last_idx],
            "observ_f1": history.get(obs_key, [0])[last_idx] if obs_key else 0,
            "angle_mae": history.get("angle_mae", [0])[last_idx],
        }
        if "type_f1" in history:
            last_metrics["type_f1"] = history["type_f1"][last_idx]
        if "activity_f1" in history:
            last_metrics["activity_f1"] = history["activity_f1"][last_idx]

        text = f"Final Metrics (Epoch {epochs[last_idx]})\n"
        text += "=" * 32 + "\n\n"

        # Format each metric with pass/fail
        criteria_check = [
            ("mAP@.5", last_metrics["mAP@.5"], TARGETS["mAP_50"], ">="),
            ("cls_f1", last_metrics["cls_f1"], TARGETS["cls_f1"], ">="),
            ("observ_f1", last_metrics["observ_f1"], TARGETS["observability_f1"], ">="),
            ("angle_mae", last_metrics["angle_mae"], TARGETS["angle_mae_deg"], "<="),
        ]

        for name, val, target, op in criteria_check:
            if op == ">=":
                ok = val >= target
                comparison = f'{val:.3f} {">=" if ok else "< "} {target}'
            else:
                ok = val <= target
                comparison = (
                    f'{val:.3f}{chr(176)} {">=" if not ok else "<="} {target}{chr(176)}'
                )
            mark = "✓" if ok else "✗"
            text += f"  {mark} {name:<12} {comparison}\n"

        # Hierarchical extras
        if "type_f1" in last_metrics or "activity_f1" in last_metrics:
            text += "\nHierarchical Heads:\n"
            if "type_f1" in last_metrics:
                text += f'    type_f1     {last_metrics["type_f1"]:.3f}\n'
            if "activity_f1" in last_metrics:
                text += f'    activity_f1 {last_metrics["activity_f1"]:.3f}\n'

        text += "\n" + "=" * 32 + "\n"
        text += "Success Criteria (thesis):\n"
        text += "  mAP@.5      >= 0.85\n"
        text += "  cls_f1      >= 0.90\n"
        text += "  observ_f1   >= 0.80\n"
        text += "  angle_mae   <= 5°\n"
        text += "  Pi5 latency <= 500ms\n"

        ax.text(
            0.0,
            1.0,
            text,
            fontsize=10,
            family="monospace",
            verticalalignment="top",
            transform=ax.transAxes,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", type=str, default=None, help="Run directory (default: latest v9 run)"
    )
    parser.add_argument(
        "--save", type=str, default=None, help="Save plot to file (e.g., plot.png)"
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Skip plt.show() (save only)"
    )
    args = parser.parse_args()

    if args.run:
        run_dir = Path(args.run)
    else:
        run_dir = find_latest_run(REPO_ROOT)

    print(f"Run dir: {run_dir}")

    history = load_metrics(run_dir)
    print(f'Epochs in history: {len(history.get("train_loss", []))}')
    print(f"Available metrics: {sorted(history.keys())}")

    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")

    save_path = args.save
    if save_path is None:
        save_path = run_dir / "training_history.png"

    plot_history(history, run_name=run_dir.name, save_path=save_path)


if __name__ == "__main__":
    main()
