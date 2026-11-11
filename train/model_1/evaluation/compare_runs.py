"""
compare_runs.py - Compare the results of 2 runs: v9.1a baseline vs v9.1b uncertainty.

Place this file in the train\ folder
Run: python compare_runs.py

Output: head-to-head comparison table + decision tree verdict.
"""

import json
import math
from pathlib import Path

import numpy as np


def find_latest_run(runs_dir, version_prefix):
    """Find the latest run folder matching the version prefix."""
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


def load_metrics(run_dir):
    """Load metrics.json and return the history dict."""
    metrics_file = run_dir / "metrics.json"
    if not metrics_file.exists():
        print(f"  WARNING: {metrics_file} not found")
        return None
    with open(metrics_file) as f:
        return json.load(f)


def best_epoch_summary(history):
    """Get the stats at the epoch with the highest composite."""
    if not history or "composite" not in history:
        return None
    composite = history["composite"]
    best_idx = int(np.argmax(composite))
    return {
        "epoch": best_idx + 1,
        "composite": composite[best_idx],
        "mAP_50": history.get("mAP_50", [0])[best_idx],
        "cls_f1": history.get("cls_f1", [0])[best_idx],
        "obs_f1": history.get("obs_f1", [0])[best_idx],
        "angle_mae": history.get("angle_mae", [float("nan")])[best_idx],
        "logvar_mean": history.get("logvar_mean", [None])[best_idx],
        "logvar_std": history.get("logvar_std", [None])[best_idx],
        "sigma_error_r": history.get("sigma_error_r", [None])[best_idx],
    }


def fmt(v, prec=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.{prec}f}"


def main():
    repo_root = Path(__file__).resolve().parents[3]  # evaluation -> model_1 -> train -> repo root
    runs_dir = repo_root / "train" / "model_1" / "checkpoint"

    print("=" * 70)
    print("FAIR COMPARE: v9.1a baseline (smooth_l1) vs v9.1b uncertainty (NLL)")
    print("=" * 70)

    # Find the 2 runs
    a_run = find_latest_run(runs_dir, "v9.1a_smooth_l1")
    b_run = find_latest_run(runs_dir, "v9.1b_gaussian_nll")

    if a_run is None:
        print("ERROR: No v9.1a_smooth_l1 run found in", runs_dir)
        return
    if b_run is None:
        print("ERROR: No v9.1b_gaussian_nll run found in", runs_dir)
        return

    print(f"\nv9.1a baseline: {a_run.name}")
    print(f"v9.1b NLL:      {b_run.name}")

    # Load metrics
    a_hist = load_metrics(a_run)
    b_hist = load_metrics(b_run)
    if a_hist is None or b_hist is None:
        return

    a_best = best_epoch_summary(a_hist)
    b_best = best_epoch_summary(b_hist)

    # ──────────────────────────────────────
    # Head-to-head table
    # ──────────────────────────────────────
    print()
    print("-" * 70)
    print(
        f'{"Metric":<22} | {"v9.1a baseline":<18} | {"v9.1b NLL":<18} | {"Delta":<10}'
    )
    print("-" * 70)

    rows = [
        ("Best epoch", a_best["epoch"], b_best["epoch"], 0),
        ("Composite score", a_best["composite"], b_best["composite"], 3),
        ("mAP@.5", a_best["mAP_50"], b_best["mAP_50"], 3),
        ("cls F1", a_best["cls_f1"], b_best["cls_f1"], 3),
        ("obs F1", a_best["obs_f1"], b_best["obs_f1"], 3),
        ("angle MAE (deg)", a_best["angle_mae"], b_best["angle_mae"], 2),
    ]
    for name, va, vb, prec in rows:
        delta = ""
        if (
            isinstance(va, (int, float))
            and isinstance(vb, (int, float))
            and not (math.isnan(va) or math.isnan(vb))
        ):
            d = vb - va
            delta = f"{d:+.{prec}f}"
        print(f"{name:<22} | {fmt(va, prec):<18} | {fmt(vb, prec):<18} | {delta:<10}")

    print("-" * 70)
    print("Uncertainty-specific (v9.1b only):")
    print(f'  logvar_mean:        {fmt(b_best["logvar_mean"], 3)}')
    print(f'  logvar_std:         {fmt(b_best["logvar_std"], 3)}')
    print(f'  sigma_error_r:      {fmt(b_best["sigma_error_r"], 3)}')
    print()

    # ──────────────────────────────────────
    # Decision tree verdict
    # ──────────────────────────────────────
    print("=" * 70)
    print("DECISION TREE VERDICT")
    print("=" * 70)

    angle_diff = b_best["angle_mae"] - a_best["angle_mae"]
    r = b_best["sigma_error_r"] if b_best["sigma_error_r"] is not None else float("nan")
    lv_std = b_best["logvar_std"] if b_best["logvar_std"] is not None else float("nan")

    print(f"Angle MAE delta (NLL - baseline): {angle_diff:+.2f} deg")
    print(f"Sigma-error correlation r:        {fmt(r, 3)}")
    print(f"Log-var std:                      {fmt(lv_std, 3)}")
    print()

    # Verdict logic from spec
    if math.isnan(r):
        verdict = "INCONCLUSIVE - missing uncertainty metrics"
        action = "Check that v9.1b actually trained with NLL (epoch > WARMUP=5)"
    elif lv_std < 0.1:
        verdict = "FAIL - log_var COLLAPSE"
        action = "Rollback or try variants (Laplace, no warm-up, no regularization)"
    elif angle_diff > 0.5:
        verdict = "FAIL - accuracy degraded > 0.5 deg"
        action = "Rollback or reduce LOGVAR_REG"
    elif r < 0.2:
        verdict = "FAIL - uncertainty has no signal (r < 0.2)"
        action = "Debug log_var head, or accept that aleatoric is too weak here"
    elif angle_diff <= 0.1 and r > 0.4:
        verdict = "SUCCESS - keep uncertainty for thesis"
        action = 'Frame as "trustworthy deployment" novelty in RQ2'
    elif r >= 0.2 and r < 0.4:
        verdict = "BORDERLINE - defendable if framed carefully"
        action = 'Use "trustworthy deployment" angle, downplay heteroscedastic claim'
    else:
        verdict = "BORDERLINE - mixed signals"
        action = "Discuss with supervisor before committing to thesis framing"

    print(f"VERDICT: {verdict}")
    print(f"ACTION:  {action}")
    print()

    # ──────────────────────────────────────
    # Pass-criteria checklist
    # ──────────────────────────────────────
    print("Pass criteria checklist:")
    checks = [
        (
            "angle_mae <= 2.1 deg",
            b_best["angle_mae"] <= 2.1 if not math.isnan(b_best["angle_mae"]) else None,
        ),
        ("logvar_std > 0.5", lv_std > 0.5 if not math.isnan(lv_std) else None),
        ("sigma_error_r >= 0.4", r >= 0.4 if not math.isnan(r) else None),
        (
            "NLL not WORSE than baseline (delta < 0.1)",
            angle_diff < 0.1 if not math.isnan(angle_diff) else None,
        ),
    ]
    for name, ok in checks:
        sym = "PASS" if ok is True else ("FAIL" if ok is False else "N/A")
        print(f"  [{sym:<4}] {name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
