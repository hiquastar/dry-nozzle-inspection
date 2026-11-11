"""
eval_test_set.py  —  Chapter 6 Script 1: test-set evaluation, canonical model

Model  : train/model_1/checkpoint/v9_20260513_013233/train_v9_v1.pth  (best-composite checkpoint)
Split  : data/splits_reverse/test.coco.json

Outputs
  train/model_1/checkpoint/v9_20260513_013233/test_results_canonical.json
  train/model_1/results/chapter6_final_numbers.json           (created / updated)

Metrics
  mAP@0.5          class-aware detection mAP
  cls_f1           weighted 4-class F1  (cell-level positive cells)
  per_class_recall recall per class [spray, no_spray, facing, big]
  obs_f1           binary F1 on S1 cells ONLY (gt class = spray | facing)
  R_obs            spray-cell recall = TP_spray / (TP_spray + FN_spray)

NOTE on angle_mae: the value computed here is cell-level (multiple FCOS
cells per GT instance; includes ALL GT-spray cells).  It is NOT valid for
Chapter 6.  Run eval_mae_split.py for MAE_primary and MAE_gated.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE.parent / "structure"))

from dataset import AFGNetDataset, afgnet_collate
from metrics import MetricsAccumulator
from model import AFGNet

from config import (
    BATCH_SIZE,
    CLASS_NAMES,
    DEVICE,
    NUM_CLASSES,
    NUM_WORKERS,
    REPO_ROOT,
    TEST_JSON,
)

CANONICAL_RUN = "v9_fixes_reverse_seed42"
CKPT_PATH = REPO_ROOT / "train" / "model_1" / "checkpoint" / CANONICAL_RUN / "train_v9_v1.pth"
OUT_RUN = REPO_ROOT / "train" / "model_1" / "checkpoint" / CANONICAL_RUN / "test_results.json"
OUT_MERGED = REPO_ROOT / "train" / "model_1" / "results" / "chapter6_final_numbers.json"
MERGED_KEY = "reverse_train_test_eval"


def load_model(ckpt_path: Path, device: torch.device) -> AFGNet:
    model = AFGNet(pretrained=False)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model"] if (isinstance(ckpt, dict) and "model" in ckpt) else ckpt
    # cell_x / cell_y are deterministic buffers computed from FEAT_H/FEAT_W.
    # PyTorch meshgrid saves them with stride-0 layout; the copy into the
    # model's own stride-0 buffer raises "multiple elements → same memory".
    # Drop them from the state dict — the freshly-init model already holds
    # the correct values.
    state.pop("head.cell_x", None)
    state.pop("head.cell_y", None)
    missing_or_extra = model.load_state_dict(state, strict=False)
    unexpected = [
        k
        for k in missing_or_extra.missing_keys
        if k not in ("head.cell_x", "head.cell_y")
    ]
    if unexpected:
        raise RuntimeError(f"Unexpected missing learned keys: {unexpected}")
    model.to(device).eval()
    if isinstance(ckpt, dict):
        ep = ckpt.get("epoch", "?")
        comp = ckpt.get("composite", float("nan"))
        print(f"  checkpoint: epoch={ep}, val_composite={comp:.4f}")
    return model


def main():
    print("=" * 65)
    print("Chapter 6 — Script 1: test-set evaluation (canonical model)")
    print("=" * 65)

    if not CKPT_PATH.exists():
        sys.exit(f"ERROR: checkpoint not found:\n  {CKPT_PATH}")
    if not TEST_JSON.exists():
        sys.exit(f"ERROR: test split not found:\n  {TEST_JSON}")

    print(f"\nCheckpoint : {CKPT_PATH}")
    print(f"Test split : {TEST_JSON}")
    print(f"Device     : {DEVICE}")

    # ── model ────────────────────────────────────────────────────────────────
    print("\nLoading model...")
    model = load_model(CKPT_PATH, DEVICE)

    # ── data ─────────────────────────────────────────────────────────────────
    print("\nLoading test split...")
    test_ds = AFGNetDataset(TEST_JSON, augment=False)
    test_ds.print_stats()

    loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=afgnet_collate,
        pin_memory=(DEVICE.type == "cuda"),
    )
    print(f"Test batches: {len(loader)}")

    # ── evaluation loop ───────────────────────────────────────────────────────
    print("\nRunning evaluation...")
    acc = MetricsAccumulator()
    model.eval()
    with torch.no_grad():
        for i, (imgs, tgts) in enumerate(loader, 1):
            imgs = imgs.to(DEVICE)
            out = model(imgs)
            acc.update_from_batch(out, tgts, DEVICE)
            if i % 10 == 0 or i == len(loader):
                print(f"  [{i}/{len(loader)}]")

    summary = acc.compute_summary()

    # ── per-class recall ──────────────────────────────────────────────────────
    if acc.cls_y_true:
        _, recall_arr, _, support_arr = precision_recall_fscore_support(
            acc.cls_y_true,
            acc.cls_y_pred,
            labels=list(range(NUM_CLASSES)),
            average=None,
            zero_division=0,
        )
        for i, name in enumerate(CLASS_NAMES):
            summary[f"recall_{name}"] = float(recall_arr[i])
            summary[f"support_{name}"] = int(support_arr[i])

    # ── rename R_obs key for clarity ──────────────────────────────────────────
    # obs_recall from MetricsAccumulator = TP_spray / (TP_spray + FN_spray)
    # i.e. fraction of GT spray cells where obs_pred > 0.5
    if "obs_recall" in summary:
        summary["R_obs_cell_level"] = summary.pop("obs_recall")

    # ── flag cell-level angle_mae as invalid for Ch6 ─────────────────────────
    if "angle_mae_deg" in summary:
        summary["angle_mae_cellevel_INVALID"] = summary.pop("angle_mae_deg")
        summary["_angle_note"] = (
            "angle_mae_cellevel is computed at FCOS cell granularity "
            "(one GT instance -> multiple cells -> multiple errors). "
            "It is NOT MAE_primary. Use eval_mae_split.py instead."
        )

    # ── print ─────────────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("TEST SET RESULTS  —  v9_20260513_013233 (best-composite ckpt)")
    print("=" * 65)
    print(f"  mAP@0.5        : {summary.get('mAP_50', float('nan')):.4f}")
    print(f"  cls_f1 (wtd)   : {summary.get('cls_f1', float('nan')):.4f}")
    print(f"  obs_f1 (S1)    : {summary.get('obs_f1', float('nan')):.4f}")
    print(f"  R_obs (cells)  : {summary.get('R_obs_cell_level', float('nan')):.4f}")
    print(f"  obs_n_samples  : {summary.get('obs_n_samples', '?')}")
    print(f"  obs_n_positive : {summary.get('obs_n_positive', '?')}")
    print()
    print("  Per-class recall:")
    for name in CLASS_NAMES:
        r = summary.get(f"recall_{name}", float("nan"))
        n = summary.get(f"support_{name}", 0)
        print(f"    {name:<22}: {r:.4f}  (n_cells={n})")
    print()
    print("  Per-class AP@0.5:")
    for i, name in enumerate(CLASS_NAMES):
        ap = summary.get(f"AP_50_cls{i}", float("nan"))
        print(f"    {name:<22}: {ap:.4f}")
    print()
    print(
        f"  [INVALID for Ch6] angle_mae (cell-level): "
        f"{summary.get('angle_mae_cellevel_INVALID', float('nan')):.3f}°  "
        f"(see eval_mae_split.py)"
    )

    # ── save ──────────────────────────────────────────────────────────────────
    payload = {
        "run": CANONICAL_RUN,
        "checkpoint": str(CKPT_PATH),
        "split": "test",
        "results": summary,
    }

    OUT_RUN.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_RUN, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {OUT_RUN}")

    merged = {}
    if OUT_MERGED.exists():
        with open(OUT_MERGED) as f:
            merged = json.load(f)
    merged[MERGED_KEY] = payload
    with open(OUT_MERGED, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Merged into: {OUT_MERGED}")


if __name__ == "__main__":
    main()
