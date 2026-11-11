"""
eval_mae_split.py  —  Chapter 6 Script 2: instance-level angle MAE split

Computes angle MAE at DETECTION INSTANCE level (not cell level):

  MAE_primary  angle MAE for all matched GT spray instances
               (IoU > 0.5 with a spray prediction, any obs value)
  MAE_gated    angle MAE for matched instances where obs > OBS_THRESH
               (model said "this measurement is valid")
  R_obs        N_gated / N_matched  (gating coverage)

Models evaluated:
  canonical  — train/model_1/checkpoint/v9_20260513_013233/train_v9_v1.pth
  abl_C      — train/model_1/checkpoint/v9_abl_C_20260512_121713/train_v9_v1_abl_no_flipswap.pth

Split: data/splits_reverse/test.coco.json

WARNING: The angle_mae values in every metrics.json training log are computed
at cell-level across ALL positive FCOS cells with GT class = spray.  One GT
spray instance maps to several cells, so the same error is counted multiple
times.  Those values MUST NOT be reported as MAE in Chapter 6.

Output: appended/merged into train/model_1/results/chapter6_final_numbers.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE.parent / "structure"))

from dataset import AFGNetDataset
from metrics import box_iou, compute_spray_angle, cxcywh_to_xyxy, decode_predictions
from model import AFGNet

from config import DEVICE, OBS_THRESH, REPO_ROOT, SPRAY_IDX, TEST_JSON

RUNS = {
    "reverse_train": {
        "run_id": "v9_fixes_reverse_seed42",
        "ckpt": REPO_ROOT / "train" / "model_1" / "checkpoint" / "v9_fixes_reverse_seed42" / "train_v9_v1.pth",
        "label": "Reverse split + 3 fixes (seed42)",
    },
}

OUT_MERGED = REPO_ROOT / "train" / "model_1" / "results" / "chapter6_final_numbers.json"
IOU_MATCH_THRESH = 0.5  # GT↔pred IoU threshold for "detected"


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
        print(
            f'    checkpoint: epoch={ckpt.get("epoch","?")}, '
            f'val_composite={ckpt.get("composite", float("nan")):.4f}'
        )
    return model


def eval_mae_split(
    model: AFGNet, test_ds: AFGNetDataset, device: torch.device, label: str
) -> dict:
    """
    Instance-level MAE split on the full test split.

    Matching strategy:
      For each GT spray instance, greedily match to the highest-IoU
      spray prediction (IoU > IOU_MATCH_THRESH) not already assigned.
      Unmatched GT instances are excluded from MAE (no prediction → no
      angle estimate), but counted in N_gt_spray and spray_det_rate.
    """
    primary_errors = []  # |angle_gt − angle_pred| for every matched instance
    gated_errors = []  # same, restricted to obs > OBS_THRESH
    n_gt_spray = 0  # total GT spray annotations in test set
    n_matched = 0  # GT sprays with an IoU-matched spray prediction
    n_gated = 0  # matched AND obs > OBS_THRESH
    matched_keys = []  # (image_path, gi) for each matched instance — for audit

    model.eval()
    with torch.no_grad():
        for idx in tqdm(range(len(test_ds)), desc=f"  {label}", unit="img"):
            sample = test_ds.samples[idx]

            # Collect GT spray indices for this image
            spray_gt_idx = [
                i for i, c in enumerate(sample["gt_classes"]) if c == SPRAY_IDX
            ]
            n_gt_spray += len(spray_gt_idx)
            if not spray_gt_idx:
                continue

            # Forward pass
            img_t, _ = test_ds[idx]
            out = model(img_t.unsqueeze(0).to(device))
            decoded = decode_predictions(out, batch_idx=0)

            # Only spray-class predictions can match GT spray instances
            spray_preds = [p for p in decoded if p["class_id"] == SPRAY_IDX]

            # Greedy IoU matching: one pred per GT, one GT per pred
            used_pred = set()
            for gi in spray_gt_idx:
                gt_kpts = sample["gt_kpts"][gi]
                gt_angle = compute_spray_angle(gt_kpts)
                if gt_angle is None:
                    # Invalid GT keypoints (all-zero annotation)
                    continue

                gt_xyxy = cxcywh_to_xyxy(sample["gt_boxes"][gi])

                best_iou = IOU_MATCH_THRESH
                best_pidx = None
                for pi, p in enumerate(spray_preds):
                    if pi in used_pred:
                        continue
                    iou = box_iou(p["box"], gt_xyxy)
                    if iou > best_iou:
                        best_iou = iou
                        best_pidx = pi

                if best_pidx is None:
                    continue  # spray not detected → exclude from MAE, keep in det_rate

                used_pred.add(best_pidx)
                pred = spray_preds[best_pidx]
                pred_angle = compute_spray_angle(pred["keypoints"])
                if pred_angle is None:
                    continue

                err = abs(gt_angle - pred_angle)
                primary_errors.append(err)
                n_matched += 1
                matched_keys.append(
                    (sample["image_path"], gi, gt_angle, pred_angle, err)
                )

                if pred["observability"] > OBS_THRESH:
                    gated_errors.append(err)
                    n_gated += 1

    mae_p = float(np.mean(primary_errors)) if primary_errors else float("nan")
    mae_g = float(np.mean(gated_errors)) if gated_errors else float("nan")
    r_obs = (n_gated / n_matched) if n_matched > 0 else float("nan")
    det_r = (n_matched / n_gt_spray) if n_gt_spray > 0 else float("nan")

    return {
        "MAE_primary_deg": mae_p,
        "MAE_gated_deg": mae_g,
        "R_obs": r_obs,
        "N_gt_spray": n_gt_spray,
        "N_matched": n_matched,
        "N_gated": n_gated,
        "spray_detection_rate": det_r,
        "_per_instance": [
            {
                "img": Path(img).name,
                "gt_idx": gi,
                "gt_angle": round(gt_a, 3),
                "pred_angle": round(pa, 3),
                "err": round(e, 3),
            }
            for img, gi, gt_a, pa, e in matched_keys
        ],
        "_note": (
            f"IoU_thresh={IOU_MATCH_THRESH}, OBS_THRESH={OBS_THRESH}. "
            "MAE_primary: all matched GT spray instances (no obs filter). "
            "MAE_gated: matched instances with obs > OBS_THRESH. "
            "R_obs = N_gated / N_matched. "
            "Undetected sprays excluded from MAE but counted in N_gt_spray."
        ),
    }


def main():
    print("=" * 65)
    print("Chapter 6 — Script 2: instance-level MAE split (primary vs gated)")
    print("=" * 65)
    print()
    print("WARNING: metrics.json angle_mae values are cell-level and INVALID")
    print("for Chapter 6. This script computes the correct instance-level MAE.")
    print()

    if not TEST_JSON.exists():
        sys.exit(f"ERROR: test split not found:\n  {TEST_JSON}")

    print(f"Test split  : {TEST_JSON}")
    print(f"Device      : {DEVICE}")
    print(f"OBS_THRESH  : {OBS_THRESH}  (from config)")
    print(f"IoU thresh  : {IOU_MATCH_THRESH}")
    print()

    print("Loading test dataset...")
    test_ds = AFGNetDataset(TEST_JSON, augment=False)
    print(f"  {len(test_ds)} images")

    results = {}
    for name, cfg in RUNS.items():
        print(f'\n── {cfg["label"]} ({cfg["run_id"]}) ──')
        if not cfg["ckpt"].exists():
            print(f'  WARN: checkpoint not found: {cfg["ckpt"]}')
            results[name] = {"error": f'checkpoint not found: {cfg["ckpt"]}'}
            continue

        model = load_model(cfg["ckpt"], DEVICE)
        metrics = eval_mae_split(model, test_ds, DEVICE, cfg["label"])

        results[name] = {"run_id": cfg["run_id"], "label": cfg["label"], **metrics}

        print(f"\n  Results:")
        print(
            f'    MAE_primary  : {metrics["MAE_primary_deg"]:.3f}°  '
            f'(N={metrics["N_matched"]})'
        )
        print(
            f'    MAE_gated    : {metrics["MAE_gated_deg"]:.3f}°  '
            f'(N={metrics["N_gated"]})'
        )
        print(
            f'    R_obs        : {metrics["R_obs"]:.4f}  '
            f'({metrics["N_gated"]} / {metrics["N_matched"]})'
        )
        print(
            f'    spray det.   : {metrics["spray_detection_rate"]:.4f}  '
            f'({metrics["N_matched"]} / {metrics["N_gt_spray"]} GT sprays)'
        )

    # Comparison table
    if len(results) == 2 and all("error" not in v for v in results.values()):
        keys = list(results.keys())
        print()
        print("── Comparison ──────────────────────────────────")
        print(f'  {"Metric":<22}  {"canonical":>10}  {"abl_C":>10}')
        print(f'  {"-"*22}  {"-"*10}  {"-"*10}')
        for metric in (
            "MAE_primary_deg",
            "MAE_gated_deg",
            "R_obs",
            "spray_detection_rate",
        ):
            vals = [results[k].get(metric, float("nan")) for k in keys]
            print(f"  {metric:<22}  {vals[0]:>10.4f}  {vals[1]:>10.4f}")

    results["_training_log_warning"] = (
        "The angle_mae field in every metrics.json training log is computed "
        "at FCOS cell level (one GT instance -> multiple cells -> inflated N). "
        "Those values are NOT MAE_primary. The canonical model metrics.json "
        "ep43 angle_mae ~2.12° is NOT directly comparable to MAE_primary here."
    )

    # Merge into chapter6_final_numbers.json
    merged = {}
    if OUT_MERGED.exists():
        with open(OUT_MERGED) as f:
            merged = json.load(f)
    merged["mae_split"] = results
    with open(OUT_MERGED, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\nSaved to: {OUT_MERGED}")


if __name__ == "__main__":
    main()
