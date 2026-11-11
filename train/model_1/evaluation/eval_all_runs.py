"""
eval_all_runs.py - Test-set evaluation of Full + 3 ablations on reverse split.

Runs cell-level (mAP / cls_f1 / obs_f1 / per-class recall) AND
instance-level (MAE_primary / MAE_gated / R_obs) for each run.

Output: train/model_1/results/chapter6_final_numbers.json   (merged into existing keys)

Run from train/train_v9_v1/ :
    python eval_all_runs.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "structure"))

from dataset import AFGNetDataset, afgnet_collate
from metrics import (
    MetricsAccumulator,
    box_iou,
    compute_spray_angle,
    cxcywh_to_xyxy,
    decode_predictions,
)
from model import AFGNet

from config import (
    BATCH_SIZE,
    CLASS_NAMES,
    DEVICE,
    NUM_CLASSES,
    NUM_WORKERS,
    OBS_THRESH,
    REPO_ROOT,
    SPRAY_IDX,
    TEST_JSON,
)

RUNS = [
    ("full", "v9_fixes_reverse_seed42", "train_v9_v1.pth", "Full model (FIX 1+2+3)"),
    (
        "abl_A",
        "v9_rabl_A_no_hier_seed42",
        "train_v9_v1_rabl_A.pth",
        "Abl A: No Hierarchical Masking",
    ),
    (
        "abl_B",
        "v9_rabl_B_no_cellkpt_seed42",
        "train_v9_v1_rabl_B.pth",
        "Abl B: No Cell-Aware KPT",
    ),
    (
        "abl_C",
        "v9_rabl_C_no_flipswap_seed42",
        "train_v9_v1_rabl_C.pth",
        "Abl C: No Flip Identity Swap",
    ),
]

OUT_MERGED = REPO_ROOT / "train" / "model_1" / "results" / "chapter6_final_numbers.json"
IOU_MATCH_THRESH = 0.5


def load_model(ckpt_path: Path, model_factory) -> AFGNet:
    """Loads checkpoint, dropping the non-leaf meshgrid buffers."""
    model = model_factory()
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = ckpt["model"]
    for k in ("head.cell_x", "head.cell_y"):
        state.pop(k, None)
    res = model.load_state_dict(state, strict=False)
    unexpected_missing = [
        k
        for k in res.missing_keys
        if k not in ("head.cell_x", "head.cell_y", "head.kpt_rho")
    ]
    if unexpected_missing:
        raise RuntimeError(f"Missing learned keys: {unexpected_missing}")
    model.to(DEVICE).eval()
    return model, ckpt


def cell_level_eval(model, test_loader):
    """Run the regular MetricsAccumulator on test set."""
    acc = MetricsAccumulator()
    with torch.no_grad():
        for imgs, tgts in tqdm(test_loader, desc="  cell-level", leave=False):
            imgs = imgs.to(DEVICE)
            out = model(imgs)
            acc.update_from_batch(out, tgts, DEVICE)

    summary = acc.compute_summary()
    if acc.cls_y_true:
        _, rec, _, sup = precision_recall_fscore_support(
            acc.cls_y_true,
            acc.cls_y_pred,
            labels=list(range(NUM_CLASSES)),
            average=None,
            zero_division=0,
        )
        for i, name in enumerate(CLASS_NAMES):
            summary[f"recall_{name}"] = float(rec[i])
            summary[f"support_{name}"] = int(sup[i])
    return summary


def instance_mae_eval(model, test_ds):
    """Per-instance IoU-matched MAE split (primary/gated/R_obs)."""
    primary, gated = [], []
    n_gt_spray = n_matched = n_gated = 0
    with torch.no_grad():
        for idx in tqdm(range(len(test_ds)), desc="  per-instance", leave=False):
            s = test_ds.samples[idx]
            spray_idxs = [i for i, c in enumerate(s["gt_classes"]) if c == SPRAY_IDX]
            n_gt_spray += len(spray_idxs)
            if not spray_idxs:
                continue
            img, _ = test_ds[idx]
            out = model(img.unsqueeze(0).to(DEVICE))
            preds = [
                p
                for p in decode_predictions(out, batch_idx=0)
                if p["class_id"] == SPRAY_IDX
            ]
            used = set()
            for gi in spray_idxs:
                gt_angle = compute_spray_angle(s["gt_kpts"][gi])
                if gt_angle is None:
                    continue
                gt_xyxy = cxcywh_to_xyxy(s["gt_boxes"][gi])
                best_iou, best_pi = IOU_MATCH_THRESH, None
                for pi, p in enumerate(preds):
                    if pi in used:
                        continue
                    iou = box_iou(p["box"], gt_xyxy)
                    if iou > best_iou:
                        best_iou, best_pi = iou, pi
                if best_pi is None:
                    continue
                used.add(best_pi)
                pred_angle = compute_spray_angle(preds[best_pi]["keypoints"])
                if pred_angle is None:
                    continue
                err = abs(gt_angle - pred_angle)
                primary.append(err)
                n_matched += 1
                if preds[best_pi]["observability"] > OBS_THRESH:
                    gated.append(err)
                    n_gated += 1
    return {
        "MAE_primary_deg": float(np.mean(primary)) if primary else float("nan"),
        "MAE_gated_deg": float(np.mean(gated)) if gated else float("nan"),
        "R_obs_instance": (n_gated / n_matched) if n_matched else float("nan"),
        "N_gt_spray": n_gt_spray,
        "N_matched": n_matched,
        "N_gated": n_gated,
        "spray_detection_rate": n_matched / n_gt_spray if n_gt_spray else float("nan"),
    }


def main():
    if not TEST_JSON.exists():
        sys.exit(f"ERROR: test split not found: {TEST_JSON}")

    print(f"Test split : {TEST_JSON}")
    print(f"Device     : {DEVICE}")
    print()
    print("Loading test dataset (used by all runs)...")
    test_ds = AFGNetDataset(TEST_JSON, augment=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=afgnet_collate,
        pin_memory=(DEVICE.type == "cuda"),
    )
    print(f"  {len(test_ds)} images, {len(test_loader)} batches")

    # Use a single model factory; each ablation has its own model.py BUT the
    # full model already imports the canonical model.py for v9 head class.
    # Each ablation imports its own model.py from its own dir — we can't load
    # all 4 models with one `model.py` import. Workaround: dynamically import
    # the model.py from each ablation's directory.
    import importlib.util

    def make_model_for(run_dir_name: str):
        train_dir_name = {
            "v9_fixes_reverse_seed42": "train_v9_v1",
            "v9_rabl_A_no_hier_seed42": "train_v9_v1_rabl_A",
            "v9_rabl_B_no_cellkpt_seed42": "train_v9_v1_rabl_B",
            "v9_rabl_C_no_flipswap_seed42": "train_v9_v1_rabl_C",
        }.get(run_dir_name)
        if train_dir_name is None:
            return AFGNet  # fallback
        model_py = REPO_ROOT / "train" / train_dir_name / "model.py"
        spec = importlib.util.spec_from_file_location(f"model_{run_dir_name}", model_py)
        mod = importlib.util.module_from_spec(spec)
        # Inject config from current dir (config is identical for all 4 runs)
        spec.loader.exec_module(mod)
        return mod.AFGNet

    all_results = {}
    for key, run_dir_name, ckpt_name, label in RUNS:
        ckpt = REPO_ROOT / "train" / "model_1" / "checkpoint" / run_dir_name / ckpt_name
        if not ckpt.exists():
            print(f"\nSKIP {label}: {ckpt} not found")
            continue

        print()
        print("=" * 70)
        print(f"  {label}")
        print("=" * 70)

        AFGNetCls = make_model_for(run_dir_name)
        model, ckpt_data = load_model(ckpt, lambda: AFGNetCls(pretrained=False))
        print(
            f'  ckpt epoch={ckpt_data["epoch"]}, val_composite={ckpt_data["composite"]:.4f}'
        )

        cell = cell_level_eval(model, test_loader)
        inst = instance_mae_eval(model, test_ds)

        # Clean obs_recall → R_obs_cell
        if "obs_recall" in cell:
            cell["R_obs_cell"] = cell.pop("obs_recall")
        if "angle_mae_deg" in cell:
            cell["angle_mae_cellevel_INVALID"] = cell.pop("angle_mae_deg")

        result = {
            "run": run_dir_name,
            "label": label,
            "checkpoint": str(ckpt),
            "ckpt_epoch": ckpt_data["epoch"],
            "val_composite": ckpt_data["composite"],
            "cell_level": cell,
            "instance_level": inst,
        }
        all_results[key] = result

        # Print summary
        print()
        print(f'  TEST mAP@0.5         : {cell["mAP_50"]:.4f}')
        print(f'  TEST cls_f1 (wtd)    : {cell["cls_f1"]:.4f}')
        print(f'  TEST obs_f1 (S1)     : {cell["obs_f1"]:.4f}')
        print(f'  TEST R_obs (cell)    : {cell["R_obs_cell"]:.4f}')
        print(f'  TEST spray recall    : {cell["recall_nozzle_spray"]:.4f}')
        print()
        print(
            f'  Inst MAE_primary     : {inst["MAE_primary_deg"]:.3f}°  (N={inst["N_matched"]})'
        )
        print(
            f'  Inst MAE_gated       : {inst["MAE_gated_deg"]:.3f}°  (N={inst["N_gated"]})'
        )
        print(f'  Inst R_obs           : {inst["R_obs_instance"]:.4f}')
        print(f'  Spray det. rate      : {inst["spray_detection_rate"]:.4f}')

    # Merge & save
    merged = {}
    if OUT_MERGED.exists():
        with open(OUT_MERGED) as f:
            merged = json.load(f)
    merged["reverse_split_all_runs"] = all_results
    with open(OUT_MERGED, "w") as f:
        json.dump(merged, f, indent=2)
    print()
    print(f"Saved: {OUT_MERGED}")

    # Final comparison table
    print()
    print("=" * 110)
    print("FINAL COMPARISON (TEST SET — reverse split)")
    print("=" * 110)
    print(
        "%-30s  %6s  %6s  %6s  %6s  %7s  %7s  %4s"
        % ("Run", "mAP", "clsF1", "obsF1", "R_obs", "MAE_p", "MAE_g", "det")
    )
    print("-" * 80)
    for k, _, _, _ in RUNS:
        r = all_results.get(k)
        if r is None:
            continue
        c = r["cell_level"]
        i = r["instance_level"]
        print(
            "%-30s  %6.3f  %6.3f  %6.3f  %6.3f  %6.2f° %6.2f° %4.2f"
            % (
                r["label"],
                c["mAP_50"],
                c["cls_f1"],
                c["obs_f1"],
                c["R_obs_cell"],
                i["MAE_primary_deg"],
                i["MAE_gated_deg"],
                i["spray_detection_rate"],
            )
        )


if __name__ == "__main__":
    main()
