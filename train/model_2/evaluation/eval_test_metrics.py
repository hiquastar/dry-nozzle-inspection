"""
eval_test_metrics.py - the standard instance-level metrics (the same
MetricsAccumulator that selects checkpoints during training) on ONE split
for ONE checkpoint, with per-class detail and a per-source breakdown.

Step 5 uses this on TEST, exactly once, for the final model:
    AFGNET_... python evaluation/eval_test_metrics.py --split test --ckpt <pth> --label <tag>

Per-source breakdown matters: 20250510 (1585 no_spray) is a dark video like
20250508 in train, so a good no_spray score there may come from the
brightness shortcut (plan section 2G) rather than from the nozzle head.

Output: results/metrics_<split>_<tag>.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import RESULTS_DIR, load_model  # also puts ../structure on sys.path
from dataset import AFGNetDataset, afgnet_collate
from metrics import MetricsAccumulator

from config import BATCH_SIZE, CLASS_NAMES, DEVICE, TEST_JSON, TRAIN_JSON, VAL_JSON

SPLITS = {"train": TRAIN_JSON, "val": VAL_JSON, "test": TEST_JSON}


def run(model, ds, indices=None):
    acc = MetricsAccumulator(ds.samples)
    sub = ds if indices is None else torch.utils.data.Subset(ds, indices)
    loader = DataLoader(sub, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=afgnet_collate)
    with torch.no_grad():
        for imgs, tgts in tqdm(loader, desc="  eval", leave=False, unit="batch"):
            out = model(imgs.to(DEVICE))
            acc.update_from_batch(out, tgts, DEVICE)
    return acc.compute_summary()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SPLITS), required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    tag = args.label or Path(args.ckpt).parent.name

    ds = AFGNetDataset(SPLITS[args.split], augment=False)
    model = load_model(Path(args.ckpt), DEVICE)

    res = {"checkpoint": args.ckpt, "split": args.split, "overall": run(model, ds), "per_source": {}}
    by_src = defaultdict(list)
    for i, s in enumerate(ds.samples):
        by_src[s["source_video"]].append(i)
    for src, idx in sorted(by_src.items()):
        res["per_source"][src] = {"n_images": len(idx), **run(model, ds, idx)}

    out = RESULTS_DIR / f"metrics_{args.split}_{tag}.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")

    o = res["overall"]
    print(f"\n{tag} on {args.split}  ({len(ds.samples)} images)")
    print(f'  mAP@0.5 {o["mAP_50"]:.3f}   cls_f1(macro) {o["cls_f1"]:.3f}   obs_f1 {o["obs_f1"]:.3f} (P {o["obs_precision"]:.2f} R {o["obs_recall"]:.2f})')
    print("  det F1 per class: " + "  ".join(f'{c.replace("nozzle_", "")} {o["det_F1_" + c]:.2f}' for c in CLASS_NAMES))
    print(f'  angle MAE {o["angle_mae_deg"]:.2f} deg on {o["angle_n"]}/{o["angle_n_gt"]} (tip-matched), tip err {o["tip_err_px"]:.1f} px')
    print("  per source (det F1 spray / no_spray / facing / big):")
    for src, r in res["per_source"].items():
        print(f'    {src:<26} n={r["n_images"]:4d}  ' + " / ".join(f'{r["det_F1_" + c]:.2f}' for c in CLASS_NAMES) + (f'   angle {r["angle_mae_deg"]:.2f} ({r["angle_n"]}/{r["angle_n_gt"]})' if r["angle_n_gt"] else ""))
    print(f"\nWritten: {out}")


if __name__ == "__main__":
    main()
