"""
map_threshold_probe.py - how much does the detection mAP depend on decoding
at the operating threshold before the AP is computed?

MetricsAccumulator decodes every image at OBJ_THRESH (0.35) and then computes
the VOC 11-point AP on those detections only, so the precision-recall curve
stops at the recall reached at 0.35. This script recomputes the class-aware
mAP@0.5 with the same code path but a lower decode threshold, to measure the
size of that truncation. It is a diagnostic: no checkpoint or design choice
depends on it, and no existing result file is overwritten.

    AFGNET_... python evaluation/map_threshold_probe.py --split val --ckpt <pth> --label <tag>

Output: results/map_threshold_<split>_<tag>.json
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import RESULTS_DIR, load_model  # also puts ../structure on sys.path
from dataset import AFGNetDataset, afgnet_collate
import metrics as M

from config import BATCH_SIZE, CLASS_NAMES, DEVICE, OBJ_THRESH, TEST_JSON, VAL_JSON

SPLITS = {"val": VAL_JSON, "test": TEST_JSON}


def map_at(model, ds, thr):
    """Collect decoded detections at `thr` and score them with the standard AP code."""
    preds_all, gts_all, gcls_all = [], [], []
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=afgnet_collate)
    with torch.no_grad():
        for imgs, tgts in tqdm(loader, desc=f"  thr {thr}", leave=False, unit="batch"):
            out = model(imgs.to(DEVICE))
            for b, i in enumerate(tgts["index"].view(-1).tolist()):
                s = ds.samples[int(i)]
                preds_all.append(M.decode_predictions(out, batch_idx=b, obj_thresh=thr))
                gts_all.append(s["gt_boxes"])
                gcls_all.append(s["gt_classes"])
    m = M.compute_map_classaware(preds_all, gts_all, gcls_all)
    return {"mAP_50": m["mAP"], "per_class": dict(zip(CLASS_NAMES, m["per_class"])),
            "n_detections": sum(len(p) for p in preds_all)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SPLITS), required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--thresholds", default=f"{OBJ_THRESH},0.05,0.01")
    args = ap.parse_args()

    ds = AFGNetDataset(SPLITS[args.split], augment=False)
    model = load_model(Path(args.ckpt), DEVICE)
    res = {"checkpoint": args.ckpt, "split": args.split, "operating_threshold": OBJ_THRESH, "by_threshold": {}}
    for thr in [float(t) for t in args.thresholds.split(",")]:
        res["by_threshold"][str(thr)] = map_at(model, ds, thr)

    out = RESULTS_DIR / f"map_threshold_{args.split}_{args.label}.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n{args.label} on {args.split}")
    for thr, r in res["by_threshold"].items():
        pc = "  ".join(f'{c.replace("nozzle_", "").replace("_nozzle", "")} {v:.3f}' for c, v in r["per_class"].items())
        print(f'  decode thr {float(thr):<5}  mAP@0.5 {r["mAP_50"]:.3f}   [{pc}]   detections {r["n_detections"]}')
    print(f"Written: {out}")


if __name__ == "__main__":
    main()
