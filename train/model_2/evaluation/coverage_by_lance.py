"""
coverage_by_lance.py - which spray lance a run fails to pair, and the position
lookup restricted to the instances it did pair.

The angle metric (eval_angle_reeval.py, metrics.tip_match) pairs each ground
truth spray instance with the nearest spray prediction by tip distance. A run
with coverage below 1 is scored on fewer instances than the baselines, which
eval_angle_reeval computes on every instance. This script repeats the pairing
for one checkpoint, reports the unpaired instances per lance (lances found by
the same box-centre clustering as the lookup), and the lookup error on the
paired instances alone, so the model can be read against a reference computed
on the same instances.

Run from train/model_2 with the flags the run was trained with, e.g.
    AFGNET_INSTANCE_NORM=0 ... python evaluation/coverage_by_lance.py \
        --ckpt checkpoint/abl_nomask_s42/afgnet_v10.pth --label abl_nomask_s42
Output: results/coverage_by_lance_val_<label>.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "structure"))

from common import load_model  # noqa: E402
from config import ANGLE_ASPECT, ANGLE_MATCH_OBJ_THRESH, SPRAY_IDX, TRAIN_JSON, VAL_JSON  # noqa: E402
from dataset import AFGNetDataset  # noqa: E402
from metrics import compute_spray_angle, decode_predictions, tip_match  # noqa: E402

import eval_angle_reeval as E  # noqa: E402

RESULTS = HERE.parent / "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    train = E.spray_instances(AFGNetDataset(TRAIN_JSON, augment=False))
    tr_ang = np.array([compute_spray_angle(k, ANGLE_ASPECT) for _, k in train])
    centres = E.cluster_centres([p for p, _ in train], E.CLUSTER_RADIUS)
    means = [float(tr_ang[[i for i, (p, _) in enumerate(train) if np.linalg.norm(p - c) <= E.CLUSTER_RADIUS]].mean())
             for c in centres]

    ds = AFGNetDataset(VAL_JSON, augment=False)
    model = load_model(Path(args.ckpt), DEVICE).eval()
    per_lance = {i: {"box_centre": c.round(3).tolist(), "n": 0, "paired": 0} for i, c in enumerate(centres)}
    lookup_all, lookup_paired = [], []
    with torch.no_grad():
        for idx, s in enumerate(ds.samples):
            gts = [i for i, c in enumerate(s["gt_classes"]) if c == SPRAY_IDX and any(v > 0 for v in s["gt_kpts"][i])]
            if not gts:
                continue
            img, _ = ds[idx]
            out = model(img.unsqueeze(0).to(DEVICE))
            preds = [p for p in decode_predictions(out, batch_idx=0, obj_thresh=ANGLE_MATCH_OBJ_THRESH)
                     if p["class_id"] == SPRAY_IDX]
            for gi in gts:
                ci = E.nearest_cluster(np.array(s["gt_boxes"][gi][:2]), centres, E.CLUSTER_RADIUS)
                a = compute_spray_angle(s["gt_kpts"][gi], ANGLE_ASPECT)
                err = abs(a - (means[ci] if ci is not None else float(tr_ang.mean())))
                lookup_all.append(err)
                paired = tip_match(preds, s["gt_kpts"][gi]) is not None
                if paired:
                    lookup_paired.append(err)
                key = ci if ci is not None else -1
                per_lance.setdefault(key, {"box_centre": None, "n": 0, "paired": 0})
                per_lance[key]["n"] += 1
                per_lance[key]["paired"] += int(paired)

    res = {
        "checkpoint": args.ckpt, "label": args.label, "split": "val",
        "coverage": f"{len(lookup_paired)}/{len(lookup_all)}",
        "per_lance": {str(k): v for k, v in per_lance.items()},
        "lookup_MAE_all": float(np.mean(lookup_all)),
        "lookup_MAE_paired": float(np.mean(lookup_paired)) if lookup_paired else float("nan"),
    }
    RESULTS.mkdir(exist_ok=True)
    out_path = RESULTS / f"coverage_by_lance_val_{args.label}.json"
    json.dump(res, open(out_path, "w"), indent=1)
    lances = ", ".join(f"{v['box_centre']} {v['paired']}/{v['n']}" for v in per_lance.values())
    print(f"{args.label}: coverage {res['coverage']}  lookup all {res['lookup_MAE_all']:.3f}  "
          f"paired {res['lookup_MAE_paired']:.3f}  | {lances}")


if __name__ == "__main__":
    main()
