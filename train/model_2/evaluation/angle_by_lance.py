"""
angle_by_lance.py - the angle error of the final model on the test set, split by
lance and by whether the paired detection would pass the operating threshold.

The angle metric of Chapter 5 pairs each annotated spray instance with the spray
detection whose tip is nearest, over detections decoded down to
ANGLE_MATCH_OBJ_THRESH (0.05), so that the angle is scored on every lance the
network gives a tip for. The deployed system keeps only detections with a score of
at least OBJ_THRESH (0.35). This script repeats the pairing of eval_mist_hybrid.py
(same filter, so the all-instance MAE equals the saved MAE_learned_head_deg) and
reports the MAE of the angle reader and of the keypoint branch per lance and for
the instances whose paired detection reaches the operating threshold.

    AFGNET_INSTANCE_NORM=0 AFGNET_TIP_RELATIVE_EDGES=1 AFGNET_MIST_HEAD=1 AFGNET_LEARNED_HEADS=1 \
      python evaluation/angle_by_lance.py [--seed 123]

Output: results/angle_by_lance_test_final_heads_s{42,123}.json
"""

import collections
import json
import sys

import numpy as np

import eval_angle_reeval as AR
import eval_mist_hybrid as MH
from common import RESULTS_DIR, load_model
from dataset import AFGNetDataset, load_frame
from metrics import compute_spray_angle, decode_predictions, tip_match

from config import ANGLE_MATCH_OBJ_THRESH, DEVICE, OBJ_THRESH, REPO_ROOT, SPRAY_IDX, TEST_JSON, TRAIN_JSON

SEED = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 42  # final design, seed 42 or 123
CKPT = REPO_ROOT / "train" / "model_2" / "checkpoint" / f"abl_heads_s{SEED}" / "afgnet_v10.pth"


def lookup_table():
    """the position lookup baseline of eval_angle_reeval.py: clusters of training box
    centres and the mean training angle of each cluster"""
    tr = AR.spray_instances(AFGNetDataset(TRAIN_JSON, augment=False))
    ang = np.array([compute_spray_angle(k) for _, k in tr])
    centres = AR.cluster_centres([p for p, _ in tr], AR.CLUSTER_RADIUS)
    means = [float(ang[[i for i, (p, _) in enumerate(tr) if np.linalg.norm(p - c) <= AR.CLUSTER_RADIUS]].mean()) for c in centres]
    return centres, means, float(ang.mean())


def main():
    ds = AFGNetDataset(TEST_JSON, augment=False)
    model = load_model(CKPT, DEVICE)
    centres, cmeans, const = lookup_table()
    rows = []
    for s in ds.samples:
        if SPRAY_IDX not in s["gt_classes"]:
            continue
        img = load_frame(s["image_path"])
        prob, out = MH.predict_map(model, ds, img)
        low = [p for p in decode_predictions(out, batch_idx=0, obj_thresh=ANGLE_MATCH_OBJ_THRESH) if p["class_id"] == SPRAY_IDX]
        for b, c, k in zip(s["gt_boxes"], s["gt_classes"], s["gt_kpts"]):
            if c != SPRAY_IDX or not any(v > 0 for v in k):
                continue
            m = tip_match(low, k)
            if m is None:
                continue
            ga = compute_spray_angle(k)
            # same filter as eval_mist_hybrid.py
            if MH.cone_from_map(prob, m["keypoints"]) is None or ga is None:
                continue
            ci = AR.nearest_cluster(np.array(b[:2]), centres, AR.CLUSTER_RADIUS)
            rows.append({
                "lance": f"({k[0]:.1f},{k[1]:.1f})",  # annotated tip, rounded: one key per lance
                "score": float(m["obj_score"]),  # the quantity the operating threshold is applied to
                "err_reader": abs(MH.head_angle(model, out, m["keypoints"]) - ga),
                "err_kpt": abs(compute_spray_angle(m["keypoints"]) - ga),
                "err_lookup": abs(ga - (const if ci is None else cmeans[ci])),
            })

    def summ(rs):
        return {"N": len(rs),
                "MAE_reader": float(np.mean([r["err_reader"] for r in rs])) if rs else None,
                "MAE_kpt": float(np.mean([r["err_kpt"] for r in rs])) if rs else None,
                "MAE_lookup": float(np.mean([r["err_lookup"] for r in rs])) if rs else None,
                "max_err_reader": float(np.max([r["err_reader"] for r in rs])) if rs else None}

    by_lance = collections.defaultdict(list)
    for r in rows:
        by_lance[r["lance"]].append(r)
    res = {
        "checkpoint": str(CKPT), "operating_threshold": OBJ_THRESH, "pairing_threshold": ANGLE_MATCH_OBJ_THRESH,
        "all": summ(rows),
        "score_at_least_operating": summ([r for r in rows if r["score"] >= OBJ_THRESH]),
        "score_below_operating": summ([r for r in rows if r["score"] < OBJ_THRESH]),
        "by_lance": {k: {**summ(v), "score_mean": float(np.mean([r["score"] for r in v]))} for k, v in sorted(by_lance.items())},
    }
    json.dump(res, open(RESULTS_DIR / f"angle_by_lance_test_final_heads_s{SEED}.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
