"""
reader_hstretch.py - the horizontal stretch probe for the learned angle reader.

eval_mist_hybrid.py probes the angle reader with the vertical stretch and the
shear only; Chapter 5 describes the stretch in both directions. This script adds
the horizontal stretch (x scaled about the tip by the factors of STRETCH), with
the same pairing and filters as eval_mist_hybrid.py: GT spray instances paired
by tip with a detection decoded at ANGLE_MATCH_OBJ_THRESH, skipped when the map
rule or the label gives no angle, tip and axis of the base prediction warped with
the frame. As a check, the first run of each call also recomputes the vertical
stretch slope, which must equal the saved probes_learned_head.stretch_y.slope.

Architecture flags must match the run (ENV_BASE, plus AFGNET_ANGLE_HEAD=cc for 7b
or AFGNET_MIST_FROM_STEM=1 for 5h); loss-only flags do not matter here.

    python evaluation/reader_hstretch.py --split val --runs abl_heads_s42 abl_nomask_s42 ...
    python evaluation/reader_hstretch.py --split test --runs abl_heads_s42 --label final_heads_s42

Output: results/reader_hstretch_{split}_{label}.json
"""

import argparse
import json

import cv2
import numpy as np

import eval_mist_hybrid as MH
from common import RESULTS_DIR, load_model
from dataset import AFGNetDataset, load_frame
from metrics import compute_spray_angle, decode_predictions, tip_match

from config import ANGLE_MATCH_OBJ_THRESH, DEVICE, IMG_H, IMG_W, REPO_ROOT, SPRAY_IDX, TEST_JSON, VAL_JSON

W, H = IMG_W, IMG_H
CKPT_DIR = REPO_ROOT / "train" / "model_2" / "checkpoint"


def matrices(tip):
    return {
        "stretch_x": lambda sc: np.array([[sc, 0, (1 - sc) * tip[0]], [0, 1, 0]], float),
        "stretch_y": lambda sc: np.array([[1, 0, 0], [0, sc, (1 - sc) * tip[1]]], float),
    }


def probe(model, ds, names):
    out = {n: [] for n in names}
    for s in ds.samples:
        if SPRAY_IDX not in s["gt_classes"]:
            continue
        img = load_frame(s["image_path"])
        prob, o = MH.predict_map(model, ds, img)
        low = [p for p in decode_predictions(o, batch_idx=0, obj_thresh=ANGLE_MATCH_OBJ_THRESH) if p["class_id"] == SPRAY_IDX]
        for c, k in zip(s["gt_classes"], s["gt_kpts"]):
            if c != SPRAY_IDX or not any(v > 0 for v in k):
                continue
            m = tip_match(low, k)
            if m is None or MH.cone_from_map(prob, m["keypoints"]) is None or compute_spray_angle(k) is None:
                continue
            tip = np.array(m["keypoints"][:2]) * [W, H]
            mk = matrices(tip)
            for name in names:
                per = {}
                for sc in MH.STRETCH:
                    M = mk[name](sc)
                    im2 = cv2.warpAffine(img, M, (W, H), borderValue=(0, 0, 0))
                    _, o2 = MH.predict_map(model, ds, im2)
                    per[sc] = (compute_spray_angle(MH.warp_kp6(k, M, W, H)), MH.head_angle(model, o2, MH.warp_kp6(m["keypoints"], M, W, H)))
                g1, c1 = per[1.0]
                out[name] += [(g - g1, p - c1) for sc, (g, p) in per.items() if sc != 1.0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--label", help="result label when a single run is scored under another name (test: final_heads_s42)")
    args = ap.parse_args()
    ds = AFGNetDataset(VAL_JSON if args.split == "val" else TEST_JSON, augment=False)
    for i, run in enumerate(args.runs):
        label = args.label if args.label and len(args.runs) == 1 else run
        model = load_model(CKPT_DIR / run / "afgnet_v10.pth", DEVICE)
        names = ["stretch_x", "stretch_y"] if i == 0 else ["stretch_x"]
        pairs = probe(model, ds, names)
        res = {"run": run, "split": args.split, "stretch_x": {"slope": MH.slope(pairs["stretch_x"]), "N": len(pairs["stretch_x"])}}
        if "stretch_y" in pairs:
            saved = json.load(open(RESULTS_DIR / f"mist_hybrid_{args.split}_{label}.json"))["probes_learned_head"]["stretch_y"]["slope"]
            res["stretch_y_check"] = {"recomputed": MH.slope(pairs["stretch_y"]), "saved": saved}
            if abs(res["stretch_y_check"]["recomputed"] - saved) > 1e-6:
                print(f"WARNING {run}: recomputed vertical slope {res['stretch_y_check']['recomputed']:.4f} != saved {saved:.4f}")
        json.dump(res, open(RESULTS_DIR / f"reader_hstretch_{args.split}_{label}.json", "w"), indent=1)
        print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
