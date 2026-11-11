"""
diag_missed_sprays.py - why is one third of the GT sprays "undetected"?

Findings this script reproduces (2026-09-16, val + test, reverse split):

  * Every missed spray is the same physical nozzle at (cx, cy) ~ (0.27, 0.02):
    it sits on the top frame border, its lance body is out of frame, so the
    GT box is clipped to ~0.04 x 0.04 (the other two lances: ~0.18 x 0.17).
  * The network fires on the right cell, classifies it as spray, but predicts
    a lance-sized box (~0.29 x 0.15).  IoU with the clipped GT box is 0.04,
    so the IoU >= 0.5 match fails and NMS drops it under a neighbour.
    Lowering OBJ_THRESH cannot fix this (see tune_decode.py).
  * Its keypoints are fine: matched by tip distance, angle error 0.82 deg,
    better than the other two nozzles (1.15 deg).

Consequence: for the ANGLE metric, match GT sprays by tip distance, not by
box IoU.  Box IoU stays the rule for detection metrics (mAP).

Output: results/diag_missed_sprays.json
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from common import RESULTS_DIR, load_model  # also puts ../structure on sys.path
from dataset import AFGNetDataset
from metrics import box_iou, compute_spray_angle, cxcywh_to_xyxy
from tune_decode import CKPT, cache_outputs, decode

from config import (
    CLASS_NAMES,
    DEVICE,
    FEAT_H,
    FEAT_W,
    OBJ_THRESH,
    ORIG_IMG_H,
    ORIG_IMG_W,
    REPO_ROOT,
    SPRAY_IDX,
    TEST_JSON,
    VAL_JSON,
)

OUT = RESULTS_DIR / "diag_missed_sprays.json"
LOW_THRESH = 0.05  # so the border nozzle has at least one candidate
TIP_MATCH_PX = 50  # tip-distance match radius, source pixels


def nozzle_key(box):
    return f"({box[0]:.1f},{box[1]:.1f})"


def annotation_stats(ds):
    """Per nozzle: GT box size, keypoint extent, keypoints-inside-box rate."""
    groups = defaultdict(list)
    for s in ds.samples:
        for b, c, k in zip(s["gt_boxes"], s["gt_classes"], s["gt_kpts"]):
            if c == SPRAY_IDX and any(v > 0 for v in k):
                groups[nozzle_key(b)].append((b, k))
    out = {}
    for key, rows in sorted(groups.items()):
        B = np.array([r[0] for r in rows])
        K = np.array([r[1] for r in rows]).reshape(-1, 3, 2)
        x1, y1 = B[:, 0] - B[:, 2] / 2, B[:, 1] - B[:, 3] / 2
        x2, y2 = B[:, 0] + B[:, 2] / 2, B[:, 1] + B[:, 3] / 2
        inside = (
            (K[:, :, 0] >= x1[:, None] - 1e-6)
            & (K[:, :, 0] <= x2[:, None] + 1e-6)
            & (K[:, :, 1] >= y1[:, None] - 1e-6)
            & (K[:, :, 1] <= y2[:, None] + 1e-6)
        )
        out[key] = {
            "n": len(rows),
            "box_w": float(B[:, 2].mean()),
            "box_h": float(B[:, 3].mean()),
            "box_touches_top_border": bool((y1 <= 1e-6).mean() > 0.5),
            "kpt_extent_w": float((K[:, :, 0].max(1) - K[:, :, 0].min(1)).mean()),
            "kpt_extent_h": float((K[:, :, 1].max(1) - K[:, :, 1].min(1)).mean()),
            "kpts_inside_box_rate": [float(inside[:, i].mean()) for i in range(3)],
        }
    return out


def model_stats(model, ds):
    cache = cache_outputs(model, ds)
    fate = Counter()
    per_nozzle = defaultdict(lambda: {"n": 0, "iou_miss": 0, "tip_err_px": [], "angle_err_iou": [], "angle_err_tip": [], "pred_box_wh": [], "obj_max": []})
    for c, s in zip(cache, ds.samples):
        preds_cur = [p for p in decode(c, OBJ_THRESH) if p["class_id"] == SPRAY_IDX]
        preds_low = [p for p in decode(c, LOW_THRESH) if p["class_id"] == SPRAY_IDX]
        for b, cl, k in zip(s["gt_boxes"], s["gt_classes"], s["gt_kpts"]):
            if cl != SPRAY_IDX:
                continue
            gk = np.array(k)
            gt = cxcywh_to_xyxy(b)
            d = per_nozzle[nozzle_key(b)]
            d["n"] += 1
            ga = compute_spray_angle(gk)

            # rule A: box IoU >= 0.5 at the current threshold (eval_mae_split rule)
            m = [p for p in preds_cur if box_iou(p["box"], gt) >= 0.5]
            if m:
                fate["matched by box IoU"] += 1
                pa = compute_spray_angle(np.array(m[0]["keypoints"]))
                if pa is not None and ga is not None:
                    d["angle_err_iou"].append(abs(pa - ga))
            else:
                fate["missed by box IoU"] += 1
                d["iou_miss"] += 1

            # rule B: nearest tip among low-threshold spray predictions
            best = min(
                preds_low,
                key=lambda p: np.hypot((p["keypoints"][0] - gk[0]) * ORIG_IMG_W, (p["keypoints"][1] - gk[1]) * ORIG_IMG_H),
                default=None,
            )
            if best is not None:
                pk = np.array(best["keypoints"])
                tip_px = float(np.hypot((pk[0] - gk[0]) * ORIG_IMG_W, (pk[1] - gk[1]) * ORIG_IMG_H))
                if tip_px <= TIP_MATCH_PX:
                    d["tip_err_px"].append(tip_px)
                    pa = compute_spray_angle(pk)
                    if pa is not None and ga is not None:
                        d["angle_err_tip"].append(abs(pa - ga))
                    bx = best["box"]
                    d["pred_box_wh"].append((bx[2] - bx[0], bx[3] - bx[1]))

            # objectness the network puts inside the GT box
            x1, y1, x2, y2 = gt
            i0, i1 = int(y1 * FEAT_H), min(FEAT_H - 1, int(y2 * FEAT_H))
            j0, j1 = int(x1 * FEAT_W), min(FEAT_W - 1, int(x2 * FEAT_W))
            d["obj_max"].append(float(c["obj"][i0 : i1 + 1, j0 : j1 + 1].max()))

    summary = {}
    for key, d in sorted(per_nozzle.items()):
        f = lambda v: float(np.mean(v)) if v else None
        summary[key] = {
            "n": d["n"],
            "missed_by_box_iou": d["iou_miss"],
            "obj_max_in_gt_box": f(d["obj_max"]),
            "pred_box_w": f([w for w, _ in d["pred_box_wh"]]),
            "pred_box_h": f([h for _, h in d["pred_box_wh"]]),
            "tip_err_px_mean": f(d["tip_err_px"]),
            "n_tip_matched": len(d["angle_err_tip"]),
            "angle_err_deg_box_iou_rule": f(d["angle_err_iou"]),
            "angle_err_deg_tip_rule": f(d["angle_err_tip"]),
        }
    return dict(fate), summary


def main():
    model = load_model(CKPT, DEVICE)
    result = {"checkpoint": str(CKPT), "OBJ_THRESH": OBJ_THRESH, "LOW_THRESH": LOW_THRESH, "TIP_MATCH_PX": TIP_MATCH_PX}
    for name, js in (("val", VAL_JSON), ("test", TEST_JSON)):
        ds = AFGNetDataset(js, augment=False)
        fate, per = model_stats(model, ds)
        result[name] = {"fate": fate, "per_nozzle_model": per, "per_nozzle_annotation": annotation_stats(ds)}
        print(f"\n[{name}] {fate}")
        print(f'{"nozzle":>12} {"n":>3} {"miss":>4} {"GTbox w x h":>13} {"top":>4} {"pred w x h":>13} {"objmax":>6} {"tip px":>6} {"ang(IoU)":>8} {"ang(tip)":>8} {"N tip":>5}')
        for key in per:
            a, m = result[name]["per_nozzle_annotation"][key], per[key]
            fmt = lambda v, w=6: f"{v:{w}.2f}" if v is not None else " " * (w - 1) + "-"
            print(
                f'{key:>12} {m["n"]:3d} {m["missed_by_box_iou"]:4d} {a["box_w"]:6.3f}x{a["box_h"]:<6.3f} {"yes" if a["box_touches_top_border"] else "no":>4} '
                f'{fmt(m["pred_box_w"]):>6}x{fmt(m["pred_box_h"]):<6} {m["obj_max_in_gt_box"]:6.3f} {fmt(m["tip_err_px_mean"]):>6} {fmt(m["angle_err_deg_box_iou_rule"], 8)} {fmt(m["angle_err_deg_tip_rule"], 8)} {m["n_tip_matched"]:5d}'
            )
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nWritten: {OUT}")


if __name__ == "__main__":
    main()
