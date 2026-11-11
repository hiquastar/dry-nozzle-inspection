"""
tune_decode.py - inference-time tuning on the VAL split, no training.

Two questions (docs/notes/PLAN-V10.md step 0d / 0e):

  0d  OBJ_THRESH / OBS_THRESH sweep.  31 of 90 test sprays go undetected at
      OBJ_THRESH = 0.35.  Is that the network or just the threshold?
      Sweep on val, pick by macro-F1, never look at test here.

  0e  Keypoint aggregation.  decode_predictions() keeps the keypoints of the
      single NMS-surviving cell.  Every cell in the box predicts the same
      keypoints, so a score-weighted mean over the top-k cells of the NMS
      cluster should be less noisy.  Sweep k on val.

Raw network outputs are cached once per image so both sweeps reuse one
forward pass.  Angles are in the corrected pixel unit.

Output: results/decode_tuning_val.json
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from common import IOU_MATCH_THRESH, MODEL1_CKPT, RESULTS_DIR, load_model  # also puts ../structure on sys.path
from dataset import AFGNetDataset
from metrics import box_iou, compute_spray_angle, cxcywh_to_xyxy, nms

from config import (
    CLASS_NAMES,
    DEVICE,
    FACING_IDX,
    NMS_IOU_THRESH,
    NUM_CLASSES,
    OBJ_THRESH,
    OBS_THRESH,
    REPO_ROOT,
    SPRAY_IDX,
    VAL_JSON,
)

CKPT = MODEL1_CKPT  # model 1 is the baseline; v10 runs pass their own path
OBJ_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]
OBS_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
K_GRID = [1, 3, 5, 9, 15, 0]  # 0 = whole NMS cluster
OUT = RESULTS_DIR / "decode_tuning_val.json"


def cache_outputs(model, ds):
    """One forward pass per image; keep numpy maps."""
    cache = []
    with torch.no_grad():
        for idx in tqdm(range(len(ds)), desc="  forward", leave=False, unit="img"):
            img, _ = ds[idx]
            out = model(img.unsqueeze(0).to(DEVICE))
            cache.append(
                {
                    "obj": (out["centerness"][0, 0] * out["iou_quality"][0, 0]).cpu().numpy(),
                    "bbox": out["bbox"][0].cpu().numpy(),
                    "cls": out["class_logits"][0].cpu().numpy(),
                    "obs": out["observability"][0, 0].cpu().numpy(),
                    "kpt": out["keypoints"][0].cpu().numpy(),
                }
            )
    return cache


def decode(c, obj_thresh, k_agg=1):
    """Same rule as metrics.decode_predictions, plus keypoint aggregation
    over the NMS cluster: score-weighted mean of the top-k_agg cells that
    the kept cell suppressed (k_agg=1 reproduces the current behaviour,
    k_agg=0 uses the whole cluster)."""
    mask = c["obj"] >= obj_thresh
    if not mask.any():
        return []
    iy, ix = np.where(mask)
    cand = []
    for r, col in zip(iy, ix):
        cx, cy, bw, bh = c["bbox"][:, r, col]
        x1, y1 = max(0.0, cx - bw / 2), max(0.0, cy - bh / 2)
        x2, y2 = min(1.0, cx + bw / 2), min(1.0, cy + bh / 2)
        if x2 <= x1 or y2 <= y1:
            continue
        probs = 1.0 / (1.0 + np.exp(-c["cls"][:, r, col]))
        cid = int(np.argmax(probs))
        cand.append(
            {
                "box": (x1, y1, x2, y2),
                "score": float(c["obj"][r, col] * probs[cid]),
                "class_id": cid,
                "observability": float(c["obs"][r, col]),
                "keypoints": c["kpt"][:, r, col].copy(),
            }
        )
    if not cand:
        return []
    boxes = np.array([d["box"] for d in cand])
    scores = np.array([d["score"] for d in cand])
    keep = nms(boxes, scores, NMS_IOU_THRESH)

    out = []
    for i in keep:
        d = dict(cand[i])
        if k_agg != 1 and d["class_id"] == SPRAY_IDX:
            members = [
                j
                for j in range(len(cand))
                if cand[j]["class_id"] == SPRAY_IDX
                and box_iou(cand[j]["box"], d["box"]) >= NMS_IOU_THRESH
            ]
            members.sort(key=lambda j: -cand[j]["score"])
            if k_agg > 0:
                members = members[:k_agg]
            w = np.array([cand[j]["score"] for j in members])
            kp = np.array([cand[j]["keypoints"] for j in members])
            d["keypoints"] = (kp * w[:, None]).sum(0) / w.sum()
            d["n_agg"] = len(members)
        out.append(d)
    return out


def match(preds, sample):
    """Greedy class-aware IoU matching. Returns per-class TP/FP/FN and the
    list of (gt_idx, pred) pairs for matched GT."""
    tp = np.zeros(NUM_CLASSES, int)
    fp = np.zeros(NUM_CLASSES, int)
    fn = np.zeros(NUM_CLASSES, int)
    pairs = []
    used = set()
    order = sorted(range(len(preds)), key=lambda i: -preds[i]["score"])
    gt_used = set()
    for pi in order:
        p = preds[pi]
        best_iou, best_gi = IOU_MATCH_THRESH, None
        for gi, (b, cl) in enumerate(zip(sample["gt_boxes"], sample["gt_classes"])):
            if gi in gt_used or cl != p["class_id"]:
                continue
            iou = box_iou(p["box"], cxcywh_to_xyxy(b))
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_gi is None:
            fp[p["class_id"]] += 1
        else:
            gt_used.add(best_gi)
            tp[p["class_id"]] += 1
            pairs.append((best_gi, p))
    for gi, cl in enumerate(sample["gt_classes"]):
        if gi not in gt_used:
            fn[cl] += 1
    return tp, fp, fn, pairs


def prf(tp, fp, fn):
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f = 2 * p * r / max(p + r, 1e-9)
    return float(p), float(r), float(f)


def sweep_obj(cache, ds):
    rows = {}
    for t in OBJ_GRID:
        TP = np.zeros(NUM_CLASSES, int)
        FP = np.zeros(NUM_CLASSES, int)
        FN = np.zeros(NUM_CLASSES, int)
        errs = []
        obs_pairs = []  # (gt_is_spray, pred_obs) on matched spraying GT
        for c, s in zip(cache, ds.samples):
            tp, fp, fn, pairs = match(decode(c, t), s)
            TP += tp
            FP += fp
            FN += fn
            for gi, p in pairs:
                cl = s["gt_classes"][gi]
                if cl in (SPRAY_IDX, FACING_IDX):
                    obs_pairs.append((cl == SPRAY_IDX, p["observability"]))
                if cl == SPRAY_IDX:
                    ga = compute_spray_angle(s["gt_kpts"][gi])
                    pa = compute_spray_angle(p["keypoints"])
                    if ga is not None and pa is not None:
                        errs.append(abs(ga - pa))
        per = {}
        for i, n in enumerate(CLASS_NAMES):
            per[n] = dict(zip(("P", "R", "F1"), prf(TP[i], FP[i], FN[i])))
        rows[str(t)] = {
            "per_class": per,
            "macro_F1": float(np.mean([per[n]["F1"] for n in CLASS_NAMES])),
            "spray_MAE_pixel": float(np.mean(errs)) if errs else float("nan"),
            "spray_N_matched": len(errs),
            "_obs_pairs": obs_pairs,
        }
    return rows


def sweep_obs(obs_pairs):
    y = np.array([g for g, _ in obs_pairs], bool)
    s = np.array([o for _, o in obs_pairs], float)
    rows = {}
    for t in OBS_GRID:
        pred = s > t
        tp = int((pred & y).sum())
        fp = int((pred & ~y).sum())
        fn = int((~pred & y).sum())
        rows[str(t)] = dict(zip(("P", "R", "F1"), prf(tp, fp, fn)))
    return rows


def sweep_k(cache, ds, obj_thresh):
    rows = {}
    for k in K_GRID:
        errs, n_agg = [], []
        for c, s in zip(cache, ds.samples):
            _, _, _, pairs = match(decode(c, obj_thresh, k_agg=k), s)
            for gi, p in pairs:
                if s["gt_classes"][gi] != SPRAY_IDX:
                    continue
                ga = compute_spray_angle(s["gt_kpts"][gi])
                pa = compute_spray_angle(p["keypoints"])
                if ga is not None and pa is not None:
                    errs.append(abs(ga - pa))
                    n_agg.append(p.get("n_agg", 1))
        e = np.array(errs)
        rows[str(k)] = {
            "MAE_pixel": float(e.mean()) if e.size else float("nan"),
            "median": float(np.median(e)) if e.size else float("nan"),
            "p90": float(np.percentile(e, 90)) if e.size else float("nan"),
            "N": int(e.size),
            "mean_cells_aggregated": float(np.mean(n_agg)) if n_agg else 1.0,
        }
    return rows


def main():
    global CKPT, OUT
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(CKPT))
    ap.add_argument("--label", default=None, help="tag for the output file (default: ckpt parent dir name)")
    args = ap.parse_args()
    CKPT = Path(args.ckpt)
    tag = args.label or CKPT.parent.name
    OUT = RESULTS_DIR / f"decode_tuning_val_{tag}.json"

    print(f"checkpoint: {CKPT}")
    print(f"val split : {VAL_JSON}")
    ds = AFGNetDataset(VAL_JSON, augment=False)
    model = load_model(CKPT, DEVICE)
    cache = cache_outputs(model, ds)

    obj_rows = sweep_obj(cache, ds)
    best_t = max(obj_rows, key=lambda t: obj_rows[t]["macro_F1"])
    obs_rows = sweep_obs(obj_rows[str(OBJ_THRESH)].pop("_obs_pairs"))
    for r in obj_rows.values():
        r.pop("_obs_pairs", None)
    best_obs = max(obs_rows, key=lambda t: obs_rows[t]["F1"])
    k_rows = sweep_k(cache, ds, OBJ_THRESH)
    k_rows_best_t = sweep_k(cache, ds, float(best_t))

    res = {
        "checkpoint": str(CKPT),
        "split": str(VAL_JSON),
        "current": {"OBJ_THRESH": OBJ_THRESH, "OBS_THRESH": OBS_THRESH},
        "obj_thresh_sweep": obj_rows,
        "obj_thresh_best_by_macro_F1": float(best_t),
        "obs_thresh_sweep_at_current_obj": obs_rows,
        "obs_thresh_best_by_F1": float(best_obs),
        "kpt_agg_sweep_at_current_obj": k_rows,
        "kpt_agg_sweep_at_best_obj": k_rows_best_t,
    }
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")

    print("\n0d  OBJ_THRESH sweep (val)")
    print(f'{"thr":>5} {"macroF1":>8} {"spray P":>8} {"spray R":>8} {"nospr F1":>9} {"facing F1":>10} {"big F1":>7} {"MAE":>6} {"N":>4}')
    for t, r in obj_rows.items():
        pc = r["per_class"]
        mark = " <- current" if float(t) == OBJ_THRESH else (" <- best" if t == best_t else "")
        print(
            f'{float(t):5.2f} {r["macro_F1"]:8.3f} {pc["nozzle_spray"]["P"]:8.3f} {pc["nozzle_spray"]["R"]:8.3f} '
            f'{pc["nozzle_no_spray"]["F1"]:9.3f} {pc["nozzle_facing"]["F1"]:10.3f} {pc["big_nozzle"]["F1"]:7.3f} '
            f'{r["spray_MAE_pixel"]:6.2f} {r["spray_N_matched"]:4d}{mark}'
        )

    print(f"\n0d  OBS_THRESH sweep at OBJ_THRESH={OBJ_THRESH} (matched spray+facing on val)")
    for t, r in obs_rows.items():
        mark = " <- current" if float(t) == OBS_THRESH else (" <- best" if t == best_obs else "")
        print(f'{float(t):5.2f}  P={r["P"]:.3f} R={r["R"]:.3f} F1={r["F1"]:.3f}{mark}')

    for label, rows in (("current OBJ_THRESH", k_rows), (f"OBJ_THRESH={best_t}", k_rows_best_t)):
        print(f"\n0e  keypoint aggregation over top-k NMS-cluster cells, {label} (val, pixel deg)")
        print(f'{"k":>4} {"MAE":>6} {"median":>7} {"p90":>6} {"N":>4} {"cells":>6}')
        for k, r in rows.items():
            print(f'{("all" if k == "0" else k):>4} {r["MAE_pixel"]:6.2f} {r["median"]:7.2f} {r["p90"]:6.2f} {r["N"]:4d} {r["mean_cells_aggregated"]:6.1f}')

    print(f"\nWritten: {OUT}")


if __name__ == "__main__":
    main()
