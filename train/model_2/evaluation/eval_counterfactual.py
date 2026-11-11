"""
eval_counterfactual.py - does the network MEASURE the spray angle, or does it
RECALL the angle of the nozzle it recognises?

The ordinary test set cannot tell: three nozzles, each at a fixed position,
each with a near-constant angle.  Position, identity and angle are one
variable.  This script builds inputs where they come apart, from the
existing frames, with no training and no new data:

  1. STRETCH   scale x about the nozzle tip (0.8 .. 1.25).  Same nozzle,
               same place, DIFFERENT apex angle (known from the transformed
               keypoints).  A measuring model tracks the new angle
               (slope ~ 1); a recalling model keeps answering the old one
               (slope ~ 0).  A position lookup has slope 0 by construction.
  2. MASK      black out the cone but keep the lance body, and vice versa.
               Measuring -> error explodes without the cone, survives
               without the body.  Recalling -> the opposite.
  3. SHIFT/ROT translate by +-2/4/6 cells, rotate +-10 deg.  Angle is
               unchanged; error should stay flat.
  4. TRACK     within one nozzle the angle drifts ~1 deg over the video.
               Pearson r between prediction and label across frames.
               Recalling -> r ~ 0.

Reading rules are fixed in READING below BEFORE any number is seen.

Matching: spray predictions at LOW_THRESH, nearest tip within TIP_MATCH_PX
(source pixels), as in diag_missed_sprays.py.  Angles in pixel degrees.
All transforms keep the 2048x1536 canvas so ANGLE_ASPECT stays valid.

Usage: python eval_counterfactual.py [--split val|test] [--ckpt PATH]
Output: results/counterfactual_<split>_<run>.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from common import MODEL1_CKPT, RESULTS_DIR, load_model  # also puts ../structure on sys.path
from dataset import AFGNetDataset, prev_frame_path
from metrics import compute_spray_angle
from tune_decode import decode

from config import (
    DEVICE,
    FEAT_H,
    FEAT_W,
    ORIG_IMG_H,
    ORIG_IMG_W,
    REPO_ROOT,
    SPRAY_IDX,
    TEST_JSON,
    TWO_FRAME,
    VAL_JSON,
)

W, H = ORIG_IMG_W, ORIG_IMG_H
LOW_THRESH = 0.05
TIP_MATCH_PX = 50
STRETCH = [0.8, 0.9, 1.0, 1.1, 1.25]
SHEAR_DEG = [-15, -8, 0, 8, 15]


def probe_matrices(tip):
    """Angle-changing transforms about the nozzle tip, keyed by probe name.

    stretch_x  the original probe. NOTE: once a model is trained with
               ANGLE_AUG (3c, horizontal stretch) this one is no longer
               independent - the model has seen it.
    stretch_y  vertical stretch: changes the angle, never used in training.
    shear      horizontal shear x' = x + tan(a)*(y - tip_y): changes the
               angle, never used in training.
    Each value: list of (param, 2x3 M); param 1.0 / 0 is the identity.
    """
    tx, ty = float(tip[0]), float(tip[1])
    out = {
        "stretch_x": [(sc, np.array([[sc, 0, (1 - sc) * tx], [0, 1, 0]], float)) for sc in STRETCH],
        "stretch_y": [(sc, np.array([[1, 0, 0], [0, sc, (1 - sc) * ty]], float)) for sc in STRETCH],
        "shear": [(a, np.array([[1, np.tan(np.radians(a)), -np.tan(np.radians(a)) * ty], [0, 1, 0]], float)) for a in SHEAR_DEG],
    }
    return out


IDENTITY_PARAM = {"stretch_x": 1.0, "stretch_y": 1.0, "shear": 0}
SHIFT_CELLS = [-6, -4, -2, 2, 4, 6]
ROT_DEG = [-10, 10]
TIP_KEEP_R = 25  # px kept visible around the tip in both mask tests
CONE_DILATE = 25
N_BOOT = 1000

READING = {
    "stretch_slope": {"measures": ">= 0.8", "recalls": "<= 0.3"},
    "stretch_y_slope / shear_slope (independent probes for 3c)": {"measures": ">= 0.5", "recalls": "<= 0.2"},
    "mask_cone_MAE": {"measures": ">= 10 deg or no spray prediction", "recalls": "<= 3 deg"},
    "mask_body_MAE": {"measures": "<= 3 deg", "recalls": ">= 10 deg"},
    "shift_rot_MAE_increase": {"measures": "< 1 deg", "recalls": ">= 3 deg"},
    "track_r": {"measures": "> 0.3", "recalls": "~ 0"},
}


# ── model I/O ────────────────────────────────────────────────────────────
def predict(model, ds, img_rgb, prev_rgb=None):
    # make_input handles TWO_FRAME (prev_rgb must have received the same
    # geometric transform / mask as img_rgb, else the "motion" is fake)
    x = ds.make_input(img_rgb, prev_rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(x)
    return {
        "obj": (out["centerness"][0, 0] * out["iou_quality"][0, 0]).cpu().numpy(),
        "bbox": out["bbox"][0].cpu().numpy(),
        "cls": out["class_logits"][0].cpu().numpy(),
        "obs": out["observability"][0, 0].cpu().numpy(),
        "kpt": out["keypoints"][0].cpu().numpy(),
    }


def match_angle(c, tip_px):
    """Nearest-tip spray prediction. Returns (pred_angle, tip_err_px) or None."""
    preds = [p for p in decode(c, LOW_THRESH) if p["class_id"] == SPRAY_IDX]
    if not preds:
        return None
    best = min(preds, key=lambda p: np.hypot((p["keypoints"][0] * W - tip_px[0]), (p["keypoints"][1] * H - tip_px[1])))
    d = float(np.hypot(best["keypoints"][0] * W - tip_px[0], best["keypoints"][1] * H - tip_px[1]))
    if d > TIP_MATCH_PX:
        return None
    return compute_spray_angle(np.array(best["keypoints"])), d


# ── geometry ─────────────────────────────────────────────────────────────
def warp(img, M, kpts_px):
    """Affine-warp image and 3x2 keypoints (pixels). Black border."""
    out = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    k = np.hstack([kpts_px, np.ones((3, 1))]) @ M.T
    return out, k


def angle_px(kpts_px):
    return compute_spray_angle((kpts_px / [W, H]).ravel())


def in_frame(k):
    return bool((k[:, 0] >= 0).all() and (k[:, 0] < W).all() and (k[:, 1] >= 0).all() and (k[:, 1] < H).all())


def mask_cone(img, kpts_px, box_px):
    hull = cv2.convexHull(kpts_px.astype(np.int32))
    m = np.zeros(img.shape[:2], np.uint8)
    cv2.fillConvexPoly(m, hull, 255)
    m = cv2.dilate(m, np.ones((2 * CONE_DILATE + 1,) * 2, np.uint8))
    x1, y1, x2, y2 = [int(v) for v in box_px]
    m[max(0, y1) : y2 + 1, max(0, x1) : x2 + 1] = 0  # keep the lance body
    cv2.circle(m, tuple(kpts_px[0].astype(int)), TIP_KEEP_R, 0, -1)  # keep the tip
    out = img.copy()
    out[m > 0] = 0
    return out


def mask_body(img, kpts_px, box_px):
    m = np.zeros(img.shape[:2], np.uint8)
    x1, y1, x2, y2 = [int(v) for v in box_px]
    m[max(0, y1) : y2 + 1, max(0, x1) : x2 + 1] = 255
    cv2.circle(m, tuple(kpts_px[0].astype(int)), TIP_KEEP_R, 0, -1)
    out = img.copy()
    out[m > 0] = 0
    return out


# ── stats ────────────────────────────────────────────────────────────────
def boot_ci(vals, fn, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    vals = list(vals)
    if len(vals) < 2:
        return [float("nan"), float("nan")]
    est = [fn([vals[i] for i in rng.integers(0, len(vals), len(vals))]) for _ in range(n)]
    return [float(np.nanpercentile(est, 2.5)), float(np.nanpercentile(est, 97.5))]


MIN_DGT = 2.0  # deg; pairs where the GT angle barely moved cannot inform a slope


def slope(pairs):
    """Least-squares slope through the origin of d_pred on d_gt, using only
    pairs where the stretch actually changed the GT angle (|d_gt| >= MIN_DGT).
    Some cone orientations are insensitive to an x-stretch; for those the
    ratio is 0/0 and would blow up."""
    a = np.array([p for p in pairs if abs(p[0]) >= MIN_DGT], float)
    if a.size == 0:
        return float("nan")
    return float((a[:, 0] * a[:, 1]).sum() / (a[:, 0] ** 2).sum())


def spurious(pairs):
    """Mean |d_pred| over pairs where the GT angle did NOT change (|d_gt| < MIN_DGT).
    A measuring model should stay put; any movement here is driven by
    appearance, not by the angle."""
    a = [abs(p[1]) for p in pairs if abs(p[0]) < MIN_DGT]
    return float(np.mean(a)) if a else float("nan")


def mae(errs):
    e = [x for x in errs if x is not None]
    return float(np.mean(e)) if e else float("nan")


def nozzle_key(b):
    return f"({b[0]:.1f},{b[1]:.1f})"


# ── main ─────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--ckpt", default=str(MODEL1_CKPT))
    args = ap.parse_args()

    js = VAL_JSON if args.split == "val" else TEST_JSON
    ds = AFGNetDataset(js, augment=False)
    model = load_model(Path(args.ckpt), DEVICE)
    run_name = Path(args.ckpt).parent.name

    frames = [s for s in ds.samples if SPRAY_IDX in s["gt_classes"]]
    print(f"{args.split}: {len(frames)} frames with spray GT")

    base = defaultdict(list)  # nozzle -> [(gt, pred)]
    probe_pairs = {p: defaultdict(list) for p in IDENTITY_PARAM}  # probe -> nozzle -> [(d_gt, d_pred)]
    stretch_raw = defaultdict(lambda: defaultdict(list))  # nozzle -> s -> [(gt_s, pred_s)]  (stretch_x only)
    mask_res = {"cone": defaultdict(list), "body": defaultdict(list)}  # nozzle -> [err or None]
    mask_nopred = {"cone": 0, "body": 0}
    shift_res = defaultdict(list)  # label -> [err]
    shift_nopred = defaultdict(int)

    for s in tqdm(frames, desc="frames", unit="frame"):
        img = cv2.cvtColor(cv2.imread(s["image_path"]), cv2.COLOR_BGR2RGB)
        assert img.shape[:2] == (H, W), img.shape
        prev = cv2.cvtColor(cv2.imread(prev_frame_path(s["image_path"])), cv2.COLOR_BGR2RGB) if TWO_FRAME else None
        aff = lambda im, M: None if im is None else cv2.warpAffine(im, M, (W, H), borderValue=(0, 0, 0))
        base_c = predict(model, ds, img, prev)

        sprays = [(b, np.array(k).reshape(3, 2) * [W, H]) for b, c, k in zip(s["gt_boxes"], s["gt_classes"], s["gt_kpts"]) if c == SPRAY_IDX and any(v > 0 for v in k)]

        # 4. TRACK + reference prediction per nozzle
        for b, kp in sprays:
            r = match_angle(base_c, kp[0])
            base[nozzle_key(b)].append((angle_px(kp), r[0] if r else None))

        # 1. ANGLE-CHANGING PROBES about each nozzle's tip (stretch_x, stretch_y, shear)
        for b, kp in sprays:
            key = nozzle_key(b)
            for probe, mats in probe_matrices(kp[0]).items():
                ident = IDENTITY_PARAM[probe]
                per_s = {}
                for param, M in mats:
                    im2, k2 = warp(img, M, kp)
                    if not in_frame(k2):
                        continue
                    r = match_angle(predict(model, ds, im2, aff(prev, M)), k2[0])
                    per_s[param] = (angle_px(k2), r[0] if r else None)
                    if probe == "stretch_x":
                        stretch_raw[key][param].append(per_s[param])
                if ident in per_s and per_s[ident][1] is not None:
                    g1, p1 = per_s[ident]
                    for param, (g, p) in per_s.items():
                        if param != ident and p is not None:
                            probe_pairs[probe][key].append((g - g1, p - p1))

        # 2. MASK
        for b, kp in sprays:
            key = nozzle_key(b)
            box_px = ((b[0] - b[2] / 2) * W, (b[1] - b[3] / 2) * H, (b[0] + b[2] / 2) * W, (b[1] + b[3] / 2) * H)
            gt = angle_px(kp)
            for kind, fn in (("cone", mask_cone), ("body", mask_body)):
                r = match_angle(predict(model, ds, fn(img, kp, box_px), None if prev is None else fn(prev, kp, box_px)), kp[0])
                if r is None:
                    mask_nopred[kind] += 1
                    mask_res[kind][key].append(None)
                else:
                    mask_res[kind][key].append(abs(r[0] - gt))

        # 3. SHIFT / ROT (whole frame; evaluate every nozzle still in frame)
        variants = []
        for n in SHIFT_CELLS:
            variants.append((f"shift_x{n:+d}", np.array([[1, 0, n * W / FEAT_W], [0, 1, 0]], float)))
            variants.append((f"shift_y{n:+d}", np.array([[1, 0, 0], [0, 1, n * H / FEAT_H]], float)))
        for a in ROT_DEG:
            variants.append((f"rot{a:+d}", cv2.getRotationMatrix2D((W / 2, H / 2), a, 1.0)))
        for label, M in variants:
            im2 = None
            for b, kp in sprays:
                k2 = (np.hstack([kp, np.ones((3, 1))]) @ M.T)
                if not in_frame(k2):
                    continue
                if im2 is None:
                    im2 = cv2.warpAffine(img, M, (W, H), borderValue=(0, 0, 0))
                    c2 = predict(model, ds, im2, aff(prev, M))
                r = match_angle(c2, k2[0])
                if r is None:
                    shift_nopred[label] += 1
                    shift_res[label].append(None)
                else:
                    shift_res[label].append(abs(r[0] - angle_px(k2)))

    # ── summarise ────────────────────────────────────────────────────────
    res = {"split": args.split, "checkpoint": args.ckpt, "n_frames": len(frames), "reading_rules": READING}

    # base MAE (tip-matched) for reference
    base_err = [abs(g - p) for v in base.values() for g, p in v if p is not None]
    res["base"] = {"MAE": mae(base_err), "N": len(base_err), "N_gt": sum(len(v) for v in base.values())}

    # 1. angle-changing probes. res["stretch"] keeps the original x-stretch
    #    block (backward compatible); stretch_y and shear are the probes that
    #    stay independent of the 3c training augmentation.
    def probe_block(pairs_by_nozzle):
        all_pairs = [p for v in pairs_by_nozzle.values() for p in v]
        return {
            "slope": slope(all_pairs),
            "slope_CI95": boot_ci(all_pairs, slope),
            "N_pairs": len(all_pairs),
            "N_pairs_informative": sum(abs(p[0]) >= MIN_DGT for p in all_pairs),
            "spurious_pred_change_deg": spurious(all_pairs),
            "N_pairs_gt_static": sum(abs(p[0]) < MIN_DGT for p in all_pairs),
            "per_nozzle": {k: {"slope": slope(v), "spurious": spurious(v), "N_informative": sum(abs(p[0]) >= MIN_DGT for p in v), "N_static": sum(abs(p[0]) < MIN_DGT for p in v)} for k, v in pairs_by_nozzle.items()},
            "gt_angle_range_deg": [float(min(p[0] for p in all_pairs)), float(max(p[0] for p in all_pairs))] if all_pairs else None,
        }

    stretch_pairs = probe_pairs["stretch_x"]
    res["stretch_y"] = probe_block(probe_pairs["stretch_y"])
    res["shear"] = probe_block(probe_pairs["shear"])
    res["stretch"] = {
        **probe_block(stretch_pairs),
        "per_scale_mean": {
            k: {str(sc): {"gt": float(np.mean([g for g, _ in v])), "pred": mae([p for _, p in v if p is not None]) if any(p is not None for _, p in v) else None, "n_pred": sum(p is not None for _, p in v)} for sc, v in d.items()}
            for k, d in stretch_raw.items()
        },
    }

    # 2. mask
    res["mask"] = {}
    for kind in ("cone", "body"):
        allv = [e for v in mask_res[kind].values() for e in v]
        res["mask"][kind] = {
            "MAE_when_predicted": mae(allv),
            "no_spray_prediction": mask_nopred[kind],
            "N": len(allv),
            "per_nozzle": {k: {"MAE": mae(v), "no_pred": sum(e is None for e in v), "N": len(v)} for k, v in mask_res[kind].items()},
        }

    # 3. shift / rot
    res["shift_rot"] = {label: {"MAE": mae(v), "no_pred": shift_nopred[label], "N": len(v), "MAE_increase_vs_base": mae(v) - res["base"]["MAE"]} for label, v in shift_res.items()}

    # 4. track
    res["track"] = {}
    for k, v in base.items():
        pairs = [(g, p) for g, p in v if p is not None]
        if len(pairs) >= 3:
            g = np.array([x for x, _ in pairs])
            p = np.array([y for _, y in pairs])
            r = float(np.corrcoef(g, p)[0, 1]) if g.std() > 0 and p.std() > 0 else float("nan")
            res["track"][k] = {"r": r, "r_CI95": boot_ci(pairs, lambda q: float(np.corrcoef([a for a, _ in q], [b for _, b in q])[0, 1]) if np.std([a for a, _ in q]) > 0 and np.std([b for _, b in q]) > 0 else float("nan")), "gt_std_deg": float(g.std()), "pred_std_deg": float(p.std()), "N": len(pairs)}

    out = RESULTS_DIR / f"counterfactual_{args.split}_{run_name}.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")

    # ── print ────────────────────────────────────────────────────────────
    print(f'\nBASE (tip-matched): MAE {res["base"]["MAE"]:.2f} deg on {res["base"]["N"]}/{res["base"]["N_gt"]}')
    st = res["stretch"]
    print(f'\n1. STRETCH  slope = {st["slope"]:.2f}  CI95 {st["slope_CI95"][0]:.2f}..{st["slope_CI95"][1]:.2f}   (N={st["N_pairs_informative"]} pairs with |dGT|>={MIN_DGT}, GT moved {st["gt_angle_range_deg"][0]:+.1f}..{st["gt_angle_range_deg"][1]:+.1f} deg)   rule: {READING["stretch_slope"]}')
    print(f'            spurious change (pred moved while GT static, N={st["N_pairs_gt_static"]}): {st["spurious_pred_change_deg"]:.2f} deg   (measuring model: ~0)')
    for k, d in st["per_nozzle"].items():
        sl = f"{d['slope']:.2f}" if not np.isnan(d["slope"]) else "  n/a"
        sp = f"{d['spurious']:.2f}" if not np.isnan(d["spurious"]) else " n/a"
        print(f"     {k}: slope {sl} (N={d['N_informative']})   spurious {sp} deg (N={d['N_static']})")
    for name in ("stretch_y", "shear"):
        b = res[name]
        rng_txt = f'GT moved {b["gt_angle_range_deg"][0]:+.1f}..{b["gt_angle_range_deg"][1]:+.1f} deg' if b["gt_angle_range_deg"] else "no pairs"
        print(f'1{"b" if name == "stretch_y" else "c"}. {name.upper():<9} slope = {b["slope"]:.2f}  CI95 {b["slope_CI95"][0]:.2f}..{b["slope_CI95"][1]:.2f}   (N={b["N_pairs_informative"]}, {rng_txt})   spurious {b["spurious_pred_change_deg"]:.2f}   [independent of ANGLE_AUG training]')
    print("\n2. MASK")
    for kind in ("cone", "body"):
        m = res["mask"][kind]
        print(f'   {kind:4s} hidden: MAE {m["MAE_when_predicted"]:.2f} deg when predicted, no spray prediction in {m["no_spray_prediction"]}/{m["N"]}   rule: {READING["mask_" + kind + "_MAE"]}')
    print("\n3. SHIFT / ROT (MAE, +increase vs base)")
    for label, d in res["shift_rot"].items():
        print(f'   {label:10s} {d["MAE"]:.2f} ({d["MAE_increase_vs_base"]:+.2f})  no_pred {d["no_pred"]}/{d["N"]}')
    print("\n4. TRACK (within-nozzle Pearson r, prediction vs label)")
    for k, d in res["track"].items():
        print(f'   {k}: r = {d["r"]:.2f}  CI95 {d["r_CI95"][0]:.2f}..{d["r_CI95"][1]:.2f}   gt std {d["gt_std_deg"]:.2f}, pred std {d["pred_std_deg"]:.2f}  (N={d["N"]})')
    print(f"\nWritten: {out}")


if __name__ == "__main__":
    main()
