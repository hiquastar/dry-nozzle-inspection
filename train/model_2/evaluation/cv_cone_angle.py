"""
cv_cone_angle.py - does the MIST BOUNDARY, found by classical CV from the
nozzle tip, agree with the annotated cone edges, and does the angle it gives
behave like a measurement (track true changes, follow shear / y-stretch)?

Hybrid idea under test: DL supplies WHERE (tip + lance axis, which it does
reliably: tip error 3-5 px), classical CV supplies HOW WIDE (the cone
boundary), because a learned regressor memorises the three angles present
in the data (plan section 3) while a geometric measurement cannot.

Method (stateless, no temporal filtering, so it can be scored like a model):
  1. sector ROI around the spray axis from the tip (radius R, +-80 deg)
  2. grayscale -> CLAHE -> blur -> subtract a large-window local median
     (removes the diffuse mist from other nozzles)
  3. polar energy: for each ray angle, mean intensity over r in [r_min, R]
  4. cone = contiguous band around the axis whose energy stays above
     `frac` of (peak - floor); boundaries = its two ends; apex = width
  5. angle is measured in the ISOTROPIC pixel plane (same as the metric)

Inputs for tip/axis: --source gt (annotated tip, axis = mean edge direction)
or --source dl --ckpt ... (tip-matched DL prediction: tip + axis from the
DL edge keypoints; DL edges are used only for direction, never for width).

Probes (same rule as eval_counterfactual): stretch_y and shear about the tip
must give slope ~ 1 if this is a measurement.

Output: results/cv_cone_<split>_<source>.json, overlays in results/cv_debug/
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from common import RESULTS_DIR, load_model  # also puts ../structure on sys.path
from dataset import AFGNetDataset
from metrics import compute_spray_angle, decode_predictions, tip_match

from config import ANGLE_MATCH_OBJ_THRESH, DEVICE, ORIG_IMG_H, ORIG_IMG_W, SPRAY_IDX, TEST_JSON, VAL_JSON

W, H = ORIG_IMG_W, ORIG_IMG_H
STRETCH = [0.8, 0.9, 1.0, 1.1, 1.25]
SHEAR_DEG = [-15, -8, 0, 8, 15]
MIN_DGT = 2.0


# ── the CV measurement ───────────────────────────────────────────────────
def cone_angle_cv(gray, tip, axis_deg, R=220, r_min=30, half_span=90, frac=0.15, step=1.0, bg_win=151, run=3):
    """Apex angle (deg) from the polar energy profile around `axis_deg`.

    The annotated cone edges are the OUTERMOST visible streaks (overlays in
    results/cv_debug), so the boundary is found by scanning from the outside
    of the sector inwards until the ray energy exceeds floor + frac*(peak-floor)
    for `run` consecutive rays. The floor is the background energy at the two
    ends of the sector (outside the fan); r_min skips the bright rim of the
    nozzle head. Returns (angle, left_deg, right_deg, profile) or None.
    """
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    g = cv2.GaussianBlur(g, (5, 5), 0).astype(np.float32)
    bg = cv2.medianBlur(g.astype(np.uint8), bg_win).astype(np.float32)
    g = np.clip(g - bg, 0, None)

    thetas = np.arange(axis_deg - half_span, axis_deg + half_span + 1e-6, step)
    rs = np.arange(r_min, R, 2.0)
    cos, sin = np.cos(np.radians(thetas)), np.sin(np.radians(thetas))
    xs = tip[0] + np.outer(cos, rs)
    ys = tip[1] + np.outer(sin, rs)
    valid = (xs >= 0) & (xs < W - 1) & (ys >= 0) & (ys < H - 1)
    if valid.mean() < 0.5:
        return None
    samp = cv2.remap(g, xs.astype(np.float32), ys.astype(np.float32), cv2.INTER_LINEAR, borderValue=0)
    energy = np.where(valid, samp, np.nan)
    prof = np.nanmean(energy, axis=1)
    prof = np.convolve(np.nan_to_num(prof), np.ones(5) / 5, mode="same")

    n = len(prof)
    edge = max(5, n // 12)
    floor = float(min(np.median(prof[:edge]), np.median(prof[-edge:])))
    peak = float(prof.max())
    if peak - floor < 3:
        return None
    thr = floor + frac * (peak - floor)
    above = prof >= thr
    # scan inwards from each end: first `run` consecutive rays above thr
    i = 0
    while i <= n - run and not above[i : i + run].all():
        i += 1
    j = n - 1
    while j >= run - 1 and not above[j - run + 1 : j + 1].all():
        j -= 1
    if i >= j:
        return None
    left, right = thetas[i], thetas[j]
    return float(right - left), float(left), float(right), prof


def to_iso(p):
    """normalised (x/W, y/H) -> isotropic pixel plane"""
    return np.array([p[0] * W, p[1] * H], float)


def axis_from_kpts(kp6):
    """spray axis (deg, isotropic plane) = bisector of the two annotated edges"""
    k = np.array(kp6, float).reshape(3, 2) * [W, H]
    v = (k[1] - k[0]) + (k[2] - k[0])
    return float(np.degrees(np.arctan2(v[1], v[0]))), k


FRAC = 0.15
AXIS_MODE = "kpts"  # or "box": axis = lance direction (box centre -> tip); does not use learned edges


def measure(gray, kp6, box=None):
    ang, k = axis_from_kpts(kp6)
    if AXIS_MODE == "box" and box is not None:
        c = np.array([box[0] * W, box[1] * H])
        v = k[0] - c
        if np.linalg.norm(v) > 1:
            ang = float(np.degrees(np.arctan2(v[1], v[0])))
    r = cone_angle_cv(gray, k[0], ang, frac=FRAC)
    if r is None:
        return None
    apex, left, right, _ = r
    # how far are the annotated edge directions from the detected boundaries (deg)?
    e1 = np.degrees(np.arctan2(*(k[1] - k[0])[::-1]))
    e2 = np.degrees(np.arctan2(*(k[2] - k[0])[::-1]))
    d1 = min(abs((e1 - left + 180) % 360 - 180), abs((e1 - right + 180) % 360 - 180))
    d2 = min(abs((e2 - left + 180) % 360 - 180), abs((e2 - right + 180) % 360 - 180))
    return {"apex": apex, "left": left, "right": right, "edge_dist_deg": [d1, d2]}


def slope(pairs):
    a = np.array([p for p in pairs if abs(p[0]) >= MIN_DGT], float)
    return float((a[:, 0] * a[:, 1]).sum() / (a[:, 0] ** 2).sum()) if a.size else float("nan")


def warp_kp6(kp6, M):
    k = np.array(kp6, float).reshape(3, 2) * [W, H]
    k2 = np.hstack([k, np.ones((3, 1))]) @ M.T
    return (k2 / [W, H]).ravel().tolist()


def nozzle_key(b):
    return f"({b[0]:.1f},{b[1]:.1f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--source", choices=["gt", "dl"], default="gt", help="where tip+axis come from")
    ap.add_argument("--ckpt", default=None, help="DL checkpoint for --source dl")
    ap.add_argument("--debug", type=int, default=6, help="number of overlay images to write")
    ap.add_argument("--frac", type=float, default=0.15, help="boundary threshold as fraction of (peak - floor)")
    ap.add_argument("--axis", choices=["kpts", "box"], default="kpts", help="spray axis from edge keypoints or from lance box centre -> tip")
    args = ap.parse_args()
    global FRAC, AXIS_MODE
    FRAC, AXIS_MODE = args.frac, args.axis

    ds = AFGNetDataset(VAL_JSON if args.split == "val" else TEST_JSON, augment=False)
    model = load_model(Path(args.ckpt), DEVICE) if args.source == "dl" else None
    dbg_dir = RESULTS_DIR / "cv_debug"
    dbg_dir.mkdir(exist_ok=True)

    rows = []  # per instance: nozzle, gt_angle, cv_angle, edge_dist, probes
    n_written = 0
    for s in ds.samples:
        sprays = [(b, k) for b, c, k in zip(s["gt_boxes"], s["gt_classes"], s["gt_kpts"]) if c == SPRAY_IDX and any(v > 0 for v in k)]
        if not sprays:
            continue
        bgr = cv2.imread(s["image_path"])
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        dl_preds = None
        if model is not None:
            import torch

            x = ds.make_input(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = model(x)
            dl_preds = [p for p in decode_predictions(out, batch_idx=0, obj_thresh=ANGLE_MATCH_OBJ_THRESH) if p["class_id"] == SPRAY_IDX]

        for b, k in sprays:
            gt_angle = compute_spray_angle(k)
            src_k = k
            box_c = (b[0], b[1])  # normalised lance-box centre (GT)
            if dl_preds is not None:
                m = tip_match(dl_preds, k)
                if m is None:
                    continue
                src_k = m["keypoints"]  # DL tip + DL edges (edges only give the axis in --axis kpts)
                x1, y1, x2, y2 = m["box"]
                box_c = ((x1 + x2) / 2, (y1 + y2) / 2)  # DL lance box centre
            r = measure(gray, src_k, box_c)
            if r is None:
                rows.append({"nozzle": nozzle_key(b), "gt": gt_angle, "cv": None})
                continue
            row = {"nozzle": nozzle_key(b), "gt": gt_angle, "cv": r["apex"], "edge_dist": r["edge_dist_deg"], "probes": {}}

            # probes: transform image + tip/axis identically, re-measure
            k_iso = np.array(src_k, float).reshape(3, 2) * [W, H]
            tip = k_iso[0]
            for name, params, mk in (
                ("stretch_y", STRETCH, lambda sc: np.array([[1, 0, 0], [0, sc, (1 - sc) * tip[1]]], float)),
                ("shear", SHEAR_DEG, lambda a: np.array([[1, np.tan(np.radians(a)), -np.tan(np.radians(a)) * tip[1]], [0, 1, 0]], float)),
            ):
                ident = 1.0 if name == "stretch_y" else 0
                per = {}
                for p in params:
                    M = mk(p)
                    g2 = cv2.warpAffine(gray, M, (W, H), borderValue=0)
                    k2 = warp_kp6(src_k, M)
                    bc = np.array([box_c[0] * W, box_c[1] * H, 1.0]) @ M.T
                    r2 = measure(g2, k2, (bc[0] / W, bc[1] / H))
                    per[p] = (compute_spray_angle(warp_kp6(k, M)), r2["apex"] if r2 else None)
                if per[ident][1] is not None:
                    g1, c1 = per[ident]
                    row["probes"][name] = [(g - g1, c - c1) for p, (g, c) in per.items() if p != ident and c is not None]
            rows.append(row)

            if n_written < args.debug:
                vis = bgr.copy()
                for ang_deg, col in ((r["left"], (0, 0, 255)), (r["right"], (0, 0, 255))):
                    end = (tip + 220 * np.array([np.cos(np.radians(ang_deg)), np.sin(np.radians(ang_deg))])).astype(int)
                    cv2.line(vis, tuple(tip.astype(int)), tuple(end), col, 2)
                kg = np.array(k, float).reshape(3, 2) * [W, H]
                cv2.line(vis, tuple(kg[0].astype(int)), tuple(kg[1].astype(int)), (0, 255, 0), 2)
                cv2.line(vis, tuple(kg[0].astype(int)), tuple(kg[2].astype(int)), (0, 255, 0), 2)
                x0, y0 = int(max(0, tip[0] - 300)), int(max(0, tip[1] - 300))
                crop = vis[y0 : y0 + 600, x0 : x0 + 600]
                cv2.putText(crop, f"GT {gt_angle:.1f}  CV {r['apex']:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.imwrite(str(dbg_dir / f"{args.split}_{args.source}_{n_written:02d}_{nozzle_key(b)}.jpg"), crop)
                n_written += 1

    ok = [r for r in rows if r.get("cv") is not None]
    err = np.array([abs(r["cv"] - r["gt"]) for r in ok])
    bias = np.array([r["cv"] - r["gt"] for r in ok])
    res = {
        "split": args.split, "source": args.source, "ckpt": args.ckpt,
        "N_gt": len(rows), "N_measured": len(ok),
        "MAE_raw_deg": float(err.mean()) if err.size else None,
        "bias_deg": float(bias.mean()) if bias.size else None,
        "MAE_after_global_bias_deg": float(np.abs(bias - bias.mean()).mean()) if bias.size else None,
        "edge_dist_deg_mean": float(np.mean([d for r in ok for d in r["edge_dist"]])) if ok else None,
        "per_nozzle": {},
        "probes": {},
    }
    by = defaultdict(list)
    for r in ok:
        by[r["nozzle"]].append(r)
    for kkey, rs in by.items():
        g = np.array([r["gt"] for r in rs])
        c = np.array([r["cv"] for r in rs])
        res["per_nozzle"][kkey] = {
            "N": len(rs), "gt_mean": float(g.mean()), "cv_mean": float(c.mean()),
            "MAE": float(np.abs(c - g).mean()), "cv_std": float(c.std()), "gt_std": float(g.std()),
            "track_r": float(np.corrcoef(g, c)[0, 1]) if g.std() > 0 and c.std() > 0 else None,
        }
    for name in ("stretch_y", "shear"):
        pairs = [p for r in ok for p in r.get("probes", {}).get(name, [])]
        res["probes"][name] = {"slope": slope(pairs), "N": len(pairs)}

    out = RESULTS_DIR / f"cv_cone_{args.split}_{args.source}.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nCV cone angle, {args.split}, tip/axis from {args.source}: measured {res['N_measured']}/{res['N_gt']}")
    print(f"  MAE raw {res['MAE_raw_deg']:.2f} deg | bias {res['bias_deg']:+.2f} | MAE after global bias {res['MAE_after_global_bias_deg']:.2f} | annotated edges vs CV boundary {res['edge_dist_deg_mean']:.2f} deg")
    for kkey, d in res["per_nozzle"].items():
        tr = f"{d['track_r']:.2f}" if d["track_r"] is not None else "n/a"
        print(f"  {kkey}: N={d['N']:2d} GT {d['gt_mean']:6.1f}+-{d['gt_std']:.2f}  CV {d['cv_mean']:6.1f}+-{d['cv_std']:.2f}  MAE {d['MAE']:.2f}  track r {tr}")
    for name, d in res["probes"].items():
        print(f"  probe {name:<9} slope {d['slope']:.2f} (N={d['N']})   [measurement -> ~1]")
    print(f"  overlays: {dbg_dir}\nWritten: {out}")


if __name__ == "__main__":
    main()
