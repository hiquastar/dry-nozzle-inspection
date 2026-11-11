"""
eval_mist_hybrid.py - score a MIST_HEAD model where everything about the
spray is DERIVED from the predicted mist map:

  state   mean mist probability in a disk at the lance tip (same geometry as
          cv_state_gate.py) > 0.5 -> spraying; DL class 'spray' kept as spray,
          everything else spraying -> facing, not spraying -> no_spray
  angle   from the DL tip + axis, polar scan of the mist map: boundary = the
          outermost rays whose mean probability exceeds MASK_FRAC; apex = width
  probes  stretch_y and shear about the tip (image warped, map re-predicted):
          a map that follows the image gives slope ~ 1; a memorised angle ~ 0
  map     quality of the map itself vs the pseudo mask inside the ROI (AUC)

    AFGNET_INSTANCE_NORM=0 AFGNET_TIP_RELATIVE_EDGES=1 AFGNET_MIST_HEAD=1 \
      python evaluation/eval_mist_hybrid.py --split val --ckpt checkpoint/abl_mist_s42/afgnet_v10.pth

Output: results/mist_hybrid_<split>_<tag>.json, overlays in results/cv_debug/mist_pred_*.jpg
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from common import RESULTS_DIR, load_model  # also puts ../structure on sys.path
from dataset import AFGNetDataset, load_frame, mist_path_for, mist_targets
from metrics import box_iou, compute_spray_angle, cxcywh_to_xyxy, decode_predictions, tip_match

from config import ANGLE_ASPECT as ANGLE_ASPECT_
from config import ANGLE_MATCH_OBJ_THRESH, CLASS_NAMES, DEVICE, FACING_IDX, IMG_H, IMG_W, LEARNED_HEADS, NO_SPRAY_IDX, SPRAY_IDX, TEST_JSON, VAL_JSON

SMALL = (SPRAY_IDX, NO_SPRAY_IDX, FACING_IDX)
STRETCH = [0.8, 0.9, 1.0, 1.1, 1.25]
SHEAR_DEG = [-15, -8, 0, 8, 15]
MIN_DGT = 2.0
MASK_FRAC = 0.3  # ray counts as inside the cone if its mean mist prob exceeds this
TIP_DISK = 0.45
STATE_THR = 0.5


def predict_map(model, ds, img_rgb):
    """mist probability map at frame resolution (480x640) + decoded detections"""
    x = ds.make_input(img_rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(x)
    prob = torch.sigmoid(out["mist_logits"])[0, 0].cpu().numpy()
    prob = cv2.resize(prob, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
    return prob, out


def tip_disk_mean(prob, box_xyxy):
    H, W = prob.shape
    x1, y1, x2, y2 = box_xyxy[0] * W, box_xyxy[1] * H, box_xyxy[2] * W, box_xyxy[3] * H
    corners = np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]])
    tip = corners[np.argsort(np.linalg.norm(corners - [W / 2, H / 2], axis=1))[:2]].mean(0)
    r = TIP_DISK * max(x2 - x1, y2 - y1)
    yy, xx = np.mgrid[0:H, 0:W]
    m = np.hypot(xx - tip[0], yy - tip[1]) < r
    return float(prob[m].mean()) if m.sum() > 10 else 0.0


def cone_from_map(prob, kp6, R=70, r_min=10, half_span=90, step=1.0, run=3):
    """Apex angle from the mist map. kp6 normalised; work at map resolution
    (480x640) but measure the angle in the isotropic plane: x scaled by W/H."""
    H, W = prob.shape
    k = np.array(kp6, float).reshape(3, 2) * [W, H]
    tip = k[0]
    v = (k[1] - tip) + (k[2] - tip)
    axis = np.degrees(np.arctan2(v[1], v[0]))
    thetas = np.arange(axis - half_span, axis + half_span + 1e-6, step)
    rs = np.arange(r_min, R, 1.0)
    xs = tip[0] + np.outer(np.cos(np.radians(thetas)), rs)
    ys = tip[1] + np.outer(np.sin(np.radians(thetas)), rs)
    valid = (xs >= 0) & (xs < W - 1) & (ys >= 0) & (ys < H - 1)
    samp = cv2.remap(prob.astype(np.float32), xs.astype(np.float32), ys.astype(np.float32), cv2.INTER_LINEAR, borderValue=0)
    prof = np.where(valid, samp, np.nan)
    prof = np.convolve(np.nan_to_num(np.nanmean(prof, axis=1)), np.ones(5) / 5, mode="same")
    above = prof >= MASK_FRAC
    n = len(prof)
    i = 0
    while i <= n - run and not above[i : i + run].all():
        i += 1
    j = n - 1
    while j >= run - 1 and not above[j - run + 1 : j + 1].all():
        j -= 1
    if i >= j:
        return None
    # convert the two boundary directions (map plane) to the isotropic plane
    def iso(theta):
        d = np.array([np.cos(np.radians(theta)) * (W / H), np.sin(np.radians(theta))])  # x scaled by W/H relative to y
        return np.degrees(np.arctan2(d[1], d[0]))
    a, b = iso(thetas[i]), iso(thetas[j])
    return float(abs((b - a + 180) % 360 - 180))


def slope(pairs):
    a = np.array([p for p in pairs if abs(p[0]) >= MIN_DGT], float)
    return float((a[:, 0] * a[:, 1]).sum() / (a[:, 0] ** 2).sum()) if a.size else float("nan")


# ── step 7: learned readers, driven by the model's OWN tips at inference ──
def box_tip_norm(box_xyxy):
    x1, y1, x2, y2 = box_xyxy
    corners = np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]])
    d = np.linalg.norm((corners - [0.5, 0.5]) * [IMG_W, IMG_H], axis=1)
    return corners[np.argsort(d)[:2]].mean(0)


def head_state(model, out, box_xyxy):
    """spraying probability from the learned state head, tip = inner end of the DL box"""
    tip = torch.tensor(box_tip_norm(box_xyxy), dtype=torch.float32, device=DEVICE).view(1, 1, 2)
    return float(model.read_state(out["mist_logits"], tip)[0, 0])


def head_angle(model, out, kp6):
    """apex angle (deg) from the learned angle head, tip + axis from the DL keypoints"""
    k = np.array(kp6, float).reshape(3, 2)
    iso = (k - k[0]) * [ANGLE_ASPECT_, 1.0]
    v = iso[1] + iso[2]
    ax = float(np.degrees(np.arctan2(v[1], v[0])))
    tip = torch.tensor(k[0], dtype=torch.float32, device=DEVICE).view(1, 1, 2)
    axis = torch.tensor([[ax]], dtype=torch.float32, device=DEVICE)
    return float(np.degrees(model.read_angle(out["mist_logits"], tip, axis)[0, 0].item()))


def warp_kp6(kp6, M, W, H):
    k = np.array(kp6, float).reshape(3, 2) * [W, H]
    return ((np.hstack([k, np.ones((3, 1))]) @ M.T) / [W, H]).ravel().tolist()


def prf(tp, fp, fn):
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--debug", type=int, default=4)
    ap.add_argument("--calib_frames", type=int, default=600, help="train frames used to set the map-gate threshold (balanced accuracy on GT small boxes)")
    args = ap.parse_args()
    tag = args.label or Path(args.ckpt).parent.name
    ds = AFGNetDataset(VAL_JSON if args.split == "val" else TEST_JSON, augment=False)
    model = load_model(Path(args.ckpt), DEVICE)
    dbg = RESULTS_DIR / "cv_debug"
    dbg.mkdir(exist_ok=True)

    # ── calibrate the map gate on TRAIN (the fixed 0.5 was wrong: the tip disk
    #    straddles lance metal (map ~0) and mist, so its mean sits well below 0.5)
    global STATE_THR
    from config import TRAIN_JSON

    tr = AFGNetDataset(TRAIN_JSON, augment=False)
    cal = [s for s in tr.samples if any(c in (NO_SPRAY_IDX, FACING_IDX) for c in s["gt_classes"])]
    step = max(1, len(cal) // args.calib_frames)
    feats, ys = [], []
    for s in cal[::step]:
        prob, _ = predict_map(model, tr, load_frame(s["image_path"]))
        for b, c in zip(s["gt_boxes"], s["gt_classes"]):
            if c in (NO_SPRAY_IDX, FACING_IDX):
                feats.append(tip_disk_mean(prob, cxcywh_to_xyxy(b))); ys.append(c == FACING_IDX)
    feats, ys = np.array(feats), np.array(ys)
    best = (0.0, 0.5)
    for t in np.unique(np.round(feats, 3)):
        pred = feats > t
        bacc = 0.5 * ((pred & ys).sum() / max(ys.sum(), 1) + (~pred & ~ys).sum() / max((~ys).sum(), 1))
        if bacc > best[0]:
            best = (bacc, float(t))
    STATE_THR = best[1]
    calib_auc = roc_auc_score(ys, feats) if ys.min() != ys.max() else float("nan")
    print(f"map gate calibrated on {len(feats)} train boxes: threshold {STATE_THR:.3f} (balanced acc {best[0]:.3f}, AUC {calib_auc:.3f})")

    # val separability of the same feature on GT boxes (is the cue in the map?)
    vf, vy = [], []
    for s in ds.samples:
        if not any(c in (NO_SPRAY_IDX, FACING_IDX) for c in s["gt_classes"]):
            continue
        prob, _ = predict_map(model, ds, load_frame(s["image_path"]))
        for b, c in zip(s["gt_boxes"], s["gt_classes"]):
            if c in (NO_SPRAY_IDX, FACING_IDX):
                vf.append(tip_disk_mean(prob, cxcywh_to_xyxy(b))); vy.append(c == FACING_IDX)
    vf, vy = np.array(vf), np.array(vy)
    val_auc = roc_auc_score(vy, vf) if len(vy) and vy.min() != vy.max() else float("nan")
    print(f"map-at-tip feature on {args.split} GT boxes: AUC {val_auc:.3f}  (no_spray mean {vf[~vy].mean():.3f}, facing mean {vf[vy].mean():.3f})")

    tp = np.zeros(4, int); fp = np.zeros(4, int); fn = np.zeros(4, int)
    tp_dl = np.zeros(4, int); fp_dl = np.zeros(4, int); fn_dl = np.zeros(4, int)
    tp_h = np.zeros(4, int); fp_h = np.zeros(4, int); fn_h = np.zeros(4, int)  # learned state head
    map_y, map_p = [], []
    angle_err = []
    angle_err_head = []
    probes = {"stretch_y": [], "shear": []}
    probes_head = {"stretch_y": [], "shear": []}
    use_heads = LEARNED_HEADS and getattr(model, "state_head", None) is not None
    n_dbg = 0
    W, H = IMG_W, IMG_H

    for s in ds.samples:
        img = load_frame(s["image_path"])
        prob, out = predict_map(model, ds, img)

        # map quality vs pseudo mask inside ROI
        pm = cv2.imread(str(mist_path_for(s["image_path"])), cv2.IMREAD_GRAYSCALE)
        if pm is not None:
            tgt, roi = mist_targets(pm, s["gt_boxes"], s["gt_classes"])
            small = cv2.resize(prob, (tgt.shape[1], tgt.shape[0]), interpolation=cv2.INTER_AREA)
            sel = roi.numpy() > 0
            if sel.sum() > 0:
                map_y.append((tgt.numpy()[sel] > 0.5)); map_p.append(small[sel])

        # state: DL boxes + mist-at-tip
        dets = [d for d in decode_predictions(out, batch_idx=0) if d["class_id"] in SMALL]
        for d in dets:
            d["hybrid_class"] = SPRAY_IDX if d["class_id"] == SPRAY_IDX else (FACING_IDX if tip_disk_mean(prob, d["box"]) > STATE_THR else NO_SPRAY_IDX)
            if use_heads:
                d["head_class"] = SPRAY_IDX if d["class_id"] == SPRAY_IDX else (FACING_IDX if head_state(model, out, d["box"]) > 0.5 else NO_SPRAY_IDX)
        for key, tpa, fpa, fna in (("class_id", tp_dl, fp_dl, fn_dl), ("hybrid_class", tp, fp, fn)) + ((("head_class", tp_h, fp_h, fn_h),) if use_heads else ()):
            used = set()
            for d in sorted(dets, key=lambda d: -d["score"]):
                best, bi = 0.5, None
                for gi, (b, c) in enumerate(zip(s["gt_boxes"], s["gt_classes"])):
                    if gi in used or c != d[key] or c not in SMALL:
                        continue
                    iou = box_iou(d["box"], cxcywh_to_xyxy(b))
                    if iou > best:
                        best, bi = iou, gi
                if bi is None:
                    fpa[d[key]] += 1
                else:
                    used.add(bi); tpa[d[key]] += 1
            for gi, c in enumerate(s["gt_classes"]):
                if c in SMALL and gi not in used:
                    fna[c] += 1

        # angle from the map (+ probes) for GT sprays, tip from DL
        sprays = [(b, k) for b, c, k in zip(s["gt_boxes"], s["gt_classes"], s["gt_kpts"]) if c == SPRAY_IDX and any(v > 0 for v in k)]
        if sprays:
            low = [p for p in decode_predictions(out, batch_idx=0, obj_thresh=ANGLE_MATCH_OBJ_THRESH) if p["class_id"] == SPRAY_IDX]
            for b, k in sprays:
                m = tip_match(low, k)
                if m is None:
                    continue
                a = cone_from_map(prob, m["keypoints"])
                ga = compute_spray_angle(k)
                if a is None or ga is None:
                    continue
                angle_err.append(abs(a - ga))
                if use_heads:
                    angle_err_head.append(abs(head_angle(model, out, m["keypoints"]) - ga))
                tip = np.array(m["keypoints"][:2]) * [W, H]
                for name, params, mk in (
                    ("stretch_y", STRETCH, lambda sc: np.array([[1, 0, 0], [0, sc, (1 - sc) * tip[1]]], float)),
                    ("shear", SHEAR_DEG, lambda ang: np.array([[1, np.tan(np.radians(ang)), -np.tan(np.radians(ang)) * tip[1]], [0, 1, 0]], float)),
                ):
                    ident = 1.0 if name == "stretch_y" else 0
                    per, per_h = {}, {}
                    for p in params:
                        M = mk(p)
                        im2 = cv2.warpAffine(img, M, (W, H), borderValue=(0, 0, 0))
                        prob2, out2 = predict_map(model, ds, im2)
                        k2 = warp_kp6(m["keypoints"], M, W, H)
                        a2 = cone_from_map(prob2, k2)
                        g2 = compute_spray_angle(warp_kp6(k, M, W, H))
                        per[p] = (g2, a2)
                        if use_heads:
                            per_h[p] = (g2, head_angle(model, out2, k2))
                    if per[ident][1] is not None:
                        g1, c1 = per[ident]
                        probes[name] += [(g - g1, c - c1) for p, (g, c) in per.items() if p != ident and c is not None]
                    if use_heads:
                        g1, c1 = per_h[ident]
                        probes_head[name] += [(g - g1, c - c1) for p, (g, c) in per_h.items() if p != ident]
                if n_dbg < args.debug:
                    vis = img.copy()
                    vis[..., 0] = np.maximum(vis[..., 0], (prob * 255).astype(np.uint8))
                    kg = np.array(k).reshape(3, 2) * [W, H]
                    cv2.line(vis, tuple(kg[0].astype(int)), tuple(kg[1].astype(int)), (0, 255, 0), 1)
                    cv2.line(vis, tuple(kg[0].astype(int)), tuple(kg[2].astype(int)), (0, 255, 0), 1)
                    cv2.putText(vis, f"GT {ga:.1f}  map {a:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.imwrite(str(dbg / f"mist_pred_{args.split}_{n_dbg:02d}.jpg"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                    n_dbg += 1

    res = {"checkpoint": args.ckpt, "split": args.split, "state": {"gate_threshold_from_train": STATE_THR, "gate_calib_auc_train": float(calib_auc), "gate_feature_auc_split": float(val_auc)}, "angle": {}, "probes": {}, "map": {}}
    print(f"\n{tag} on {args.split}")
    print("STATE per class F1: network head -> mist-map gate")
    for c in SMALL:
        f_dl, f_h = prf(tp_dl[c], fp_dl[c], fn_dl[c])[2], prf(tp[c], fp[c], fn[c])[2]
        res["state"][CLASS_NAMES[c]] = {"F1_network": f_dl, "F1_mist": f_h}
        print(f"  {CLASS_NAMES[c].replace('nozzle_', ''):<10} {f_dl:.3f} -> {f_h:.3f}")
    res["state"]["macro_small3"] = {"network": float(np.mean([res['state'][CLASS_NAMES[c]]['F1_network'] for c in SMALL])), "mist": float(np.mean([res['state'][CLASS_NAMES[c]]['F1_mist'] for c in SMALL]))}
    print(f"  macro(3)   {res['state']['macro_small3']['network']:.3f} -> {res['state']['macro_small3']['mist']:.3f}")
    if use_heads:
        print("STATE per class F1 with the LEARNED state head (blindfolded, reads the map window at the DL tip)")
        res["state"]["learned_head"] = {}
        for c in SMALL:
            f_h = prf(tp_h[c], fp_h[c], fn_h[c])[2]
            res["state"]["learned_head"][CLASS_NAMES[c]] = f_h
            print(f"  {CLASS_NAMES[c].replace('nozzle_', ''):<10} {f_h:.3f}")
        res["state"]["learned_head"]["macro_small3"] = float(np.mean([res["state"]["learned_head"][CLASS_NAMES[c]] for c in SMALL]))
        print(f"  macro(3)   {res['state']['learned_head']['macro_small3']:.3f}")
    e = np.array(angle_err)
    res["angle"] = {"MAE_from_map_deg": float(e.mean()) if e.size else None, "N": int(e.size)}
    print(f"ANGLE from mist map (fixed decoder): MAE {res['angle']['MAE_from_map_deg']:.2f} deg on {e.size}" if e.size else "ANGLE: no measurements")
    for name, pairs in probes.items():
        res["probes"][name] = {"slope": slope(pairs), "N": len(pairs)}
        print(f"  probe {name:<9} slope {res['probes'][name]['slope']:.2f} (N={len(pairs)})   [map follows the image -> ~1]")
    if use_heads:
        eh = np.array(angle_err_head)
        res["angle"]["MAE_learned_head_deg"] = float(eh.mean()) if eh.size else None
        print(f"ANGLE from the LEARNED angle head: MAE {res['angle']['MAE_learned_head_deg']:.2f} deg on {eh.size}" if eh.size else "ANGLE head: no measurements")
        res["probes_learned_head"] = {}
        for name, pairs in probes_head.items():
            res["probes_learned_head"][name] = {"slope": slope(pairs), "N": len(pairs)}
            print(f"  probe {name:<9} slope {res['probes_learned_head'][name]['slope']:.2f} (N={len(pairs)})   [learned head: measures -> ~1, memorises -> ~0]")
    if map_y:
        y = np.concatenate(map_y); p = np.concatenate(map_p)
        res["map"] = {"AUC_vs_pseudo_in_roi": float(roc_auc_score(y, p)) if y.min() != y.max() else None, "n_px": int(y.size)}
        print(f"MAP vs pseudo mask (inside ROI): AUC {res['map']['AUC_vs_pseudo_in_roi']:.3f}")
    out = RESULTS_DIR / f"mist_hybrid_{args.split}_{tag}.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"Written: {out}")


if __name__ == "__main__":
    main()
