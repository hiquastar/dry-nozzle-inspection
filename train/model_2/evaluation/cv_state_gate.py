"""
cv_state_gate.py - "is this small nozzle spraying?" decided by classical CV
from a mist disk at the lance tip, instead of by the network's class head.

Why (plan section 2G): the network separates facing from no_spray by absolute
brightness, which does not transfer between cameras (AUC 0.62 on CAM1), and
loses the class entirely when brightness is augmented away. A human sees the
state at once: a bright mist blob sits AT THE TIP of a spraying lance. The
earlier CV probes failed only because they looked around the whole lance box.

Feature: mean local-excess brightness (frame minus a wide Gaussian
background, so ambient mist and exposure cancel) inside a disk of radius
0.45*max(box side) centred at the box end nearest the image centre (lances
enter from the border, so that end is the tip).

Rule: mist > threshold -> spraying (facing, or spray if a cone is visible),
else no_spray. Threshold chosen on TRAIN by balanced accuracy; reported on VAL.

Modes:
  --boxes gt   feature on annotated lance boxes (is the cue there?)
  --boxes dl   feature on the network's boxes (end-to-end hybrid):
               DL says spray -> spray; otherwise gate decides facing/no_spray

Output: results/cv_state_gate_<boxes>.json
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from common import RESULTS_DIR, load_model  # also puts ../structure on sys.path
from dataset import AFGNetDataset, load_frame
from metrics import box_iou, cxcywh_to_xyxy, decode_predictions

from config import CLASS_NAMES, DEVICE, FACING_IDX, NO_SPRAY_IDX, SPRAY_IDX, TRAIN_JSON, VAL_JSON

SMALL = (SPRAY_IDX, NO_SPRAY_IDX, FACING_IDX)
BG_SIGMA = 40  # on the 640x480 frame
DISK = 0.45


def mist_at_tip(gray, box_xyxy):
    """Local-excess brightness in a disk at the lance tip. gray: 480x640 uint8."""
    H, W = gray.shape
    gf = gray.astype(np.float32)
    bg = cv2.GaussianBlur(gf, (0, 0), BG_SIGMA)
    hp = np.clip(gf - bg, 0, None)
    x1, y1, x2, y2 = box_xyxy[0] * W, box_xyxy[1] * H, box_xyxy[2] * W, box_xyxy[3] * H
    corners = np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]])
    centre = np.array([W / 2, H / 2])
    tip = corners[np.argsort(np.linalg.norm(corners - centre, axis=1))[:2]].mean(0)
    r = DISK * max(x2 - x1, y2 - y1)
    yy, xx = np.mgrid[0:H, 0:W]
    m = np.hypot(xx - tip[0], yy - tip[1]) < r
    return float(hp[m].mean()) if m.sum() > 10 else 0.0


def gt_features(ds):
    rows = []
    for s in ds.samples:
        bc = [(b, c) for b, c in zip(s["gt_boxes"], s["gt_classes"]) if c in (NO_SPRAY_IDX, FACING_IDX)]
        if not bc:
            continue
        g = cv2.cvtColor(load_frame(s["image_path"]), cv2.COLOR_RGB2GRAY)
        for b, c in bc:
            rows.append((c, mist_at_tip(g, cxcywh_to_xyxy(b))))
    return rows


def pick_threshold(rows):
    y = np.array([c == FACING_IDX for c, _ in rows])
    x = np.array([f for _, f in rows])
    best = (0, None)
    for t in np.unique(np.round(x, 2)):
        pred = x > t
        bacc = 0.5 * ((pred & y).sum() / max(y.sum(), 1) + (~pred & ~y).sum() / max((~y).sum(), 1))
        if bacc > best[0]:
            best = (bacc, float(t))
    return best[1], best[0]


def prf(tp, fp, fn):
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boxes", choices=["gt", "dl"], default="gt")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--split", choices=["val", "test"], default="val", help="threshold always comes from train; test only once, for the final model")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    from config import TEST_JSON

    train_ds = AFGNetDataset(TRAIN_JSON, augment=False)
    val_ds = AFGNetDataset(VAL_JSON if args.split == "val" else TEST_JSON, augment=False)

    thr, bacc = pick_threshold(gt_features(train_ds))
    res = {"threshold_from_train": thr, "train_balanced_acc": bacc, "boxes": args.boxes}
    print(f"threshold (train, balanced acc {bacc:.3f}): mist > {thr:.2f} -> spraying")

    if args.boxes == "gt":
        rows = gt_features(val_ds)
        y = np.array([c == FACING_IDX for c, _ in rows])
        pred = np.array([f > thr for _, f in rows])
        tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
        tn = int((~pred & ~y).sum())
        p, r, f = prf(tp, fp, fn)
        p0, r0, f0 = prf(tn, fn, fp)
        res["val"] = {"facing": {"P": p, "R": r, "F1": f}, "no_spray": {"P": p0, "R": r0, "F1": f0}, "n": len(rows), "accuracy": (tp + tn) / len(rows)}
        print(f"VAL (GT boxes, n={len(rows)}): facing P {p:.3f} R {r:.3f} F1 {f:.3f} | no_spray P {p0:.3f} R {r0:.3f} F1 {f0:.3f} | acc {(tp+tn)/len(rows):.3f}")
    else:
        model = load_model(Path(args.ckpt), DEVICE)
        tp = np.zeros(4, int); fp = np.zeros(4, int); fn = np.zeros(4, int)
        tp_dl = np.zeros(4, int); fp_dl = np.zeros(4, int); fn_dl = np.zeros(4, int)
        with torch.no_grad():
            for s in val_ds.samples:
                img, prev = val_ds.frames_for(s)
                out = model(val_ds.make_input(img, prev).unsqueeze(0).to(DEVICE))
                dets = [d for d in decode_predictions(out, batch_idx=0) if d["class_id"] in SMALL]
                g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                # hybrid relabel: DL spray stays spray; other small-class boxes -> gate
                for d in dets:
                    d["hybrid_class"] = d["class_id"] if d["class_id"] == SPRAY_IDX else (FACING_IDX if mist_at_tip(g, d["box"]) > thr else NO_SPRAY_IDX)
                # class-aware greedy matching for both label sets
                for key, tpa, fpa, fna in (("class_id", tp_dl, fp_dl, fn_dl), ("hybrid_class", tp, fp, fn)):
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
        res["val"] = {}
        print("VAL, DL boxes  (per class F1: network head  ->  hybrid CV gate)")
        for c in SMALL:
            f_dl = prf(tp_dl[c], fp_dl[c], fn_dl[c])[2]; f_h = prf(tp[c], fp[c], fn[c])[2]
            res["val"][CLASS_NAMES[c]] = {"F1_network": f_dl, "F1_hybrid": f_h}
            print(f"  {CLASS_NAMES[c].replace('nozzle_', ''):<10} {f_dl:.3f}  ->  {f_h:.3f}")
        m_dl = np.mean([res["val"][CLASS_NAMES[c]]["F1_network"] for c in SMALL]); m_h = np.mean([res["val"][CLASS_NAMES[c]]["F1_hybrid"] for c in SMALL])
        res["val"]["macro_small3"] = {"network": m_dl, "hybrid": m_h}
        print(f"  macro(3)   {m_dl:.3f}  ->  {m_h:.3f}")

    tag = f"{args.split}_{args.boxes}" + (f"_{args.label}" if args.label else "")
    res["split"] = args.split
    out = RESULTS_DIR / f"cv_state_gate_{tag}.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"Written: {out}")


if __name__ == "__main__":
    main()
