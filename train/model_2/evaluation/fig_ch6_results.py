"""
fig_ch6_results.py - the result figures of thesis Chapter 6, all from the
final model on the TEST set (the same checkpoint and the same code paths as
the reported numbers; the test set is only re-read, no choice depends on it).

  test_grid.png      one test frame per source: predicted boxes coloured by the
                     reported state (spray kept from the class head, the other
                     small lances decided by the mist gate), and on spray
                     lances the learned-head angle next to the annotated one,
                     drawn as two edges about the predicted axis
  probe_slopes.png   change of the prediction against change of the true angle
                     under the vertical stretch and the shear, for the keypoint
                     branch (eval_counterfactual.py path) and the learned angle
                     head (eval_mist_hybrid.py path); slopes are checked against
                     the saved results
  probe_lance.png    one test lance under the probes, with the true angle, the
                     keypoint-branch angle and the learned-head angle
  failures.png       the three kinds of error on test: (a) a lance at the top
                     edge of CAM3 not detected, (b) a facing lance the gate
                     reports as no_spray, (c) a big lance not detected

Frames are shown at source resolution; all predictions come from the 640x480
input. Optional arguments select figures: grid lance failures slopes.

    AFGNET_INSTANCE_NORM=0 AFGNET_TIP_RELATIVE_EDGES=1 AFGNET_MIST_HEAD=1 AFGNET_LEARNED_HEADS=1 \
      python evaluation/fig_ch6_results.py

Output: papers/thesis/gfx/{test_grid,probe_slopes,probe_lance,failures}.png
"""

import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import eval_counterfactual as CF
import eval_mist_hybrid as MH
from common import RESULTS_DIR, load_model
from dataset import AFGNetDataset, load_frame
from metrics import compute_spray_angle, cxcywh_to_xyxy, decode_predictions, tip_match
from tune_decode import decode as cf_decode

from config import ANGLE_MATCH_OBJ_THRESH, BIG_IDX, DEVICE, FACING_IDX, IMG_H, IMG_W, NO_SPRAY_IDX, REPO_ROOT, SPRAY_IDX, TEST_JSON

CKPT = REPO_ROOT / "train" / "model_2" / "checkpoint" / "abl_heads_s42" / "afgnet_v10.pth"
GFX = REPO_ROOT / "papers" / "thesis" / "gfx"
MH_JSON = RESULTS_DIR / "mist_hybrid_test_final_heads_s42.json"
CF_JSON = RESULTS_DIR / "counterfactual_test_abl_heads_s42.json"
SOURCE_LABEL = {"20250509_frames_275-304": "CAM3 V2", "20250510": "CAM3 V3", "CAM1 - V2": "CAM1 V2", "CAM1 - V3": "CAM1 V3"}
COL = {SPRAY_IDX: "#ff8c1a", FACING_IDX: "#33c3ff", NO_SPRAY_IDX: "#e8e8e8", BIG_IDX: "#c77dff"}
NAME = {SPRAY_IDX: "spray", FACING_IDX: "facing", NO_SPRAY_IDX: "no_spray", BIG_IDX: "big"}
W, H = IMG_W, IMG_H


# ── the reported system on one 640x480 frame ─────────────────────────────
def system(model, ds, img, thr):
    prob, out = MH.predict_map(model, ds, img)
    res = []
    for d in decode_predictions(out, batch_idx=0):
        c = d["class_id"]
        r = {"box": d["box"], "class_head": c, "kp": d["keypoints"]}
        if c == SPRAY_IDX:
            r["state"] = SPRAY_IDX
            r["angle_head"] = MH.head_angle(model, out, d["keypoints"])
            r["angle_kpt"] = compute_spray_angle(d["keypoints"])
        elif c == BIG_IDX:
            r["state"] = BIG_IDX
        else:
            r["state"] = FACING_IDX if MH.tip_disk_mean(prob, d["box"]) > thr else NO_SPRAY_IDX
        res.append(r)
    return prob, out, res


def head_edges(kp6, angle):
    """two cone edges at +-angle/2 about the predicted axis (the axis head_angle uses), in 640x480 pixels"""
    k = np.array(kp6, float).reshape(3, 2) * [W, H]
    v = (k[1] - k[0]) + (k[2] - k[0])  # as MH.head_angle: pixel space equals its aspect-corrected space up to scale
    ax = np.arctan2(v[1], v[0])
    L = 0.5 * (np.linalg.norm(k[1] - k[0]) + np.linalg.norm(k[2] - k[0]))
    return k[0], [k[0] + L * np.array([np.cos(ax + s * np.radians(angle) / 2), np.sin(ax + s * np.radians(angle) / 2)]) for s in (-1, 1)]


def draw(ax, img, dets, gts=None, label_fn=None, crop=None, img_hi=None):
    """img: the 640x480 frame the model saw; img_hi: the source frame, shown instead for sharpness"""
    ax.imshow(img if img_hi is None else img_hi, extent=(-0.5, W - 0.5, H - 0.5, -0.5))
    vx0, vy0, vx1, vy1 = crop if crop else (0, 0, W, H)
    for r in dets:
        x1, y1, x2, y2 = r["box"][0] * W, r["box"][1] * H, r["box"][2] * W, r["box"][3] * H
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=1.0, ec=COL[r["state"]]))
        if r["state"] == SPRAY_IDX:
            t, es = head_edges(r["kp"], r["angle_head"])
            for e in es:
                ax.plot([t[0], e[0]], [t[1], e[1]], color=COL[SPRAY_IDX], lw=1.3)
        if x2 < vx0 or x1 > vx1 or y2 < vy0 or y1 > vy1:
            continue  # box outside the view: no label
        txt = label_fn(r) if label_fn else NAME[r["state"]]
        # label above the box, or inside it when the box touches the top of the view;
        # right-aligned for boxes in the right half; always kept inside the view
        top = y1 - vy0 < 0.06 * (vy1 - vy0)
        right = (x1 + x2) / 2 > (vx0 + vx1) / 2
        m = 0.01 * (vx1 - vx0)
        tx = min(x2, vx1) - m if right else max(x1, vx0) + m
        ty = max(y1, vy0) + m if top else y1 - m
        ax.text(tx, ty, txt, color=COL[r["state"]], fontsize=6.5, clip_on=True,  # figures are drawn at about the text width
                va="top" if top else "bottom", ha="right" if right else "left",
                bbox=dict(facecolor="black", alpha=0.6, pad=1, lw=0))
    for b in gts or []:
        x1, y1, x2, y2 = cxcywh_to_xyxy(b)
        ax.add_patch(Rectangle((x1 * W, y1 * H), (x2 - x1) * W, (y2 - y1) * H, fill=False, lw=1.0, ec="#ffd23f", ls="--"))
    # fixed limits: ax.plot would otherwise autoscale with a margin and show a white border
    ax.set_xlim(vx0 - 0.5, vx1 - 0.5); ax.set_ylim(vy1 - 0.5, vy0 - 0.5)
    ax.set_xticks([]); ax.set_yticks([])


def load_hi(s):
    return cv2.cvtColor(cv2.imread(s["image_path"]), cv2.COLOR_BGR2RGB)


def gt_angle_for(s, kp6_pred):
    """annotated angle of the GT spray lance whose tip is nearest the predicted tip"""
    best, ang = 1e9, None
    for c, k in zip(s["gt_classes"], s["gt_kpts"]):
        if c == SPRAY_IDX and any(v > 0 for v in k):
            d = np.hypot((k[0] - kp6_pred[0]) * W, (k[1] - kp6_pred[1]) * H)
            if d < best:
                best, ang = d, compute_spray_angle(k)
    return ang if best < 16 else None


# ── figure 1: one test frame per source ──────────────────────────────────
def seed_model(sd):
    """the final design with seed sd, and its gate threshold calibrated on the training set"""
    model = load_model(CKPT.parent.parent / f"abl_heads_s{sd}" / CKPT.name, DEVICE)
    thr = json.load(open(RESULTS_DIR / f"mist_hybrid_test_final_heads_s{sd}.json"))["state"]["gate_threshold_from_train"]
    return model, thr


def fig_grid(ds, seeds=(42, 123)):
    """one test frame per recording (rows), the same frame for each seed (columns)"""
    by_src = {}
    for s in ds.samples:
        by_src.setdefault(s["source_video"], []).append(s)
    order = ["CAM1 - V2", "CAM1 - V3", "20250510", "20250509_frames_275-304"]
    frames = {}
    for src in order:
        fr = by_src[src]
        if src.startswith("20250509"):  # the frame with the most spray lances, middle of the window
            n = max(sum(c == SPRAY_IDX for c in f["gt_classes"]) for f in fr)
            cand = [f for f in fr if sum(c == SPRAY_IDX for c in f["gt_classes"]) == n]
            frames[src] = cand[len(cand) // 2]
        else:
            frames[src] = fr[len(fr) // 2]
    fig, axes = plt.subplots(len(order), len(seeds), figsize=(7.2, 11.0))
    for j, sd in enumerate(seeds):
        model, thr = seed_model(sd)
        for i, src in enumerate(order):
            s, ax = frames[src], axes[i, j]
            img = load_frame(s["image_path"])
            _, _, dets = system(model, ds, img, thr)

            def lab(r, s=s):
                if r["state"] != SPRAY_IDX:
                    return NAME[r["state"]]
                g = gt_angle_for(s, r["kp"])
                return f"spray {r['angle_head']:.1f}$^\\circ$" + (f" (true {g:.1f}$^\\circ$)" if g else "")

            draw(ax, img, dets, label_fn=lab, img_hi=load_hi(s))
            if i == 0:
                ax.set_title(f"seed {sd}", fontsize=9.5, pad=6)
            if j == 0:
                ax.set_ylabel(SOURCE_LABEL[src], fontsize=9)
    handles = [plt.Line2D([0], [0], color=COL[c], lw=3, label=NAME[c]) for c in (SPRAY_IDX, FACING_IDX, NO_SPRAY_IDX, BIG_IDX)]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5, frameon=False)
    fig.tight_layout(rect=(0, 0.025, 1, 1), h_pad=1.0, w_pad=0.4)
    fig.savefig(GFX / "test_grid.png", dpi=260)
    plt.close(fig)


# ── probe pairs, reproducing the two evaluation scripts ──────────────────
def pairs_keypoint(model, ds, frames):
    """eval_counterfactual.py path: source resolution, keypoints re-predicted on every probe"""
    out = {"stretch_x": [], "stretch_y": [], "shear": []}
    for s in frames:
        img = cv2.cvtColor(cv2.imread(s["image_path"]), cv2.COLOR_BGR2RGB)
        sprays = [np.array(k).reshape(3, 2) * [CF.W, CF.H] for c, k in zip(s["gt_classes"], s["gt_kpts"]) if c == SPRAY_IDX and any(v > 0 for v in k)]
        for kp in sprays:
            for probe, mats in CF.probe_matrices(kp[0]).items():
                if probe not in out:
                    continue
                ident = CF.IDENTITY_PARAM[probe]
                per = {}
                for param, M in mats:
                    im2, k2 = CF.warp(img, M, kp)
                    if not CF.in_frame(k2):
                        continue
                    r = CF.match_angle(CF.predict(model, ds, im2), k2[0])
                    per[param] = (CF.angle_px(k2), r[0] if r else None)
                if ident in per and per[ident][1] is not None:
                    g1, p1 = per[ident]
                    out[probe] += [(g - g1, p - p1) for param, (g, p) in per.items() if param != ident and p is not None]
    return out


def pairs_head(model, ds, frames):
    """eval_mist_hybrid.py path (horizontal stretch as in reader_hstretch.py): 640x480, tip and
    axis of the base prediction warped with the frame"""
    out = {"stretch_x": [], "stretch_y": [], "shear": []}
    for s in frames:
        img = load_frame(s["image_path"])
        prob, o = MH.predict_map(model, ds, img)
        low = [p for p in decode_predictions(o, batch_idx=0, obj_thresh=ANGLE_MATCH_OBJ_THRESH) if p["class_id"] == SPRAY_IDX]
        for c, k in zip(s["gt_classes"], s["gt_kpts"]):
            if c != SPRAY_IDX or not any(v > 0 for v in k):
                continue
            m = tip_match(low, k)
            # same filter as eval_mist_hybrid.py: skip when the map rule or the label gives no angle
            if m is None or MH.cone_from_map(prob, m["keypoints"]) is None or compute_spray_angle(k) is None:
                continue
            tip = np.array(m["keypoints"][:2]) * [W, H]
            for name, params, mk in (
                ("stretch_x", MH.STRETCH, lambda sc: np.array([[sc, 0, (1 - sc) * tip[0]], [0, 1, 0]], float)),
                ("stretch_y", MH.STRETCH, lambda sc: np.array([[1, 0, 0], [0, sc, (1 - sc) * tip[1]]], float)),
                ("shear", MH.SHEAR_DEG, lambda a: np.array([[1, np.tan(np.radians(a)), -np.tan(np.radians(a)) * tip[1]], [0, 1, 0]], float)),
            ):
                ident = 0 if name == "shear" else 1.0
                per = {}
                for p in params:
                    M = mk(p)
                    im2 = cv2.warpAffine(img, M, (W, H), borderValue=(0, 0, 0))
                    _, o2 = MH.predict_map(model, ds, im2)
                    k2 = MH.warp_kp6(m["keypoints"], M, W, H)
                    per[p] = (compute_spray_angle(MH.warp_kp6(k, M, W, H)), MH.head_angle(model, o2, k2))
                g1, c1 = per[ident]
                out[name] += [(g - g1, c - c1) for p, (g, c) in per.items() if p != ident]
    return out


def fig_slopes(ds, seeds=(42, 123)):
    """one landscape grid for both seeds of the final design on test: rows = the three
    angle probes, columns = keypoint branch / angle reader of each seed, with a group
    title per seed; each fitted slope is checked against the saved result of that seed"""
    frames = [s for s in ds.samples if SPRAY_IDX in s["gt_classes"]]
    probes = (("stretch_x", "horizontal stretch"), ("stretch_y", "vertical stretch"), ("shear", "shear"))
    fig, axes = plt.subplots(len(probes), 2 * len(seeds), figsize=(7.2, 6.3), sharex=True, sharey=True)
    lim, xs, out = 12, np.array([-12, 12]), {}
    for si, sd in enumerate(seeds):
        model = load_model(CKPT.parent.parent / f"abl_heads_s{sd}" / CKPT.name, DEVICE)
        pk, ph = pairs_keypoint(model, ds, frames), pairs_head(model, ds, frames)
        saved_cf = json.load(open(RESULTS_DIR / f"counterfactual_test_abl_heads_s{sd}.json"))
        saved_mh = json.load(open(RESULTS_DIR / f"mist_hybrid_test_final_heads_s{sd}.json"))
        saved_hx = json.load(open(RESULTS_DIR / f"reader_hstretch_test_final_heads_s{sd}.json"))
        cf_key = {"stretch_x": "stretch"}  # the horizontal stretch is saved as "stretch" by eval_counterfactual.py
        rows = [("keypoint branch", pk, lambda n: saved_cf[cf_key.get(n, n)]["slope"]),
                ("angle reader", ph, lambda n: saved_hx["stretch_x"]["slope"] if n == "stretch_x"
                 else saved_mh["probes_learned_head"][n]["slope"])]
        for r, (rname, pairs, saved) in enumerate(rows):
            j = 2 * si + r  # column: seed x output
            for i, (probe, pname) in enumerate(probes):  # row: probe
                ax = axes[i, j]
                a = np.array(pairs[probe], float)
                inf = a[np.abs(a[:, 0]) >= MH.MIN_DGT]
                sl = MH.slope(pairs[probe])
                if abs(sl - saved(probe)) > 0.02:
                    print(f"WARNING seed {sd} {rname} {probe}: recomputed slope {sl:.3f} differs from saved {saved(probe):.3f}")
                out[f"s{sd} {rname} {probe}"] = sl
                ax.scatter(a[:, 0], a[:, 1], s=4, alpha=0.35, color="#888888", lw=0)
                ax.scatter(inf[:, 0], inf[:, 1], s=5, alpha=0.6, color="#1f6fb2", lw=0)
                ax.plot(xs, xs, color="black", lw=0.7, ls="--")
                ax.plot(xs, 0 * xs, color="black", lw=0.7, ls=":")
                ax.plot(xs, sl * xs, color="#d62728", lw=1.3)
                ax.text(-lim + 1, lim - 1.5, f"slope {sl:.2f}", color="#d62728", fontsize=6.5, va="top")
                ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
                ax.tick_params(labelsize=6)
                if i == 0:
                    ax.set_title(rname, fontsize=7.5)
                if i == len(probes) - 1:
                    ax.set_xlabel("true change ($^\\circ$)", fontsize=7)
                if j == 0:
                    ax.set_ylabel(f"{pname}\npredicted change ($^\\circ$)", fontsize=7)
    handles = [plt.Line2D([0], [0], color="black", lw=0.8, ls="--", label="measuring, slope 1"),
               plt.Line2D([0], [0], color="black", lw=0.8, ls=":", label="recalling, slope 0"),
               plt.Line2D([0], [0], color="#d62728", lw=1.5, label="fitted slope"),
               plt.Line2D([0], [0], marker="o", ls="", color="#1f6fb2", markersize=4, label="used in the fit"),
               plt.Line2D([0], [0], marker="o", ls="", color="#888888", markersize=4, label="true change below $2^\\circ$")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=6.5, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95), w_pad=0.3, h_pad=0.5)
    # group title over the two columns of each seed, with a rule under it, as in the grouped tables
    for si, sd in enumerate(seeds):
        l, r = axes[0, 2 * si].get_position(), axes[0, 2 * si + 1].get_position()
        y = l.y1 + 0.045
        fig.text((l.x0 + r.x1) / 2, y + 0.006, f"seed {sd}", ha="center", va="bottom", fontsize=8.5)
        fig.add_artist(plt.Line2D([l.x0 + 0.01, r.x1 - 0.01], [y, y], color="black", lw=0.6, transform=fig.transFigure))
    fig.savefig(GFX / "probe_slopes.png", dpi=260)
    plt.close(fig)
    return out


# ── figure 3: one lance under the probes ─────────────────────────────────
def pick_lance(ds):
    frames = [s for s in ds.samples if SPRAY_IDX in s["gt_classes"]]
    s = frames[len(frames) // 2]
    best = None
    for b, c, k in zip(s["gt_boxes"], s["gt_classes"], s["gt_kpts"]):
        if c != SPRAY_IDX or not any(v > 0 for v in k):
            continue
        x1, y1, x2, y2 = cxcywh_to_xyxy(b)
        if x1 > 0.01 and y1 > 0.01 and x2 < 0.99 and y2 < 0.99:
            a = compute_spray_angle(k)
            if best is None or a > best[2]:
                best = (b, k, a)
    return s, best[0], best[1]


def kpt_pred_src(model, ds, img_src, tip_src):
    """keypoint-branch prediction nearest the given tip, as in CF.match_angle, with its keypoints"""
    preds = [p for p in cf_decode(CF.predict(model, ds, img_src), CF.LOW_THRESH) if p["class_id"] == SPRAY_IDX]
    if not preds:
        return None
    b = min(preds, key=lambda p: np.hypot(p["keypoints"][0] * CF.W - tip_src[0], p["keypoints"][1] * CF.H - tip_src[1]))
    if np.hypot(b["keypoints"][0] * CF.W - tip_src[0], b["keypoints"][1] * CF.H - tip_src[1]) > CF.TIP_MATCH_PX:
        return None
    return b


def fig_probe_lance(ds, seeds=(42, 123)):
    """one test lance under the probes; the inputs are the same for both seeds of the final
    design, so both seeds are drawn on the same panels (keypoint edges in two colours,
    angles as seed 42 / seed 123 in the titles)"""
    s, b, k = pick_lance(ds)
    img_src = cv2.cvtColor(cv2.imread(s["image_path"]), cv2.COLOR_BGR2RGB)
    kp_src = np.array(k).reshape(3, 2) * [CF.W, CF.H]
    box_src = ((b[0] - b[2] / 2) * CF.W, (b[1] - b[3] / 2) * CF.H, (b[0] + b[2] / 2) * CF.W, (b[1] + b[3] / 2) * CF.H)
    models = {sd: load_model(CKPT.parent.parent / f"abl_heads_s{sd}" / CKPT.name, DEVICE) for sd in seeds}
    bases = {sd: kpt_pred_src(models[sd], ds, img_src, kp_src[0]) for sd in seeds}
    colours = {42: "#33c3ff", 123: "#ff5fd2"}
    mats = {n: dict(v) for n, v in CF.probe_matrices(kp_src[0]).items()}
    variants = [
        ("original", img_src, kp_src, np.eye(3)[:2]),
        ("vertical stretch 0.8", *CF.warp(img_src, mats["stretch_y"][0.8], kp_src), mats["stretch_y"][0.8]),
        ("vertical stretch 1.25", *CF.warp(img_src, mats["stretch_y"][1.25], kp_src), mats["stretch_y"][1.25]),
        ("shear 15$^\\circ$", *CF.warp(img_src, mats["shear"][15], kp_src), mats["shear"][15]),
        ("cone masked", CF.mask_cone(img_src, kp_src, box_src), kp_src, None),
        ("body masked", CF.mask_body(img_src, kp_src, box_src), kp_src, None),
    ]
    cw, ch = 900, 675
    x0 = int(np.clip(kp_src[0, 0] - cw / 2, 0, CF.W - cw)); y0 = int(np.clip(kp_src[0, 1] - ch / 2, 0, CF.H - ch))
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.6))
    table = {}
    for ax, (name, im, k2, M) in zip(axes.ravel(), variants):
        ax.imshow(im[y0:y0 + ch, x0:x0 + cw])
        t, l, r = k2 - [x0, y0]
        for e in (l, r):
            ax.plot([t[0], e[0]], [t[1], e[1]], color="#ffd23f", lw=1.6)
        kp_txt, head_txt = [], []
        for sd in seeds:
            model = models[sd]
            p = kpt_pred_src(model, ds, im, k2[0])
            kp = None
            if p is not None:
                pk = np.array(p["keypoints"]).reshape(3, 2) * [CF.W, CF.H] - [x0, y0]
                for e in (pk[1], pk[2]):
                    ax.plot([pk[0, 0], e[0]], [pk[0, 1], e[1]], color=colours[sd], lw=1.2, ls="--")
                kp = compute_spray_angle(np.array(p["keypoints"]))
            hd_ = None
            if M is not None and bases[sd] is not None:
                small = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
                _, o2 = MH.predict_map(model, ds, small)
                sc = np.diag([W / CF.W, H / CF.H])  # the matrix was built in source pixels
                M_small = np.hstack([sc @ M[:, :2] @ np.linalg.inv(sc), (sc @ M[:, 2]).reshape(2, 1)])
                hd_ = MH.head_angle(model, o2, MH.warp_kp6(bases[sd]["keypoints"], M_small, W, H))
            kp_txt.append("none" if kp is None else f"{kp:.1f}$^\\circ$")
            head_txt.append("not run" if hd_ is None else f"{hd_:.1f}$^\\circ$")
            table[f"{name} s{sd}"] = {"true": CF.angle_px(k2), "keypoint": kp, "reader": hd_}
        # note above the panel in two columns: quantity on the left, value on the right
        tr = ax.transAxes
        ax.text(0.5, 1.27, name, ha="center", va="bottom", fontsize=7.5, transform=tr)
        ax.text(0.17, 1.02, "true\nkeypoint\nreader", ha="left", va="bottom", fontsize=7, linespacing=1.35, transform=tr)
        ax.text(0.45, 1.02, f"{CF.angle_px(k2):.1f}$^\\circ$\n{' / '.join(kp_txt)}\n{' / '.join(head_txt)}",
                ha="left", va="bottom", fontsize=7, linespacing=1.35, transform=tr)
        ax.set_xticks([]); ax.set_yticks([])
    handles = [plt.Line2D([0], [0], color="#ffd23f", lw=2, label="annotated cone")] + \
              [plt.Line2D([0], [0], color=colours[sd], lw=1.5, ls="--", label=f"keypoint branch, seed {sd}") for sd in seeds]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=7.5, frameon=False,
               title="angles: seed 42 / seed 123", title_fontsize=7.5)
    # the notes are plain texts above the axes, so the spacing is set by hand
    fig.subplots_adjust(left=0.01, right=0.99, top=0.87, bottom=0.1, wspace=0.04, hspace=0.5)
    fig.savefig(GFX / "probe_lance.png", dpi=260)
    plt.close(fig)
    return table


# ── figure 4: failure cases ──────────────────────────────────────────────
def crop_43(box_xyxy, frac):
    """a 4:3 view of width frac*W centred on a normalised box, kept inside the frame"""
    cw, ch = frac * W, frac * H
    cx, cy = (box_xyxy[0] + box_xyxy[2]) / 2 * W, (box_xyxy[1] + box_xyxy[3]) / 2 * H
    x0, y0 = float(np.clip(cx - cw / 2, 0, W - cw)), float(np.clip(cy - ch / 2, 0, H - ch))
    return (x0, y0, x0 + cw, y0 + ch)


def fig_failures(ds):
    """one example of each kind of error of the reported system on the test set. Seed 42:
    (a) the spraying lance at the top edge of CAM3 not detected, (b) a facing lance the gate
    reports as no_spray (CAM1 V3), (c) a big lance not detected (the frame where it stands out
    most from its surroundings, so it is visible), (d) the same edge lance idle, not detected.
    Seed 123, the kinds that occur only for it: (e) a small lance reported as big (CAM1 V2),
    (f) an idle lance reported as facing, (g) an idle lance not detected (both CAM1 V3).
    First frame of each kind, found with each seed's own gate threshold."""
    def contrast(s, b):
        g = cv2.cvtColor(cv2.imread(s["image_path"]), cv2.COLOR_BGR2GRAY).astype(float)
        x1, y1, x2, y2 = (np.clip(cxcywh_to_xyxy(b), 0, 1) * np.tile([g.shape[1], g.shape[0]], 2)).astype(int)
        pw, ph = (x2 - x1) // 4, (y2 - y1) // 4
        ring = g[max(0, y1 - ph):y2 + ph, max(0, x1 - pw):x2 + pw]
        inside = g[y1:y2, x1:x2]
        return abs((ring.sum() - inside.sum()) / max(ring.size - inside.size, 1) - inside.mean())

    SMALL = (FACING_IDX, NO_SPRAY_IDX)
    # seed 42: the four kinds of error of Table 12; seed 123: the kinds that occur only for it
    rules = {
        42: {"c": lambda s, c, b, hit, m: s["source_video"] == "20250509_frames_275-304" and c == SPRAY_IDX and not hit,
             "b": lambda s, c, b, hit, m: s["source_video"] == "CAM1 - V3" and c == FACING_IDX and hit and m["state"] == NO_SPRAY_IDX,
             "d": lambda s, c, b, hit, m: s["source_video"] == "20250510" and c == NO_SPRAY_IDX and not hit and b[1] < 0.2},
        123: {"e": lambda s, c, b, hit, m: s["source_video"] == "CAM1 - V2" and c in SMALL and hit and m["state"] == BIG_IDX,
              "f": lambda s, c, b, hit, m: s["source_video"] == "CAM1 - V3" and c == NO_SPRAY_IDX and hit and m["state"] == FACING_IDX},
    }
    cases, thrs = {}, {}
    for sd, rl in rules.items():
        model, thr = seed_model(sd)
        thrs[sd], best_c = thr, -1.0
        for s in ds.samples:
            img = load_frame(s["image_path"])
            prob, _, dets = system(model, ds, img, thr)
            for b, c in zip(s["gt_boxes"], s["gt_classes"]):
                g = cxcywh_to_xyxy(b)
                m = max(dets, key=lambda d: MH.box_iou(d["box"], g), default=None)
                hit = m is not None and MH.box_iou(m["box"], g) > 0.5
                for key, rule in rl.items():
                    if key not in cases and rule(s, c, b, hit, m):
                        extra = (m, MH.tip_disk_mean(prob, m["box"])) if key in ("b", "f") else None
                        cases[key] = (sd, s, img, dets, b, extra)
                if sd == 42 and c == BIG_IDX and not hit:
                    k = contrast(s, b)
                    if k > best_c:
                        best_c, cases["a"] = k, (sd, s, img, dets, b, None)
    # 2 x 4: seed 42 in the upper row (a-d), seed 123 in the lower row (e-f and the legend)
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8))
    titles = {"a": "big lance,\nnot detected", "b": "facing lance\nreported as no_spray",
              "c": "spraying edge lance,\nnot detected", "d": "idle edge lance,\nnot detected",
              "e": "small lance\nreported as big", "f": "idle lance\nreported as facing"}
    fracs = {"a": 0.7, "b": 0.45, "c": 0.4, "d": 0.4, "e": 0.55, "f": 0.45}
    for ax, key in zip(axes.ravel(), "abcdef"):
        sd, s, img, dets, b, extra = cases[key]
        crop = crop_43(cxcywh_to_xyxy(b), fracs[key])
        lab = None
        if extra is not None:
            m, mean = extra
            sign = "<" if mean < thrs[sd] else ">"
            lab = lambda d, m=m, mean=mean, sign=sign, thr=thrs[sd]: (
                f"gate: {NAME[d['state']]}\n{mean:.3f} {sign} {thr:.3f}" if d is m else NAME[d["state"]])
        draw(ax, img, dets, gts=[b], label_fn=lab, crop=crop, img_hi=load_hi(s))
        ax.set_title(f"({key}) seed {sd}, {SOURCE_LABEL[s['source_video']]}\n{titles[key]}", fontsize=7, pad=3)
    handles = [plt.Line2D([0], [0], color="#ffd23f", lw=1.5, ls="--", label="annotated box of the missed or wrong lance")] + \
              [plt.Line2D([0], [0], color=COL[c], lw=3, label=f"reported {NAME[c]}") for c in (SPRAY_IDX, FACING_IDX, NO_SPRAY_IDX, BIG_IDX)]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0.075, 1, 1), h_pad=3.2, w_pad=0.3)
    fig.savefig(GFX / "failures.png", dpi=260)
    plt.close(fig)
    return {k: (v[0], v[1]["image_path"].split("\\")[-1].split("/")[-1]) for k, v in cases.items()}


def error_counts(model, ds, thr, seed=42):
    """per source and class, how every annotated lance ends in the reported system:
    correct, not detected (no detection with IoU > 0.5), or detected with a wrong
    state; plus detections that match no annotated lance. Written to
    results/test_errors_final_heads_s{seed}.json for the numbers of Chapter 6."""
    import collections
    cnt = collections.defaultdict(collections.Counter)
    for s in ds.samples:
        img = load_frame(s["image_path"])
        _, _, dets = system(model, ds, img, thr)
        src = SOURCE_LABEL[s["source_video"]]
        for b, c in zip(s["gt_boxes"], s["gt_classes"]):
            g = cxcywh_to_xyxy(b)
            m = max(dets, key=lambda d: MH.box_iou(d["box"], g), default=None)
            if m is None or MH.box_iou(m["box"], g) <= 0.5:
                cnt[f"{src} {NAME[c]}"]["not detected"] += 1
            elif m["state"] != c:
                cnt[f"{src} {NAME[c]}"][f"reported {NAME[m['state']]}"] += 1
            else:
                cnt[f"{src} {NAME[c]}"]["correct"] += 1
        for d in dets:
            if all(MH.box_iou(d["box"], cxcywh_to_xyxy(b)) <= 0.5 for b in s["gt_boxes"]):
                cnt[f"{src} {NAME[d['state']]}"]["unmatched detection"] += 1
    out = {k: dict(v) for k, v in sorted(cnt.items())}
    tot = collections.Counter()
    for v in out.values():
        tot.update(v)
    out["total"] = dict(tot)
    json.dump(out, open(RESULTS_DIR / f"test_errors_final_heads_s{seed}.json", "w"), indent=1)
    return out


def main():
    # "seed=123" scores the other seed of the final design (errors only; the figures stay on seed 42);
    # the gate threshold is the one calibrated for that model on the training set
    seed = next((int(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("seed=")), 42)
    ds = AFGNetDataset(TEST_JSON, augment=False)
    model = load_model(CKPT.parent.parent / f"abl_heads_s{seed}" / CKPT.name, DEVICE)
    thr = json.load(open(RESULTS_DIR / f"mist_hybrid_test_final_heads_s{seed}.json"))["state"]["gate_threshold_from_train"]
    GFX.mkdir(parents=True, exist_ok=True)
    only = {a for a in sys.argv[1:] if not a.startswith("seed=")} or {"grid", "lance", "failures", "slopes"}
    if seed != 42 and only - {"errors"}:
        raise SystemExit("the figures are drawn for seed 42 only; use: errors seed=123")
    if "grid" in only:
        fig_grid(ds); print("test_grid.png")
    if "lance" in only:
        print("probe_lance.png", json.dumps(fig_probe_lance(ds), indent=1))
    if "failures" in only:
        print("failures.png", fig_failures(ds))
    if "slopes" in only:
        sl = fig_slopes(ds); print("probe_slopes.png", json.dumps(sl))
    if "errors" in only:  # not part of the default set
        print(json.dumps(error_counts(model, ds, thr, seed), indent=1))


if __name__ == "__main__":
    main()
