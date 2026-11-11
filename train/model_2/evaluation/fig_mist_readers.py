"""
fig_mist_readers.py - one picture of how the final model reads its mist map.

Panels, all from the FINAL checkpoint on one validation frame of the spray
recording (GT boxes / keypoints only fix WHERE to look; every value shown is
predicted):

  (a) frame with GT boxes            (b) predicted mist map m = sigmoid(logits), 120x160 -> 480x640
  (c) S1 gate: disk of radius 0.45*max(w,h) at each small-lance tip,
      mean(m) vs tau_m from the val json (spraying iff mean > tau_m)
  (d) S2 reader: 180 rays over axis +- 90 deg, r in [10, 70] px, on the spray lance
  (e) the 180-bin profile the learned angle head sees, with its answer vs the label

    AFGNET_INSTANCE_NORM=0 AFGNET_TIP_RELATIVE_EDGES=1 AFGNET_MIST_HEAD=1 AFGNET_LEARNED_HEADS=1 \
      python evaluation/fig_mist_readers.py --ckpt checkpoint/abl_heads_s42/afgnet_v10.pth

Output: papers/thesis/gfx/mist_readers.png and results/cv_debug/mist_readers.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle, Rectangle

from common import RESULTS_DIR, load_model  # also puts ../structure on sys.path
from dataset import AFGNetDataset, load_frame
from eval_mist_hybrid import TIP_DISK, predict_map, tip_disk_mean
from heads import HEAD_RAD, HEAD_RAYS, polar_points
from metrics import compute_spray_angle, cxcywh_to_xyxy

from config import ANGLE_ASPECT, CLASS_NAMES, DEVICE, FACING_IDX, IMG_H, IMG_W, MIST_STRIDE, NO_SPRAY_IDX, SPRAY_IDX, VAL_JSON

SMALL = (SPRAY_IDX, NO_SPRAY_IDX, FACING_IDX)
COL = {SPRAY_IDX: "#e6194b", NO_SPRAY_IDX: "#4363d8", FACING_IDX: "#3cb44b"}
THESIS_GFX = Path(__file__).resolve().parents[3] / "papers" / "thesis" / "gfx"


def tip_px(box_xyxy):
    """inner end of a box (the two corners nearest the frame centre), in px"""
    x1, y1, x2, y2 = box_xyxy[0] * IMG_W, box_xyxy[1] * IMG_H, box_xyxy[2] * IMG_W, box_xyxy[3] * IMG_H
    corners = np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]])
    d = np.linalg.norm(corners - [IMG_W / 2, IMG_H / 2], axis=1)
    return corners[np.argsort(d)[:2]].mean(0), TIP_DISK * max(x2 - x1, y2 - y1)


def axis_deg_from_kpts(kp6):
    k = np.array(kp6, float).reshape(3, 2)
    iso = (k - k[0]) * [ANGLE_ASPECT, 1.0]
    v = iso[1] + iso[2]
    return float(np.degrees(np.arctan2(v[1], v[0])))


def pick_frame(ds):
    """first val frame with a spray lance (3 keypoints) next to a facing lance,
    so the same picture shows the gate saying yes and no"""
    for s in ds.samples:
        cls = s["gt_classes"]
        has_spray = any(c == SPRAY_IDX and np.any(np.array(k) > 0) for c, k in zip(cls, s["gt_kpts"]))
        if has_spray and (FACING_IDX in cls or NO_SPRAY_IDX in cls):
            return s
    raise SystemExit("no val frame with spray + facing/no_spray")


def pick_gate_frame(ds):
    """first val frame with a facing AND a no_spray lance: the two classes the gate separates"""
    for s in ds.samples:
        if FACING_IDX in s["gt_classes"] and NO_SPRAY_IDX in s["gt_classes"]:
            return s
    raise SystemExit("no val frame with facing + no_spray")


def gate_rows_for(prob, s, tau):
    rows = []
    for box, c in zip(s["gt_boxes"], s["gt_classes"]):
        if c not in SMALL:
            continue
        xyxy = np.array(cxcywh_to_xyxy(np.array(box, float)), float)
        tip, r = tip_px(xyxy)
        mean = tip_disk_mean(prob, xyxy)
        rows.append((xyxy, c, tip, r, mean, mean > tau))
    return rows


def draw_gate(ax, prob, rows, tau, title):
    ax.imshow(prob, cmap="magma", vmin=0, vmax=1)
    for xyxy, c, tip, r, mean, on in rows:
        col = COL[c]
        ax.add_patch(Circle(tip, r, fill=False, lw=1.6, ec=col, ls="-" if on else "--"))
        ax.text(tip[0] + r + 3, tip[1], f"{mean:.2f}", color=col, fontsize=7, va="center", fontweight="bold")
    ax.set_title(title, fontsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoint/abl_heads_s42/afgnet_v10.pth")
    ap.add_argument("--gate_json", default=str(RESULTS_DIR / "mist_hybrid_val_abl_heads_s42.json"))
    ap.add_argument("--frame", type=int, default=-1, help="val sample index (-1 = auto pick)")
    args = ap.parse_args()

    tau = json.load(open(args.gate_json))["state"]["gate_threshold_from_train"]
    model = load_model(Path(args.ckpt), DEVICE).eval()
    ds = AFGNetDataset(VAL_JSON, augment=False)
    s = ds.samples[args.frame] if args.frame >= 0 else pick_frame(ds)
    img = load_frame(s["image_path"])
    prob, out = predict_map(model, ds, img)  # prob 480x640, out["mist_logits"] [1,1,120,160]
    show = np.asarray(torch.nn.functional.interpolate(torch.from_numpy(img).permute(2, 0, 1)[None].float(), size=(IMG_H, IMG_W), mode="bilinear", align_corners=False)[0].permute(1, 2, 0).byte())
    pm = torch.sigmoid(out["mist_logits"])

    # ---- S1: gate per small lance, on this frame and on a facing + no_spray frame
    gate_rows = gate_rows_for(prob, s, tau)
    s2 = pick_gate_frame(ds)
    prob2, _ = predict_map(model, ds, load_frame(s2["image_path"]))
    gate_rows2 = gate_rows_for(prob2, s2, tau)

    # ---- S2: the spray lance with keypoints
    j = next(i for i, (c, k) in enumerate(zip(s["gt_classes"], s["gt_kpts"])) if c == SPRAY_IDX and np.any(np.array(k) > 0))
    kp6 = s["gt_kpts"][j]
    k = np.array(kp6).reshape(3, 2) * [IMG_W, IMG_H]
    axis = axis_deg_from_kpts(kp6)
    tips = torch.tensor(kp6[:2], dtype=torch.float32, device=DEVICE).view(1, 1, 2)
    ax_t = torch.tensor([[axis]], dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        prof = model.angle_head.profile(pm, tips, ax_t)[0, 0].cpu().numpy()  # 180 bins
        pred = float(np.degrees(model.read_angle(out["mist_logits"], tips, ax_t)[0, 0].item()))
    gt = compute_spray_angle(kp6)
    pts = polar_points(tips, ax_t)[0, 0].cpu().numpy() * [IMG_W, IMG_H]  # [RAYS, RAD_N, 2]

    # ---- figure
    fig = plt.figure(figsize=(13, 8.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])
    axA, axB, axC = (fig.add_subplot(gs[0, i]) for i in range(3))
    axD, axE, axF = (fig.add_subplot(gs[1, i]) for i in range(3))

    axA.imshow(show)
    for box, c in zip(s["gt_boxes"], s["gt_classes"]):
        x1, y1, x2, y2 = np.array(cxcywh_to_xyxy(np.array(box, float)), float) * [IMG_W, IMG_H, IMG_W, IMG_H]
        axA.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=1.2, ec=COL.get(c, "w")))
    axA.set_title("(a) frame 640x480, GT boxes\nred spray, blue no_spray, green facing", fontsize=9)

    im = axB.imshow(prob, cmap="magma", vmin=0, vmax=1)
    axB.set_title(f"(b) predicted mist map m, {pm.shape[-2]}x{pm.shape[-1]} cells (stride {MIST_STRIDE})", fontsize=9)
    fig.colorbar(im, ax=axB, fraction=0.035, pad=0.02)

    draw_gate(axC, prob, gate_rows, tau,
              f"(c) S1 gate on (b): mean of m in a disk at the tip, r = {TIP_DISK}*max(w,h)\n"
              f"spraying iff mean > tau = {tau:.3f} (solid = yes)")
    draw_gate(axF, prob2, gate_rows2, tau,
              f"(f) S1 gate on val frame {Path(s2['image_path']).name}\n"
              "green facing (spraying at the camera) vs blue no_spray (idle)")

    # zoom on the spray lance for the rays
    cx, cy = k[0]
    pad = 110
    axD.imshow(prob, cmap="magma", vmin=0, vmax=1)
    for i in range(0, HEAD_RAYS, 6):
        axD.plot(pts[i, :, 0], pts[i, :, 1], color="cyan", lw=0.5, alpha=0.8)
    axD.plot(pts[0, :, 0], pts[0, :, 1], color="yellow", lw=1.2)
    axD.plot(pts[-1, :, 0], pts[-1, :, 1], color="yellow", lw=1.2)
    axD.plot(k[[0, 1], 0], k[[0, 1], 1], "w-", lw=1.5)
    axD.plot(k[[0, 2], 0], k[[0, 2], 1], "w-", lw=1.5)
    axD.plot(k[0, 0], k[0, 1], "wo", ms=4)
    axD.set_xlim(cx - pad, cx + pad)
    axD.set_ylim(cy + pad * 0.75, cy - pad * 0.75)
    axD.set_title(f"(d) S2 reader: {HEAD_RAYS} rays over axis +- 90 deg, r in [{HEAD_RAD[0]*MIST_STRIDE:.0f}, {HEAD_RAD[1]*MIST_STRIDE:.0f}] px\n"
                  "(every 6th drawn, yellow = +- 90 deg); white = GT edges", fontsize=9)

    bins = np.linspace(-90, 90, HEAD_RAYS)
    axE.plot(bins, prof, color="k", lw=1.2)
    axE.axvspan(-gt / 2, gt / 2, color="#e6194b", alpha=0.15, label=f"label {gt:.1f}°")
    axE.axvline(-pred / 2, color="cyan", lw=1.2); axE.axvline(pred / 2, color="cyan", lw=1.2, label=f"learned head {pred:.1f}°")
    axE.set_xlabel("ray angle relative to axis (deg)"); axE.set_ylabel("mean m along the ray")
    axE.set_ylim(0, 1); axE.set_xlim(-90, 90); axE.legend(fontsize=8, loc="upper right")
    axE.set_title("(e) profile the angle head sees\n(1x180 vector, map detached)", fontsize=9)

    for a in (axA, axB, axC, axD, axF):
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"Final model ({Path(args.ckpt).parent.name}) on val frame {Path(s['image_path']).name}", fontsize=10)
    fig.tight_layout()
    for dst in (THESIS_GFX / "mist_readers.png", RESULTS_DIR / "cv_debug" / "mist_readers.png"):
        dst.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dst, dpi=160)
        print("wrote", dst)

    # ---- focused 2-panel figure for the apex-angle reader alone (thesis Section 4.5.6)
    fig2, (bx, by) = plt.subplots(1, 2, figsize=(9.5, 4.3))
    bx.imshow(prob, cmap="magma", vmin=0, vmax=1)
    for i in range(0, HEAD_RAYS, 6):
        bx.plot(pts[i, :, 0], pts[i, :, 1], color="cyan", lw=0.5, alpha=0.8)
    bx.plot(pts[0, :, 0], pts[0, :, 1], color="yellow", lw=1.2)
    bx.plot(pts[-1, :, 0], pts[-1, :, 1], color="yellow", lw=1.2)
    bx.plot(k[[0, 1], 0], k[[0, 1], 1], "w-", lw=1.5)
    bx.plot(k[[0, 2], 0], k[[0, 2], 1], "w-", lw=1.5)
    bx.plot(k[0, 0], k[0, 1], "wo", ms=4)
    bx.set_xlim(cx - pad, cx + pad)
    bx.set_ylim(cy + pad * 0.75, cy - pad * 0.75)
    bx.set_xticks([]); bx.set_yticks([])
    bx.set_title(f"(a) polar rays on the mist map\n{HEAD_RAYS} rays, axis $\\pm90^\\circ$, $r\\in[{HEAD_RAD[0]*MIST_STRIDE:.0f},{HEAD_RAD[1]*MIST_STRIDE:.0f}]$ px", fontsize=10)

    by.plot(bins, prof, color="k", lw=1.4)
    by.axvspan(-gt / 2, gt / 2, color="#e6194b", alpha=0.15, label=f"label {gt:.1f}$^\\circ$")
    by.axvline(-pred / 2, color="#0072B2", lw=1.4)
    by.axvline(pred / 2, color="#0072B2", lw=1.4, label=f"learned head {pred:.1f}$^\\circ$")
    by.set_xlabel("ray angle relative to axis (deg)")
    by.set_ylabel("mean $m$ along the ray")
    by.set_ylim(0, 1); by.set_xlim(-90, 90)
    by.legend(fontsize=9, loc="upper right")
    by.set_title("(b) resulting profile\n180-bin vector, map detached", fontsize=10)

    fig2.tight_layout()
    for dst in (THESIS_GFX / "angle_reader.png", RESULTS_DIR / "cv_debug" / "angle_reader.png"):
        dst.parent.mkdir(parents=True, exist_ok=True)
        fig2.savefig(dst, dpi=200)
        print("wrote", dst)

    summary = {"frame": s["image_path"], "tau": tau,
               "gate": [{"class": CLASS_NAMES[c], "mean": round(m, 3), "spraying": bool(o)} for _, c, _, _, m, o in gate_rows],
               "gate_frame2": s2["image_path"],
               "gate2": [{"class": CLASS_NAMES[c], "mean": round(m, 3), "spraying": bool(o)} for _, c, _, _, m, o in gate_rows2],
               "angle": {"label_deg": gt, "learned_head_deg": pred}}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
