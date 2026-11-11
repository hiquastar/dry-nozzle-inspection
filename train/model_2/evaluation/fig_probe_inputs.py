"""
fig_probe_inputs.py - one picture of the inputs that the measuring-or-recalling
probes build (thesis Chapter 5). Inputs only: no prediction is shown, so the
figure belongs to the setup, not to the results.

It imports the transforms from eval_counterfactual.py, so every panel is the
exact input the evaluation feeds to the network. One validation frame with a
spray lance whose box lies fully inside the frame is picked; every panel shows
the same crop window, so the shift and rotation panels show the lance moving.

Panels: original | horizontal stretch 1.25 | vertical stretch 1.25 | shear 15 deg
        cone masked | body masked | shift +6 cells | rotation +10 deg
Each panel draws the (transformed) annotated cone and its true apex angle.

    python evaluation/fig_probe_inputs.py [--frame N]

Output: papers/thesis/gfx/probe_inputs.png
"""

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval_counterfactual import H, W, FEAT_H, FEAT_W, angle_px, mask_body, mask_cone, probe_matrices, warp
from dataset import AFGNetDataset

from config import REPO_ROOT, SPRAY_IDX, VAL_JSON

OUT = REPO_ROOT / "papers" / "thesis" / "gfx" / "probe_inputs.png"
CROP_W, CROP_H = 900, 675  # source pixels, 4:3


def spray_lances(s):
    out = []
    for b, c, k in zip(s["gt_boxes"], s["gt_classes"], s["gt_kpts"]):
        if c == SPRAY_IDX and any(v > 0 for v in k):
            box_px = ((b[0] - b[2] / 2) * W, (b[1] - b[3] / 2) * H, (b[0] + b[2] / 2) * W, (b[1] + b[3] / 2) * H)
            out.append((box_px, np.array(k, float).reshape(3, 2) * [W, H]))
    return out


def inside(box, margin=5):
    x1, y1, x2, y2 = box
    return x1 > margin and y1 > margin and x2 < W - margin and y2 < H - margin


def pick(ds, frame):
    frames = [s for s in ds.samples if SPRAY_IDX in s["gt_classes"]]
    cands = []
    for fi, s in enumerate(frames):
        for box, kp in spray_lances(s):
            if inside(box):
                cands.append((fi, s, box, kp))
    if frame >= 0:
        c = [x for x in cands if x[0] == frame]
        return c[0]
    # the frame in the middle of the window, lance with the widest apex angle
    mid = sorted({c[0] for c in cands})[len({c[0] for c in cands}) // 2]
    c = [x for x in cands if x[0] == mid]
    return max(c, key=lambda x: angle_px(x[3]))


def draw_cone(ax, kp, x0, y0, color="#ffd23f"):
    t, l, r = kp - [x0, y0]
    for e in (l, r):
        ax.plot([t[0], e[0]], [t[1], e[1]], color=color, lw=1.6)
    ax.plot(*t, "o", color=color, ms=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=-1, help="index among validation frames with a spray lance")
    args = ap.parse_args()

    ds = AFGNetDataset(VAL_JSON, augment=False)
    fi, s, box, kp = pick(ds, args.frame)
    img = cv2.cvtColor(cv2.imread(s["image_path"]), cv2.COLOR_BGR2RGB)
    tip = kp[0]
    x0 = int(np.clip(tip[0] - CROP_W / 2, 0, W - CROP_W))
    y0 = int(np.clip(tip[1] - CROP_H / 2, 0, H - CROP_H))

    mats = {name: dict(v) for name, v in probe_matrices(tip).items()}
    shift = np.array([[1, 0, 6 * W / FEAT_W], [0, 1, 0]], float)
    rot = cv2.getRotationMatrix2D((W / 2, H / 2), 10, 1.0)
    panels = [("original", img, kp)]
    panels.append(("horizontal stretch 1.25", *warp(img, mats["stretch_x"][1.25], kp)))
    panels.append(("vertical stretch 1.25", *warp(img, mats["stretch_y"][1.25], kp)))
    panels.append(("shear 15$^\\circ$", *warp(img, mats["shear"][15], kp)))
    panels.append(("cone masked", mask_cone(img, kp, box), kp))
    panels.append(("body masked", mask_body(img, kp, box), kp))
    panels.append(("shift 6 cells", *warp(img, shift, kp)))
    panels.append(("rotation 10$^\\circ$", *warp(img, rot, kp)))

    fig, axes = plt.subplots(2, 4, figsize=(12, 6.6))
    for ax, (name, im, k) in zip(axes.ravel(), panels):
        ax.imshow(im[y0 : y0 + CROP_H, x0 : x0 + CROP_W])
        draw_cone(ax, k, x0, y0)
        ax.set_title(f"{name}\ntrue angle {angle_px(k):.1f}$^\\circ$", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170)
    print(f"frame {fi} ({Path(s['image_path']).name}), base angle {angle_px(kp):.1f} deg -> {OUT}")


if __name__ == "__main__":
    main()
