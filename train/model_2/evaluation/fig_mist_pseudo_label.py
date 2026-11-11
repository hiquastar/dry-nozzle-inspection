"""
fig_mist_pseudo_label.py - thesis figure for Section 4.5 (mist evidence map):
how the pseudo target m* and the loss region R are derived from one frame.

Writes papers/thesis/gfx/mist_pseudo_label.png (matplotlib, titles above the panels) with two rows (a spray frame from
CAM3-V2 and a facing/no_spray frame from CAM2) and four panels each:
  (a) grey frame with the annotated boxes
  (b) excess brightness s = clip((I - G40*I) / 2T, 0, 1)
  (c) edge suppression 1 - e
  (d) target m* = s (1 - e) with the loss region R outlined

Run from train/model_2:  python evaluation/fig_mist_pseudo_label.py
"""

import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "structure"))
from config import BIG_IDX, CLASS_TO_IDX, IMG_H, IMG_W, MIST_BG_SIGMA, MIST_EDGE_G, MIST_STRIDE, MIST_T, REPO_ROOT, TRAIN_JSON  # noqa: E402
from dataset import cache_path_for, mist_targets, resolve_image_path  # noqa: E402

OUT = REPO_ROOT / "papers" / "thesis" / "gfx" / "mist_pseudo_label.png"
PICK = {"20250509_frames_0-213": "20250509_000100.jpg", "CAM2 - V1": None}  # None = first frame of that source


def parts(gray_u8):
    """the three factors of the pseudo label, at full 640x480 resolution"""
    gf = gray_u8.astype(np.float32)
    hp = np.clip(gf - cv2.GaussianBlur(gf, (0, 0), MIST_BG_SIGMA), 0, None)
    s = np.clip(hp / (2.0 * MIST_T), 0.0, 1.0)
    sm = cv2.GaussianBlur(gf, (0, 0), 2.0)
    gmag = np.hypot(cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3))
    gmag = cv2.GaussianBlur(gmag, (0, 0), 3.0)
    e = np.clip(gmag / MIST_EDGE_G, 0.0, 1.0)
    return s, 1.0 - e, s * (1.0 - e)


def heat(x):
    return cv2.applyColorMap((np.clip(x, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def main():
    d = json.load(open(TRAIN_JSON, encoding="utf-8"))
    imgs = {i["id"]: i for i in d["images"]}
    anns = {}
    for a in d["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    rows = []
    for src, want in PICK.items():
        cands = [i for i in d["images"] if i.get("source_video") == src and (want is None or i["file_name"] == want)]
        im = sorted(cands, key=lambda i: i["file_name"])[0]
        bgr = cv2.imread(str(cache_path_for(resolve_image_path(im["file_name"], src))))
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        s, keep, m = parts(gray)
        boxes, classes = [], []
        for a in anns[im["id"]]:
            name = (a.get("_meta") or {}).get("original_class")
            if name not in CLASS_TO_IDX:
                continue
            x, y, w, h = a["bbox"]
            boxes.append(((x + w / 2) / im["width"], (y + h / 2) / im["height"], w / im["width"], h / im["height"]))
            classes.append(CLASS_TO_IDX[name])
        _, roi = mist_targets((m * 255).astype(np.uint8), boxes, classes)
        roi_full = cv2.resize(roi.numpy(), (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)

        a_ = bgr.copy()
        for (cx, cy, bw, bh), c in zip(boxes, classes):
            col = (255, 200, 0) if c == BIG_IDX else (0, 255, 0)
            cv2.rectangle(a_, (int((cx - bw / 2) * IMG_W), int((cy - bh / 2) * IMG_H)),
                          (int((cx + bw / 2) * IMG_W), int((cy + bh / 2) * IMG_H)), col, 2)
        d_ = heat(m)
        contours, _ = cv2.findContours((roi_full > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(d_, contours, -1, (255, 255, 255), 2)
        rows.append([a_, heat(s), heat(keep), d_])

    titles = ["(a) frame and boxes", "(b) excess brightness $s$",
              "(c) edge suppression $1 - e$", "(d) target $m^*$ and region $\\mathcal{R}$"]
    row_names = ["spraying frame, CAM3 V2", "idle frame, CAM2 V1"]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.6))
    for r, (row, name) in enumerate(zip(rows, row_names)):
        for c, (ax, im) in enumerate(zip(axes[r], row)):
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            ax.set_title(titles[c], fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[r][0].set_ylabel(name, fontsize=10)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
