"""
viz_mist_map.py - what the MIST_HEAD model sees: frame | predicted mist map | overlay
for a handful of val frames from both cameras, plus tip crops with the
gate value written on them.

    AFGNET_INSTANCE_NORM=0 AFGNET_TIP_RELATIVE_EDGES=1 AFGNET_MIST_HEAD=1 \
      python evaluation/viz_mist_map.py --ckpt checkpoint/abl_mist_s42/afgnet_v10.pth [--split val] [--n 6]

Output: results/cv_debug/mist_seen_<split>.jpg and mist_seen_tips_<split>.jpg
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from common import RESULTS_DIR, load_model  # also puts ../structure on sys.path
from dataset import AFGNetDataset, load_frame
from eval_mist_hybrid import predict_map, tip_disk_mean
from metrics import cxcywh_to_xyxy

from config import CLASS_NAMES, DEVICE, FACING_IDX, NO_SPRAY_IDX, SPRAY_IDX, TEST_JSON, VAL_JSON


def heat(prob):
    return cv2.applyColorMap((prob * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()
    ds = AFGNetDataset(VAL_JSON if args.split == "val" else TEST_JSON, augment=False)
    model = load_model(Path(args.ckpt), DEVICE)
    out_dir = RESULTS_DIR / "cv_debug"

    # pick frames: half with a spray nozzle (video), half from the still camera
    by_src = {}
    for s in ds.samples:
        by_src.setdefault(s["source_video"], []).append(s)
    picks = []
    for src, ss in by_src.items():
        k = max(1, args.n // len(by_src))
        picks += ss[:: max(1, len(ss) // k)][:k]

    rows, tips = [], []
    for s in picks:
        img = load_frame(s["image_path"])
        prob, _ = predict_map(model, ds, img)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        over = bgr.copy()
        over[..., 2] = np.maximum(over[..., 2], (prob * 255).astype(np.uint8))
        H, W = img.shape[:2]
        for b, c in zip(s["gt_boxes"], s["gt_classes"]):
            x1, y1, x2, y2 = [int(v) for v in np.array(cxcywh_to_xyxy(b)) * [W, H, W, H]]
            col = {SPRAY_IDX: (0, 255, 0), FACING_IDX: (0, 200, 255), NO_SPRAY_IDX: (255, 80, 80)}.get(c, (200, 200, 200))
            cv2.rectangle(over, (x1, y1), (x2, y2), col, 1)
            cv2.putText(over, CLASS_NAMES[c].replace("nozzle_", ""), (x1, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
            if c in (SPRAY_IDX, FACING_IDX, NO_SPRAY_IDX):
                # tip crop with the gate value
                xx1, yy1, xx2, yy2 = np.array(cxcywh_to_xyxy(b)) * [W, H, W, H]
                corners = np.array([[xx1, yy1], [xx2, yy1], [xx1, yy2], [xx2, yy2]])
                tip = corners[np.argsort(np.linalg.norm(corners - [W / 2, H / 2], axis=1))[:2]].mean(0)
                r = int(max(xx2 - xx1, yy2 - yy1) * 0.9)
                x0, y0 = int(max(0, tip[0] - r)), int(max(0, tip[1] - r))
                crop = np.hstack([bgr[y0 : y0 + 2 * r, x0 : x0 + 2 * r], heat(prob)[y0 : y0 + 2 * r, x0 : x0 + 2 * r]])
                if crop.size:
                    crop = cv2.resize(crop, (400, 200))
                    cv2.putText(crop, f"{CLASS_NAMES[c].replace('nozzle_', '')}  mist@tip {tip_disk_mean(prob, cxcywh_to_xyxy(b)):.2f}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    tips.append(crop)
        cv2.putText(bgr, s["source_video"], (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        rows.append(np.hstack([bgr, heat(prob), over]))

    cv2.imwrite(str(out_dir / f"mist_seen_{args.split}.jpg"), np.vstack(rows))
    if tips:
        while len(tips) % 3:
            tips.append(np.zeros_like(tips[0]))
        cv2.imwrite(str(out_dir / f"mist_seen_tips_{args.split}.jpg"), np.vstack([np.hstack(tips[i : i + 3]) for i in range(0, len(tips), 3)]))
    print("wrote", out_dir / f"mist_seen_{args.split}.jpg", "and", out_dir / f"mist_seen_tips_{args.split}.jpg")


if __name__ == "__main__":
    main()
