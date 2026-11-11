"""
build_mist_masks.py - pseudo-labels for the MIST head (step 6 / option B).

No hand annotation exists for mist. The cue that separates spraying from
idle nozzles across cameras with AUC 1.0 (evaluation/cv_state_gate.py) is
LOCAL EXCESS BRIGHTNESS: frame minus a wide Gaussian background. Thresholded
per pixel it gives a mist mask that the network can learn to reproduce from
shape and texture - and then the mask, not a regressed number, drives the
state rule and the cone-angle measurement.

For every cached 640x480 frame:
    hp   = clip(gray - GaussianBlur(gray, sigma=40), 0)
    soft = clip(hp / (2*T), 0, 1)        # T = MIST_T; soft target, 0.5 at T
stored as uint8 PNG (0..255) at 640x480 in data/cache_640x480_mist/<rel>.png.
The loader warps it with the same affine as the frame and downsamples to the
head's stride-4 grid (160x120).

    python train/model_2/structure/build_mist_masks.py [--preview N]
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from dataset import AFGNetDataset, cache_path_for, mist_path_for  # noqa: E402

from config import MIST_BG_SIGMA, MIST_EDGE_G, MIST_T, TEST_JSON, TRAIN_JSON, VAL_JSON  # noqa: E402


def mist_soft(gray_u8):
    """Local excess brightness, suppressed where the image has sharp edges.

    Mist is diffuse (low gradient); glare on lances, rim and the central
    nozzle body is bright but edgy. Without the edge term the pseudo mask
    paints every reflection (see cv_debug/mist_pseudo_preview.jpg, v1)."""
    gf = gray_u8.astype(np.float32)
    hp = np.clip(gf - cv2.GaussianBlur(gf, (0, 0), MIST_BG_SIGMA), 0, None)
    soft = np.clip(hp / (2.0 * MIST_T), 0.0, 1.0)
    sm = cv2.GaussianBlur(gf, (0, 0), 2.0)
    gmag = np.hypot(cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3))
    gmag = cv2.GaussianBlur(gmag, (0, 0), 3.0)  # edginess of the neighbourhood
    edgy = np.clip(gmag / MIST_EDGE_G, 0.0, 1.0)
    return soft * (1.0 - edgy)


def build_one(src):
    dst = mist_path_for(src)
    if dst.exists():
        return "skip"
    c = cache_path_for(src)
    img = cv2.imread(str(c) if c.exists() else src, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "missing"
    if img.shape != (480, 640):
        img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), (mist_soft(img) * 255).astype(np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", type=int, default=0, help="write N overlay previews to results/cv_debug/")
    args = ap.parse_args()
    paths = []
    for js in (TRAIN_JSON, VAL_JSON, TEST_JSON):
        paths += [s["image_path"] for s in AFGNetDataset(js, augment=False).samples]
    paths = sorted(set(paths))
    counts = {"ok": 0, "skip": 0, "missing": 0}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for r in tqdm(ex.map(build_one, paths), total=len(paths), unit="img"):
            counts[r] += 1
    print(counts)

    if args.preview:
        out_dir = _HERE.parent / "results" / "cv_debug"
        out_dir.mkdir(parents=True, exist_ok=True)
        ds = AFGNetDataset(VAL_JSON, augment=False)
        srcs = {}
        for s in ds.samples:
            srcs.setdefault(s["source_video"], []).append(s)
        tiles = []
        for src, ss in srcs.items():
            for s in ss[:: max(1, len(ss) // max(1, args.preview // len(srcs)))][: args.preview // len(srcs)]:
                img = cv2.imread(str(cache_path_for(s["image_path"])))
                m = cv2.imread(str(mist_path_for(s["image_path"])), cv2.IMREAD_GRAYSCALE)
                over = img.copy()
                over[..., 2] = np.maximum(over[..., 2], m)  # mist in red
                tiles.append(np.hstack([img, over]))
        cv2.imwrite(str(out_dir / "mist_pseudo_preview.jpg"), np.vstack(tiles))
        print("preview:", out_dir / "mist_pseudo_preview.jpg")


if __name__ == "__main__":
    main()
