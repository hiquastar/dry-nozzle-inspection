"""
build_cache.py - pre-resize every frame of the three splits to IMG_W x IMG_H.

Why: the network sees 640x480, but the loader decoded 2048x1536 PNG/JPG and
ran the geometric augmentation at that size for every sample of every
epoch. That kept the GPU at ~37%. Resizing once (cv2.INTER_AREA) and
reading the small file instead is a pure speed change: boxes and keypoints
are normalised coordinates and do not depend on the pixel size.

Cache layout: IMG_CACHE_DIR / <path relative to REPO_ROOT, extension .png>

    python train/model_2/structure/build_cache.py
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from dataset import AFGNetDataset, cache_path_for, prev_frame_path  # noqa: E402

from config import IMG_CACHE_DIR, IMG_H, IMG_W, TEST_JSON, TRAIN_JSON, VAL_JSON  # noqa: E402

# Step 4 (TWO_FRAME) also needs frame t-k for every annotated frame; cache the
# neighbours for the gaps that will be tried so training never falls back to
# decoding a 2048x1536 source frame.
NEIGHBOUR_GAPS = (1, 3, 5)


def resize_one(src: str):
    dst = cache_path_for(src)
    if dst.exists():
        return "skip"
    img = cv2.imread(src, cv2.IMREAD_COLOR)
    if img is None:
        return "missing"
    if img.shape[1] != IMG_W or img.shape[0] != IMG_H:
        img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return "ok"


def main():
    paths = []
    for js in (TRAIN_JSON, VAL_JSON, TEST_JSON):
        ds = AFGNetDataset(js, augment=False)
        paths += [s["image_path"] for s in ds.samples]
    annotated = sorted(set(paths))
    for p in annotated:
        for g in NEIGHBOUR_GAPS:
            paths.append(prev_frame_path(p, g))
    paths = sorted(set(paths))
    print(f"{len(annotated)} annotated + neighbours (gaps {NEIGHBOUR_GAPS}) = {len(paths)} frames -> {IMG_CACHE_DIR}  ({IMG_W}x{IMG_H})")
    counts = {"ok": 0, "skip": 0, "missing": 0}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for r in tqdm(ex.map(resize_one, paths), total=len(paths), unit="img"):
            counts[r] += 1
    print(counts)


if __name__ == "__main__":
    main()
