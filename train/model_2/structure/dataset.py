"""
dataset.py - AFGNet v9 dataset (loads COCO splits)

PATCHED (May 11): Horizontal flip now correctly swaps left <-> right
keypoint identities. The original code only flipped x-coordinates,
which caused contradictory supervision (the "left" label sometimes
pointed to right-side pixels and vice versa).

Loads COCO split JSON (train/model_2/data/train.coco.json etc; same reverse split as data/splits_reverse) and builds
cxcywh bbox-format targets.

Maps original_class to v9 class index:
  big_nozzle      -> 3 (BIG_IDX)
  nozzle_no_spray -> 1 (NO_SPRAY_IDX)
  nozzle_spray    -> 0 (SPRAY_IDX)  + has keypoints + obs=1
  nozzle_facing   -> 2 (FACING_IDX) + obs=0
  nozzle_buildup  -> SKIPPED (excluded)
"""

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from targets import build_fcos_targets
from torch.utils.data import Dataset

import torchvision.transforms.functional as TF

from config import (
    ANGLE_ASPECT,
    ANGLE_AUG,
    ANGLE_AUG_SCALE,
    BIG_IDX,
    PHOTO_AUG,
    PHOTO_AUG_BRIGHTNESS,
    PHOTO_AUG_CONTRAST,
    PHOTO_AUG_GAMMA,
    PHOTO_AUG_GRAY_P,
    GEO_AUG,
    GEO_AUG_ROT_DEG,
    GEO_AUG_SCALE,
    GEO_AUG_SHIFT_FRAC,
    FRAME_GAP,
    IMG_CACHE_DIR,
    IMG_H,
    IMG_W,
    LEARNED_HEADS,
    MAX_INST,
    MIST_CACHE_DIR,
    MIST_HEAD,
    MIST_ROI_FACTOR,
    MIST_STRIDE,
    REPO_ROOT,
    TWO_FRAME,
    USE_IMG_CACHE,
    CLASS_NAMES,
    CLASS_TO_IDX,
    CONVERTED_IMG_ROOT,
    EXCLUDED_CATS,
    FACING_IDX,
    IMG_SIZE,
    NO_SPRAY_IDX,
    NUM_KEYPOINTS,
    ORIGINAL_IMG_ROOT,
    SPRAY_IDX,
)


def resolve_image_path(file_name, source_video_hint=None):
    """Resolve an image file by name across the dataset roots/source-video folders."""
    candidates = []

    if source_video_hint:
        clean_hint = source_video_hint.split("_frames_")[0]
        candidates.append(ORIGINAL_IMG_ROOT / clean_hint / file_name)

    if file_name.startswith("20250508"):
        candidates.append(ORIGINAL_IMG_ROOT / "20250508" / file_name)
    elif file_name.startswith("frame_"):
        candidates.append(ORIGINAL_IMG_ROOT / "20250508" / file_name)
    elif file_name.startswith("20250509"):
        candidates.append(ORIGINAL_IMG_ROOT / "20250509" / file_name)
    elif file_name.startswith("20250510"):
        candidates.append(ORIGINAL_IMG_ROOT / "20250510" / file_name)
    elif file_name.startswith("ShortAll"):
        candidates.append(ORIGINAL_IMG_ROOT / "ShortAll" / file_name)
    elif "CAM#1" in file_name:
        for v in ["V1", "V2", "V3"]:
            if f"-{v} CAM" in file_name:
                candidates.append(ORIGINAL_IMG_ROOT / f"CAM1 - {v}" / file_name)
                break
    elif "CAM#2" in file_name:
        for v in ["V1", "V2", "V3"]:
            if f"-{v} CAM" in file_name:
                candidates.append(ORIGINAL_IMG_ROOT / f"CAM2 - {v}" / file_name)
                break

    for c in candidates:
        if c.exists():
            return c

    matches = list(ORIGINAL_IMG_ROOT.rglob(file_name))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Cannot find: {file_name}")


def map_original_class_to_v9(orig_class):
    """Map original class name to (cls_id, has_kpts, observability)."""
    if orig_class in EXCLUDED_CATS:
        return None

    if orig_class == "nozzle_spray":
        return (SPRAY_IDX, True, 1.0)
    if orig_class == "nozzle_no_spray":
        return (NO_SPRAY_IDX, False, 0.0)
    if orig_class == "nozzle_facing":
        return (FACING_IDX, False, 0.0)
    if orig_class == "big_nozzle":
        return (BIG_IDX, False, 0.0)

    return None


def flip_keypoints_horizontal(kp):
    """
    Horizontally flip keypoints AND swap left <-> right identity.

    Keypoint layout: [tip_x, tip_y, left_x, left_y, right_x, right_y]

    When the image is mirrored, the pixel previously labeled "left edge of
    cone" is now physically on the right side, so its identity must become
    "right edge". Without this swap, the model receives contradictory
    supervision and collapses to predicting averaged positions.

    Mutates `kp` in place.
    """
    tip_x, tip_y = 1.0 - kp[0], kp[1]
    # Pixel that was 'left' (kp[2:4]) is now on the right -> becomes right kp
    new_right_x, new_right_y = 1.0 - kp[2], kp[3]
    # Pixel that was 'right' (kp[4:6]) is now on the left -> becomes left kp
    new_left_x, new_left_y = 1.0 - kp[4], kp[5]
    kp[0], kp[1] = tip_x, tip_y
    kp[2], kp[3] = new_left_x, new_left_y
    kp[4], kp[5] = new_right_x, new_right_y


def cache_path_for(image_path):
    """Where build_cache.py stores the pre-resized copy of a source frame."""
    p = Path(image_path).resolve()
    try:
        rel = p.relative_to(REPO_ROOT)
    except ValueError:
        rel = Path(p.name)
    return IMG_CACHE_DIR / rel.with_suffix(".png")


def mist_path_for(image_path):
    """Where build_mist_masks.py stores the pseudo mist mask (640x480 uint8)."""
    p = Path(image_path).resolve()
    try:
        rel = p.relative_to(REPO_ROOT)
    except ValueError:
        rel = Path(p.name)
    return MIST_CACHE_DIR / rel.with_suffix(".png")


def load_frame(image_path):
    """RGB uint8 frame. With USE_IMG_CACHE the pre-resized copy is used when
    it exists (falls back to the source file, so a partial cache never
    changes labels - only speed)."""
    src = image_path
    if USE_IMG_CACHE:
        c = cache_path_for(image_path)
        if c.exists():
            src = str(c)
    image = cv2.imread(src, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot load: {src}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


import re

_NUM_RE = re.compile(r"^(.*?)(\d+)(\D*)$")


def prev_frame_path(image_path, gap=FRAME_GAP):
    """Path of frame t-gap in the same directory (step 4, TWO_FRAME).

    Every source video is extracted with contiguous numbering
    (frame_000123.jpg, 20250509_000230.jpg ...), so the previous frame is the
    same name with the number decremented. Falls back to the nearest earlier
    frame that exists, then to the frame itself (zero motion) so no sample is
    ever dropped. The neighbour need not be annotated - only its pixels are used.
    """
    p = Path(image_path)
    m = _NUM_RE.match(p.name)
    if not m:
        return image_path
    head, num, tail = m.groups()
    for g in range(gap, 0, -1):
        n = int(num) - g
        if n < 0:
            break
        cand = p.with_name(f"{head}{n:0{len(num)}d}{tail}")
        if cand.exists():
            return str(cand)
    return image_path


class RandomGamma:
    """Picklable random gamma (a lambda inside the Dataset cannot be sent to
    spawned DataLoader workers on Windows - that crashed the first 2G run)."""

    def __init__(self, lo, hi):
        self.lo, self.hi = lo, hi

    def __call__(self, t):
        return TF.adjust_gamma(t, random.uniform(self.lo, self.hi))


def photometric_ops():
    """Train-time colour transforms, applied to the [0,1] tensor before
    Normalize. Default = the v9 ColorJitter. PHOTO_AUG (2G) makes absolute
    brightness an unreliable cue: random gamma, stronger brightness/contrast
    jitter, and occasional grayscale."""
    if not PHOTO_AUG:
        return [T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)]
    return [
        RandomGamma(*PHOTO_AUG_GAMMA),
        T.ColorJitter(brightness=PHOTO_AUG_BRIGHTNESS, contrast=PHOTO_AUG_CONTRAST, saturation=0.2),
        T.RandomGrayscale(p=PHOTO_AUG_GRAY_P),
    ]


def mist_targets(mist_u8, gt_boxes, gt_classes):
    """Pseudo mist target and loss ROI on the mist head's grid.

    target: soft mask (0..1), area-averaged to IMG/MIST_STRIDE.
    roi:    1 inside a disk of MIST_ROI_FACTOR * box size around every
            labelled SMALL nozzle (spray / no_spray / facing - the ones whose
            state we must decide), EXCLUDING every box interior (lance body
            glare must not be learned as mist), 0 elsewhere - so the big
            central spray column and rim reflections never enter the loss.
    """
    h, w = IMG_H // MIST_STRIDE, IMG_W // MIST_STRIDE
    tgt = cv2.resize(mist_u8, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    roi = np.zeros((h, w), np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    for (cx, cy, bw, bh), c in zip(gt_boxes, gt_classes):
        if c == BIG_IDX:
            continue
        cxp, cyp, bwp, bhp = cx * w, cy * h, bw * w, bh * h
        r = MIST_ROI_FACTOR * max(bwp, bhp)
        roi[np.hypot(xx + 0.5 - cxp, yy + 0.5 - cyp) < r] = 1.0
    for cx, cy, bw, bh in gt_boxes:
        cxp, cyp, bwp, bhp = cx * w, cy * h, bw * w, bh * h
        roi[(np.abs(xx + 0.5 - cxp) < bwp / 2) & (np.abs(yy + 0.5 - cyp) < bhp / 2)] = 0.0
    return torch.from_numpy(tgt), torch.from_numpy(roi)


def tip_of_box(box):
    """Tip of a lance box = midpoint of the box edge nearest the image centre
    (lances enter from the border). Normalised cxcywh -> (x, y) normalised."""
    cx, cy, bw, bh = box
    corners = np.array([[cx - bw / 2, cy - bh / 2], [cx + bw / 2, cy - bh / 2], [cx - bw / 2, cy + bh / 2], [cx + bw / 2, cy + bh / 2]])
    d = np.linalg.norm((corners - [0.5, 0.5]) * [IMG_W, IMG_H], axis=1)
    return corners[np.argsort(d)[:2]].mean(0)


def instance_targets(gt_boxes, gt_classes, gt_kpts):
    """Fixed-size per-image targets for the learned map readers (step 7).

    inst_tip   [MAX_INST,2] normalised tip: annotated tip for spray, box tip otherwise
    inst_axis  [MAX_INST]   spray axis in deg (isotropic plane): bisector of the
                            annotated edges for spray; box-centre -> tip otherwise
    inst_state [MAX_INST]   1 spraying (spray/facing), 0 no_spray, -1 ignore (big / empty slot)
    inst_angle [MAX_INST]   apex angle in rad for spray, -1 otherwise
    inst_valid [MAX_INST]   1 for a real small-nozzle instance
    """
    tip = np.zeros((MAX_INST, 2), np.float32)
    axis = np.zeros(MAX_INST, np.float32)
    state = np.full(MAX_INST, -1.0, np.float32)
    angle = np.full(MAX_INST, -1.0, np.float32)
    valid = np.zeros(MAX_INST, np.float32)
    n = 0
    for b, c, k in zip(gt_boxes, gt_classes, gt_kpts):
        if n >= MAX_INST or c == BIG_IDX:
            continue
        if c == SPRAY_IDX and any(v > 0 for v in k):
            kp = np.array(k, np.float64).reshape(3, 2)
            t = kp[0]
            iso = (kp - t) * [ANGLE_ASPECT, 1.0]  # to isotropic plane
            v = iso[1] + iso[2]
            ax = math.degrees(math.atan2(v[1], v[0]))
            a = _apex_angle_px(kp * [IMG_W, IMG_H])
            angle[n] = math.radians(a) if a is not None else -1.0
        else:
            t = tip_of_box(b)
            v = (t - np.array(b[:2])) * [ANGLE_ASPECT, 1.0]
            ax = math.degrees(math.atan2(v[1], v[0])) if np.linalg.norm(v) > 1e-6 else 0.0
        tip[n] = t
        axis[n] = ax
        state[n] = 0.0 if c == NO_SPRAY_IDX else 1.0
        valid[n] = 1.0
        n += 1
    return {
        "inst_tip": torch.from_numpy(tip),
        "inst_axis": torch.from_numpy(axis),
        "inst_state": torch.from_numpy(state),
        "inst_angle": torch.from_numpy(angle),
        "inst_valid": torch.from_numpy(valid),
    }


def _apex_angle_px(kp):
    """Apex angle (deg) from a 3x2 pixel keypoint array [tip, left, right]."""
    v1, v2 = kp[1] - kp[0], kp[2] - kp[0]
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-9:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))))


def random_geo_affine(image, gt_boxes, gt_kpts, gt_classes, rng=random, max_tries=5, extra=None):
    """v10 GEO_AUG: random rotation + isotropic scale + shift, in source pixels.

    Boxes become the axis-aligned bbox of their transformed corners, then are
    clipped to the frame exactly like the loader clips border instances.
    Keypoints are mapped with the same affine. A draw is rejected (and
    redrawn) if any box centre or any spray keypoint leaves the frame; after
    max_tries the image is returned untouched.

    Guarantees checked on every accepted draw:
      * the tip stays inside its (clipped) box
      * the apex angle is unchanged (< 0.1 deg) - rotation and isotropic
        scale cannot alter it; this assert is what protects against an
        accidental anisotropic scale.
    Edge keypoints are NOT required to lie inside the box: by annotation
    convention they never do.

    `extra`: optional list of companion images (e.g. the previous frame for
    TWO_FRAME) warped with the SAME affine; returned as a 4th element.
    """
    H, W = image.shape[:2]
    for _ in range(max_tries):
        ang = rng.uniform(-GEO_AUG_ROT_DEG, GEO_AUG_ROT_DEG)
        sc = rng.uniform(*GEO_AUG_SCALE)
        M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), ang, sc)
        M[0, 2] += rng.uniform(-GEO_AUG_SHIFT_FRAC, GEO_AUG_SHIFT_FRAC) * W
        M[1, 2] += rng.uniform(-GEO_AUG_SHIFT_FRAC, GEO_AUG_SHIFT_FRAC) * H
        aniso = False
        if ANGLE_AUG and any(c == SPRAY_IDX for c in gt_classes):
            # 3c: horizontal stretch about the image centre, composed AFTER the
            # rigid part. This changes the apex angle on purpose; the label
            # follows from the transformed keypoints, so the invariance assert
            # below is skipped for this draw.
            sx = rng.uniform(*ANGLE_AUG_SCALE)
            S = np.array([[sx, 0.0, (1.0 - sx) * W / 2.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            M = (S @ np.vstack([M, [0.0, 0.0, 1.0]]))[:2]
            aniso = True
        A, t = M[:, :2].T, M[:, 2]

        new_boxes, new_kpts, ok = [], [], True
        for b, k, c in zip(gt_boxes, gt_kpts, gt_classes):
            cx, cy, bw, bh = b[0] * W, b[1] * H, b[2] * W, b[3] * H
            corners = np.array(
                [[cx - bw / 2, cy - bh / 2], [cx + bw / 2, cy - bh / 2],
                 [cx + bw / 2, cy + bh / 2], [cx - bw / 2, cy + bh / 2]]
            )
            pc = corners @ A + t
            x1, y1, x2, y2 = pc[:, 0].min(), pc[:, 1].min(), pc[:, 0].max(), pc[:, 1].max()
            if not (0 <= (x1 + x2) / 2 < W and 0 <= (y1 + y2) / 2 < H):
                ok = False
                break
            x1c, y1c, x2c, y2c = max(0.0, x1), max(0.0, y1), min(float(W), x2), min(float(H), y2)
            if x2c - x1c < 2 or y2c - y1c < 2:
                ok = False
                break
            new_boxes.append(
                [(x1c + x2c) / 2 / W, (y1c + y2c) / 2 / H, (x2c - x1c) / W, (y2c - y1c) / H]
            )

            if c == SPRAY_IDX and any(v > 0 for v in k):
                kp = np.array(k, dtype=np.float64).reshape(3, 2) * [W, H]
                pk = kp @ A + t
                if not ((pk[:, 0] >= 0) & (pk[:, 0] < W) & (pk[:, 1] >= 0) & (pk[:, 1] < H)).all():
                    ok = False
                    break
                assert x1c - 1e-6 <= pk[0, 0] <= x2c + 1e-6 and y1c - 1e-6 <= pk[0, 1] <= y2c + 1e-6, "tip left its box"
                a0, a1 = _apex_angle_px(kp), _apex_angle_px(pk)
                assert aniso or a0 is None or a1 is None or abs(a0 - a1) < 0.1, f"apex angle changed {a0:.2f}->{a1:.2f}"
                new_kpts.append((pk / [W, H]).ravel().tolist())
            else:
                new_kpts.append(list(k))

        if ok:
            warp = lambda im: cv2.warpAffine(im, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
            image = warp(image)
            if extra is not None:
                return image, new_boxes, new_kpts, [warp(e) for e in extra]
            return image, new_boxes, new_kpts
    if extra is not None:
        return image, gt_boxes, gt_kpts, list(extra)
    return image, gt_boxes, gt_kpts


class AFGNetDataset(Dataset):
    """v10 dataset: loads COCO split JSON and uses cxcywh bbox format."""

    def __init__(self, json_path, augment=False, image_size=IMG_SIZE):
        self.json_path = Path(json_path)
        self.augment = augment
        self.image_size = image_size
        self.samples = []
        self.skipped_no_image = 0
        self.skipped_no_anns = 0
        self.aspect_mismatch = 0  # frames whose W/H differs from ANGLE_ASPECT

        # (IMG_H, IMG_W) = 480x640 with LETTERBOX: the 4:3 frame is resized
        # without distortion, so normalised coordinates are unchanged and
        # grid cells are square in source pixels (51.2 px).
        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Resize((IMG_H, IMG_W), antialias=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.aug_transform = T.Compose(
            [
                T.ToTensor(),
                T.Resize((IMG_H, IMG_W), antialias=True),
                *photometric_ops(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self._load_split()
        self._two_frame_audit()

    # ── input construction (shared by training and every evaluation script) ──
    def make_input(self, image, prev=None, augment=False):
        """RGB uint8 frame(s) -> model tensor. With TWO_FRAME the two frames are
        stacked vertically, pushed through ONE transform (so ColorJitter is
        identical for both - a differing jitter would be fake motion), then
        split and concatenated along channels -> [6, H, W]."""
        tf = self.aug_transform if augment else self.transform
        if prev is None:
            return tf(image)
        if not hasattr(self, "_tf2"):
            mk = lambda jitter: T.Compose(
                [T.ToTensor(), T.Resize((2 * IMG_H, IMG_W), antialias=True)]
                + (photometric_ops() if jitter else [])
                + [T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            )
            self._tf2 = {False: mk(False), True: mk(True)}
        if prev.shape != image.shape:
            prev = cv2.resize(prev, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
        t = self._tf2[augment](np.concatenate([image, prev], axis=0))
        return torch.cat([t[:, :IMG_H], t[:, IMG_H:]], dim=0)

    @staticmethod
    def frames_for(sample):
        """(frame, previous frame or None) as RGB uint8, cache-aware."""
        img = load_frame(sample["image_path"])
        prev = load_frame(prev_frame_path(sample["image_path"])) if TWO_FRAME else None
        return img, prev

    def _two_frame_audit(self):
        """TWO_FRAME only makes sense where a real earlier frame exists. The
        CAM1/CAM2 sources are 15-second stills named
        '20250429-140936-V1 CAM#2.png' (no frame counter), so prev_frame_path
        falls back to the frame itself there. Say so loudly instead of
        training on duplicated frames (this silently invalidated the first
        step-4 run on 2026-09-17)."""
        if not TWO_FRAME:
            return
        per_src = defaultdict(lambda: [0, 0])
        for s in self.samples:
            same = prev_frame_path(s["image_path"]) == s["image_path"]
            per_src[s["source_video"]][0] += int(same)
            per_src[s["source_video"]][1] += 1
        print(f"  TWO_FRAME audit ({self.json_path.name}): frames whose previous frame is MISSING (fallback = same frame):")
        for src, (same, n) in sorted(per_src.items()):
            flag = "   <-- no motion signal for this source" if same == n else ""
            print(f"    {src:<28} {same:5d}/{n:<5d}{flag}")

    def _load_split(self):
        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        images_lookup = {img["id"]: img for img in data.get("images", [])}

        img_to_anns = defaultdict(list)
        for ann in data.get("annotations", []):
            img_to_anns[ann["image_id"]].append(ann)

        for img_id, anns in img_to_anns.items():
            info = images_lookup.get(img_id)
            if info is None:
                continue

            try:
                source_hint = info.get("source_video", None)
                img_path = resolve_image_path(info["file_name"], source_hint)
            except FileNotFoundError:
                self.skipped_no_image += 1
                continue

            orig_w = float(info.get("width", 1))
            orig_h = float(info.get("height", 1))
            if abs(orig_w / orig_h - ANGLE_ASPECT) > 1e-3:
                if self.aspect_mismatch == 0:
                    print(
                        f"WARNING: {info['file_name']} is {orig_w:.0f}x{orig_h:.0f}; "
                        f"aspect differs from ANGLE_ASPECT={ANGLE_ASPECT:.4f}. "
                        "Spray angles for such frames will be biased."
                    )
                self.aspect_mismatch += 1
            gt_boxes, gt_classes = [], []
            gt_kpts, gt_obs = [], []

            for ann in anns:
                raw_bbox = ann.get("bbox", [])
                if len(raw_bbox) < 4:
                    continue
                try:
                    bbox = [float(v) for v in raw_bbox[:4]]
                except (ValueError, TypeError):
                    continue
                if bbox[2] <= 1 or bbox[3] <= 1:
                    continue

                x, y, bw, bh = bbox

                # Clamp to image bounds
                x1 = max(0.0, x)
                y1 = max(0.0, y)
                x2 = min(orig_w, x + bw)
                y2 = min(orig_h, y + bh)
                clamped_w = x2 - x1
                clamped_h = y2 - y1
                if clamped_w <= 1 or clamped_h <= 1:
                    continue

                cx = (x1 + clamped_w / 2.0) / orig_w
                cy = (y1 + clamped_h / 2.0) / orig_h
                nw = clamped_w / orig_w
                nh = clamped_h / orig_h

                cx = float(np.clip(cx, 0.0, 1.0))
                cy = float(np.clip(cy, 0.0, 1.0))
                nw = float(np.clip(nw, 0.001, 1.0))
                nh = float(np.clip(nh, 0.001, 1.0))

                # Get original class
                orig_class = ann.get("_meta", {}).get("original_class", None)
                if orig_class is None:
                    continue

                mapped = map_original_class_to_v9(orig_class)
                if mapped is None:
                    continue

                cls_id, has_kpts, obs_val = mapped

                # Keypoints
                if has_kpts:
                    raw_kpts = ann.get("keypoints", [])
                    kp_arr = []
                    for i in range(NUM_KEYPOINTS):
                        try:
                            if len(raw_kpts) >= (i + 1) * 3:
                                kx = float(raw_kpts[i * 3]) / orig_w
                                ky = float(raw_kpts[i * 3 + 1]) / orig_h
                            else:
                                kx = ky = 0.0
                        except (ValueError, TypeError):
                            kx = ky = 0.0
                        kp_arr.extend(
                            [
                                float(np.clip(kx, 0.0, 1.0)),
                                float(np.clip(ky, 0.0, 1.0)),
                            ]
                        )
                else:
                    kp_arr = [0.0] * (NUM_KEYPOINTS * 2)

                gt_boxes.append([cx, cy, nw, nh])
                gt_classes.append(cls_id)
                gt_kpts.append(kp_arr)
                gt_obs.append(obs_val)

            if not gt_boxes:
                self.skipped_no_anns += 1
                continue

            self.samples.append(
                {
                    "image_path": str(img_path),
                    "gt_boxes": gt_boxes,
                    "gt_classes": gt_classes,
                    "gt_kpts": gt_kpts,
                    "gt_obs": gt_obs,
                    "source_video": info.get("source_video", "unknown"),
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        image, prev = self.frames_for(s)
        mist = None
        if MIST_HEAD:
            mist = cv2.imread(str(mist_path_for(s["image_path"])), cv2.IMREAD_GRAYSCALE)
            if mist is None:
                raise FileNotFoundError(f"mist pseudo-mask missing (run build_mist_masks.py): {mist_path_for(s['image_path'])}")
            if mist.shape != image.shape[:2]:
                mist = cv2.resize(mist, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)

        gt_boxes = [list(b) for b in s["gt_boxes"]]
        gt_kpts = [list(k) for k in s["gt_kpts"]]
        gt_classes = list(s["gt_classes"])
        gt_obs = list(s["gt_obs"])

        if self.augment and GEO_AUG:
            extra = [e for e in (prev, mist) if e is not None]
            if not extra:
                image, gt_boxes, gt_kpts = random_geo_affine(image, gt_boxes, gt_kpts, gt_classes)
            else:
                image, gt_boxes, gt_kpts, warped = random_geo_affine(image, gt_boxes, gt_kpts, gt_classes, extra=extra)
                if prev is not None:
                    prev = warped.pop(0)
                if mist is not None:
                    mist = warped.pop(0)

        if self.augment:
            # Horizontal flip (p=0.5) with proper left<->right kpt swap
            if random.random() < 0.5:
                image = image[:, ::-1, :].copy()
                if prev is not None:
                    prev = prev[:, ::-1, :].copy()
                if mist is not None:
                    mist = mist[:, ::-1].copy()
                for b in gt_boxes:
                    b[0] = 1.0 - b[0]
                for kp, cls_id in zip(gt_kpts, gt_classes):
                    # Only swap for classes that actually have keypoints (spray)
                    if cls_id == SPRAY_IDX:
                        flip_keypoints_horizontal(kp)
                    # For other classes kp is all zeros -> nothing to flip

        img_t = self.make_input(image, prev, augment=self.augment)

        targets = build_fcos_targets(gt_boxes, gt_classes, gt_kpts, gt_obs)
        # Sample index so the standard (instance-level) evaluation can read the
        # raw GT of this image instead of reconstructing it from cell maps.
        targets["index"] = torch.tensor(idx, dtype=torch.long)
        if mist is not None:
            targets["mist"], targets["mist_roi"] = mist_targets(mist, gt_boxes, gt_classes)
        if LEARNED_HEADS:
            targets.update(instance_targets(gt_boxes, gt_classes, gt_kpts))
        return img_t, targets

    def print_stats(self):
        class_counter = Counter()
        kpt_count = 0
        per_source = defaultdict(int)

        for s in self.samples:
            for cls_id in s["gt_classes"]:
                class_counter[CLASS_NAMES[cls_id]] += 1
            kpt_count += sum(int(o > 0.5) for o in s["gt_obs"])
            per_source[s["source_video"]] += len(s["gt_classes"])

        total = sum(class_counter.values())
        print(f"\nDataset: {self.json_path.name}")
        print(f"  Samples loaded:       {len(self.samples)}")
        print(f"  Skipped (no image):   {self.skipped_no_image}")
        print(f"  Skipped (no anns):    {self.skipped_no_anns}")
        print(f"  Total annotations:    {total}")
        print(f"  Spray (with kpts):    {kpt_count}")
        print(f"  Class distribution:")
        for name in CLASS_NAMES:
            n = class_counter.get(name, 0)
            pct = 100 * n / total if total else 0
            print(f"    {name:<22}: {n:>6} ({pct:.1f}%)")
        if len(per_source) <= 12:
            print(f"  Per-source breakdown:")
            for src in sorted(per_source.keys()):
                print(f"    {src:<35}: {per_source[src]} total")


def afgnet_collate(batch):
    images = torch.stack([b[0] for b in batch], dim=0)
    keys = batch[0][1].keys()
    targets = {k: torch.stack([b[1][k] for b in batch], dim=0) for k in keys}
    return images, targets


if __name__ == "__main__":
    from config import TEST_JSON, TRAIN_JSON, VAL_JSON

    for name, path in [("TRAIN", TRAIN_JSON), ("VAL", VAL_JSON), ("TEST", TEST_JSON)]:
        if path.exists():
            print(f'\n{"=" * 60}\nTesting {name}: {path}\n{"=" * 60}')
            ds = AFGNetDataset(path, augment=False)
            ds.print_stats()

            if len(ds) > 0:
                img, tgts = ds[0]
                print(f"\n  First sample:")
                print(f"    image shape: {tuple(img.shape)}")
                print(f'    bbox shape: {tuple(tgts["bbox"].shape)}')
                print(f'    pos_mask sum: {tgts["pos_mask"].sum().item():.0f}')
        else:
            print(f"\n{name}: file not found - {path}")
