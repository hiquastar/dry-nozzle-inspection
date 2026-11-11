"""
dataset.py - AFGNet v9 dataset (loads COCO splits)

PATCHED (May 11): Horizontal flip now correctly swaps left <-> right
keypoint identities. The original code only flipped x-coordinates,
which caused contradictory supervision (the "left" label sometimes
pointed to right-side pixels and vice versa).

Loads COCO split JSON (data/splits_reverse/train.coco.json etc; reverse split, CAM2 train / CAM1 val+test) and builds
cxcywh bbox-format targets.

Maps original_class to v9 class index:
  big_nozzle      -> 3 (BIG_IDX)
  nozzle_no_spray -> 1 (NO_SPRAY_IDX)
  nozzle_spray    -> 0 (SPRAY_IDX)  + has keypoints + obs=1
  nozzle_facing   -> 2 (FACING_IDX) + obs=0
  nozzle_buildup  -> SKIPPED (excluded)
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from targets import build_fcos_targets
from torch.utils.data import Dataset

from config import (
    ANGLE_ASPECT,
    BIG_IDX,
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


class AFGNetDataset(Dataset):
    """v9 dataset: loads COCO split JSON and uses cxcywh bbox format."""

    def __init__(self, json_path, augment=False, image_size=IMG_SIZE):
        self.json_path = Path(json_path)
        self.augment = augment
        self.image_size = image_size
        self.samples = []
        self.skipped_no_image = 0
        self.skipped_no_anns = 0
        self.aspect_mismatch = 0  # frames whose W/H differs from ANGLE_ASPECT

        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Resize((image_size, image_size), antialias=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.aug_transform = T.Compose(
            [
                T.ToTensor(),
                T.Resize((image_size, image_size), antialias=True),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self._load_split()

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
        image = cv2.imread(s["image_path"], cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot load: {s['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        gt_boxes = [list(b) for b in s["gt_boxes"]]
        gt_kpts = [list(k) for k in s["gt_kpts"]]
        gt_classes = list(s["gt_classes"])
        gt_obs = list(s["gt_obs"])

        if self.augment:
            # Horizontal flip (p=0.5) with proper left<->right kpt swap
            if random.random() < 0.5:
                image = image[:, ::-1, :].copy()
                for b in gt_boxes:
                    b[0] = 1.0 - b[0]
                for kp, cls_id in zip(gt_kpts, gt_classes):
                    # Only swap for classes that actually have keypoints (spray)
                    if cls_id == SPRAY_IDX:
                        flip_keypoints_horizontal(kp)
                    # For other classes kp is all zeros -> nothing to flip

            img_t = self.aug_transform(image)
        else:
            img_t = self.transform(image)

        targets = build_fcos_targets(gt_boxes, gt_classes, gt_kpts, gt_obs)
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
