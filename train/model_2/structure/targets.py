"""
targets.py - FCOS target builder for v10

CRITICAL: Uses cxcywh bbox format. NOT ltrb format!

For each cell in the FEAT_H x FEAT_W feature map:
  - If cell center is inside any GT box -> positive
  - Multiple overlapping -> assign to smallest area (FCOS heuristic)
  - Centerness = sqrt(min(l,r)/max(l,r) * min(t,b)/max(t,b))
  - bbox target = (cx, cy, w, h) of assigned box (NOT ltrb distances!)

v10 addition: `inst_weight` = 1 / (number of cells assigned to that instance).
A 0.04 x 0.04 box (the border nozzle) gets ~2 positive cells, a 0.18 x 0.17
box gets ~90, so without this the small instance contributes ~2% of the
gradient of a large one. The loss uses it only when config.INSTANCE_NORM
is on; the map is always built so the target dict has a fixed schema.
"""

import math

import numpy as np
import torch

from config import FEAT_H, FEAT_W, NUM_KEYPOINTS


def build_fcos_targets(gt_boxes, gt_classes, gt_keypoints, gt_observability):
    """Build per-cell FCOS targets for one image (cxcywh format).

    Args:
        gt_boxes: list of [cx, cy, bw, bh] in [0, 1] image coords
        gt_classes: list of int class IDs (0-3)
        gt_keypoints: list of [k1x, k1y, k2x, k2y, k3x, k3y] in [0, 1]
        gt_observability: list of {0, 1} (1 = measurable, 0 = not)

    Returns:
        dict of torch tensors:
          centerness:    [H, W]
          pos_mask:      [H, W]
          bbox:          [H, W, 4]  (cxcywh, NOT ltrb)
          class_ids:     [H, W]
          keypoints:     [H, W, 6]
          observability: [H, W]
          inst_weight:   [H, W]  1/n_cells of the assigned instance, 0 elsewhere
    """
    centerness_map = np.zeros((FEAT_H, FEAT_W), dtype=np.float32)
    pos_mask = np.zeros((FEAT_H, FEAT_W), dtype=np.float32)
    bbox_map = np.zeros((FEAT_H, FEAT_W, 4), dtype=np.float32)
    class_map = np.zeros((FEAT_H, FEAT_W), dtype=np.int64)
    kpt_map = np.zeros((FEAT_H, FEAT_W, NUM_KEYPOINTS * 2), dtype=np.float32)
    obs_map = np.zeros((FEAT_H, FEAT_W), dtype=np.float32)
    assigned_area = np.full((FEAT_H, FEAT_W), 1e9, dtype=np.float32)
    assigned_idx = np.full((FEAT_H, FEAT_W), -1, dtype=np.int64)

    for idx, (cx, cy, bw, bh) in enumerate(gt_boxes):
        if bw <= 0 or bh <= 0:
            continue
        x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
        j_min = max(0, math.floor(x1 * FEAT_W))
        j_max = min(FEAT_W - 1, math.ceil(x2 * FEAT_W - 1))
        i_min = max(0, math.floor(y1 * FEAT_H))
        i_max = min(FEAT_H - 1, math.ceil(y2 * FEAT_H - 1))
        area = bw * bh

        for i in range(i_min, i_max + 1):
            for j in range(j_min, j_max + 1):
                ccx = (j + 0.5) / FEAT_W
                ccy = (i + 0.5) / FEAT_H
                if not (x1 <= ccx <= x2 and y1 <= ccy <= y2):
                    continue
                if area >= assigned_area[i, j]:
                    continue

                assigned_area[i, j] = area
                assigned_idx[i, j] = idx
                l, r = ccx - x1, x2 - ccx
                t, b = ccy - y1, y2 - ccy
                centerness_map[i, j] = math.sqrt(
                    min(l, r) / (max(l, r) + 1e-8) * min(t, b) / (max(t, b) + 1e-8)
                )
                pos_mask[i, j] = 1.0
                # CRITICAL: Store box in cxcywh (NOT ltrb)
                bbox_map[i, j] = [cx, cy, bw, bh]
                class_map[i, j] = gt_classes[idx]
                kpt_map[i, j] = gt_keypoints[idx]
                obs_map[i, j] = gt_observability[idx]

    # Per-instance weight: cells of the same instance share 1/n_cells.
    inst_weight = np.zeros((FEAT_H, FEAT_W), dtype=np.float32)
    if len(gt_boxes):
        counts = np.bincount(assigned_idx[assigned_idx >= 0], minlength=len(gt_boxes))
        pos = assigned_idx >= 0
        inst_weight[pos] = 1.0 / counts[assigned_idx[pos]]

    return {
        "centerness": torch.from_numpy(centerness_map),
        "pos_mask": torch.from_numpy(pos_mask),
        "bbox": torch.from_numpy(bbox_map),
        "class_ids": torch.from_numpy(class_map),
        "keypoints": torch.from_numpy(kpt_map),
        "observability": torch.from_numpy(obs_map),
        "inst_weight": torch.from_numpy(inst_weight),
    }


if __name__ == "__main__":
    fake_boxes = [[0.5, 0.5, 0.2, 0.2], [0.3, 0.7, 0.15, 0.15], [0.9, 0.05, 0.04, 0.04]]
    fake_classes = [0, 2, 0]
    fake_kpts = [
        [0.5, 0.45, 0.45, 0.55, 0.55, 0.55],
        [0.0] * 6,
        [0.9, 0.05, 0.85, 0.15, 0.95, 0.15],
    ]
    fake_obs = [1, 0, 1]

    targets = build_fcos_targets(fake_boxes, fake_classes, fake_kpts, fake_obs)
    print(f"grid {FEAT_H}x{FEAT_W}")
    for k, v in targets.items():
        print(f"  {k:<15}: shape={tuple(v.shape)}, sum={v.sum().item():.3f}")
    n_pos = int(targets["pos_mask"].sum().item())
    w_sum = targets["inst_weight"].sum().item()
    print(f"positive cells: {n_pos}; inst_weight sums to {w_sum:.3f} (= number of instances with >=1 cell)")
