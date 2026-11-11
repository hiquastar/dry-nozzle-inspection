"""
metrics.py - Evaluation metrics for AFGNet v9 (single-scale + class-aware)

FIXES (vs v9_v1 metrics):
  1. CLASS-AWARE mAP: matching now considers class_id (was class-agnostic, stuck at 0.7273)
  2. Combined score: score = centerness * iou_quality * cls_confidence
     (was just centerness * iou_quality - cls improvements not reflected)
  3. GT classes tracked per image for proper class-wise AP computation
  4. Standard COCO/PASCAL VOC per-class AP averaged to mAP

KEEPS v9_v1 single-scale architecture - decode just one feature map.
"""

import math

import numpy as np
import torch

from config import (
    ANGLE_ASPECT,
    FACING_IDX,
    FEAT_H,
    FEAT_W,
    NMS_IOU_THRESH,
    NUM_CLASSES,
    NUM_KEYPOINTS,
    OBJ_THRESH,
    OBS_THRESH,
    SPRAY_IDX,
)


def box_iou(a, b):
    """IoU between 2 boxes in (x1, y1, x2, y2) format."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-7)


def cxcywh_to_xyxy(box):
    """(cx, cy, w, h) -> (x1, y1, x2, y2)."""
    cx, cy, w, h = box
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def nms(boxes, scores, iou_thresh=0.5):
    """Greedy NMS. Returns kept indices."""
    if len(boxes) == 0:
        return []
    order = np.argsort(-scores)
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        ious = np.array([box_iou(boxes[i], boxes[j]) for j in rest])
        order = rest[ious < iou_thresh]
    return keep


def decode_predictions(
    preds, batch_idx=0, obj_thresh=OBJ_THRESH, nms_iou=NMS_IOU_THRESH
):
    """Decode raw model output -> list of dicts for one image.

    SINGLE-SCALE decode (keeps v9_v1 architecture).

    FIX: score now combines objectness with class confidence:
         score = centerness * iou_quality * max_class_prob
    """
    centerness = preds["centerness"][batch_idx, 0].detach().cpu().numpy()
    bbox = preds["bbox"][batch_idx].detach().cpu().numpy()
    iou_q = preds["iou_quality"][batch_idx, 0].detach().cpu().numpy()
    cls_logits = preds["class_logits"][batch_idx].detach().cpu().numpy()
    # Abl D removes the observability head. Fall back to NaN sentinel so
    # downstream consumers that gate by 'observability' can detect absence.
    if "observability" in preds:
        obs = preds["observability"][batch_idx, 0].detach().cpu().numpy()
    else:
        obs = np.full_like(centerness, float("nan"))
    keypoints = preds["keypoints"][batch_idx].detach().cpu().numpy()

    H, W = centerness.shape
    obj_score_map = centerness * iou_q

    mask = obj_score_map >= obj_thresh
    if not mask.any():
        return []

    iy, ix = np.where(mask)
    candidates = []

    for r, c in zip(iy, ix):
        cx = float(bbox[0, r, c])
        cy = float(bbox[1, r, c])
        bw = float(bbox[2, r, c])
        bh = float(bbox[3, r, c])

        x1 = max(0.0, cx - bw / 2)
        y1 = max(0.0, cy - bh / 2)
        x2 = min(1.0, cx + bw / 2)
        y2 = min(1.0, cy + bh / 2)
        if x2 <= x1 or y2 <= y1:
            continue

        cls_probs = 1.0 / (1.0 + np.exp(-cls_logits[:, r, c]))
        cls_id = int(np.argmax(cls_probs))
        cls_conf = float(cls_probs[cls_id])

        # FIX: Combined score includes class confidence
        obj_score = float(obj_score_map[r, c])
        combined_score = obj_score * cls_conf

        candidates.append(
            {
                "box": (x1, y1, x2, y2),
                "score": combined_score,
                "obj_score": obj_score,
                "cls_conf": cls_conf,
                "class_id": cls_id,
                "class_probs": cls_probs.tolist(),
                "observability": float(obs[r, c]),
                "keypoints": keypoints[:, r, c].tolist(),
            }
        )

    if not candidates:
        return []

    boxes = np.array([c["box"] for c in candidates])
    scores = np.array([c["score"] for c in candidates])
    keep = nms(boxes, scores, nms_iou)

    return [candidates[i] for i in keep]


def compute_spray_angle(kpts, aspect=ANGLE_ASPECT):
    """Compute spray apex angle (deg) from 6 normalised keypoint coords.

    Keypoints are normalised by W and H separately, so x is multiplied by
    `aspect` (= W / H) to return to an isotropic (pixel-like) plane before
    taking the angle. Pass aspect=1.0 to reproduce the pre-2026-09-16
    behaviour (angle measured in the squashed plane).
    """
    kpts = np.asarray(kpts, dtype=np.float64)
    if not np.all(np.isfinite(kpts)) or len(kpts) < 6:
        return None
    kpts = kpts.copy()
    kpts[0::2] *= aspect
    tip = kpts[0:2]
    left = kpts[2:4]
    right = kpts[4:6]
    v1 = left - tip
    v2 = right - tip
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos_a = np.dot(v1, v2) / (n1 * n2)
    cos_a = np.clip(cos_a, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


class MetricsAccumulator:
    """Accumulates predictions across validation epoch (single-scale)."""

    def __init__(self):
        self.cls_y_true = []
        self.cls_y_pred = []
        self.obs_y_true = []
        self.obs_y_pred = []

        self.pred_per_image = []
        self.gt_per_image = []
        self.gt_classes_per_image = []  # FIX: track classes per GT box

        self.angle_errors = []

    def update_from_batch(self, preds, targets, device):
        """Aggregate metrics from one batch."""
        B = preds["centerness"].shape[0]

        pos_mask = targets["pos_mask"].to(device).bool().view(-1)
        if pos_mask.sum() == 0:
            return

        def flat(t, ch):
            return t.permute(0, 2, 3, 1).reshape(-1, ch)[pos_mask]

        # Class
        cls_true = targets["class_ids"].to(device).view(-1)[pos_mask].cpu().numpy()
        cls_pred = flat(preds["class_logits"], NUM_CLASSES).argmax(dim=1).cpu().numpy()
        self.cls_y_true.extend(cls_true.tolist())
        self.cls_y_pred.extend(cls_pred.tolist())

        # Observability (only for spraying classes).
        # Skip entirely for Abl D, which has no observability head.
        if "observability" in preds:
            spraying_mask = (cls_true == SPRAY_IDX) | (cls_true == FACING_IDX)
            if spraying_mask.sum() > 0:
                obs_pred_all = flat(preds["observability"], 1).squeeze(-1).cpu().numpy()
                obs_true_all = (
                    targets["observability"].to(device).view(-1)[pos_mask].cpu().numpy()
                )
                obs_true_filt = obs_true_all[spraying_mask]
                obs_pred_filt = (obs_pred_all[spraying_mask] > 0.5).astype(int)
                self.obs_y_true.extend(obs_true_filt.astype(int).tolist())
                self.obs_y_pred.extend(obs_pred_filt.tolist())

        # Detection mAP
        for b in range(B):
            decoded = decode_predictions(preds, batch_idx=b)
            self.pred_per_image.append(decoded)

            # FIX: GT boxes AND classes
            gt_boxes = []
            gt_classes = []
            pm = targets["pos_mask"][b].bool().view(-1)
            if pm.sum() > 0:
                bb = targets["bbox"][b].view(-1, 4)[pm].cpu().numpy()
                cc = targets["class_ids"][b].view(-1)[pm].cpu().numpy()
                seen = set()
                for box, cls_id in zip(bb, cc):
                    key = tuple(np.round(box, 4))
                    if key not in seen:
                        seen.add(key)
                        gt_boxes.append(box)
                        gt_classes.append(int(cls_id))
            self.gt_per_image.append(gt_boxes)
            self.gt_classes_per_image.append(gt_classes)

        # Angle MAE
        spray_mask = cls_true == SPRAY_IDX
        if spray_mask.sum() > 0:
            kpt_pred = (
                flat(preds["keypoints"], NUM_KEYPOINTS * 2)[
                    torch.from_numpy(spray_mask).bool()
                ]
                .cpu()
                .numpy()
            )
            kpt_true = (
                targets["keypoints"]
                .to(device)
                .view(-1, NUM_KEYPOINTS * 2)[pos_mask][
                    torch.from_numpy(spray_mask).bool()
                ]
                .cpu()
                .numpy()
            )
            for kp_p, kp_t in zip(kpt_pred, kpt_true):
                a_p = compute_spray_angle(kp_p)
                a_t = compute_spray_angle(kp_t)
                if a_p is not None and a_t is not None:
                    self.angle_errors.append(abs(a_p - a_t))

    def compute_summary(self, iou_thresh_map=0.5):
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        results = {}

        if self.cls_y_true:
            p, r, f1, _ = precision_recall_fscore_support(
                self.cls_y_true,
                self.cls_y_pred,
                labels=list(range(NUM_CLASSES)),
                average="weighted",
                zero_division=0,
            )
            results["cls_precision"] = float(p)
            results["cls_recall"] = float(r)
            results["cls_f1"] = float(f1)
            results["cls_accuracy"] = float(
                accuracy_score(self.cls_y_true, self.cls_y_pred)
            )

        if self.obs_y_true:
            p, r, f1, _ = precision_recall_fscore_support(
                self.obs_y_true,
                self.obs_y_pred,
                average="binary",
                zero_division=0,
            )
            results["obs_precision"] = float(p)
            results["obs_recall"] = float(r)
            results["obs_f1"] = float(f1)
            results["obs_n_samples"] = len(self.obs_y_true)
            results["obs_n_positive"] = int(sum(self.obs_y_true))

        if self.angle_errors:
            results["angle_mae_deg"] = float(np.mean(self.angle_errors))
            results["angle_n"] = len(self.angle_errors)
        else:
            results["angle_mae_deg"] = float("nan")

        # FIX: Class-aware mAP
        map_results = compute_map_classaware(
            self.pred_per_image,
            self.gt_per_image,
            self.gt_classes_per_image,
            iou_thresh=iou_thresh_map,
            num_classes=NUM_CLASSES,
        )
        results["mAP_50"] = map_results["mAP"]
        for cls_id in range(NUM_CLASSES):
            results[f"AP_50_cls{cls_id}"] = map_results["per_class"][cls_id]

        return results


def compute_ap_single_class(pred_per_image, gt_per_image, iou_thresh=0.5):
    """Compute AP for ONE class. Pascal VOC 11-point interpolation."""
    total_gt = sum(len(gt) for gt in gt_per_image)
    if total_gt == 0:
        return float("nan")

    all_preds = []
    for img_i, preds in enumerate(pred_per_image):
        for p in preds:
            all_preds.append((img_i, p["score"], p["box"]))

    if not all_preds:
        return 0.0

    all_preds.sort(key=lambda x: -x[1])

    gt_matched = [np.zeros(len(gt), dtype=bool) for gt in gt_per_image]
    tp, fp = [], []

    for img_i, score, pred_box in all_preds:
        gt_boxes = gt_per_image[img_i]
        if len(gt_boxes) == 0:
            tp.append(0)
            fp.append(1)
            continue

        ious = np.array([box_iou(pred_box, cxcywh_to_xyxy(gb)) for gb in gt_boxes])
        order = np.argsort(-ious)
        matched_idx = -1
        for j in order:
            if ious[j] < iou_thresh:
                break
            if not gt_matched[img_i][j]:
                matched_idx = j
                break

        if matched_idx >= 0:
            gt_matched[img_i][matched_idx] = True
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    recall = tp / (total_gt + 1e-7)
    precision = tp / (tp + fp + 1e-7)

    ap = 0.0
    for t in np.arange(0, 1.01, 0.1):
        if (recall >= t).any():
            ap += precision[recall >= t].max()
    ap /= 11.0

    return float(ap)


def compute_map_classaware(
    pred_per_image,
    gt_per_image,
    gt_classes_per_image,
    iou_thresh=0.5,
    num_classes=NUM_CLASSES,
):
    """Class-aware mAP. Per-class AP averaged across present classes."""
    if not pred_per_image:
        return {"mAP": 0.0, "per_class": [float("nan")] * num_classes}

    per_class_ap = []
    for cls_id in range(num_classes):
        cls_preds = [
            [p for p in img_preds if p["class_id"] == cls_id]
            for img_preds in pred_per_image
        ]
        cls_gts = [
            [gb for gb, gc in zip(gt_boxes, gt_classes) if gc == cls_id]
            for gt_boxes, gt_classes in zip(gt_per_image, gt_classes_per_image)
        ]

        ap_c = compute_ap_single_class(cls_preds, cls_gts, iou_thresh)
        per_class_ap.append(ap_c)

    valid_aps = [ap for ap in per_class_ap if not math.isnan(ap)]
    mean_ap = float(np.mean(valid_aps)) if valid_aps else 0.0

    return {
        "mAP": mean_ap,
        "per_class": per_class_ap,
    }


# Backward-compat alias
def compute_map(
    pred_per_image, gt_per_image, gt_classes_per_image=None, iou_thresh=0.5
):
    """Class-aware mAP alias."""
    if gt_classes_per_image is None:
        import warnings

        warnings.warn(
            "compute_map called without gt_classes_per_image - falling back to legacy.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _compute_map_classagnostic_LEGACY(
            pred_per_image, gt_per_image, iou_thresh
        )

    result = compute_map_classaware(
        pred_per_image,
        gt_per_image,
        gt_classes_per_image,
        iou_thresh=iou_thresh,
        num_classes=NUM_CLASSES,
    )
    return result["mAP"]


def _compute_map_classagnostic_LEGACY(pred_per_image, gt_per_image, iou_thresh=0.5):
    """LEGACY class-agnostic mAP - backward compat only."""
    if not pred_per_image:
        return 0.0

    all_preds = []
    for img_i, preds in enumerate(pred_per_image):
        for p in preds:
            all_preds.append((img_i, p["score"], p["box"]))
    all_preds.sort(key=lambda x: -x[1])

    total_gt = sum(len(gt) for gt in gt_per_image)
    if total_gt == 0:
        return 0.0

    gt_matched = [np.zeros(len(gt), dtype=bool) for gt in gt_per_image]
    tp, fp = [], []

    for img_i, score, pred_box in all_preds:
        gt_boxes = gt_per_image[img_i]
        if len(gt_boxes) == 0:
            tp.append(0)
            fp.append(1)
            continue

        ious = np.array([box_iou(pred_box, cxcywh_to_xyxy(gb)) for gb in gt_boxes])
        order = np.argsort(-ious)
        matched_idx = -1
        for j in order:
            if ious[j] < iou_thresh:
                break
            if not gt_matched[img_i][j]:
                matched_idx = j
                break

        if matched_idx >= 0:
            gt_matched[img_i][matched_idx] = True
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    recall = tp / (total_gt + 1e-7)
    precision = tp / (tp + fp + 1e-7)

    ap = 0.0
    for t in np.arange(0, 1.01, 0.1):
        if (recall >= t).any():
            ap += precision[recall >= t].max()
    ap /= 11.0

    return float(ap)


if __name__ == "__main__":
    print("metrics.py (class-aware) loaded")

    kpts = [0.5, 0.5, 0.4, 0.6, 0.6, 0.6]
    a = compute_spray_angle(kpts)
    print(f"Test angle: {a:.2f} degrees")

    print("\nSmoke test: class-aware mAP")
    gt_boxes = [[(0.5, 0.5, 0.2, 0.2)]]
    gt_classes = [[0]]
    pred_correct = [[{"box": (0.4, 0.4, 0.6, 0.6), "score": 0.9, "class_id": 0}]]
    pred_wrong = [[{"box": (0.4, 0.4, 0.6, 0.6), "score": 0.9, "class_id": 3}]]

    map_correct = compute_map(pred_correct, gt_boxes, gt_classes, iou_thresh=0.5)
    map_wrong = compute_map(pred_wrong, gt_boxes, gt_classes, iou_thresh=0.5)
    print(f"  Correct class: mAP = {map_correct:.4f}")
    print(f"  Wrong class:   mAP = {map_wrong:.4f}")
    if map_correct > map_wrong:
        print("  PASS: class info now affects mAP")
    else:
        print("  FAIL: still buggy")
