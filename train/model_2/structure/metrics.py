"""
metrics.py - the ONE standard evaluation for AFGNet v10 (model_2)

Everything here is INSTANCE level. The v9 cell-level metrics (one GT box
counted once per positive cell, angle error per cell, observability per
cell) are gone: they inflated N by ~90x, hid a border nozzle that is never
box-matched, and measured the angle in a squashed plane.

Rules (fixed in docs/notes/PLAN-V10.md, section 5):
  detection   class-aware greedy match at IoU >= 0.5, VOC 11-point AP,
              per-class P/R/F1 from the same matching; cls_f1 = macro F1
  observability  on ALL GT spray/facing instances; box-matched to ANY class
              (class-agnostic, so the metric is not a copy of the class),
              pred obs > OBS_THRESH vs GT (spray = 1, facing = 0); a GT
              spray with no box at all counts as a miss (FN)
  angle       GT spray <-> nearest spray prediction by TIP distance
              (<= TIP_MATCH_PX source px, decoded at ANGLE_MATCH_OBJ_THRESH),
              apex angle in PIXEL degrees (x rescaled by W/H)

The same MetricsAccumulator drives checkpoint selection in train.py and
the evaluation scripts, so the model is selected by the metric it is
reported on.
"""

import math

import numpy as np

from config import (
    ANGLE_ASPECT,
    ANGLE_MATCH_OBJ_THRESH,
    CLASS_NAMES,
    FACING_IDX,
    NMS_IOU_THRESH,
    NUM_CLASSES,
    OBJ_THRESH,
    OBS_THRESH,
    ORIG_IMG_H,
    ORIG_IMG_W,
    SPRAY_IDX,
    TIP_MATCH_PX,
)

IOU_MATCH_THRESH = 0.5


# ── geometry ─────────────────────────────────────────────────────────────
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


def compute_spray_angle(kpts, aspect=ANGLE_ASPECT):
    """Apex angle (deg) from 6 normalised keypoint coords [tip, left, right].

    Keypoints are normalised by W and H separately, so x is multiplied by
    `aspect` (= W / H) to return to an isotropic (pixel-like) plane before
    taking the angle. aspect=1.0 reproduces the pre-2026-09-16 squashed unit.
    """
    kpts = np.asarray(kpts, dtype=np.float64)
    if not np.all(np.isfinite(kpts)) or len(kpts) < 6:
        return None
    kpts = kpts.copy()
    kpts[0::2] *= aspect
    tip, left, right = kpts[0:2], kpts[2:4], kpts[4:6]
    v1, v2 = left - tip, right - tip
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


# ── decoding ─────────────────────────────────────────────────────────────
def decode_predictions(preds, batch_idx=0, obj_thresh=OBJ_THRESH, nms_iou=NMS_IOU_THRESH):
    """Raw model output -> list of detections for one image.

    score = centerness * iou_quality * max_class_prob; class-agnostic NMS.
    Keypoints are those of the surviving cell (top-k aggregation was tested
    and rejected: it triples the angle error, see decode_tuning_val.json).
    """
    centerness = preds["centerness"][batch_idx, 0].detach().cpu().numpy()
    bbox = preds["bbox"][batch_idx].detach().cpu().numpy()
    iou_q = preds["iou_quality"][batch_idx, 0].detach().cpu().numpy()
    cls_logits = preds["class_logits"][batch_idx].detach().cpu().numpy()
    if "observability" in preds:
        obs = preds["observability"][batch_idx, 0].detach().cpu().numpy()
    else:
        obs = np.full_like(centerness, float("nan"))
    keypoints = preds["keypoints"][batch_idx].detach().cpu().numpy()

    obj_score_map = centerness * iou_q
    mask = obj_score_map >= obj_thresh
    if not mask.any():
        return []

    candidates = []
    for r, c in zip(*np.where(mask)):
        cx, cy, bw, bh = (float(v) for v in bbox[:, r, c])
        x1, y1 = max(0.0, cx - bw / 2), max(0.0, cy - bh / 2)
        x2, y2 = min(1.0, cx + bw / 2), min(1.0, cy + bh / 2)
        if x2 <= x1 or y2 <= y1:
            continue
        cls_probs = 1.0 / (1.0 + np.exp(-cls_logits[:, r, c]))
        cls_id = int(np.argmax(cls_probs))
        cls_conf = float(cls_probs[cls_id])
        obj_score = float(obj_score_map[r, c])
        candidates.append(
            {
                "box": (x1, y1, x2, y2),
                "score": obj_score * cls_conf,
                "obj_score": obj_score,
                "cls_conf": cls_conf,
                "class_id": cls_id,
                "class_probs": cls_probs.tolist(),
                "observability": float(obs[r, c]),
                "keypoints": keypoints[:, r, c].tolist(),
                "cell": (int(r), int(c)),
            }
        )
    if not candidates:
        return []
    boxes = np.array([d["box"] for d in candidates])
    scores = np.array([d["score"] for d in candidates])
    return [candidates[i] for i in nms(boxes, scores, nms_iou)]


# ── matching ─────────────────────────────────────────────────────────────
def match_detections(preds, gt_boxes, gt_classes, iou_thresh=IOU_MATCH_THRESH, class_aware=True):
    """Greedy matching by descending score.

    Returns (tp, fp, fn) as per-class int arrays and `pairs` = list of
    (gt_idx, pred) for every matched GT.
    """
    tp = np.zeros(NUM_CLASSES, int)
    fp = np.zeros(NUM_CLASSES, int)
    fn = np.zeros(NUM_CLASSES, int)
    pairs, gt_used = [], set()
    gt_xyxy = [cxcywh_to_xyxy(b) for b in gt_boxes]
    for pi in sorted(range(len(preds)), key=lambda i: -preds[i]["score"]):
        p = preds[pi]
        best_iou, best_gi = iou_thresh, None
        for gi, (g, cl) in enumerate(zip(gt_xyxy, gt_classes)):
            if gi in gt_used or (class_aware and cl != p["class_id"]):
                continue
            iou = box_iou(p["box"], g)
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_gi is None:
            fp[p["class_id"]] += 1
        else:
            gt_used.add(best_gi)
            tp[p["class_id"]] += 1
            pairs.append((best_gi, p))
    for gi, cl in enumerate(gt_classes):
        if gi not in gt_used:
            fn[cl] += 1
    return tp, fp, fn, pairs


def tip_distance_px(pred_kpts, gt_kpts):
    return float(
        np.hypot(
            (pred_kpts[0] - gt_kpts[0]) * ORIG_IMG_W,
            (pred_kpts[1] - gt_kpts[1]) * ORIG_IMG_H,
        )
    )


def tip_match(spray_preds, gt_kpts, max_px=TIP_MATCH_PX):
    """Nearest spray prediction by tip distance, or None if none within max_px."""
    if not spray_preds:
        return None
    best = min(spray_preds, key=lambda p: tip_distance_px(p["keypoints"], gt_kpts))
    return best if tip_distance_px(best["keypoints"], gt_kpts) <= max_px else None


def _prf(tp, fp, fn):
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f = 2 * p * r / max(p + r, 1e-9)
    return float(p), float(r), float(f)


# ── AP ───────────────────────────────────────────────────────────────────
def compute_ap_single_class(pred_per_image, gt_per_image, iou_thresh=IOU_MATCH_THRESH):
    """AP for ONE class, Pascal VOC 11-point interpolation."""
    total_gt = sum(len(gt) for gt in gt_per_image)
    if total_gt == 0:
        return float("nan")
    all_preds = [(i, p["score"], p["box"]) for i, preds in enumerate(pred_per_image) for p in preds]
    if not all_preds:
        return 0.0
    all_preds.sort(key=lambda x: -x[1])
    gt_matched = [np.zeros(len(gt), dtype=bool) for gt in gt_per_image]
    tp, fp = [], []
    for img_i, _, pred_box in all_preds:
        gts = gt_per_image[img_i]
        hit = -1
        if gts:
            ious = np.array([box_iou(pred_box, cxcywh_to_xyxy(g)) for g in gts])
            for j in np.argsort(-ious):
                if ious[j] < iou_thresh:
                    break
                if not gt_matched[img_i][j]:
                    hit = j
                    break
        if hit >= 0:
            gt_matched[img_i][hit] = True
        tp.append(1 if hit >= 0 else 0)
        fp.append(0 if hit >= 0 else 1)
    tp, fp = np.cumsum(tp), np.cumsum(fp)
    recall = tp / (total_gt + 1e-7)
    precision = tp / (tp + fp + 1e-7)
    ap = 0.0
    for t in np.arange(0, 1.01, 0.1):
        if (recall >= t).any():
            ap += precision[recall >= t].max()
    return float(ap / 11.0)


def compute_map_classaware(pred_per_image, gt_per_image, gt_classes_per_image, iou_thresh=IOU_MATCH_THRESH):
    if not pred_per_image:
        return {"mAP": 0.0, "per_class": [float("nan")] * NUM_CLASSES}
    per_class = []
    for cid in range(NUM_CLASSES):
        cls_preds = [[p for p in preds if p["class_id"] == cid] for preds in pred_per_image]
        cls_gts = [[b for b, c in zip(bs, cs) if c == cid] for bs, cs in zip(gt_per_image, gt_classes_per_image)]
        per_class.append(compute_ap_single_class(cls_preds, cls_gts, iou_thresh))
    valid = [a for a in per_class if not math.isnan(a)]
    return {"mAP": float(np.mean(valid)) if valid else 0.0, "per_class": per_class}


# ── accumulator ──────────────────────────────────────────────────────────
class MetricsAccumulator:
    """Instance-level metrics over a validation/test pass.

    `samples` is the dataset's raw sample list; each batch carries
    targets["index"] so the GT of every image is read from it directly.
    """

    def __init__(self, samples):
        self.samples = samples
        self.pred_per_image = []
        self.gt_per_image = []
        self.gt_classes_per_image = []
        self.tp = np.zeros(NUM_CLASSES, int)
        self.fp = np.zeros(NUM_CLASSES, int)
        self.fn = np.zeros(NUM_CLASSES, int)
        self.obs_true, self.obs_pred = [], []
        self.angle_errors = []
        self.tip_errors_px = []
        self.n_gt_spray = 0

    def update_from_batch(self, preds, targets, device=None):
        idx = targets["index"].view(-1).tolist()
        for b, i in enumerate(idx):
            s = self.samples[int(i)]
            self.update_image(preds, b, s)

    def update_image(self, preds, b, s):
        gt_boxes, gt_classes, gt_kpts = s["gt_boxes"], s["gt_classes"], s["gt_kpts"]
        det = decode_predictions(preds, batch_idx=b)

        # detection / classification
        self.pred_per_image.append(det)
        self.gt_per_image.append(gt_boxes)
        self.gt_classes_per_image.append(gt_classes)
        tp, fp, fn, _ = match_detections(det, gt_boxes, gt_classes)
        self.tp += tp
        self.fp += fp
        self.fn += fn

        # observability: every GT spray/facing instance, matched class-agnostically.
        # A GT spray with NO matching box is a missed measurable nozzle -> FN.
        # (Before 2026-09-17 unmatched sprays were simply dropped, which made
        # obs_f1 = 1.0 whenever the few detected sprays were right and let the
        # composite pick checkpoints that had lost a whole class.)
        _, _, _, pairs = match_detections(det, gt_boxes, gt_classes, class_aware=False)
        matched = {gi: p for gi, p in pairs}
        for gi, cl in enumerate(gt_classes):
            if cl not in (SPRAY_IDX, FACING_IDX):
                continue
            p = matched.get(gi)
            if p is not None and not math.isnan(p["observability"]):
                self.obs_true.append(int(cl == SPRAY_IDX))
                self.obs_pred.append(int(p["observability"] > OBS_THRESH))
            elif cl == SPRAY_IDX:
                self.obs_true.append(1)
                self.obs_pred.append(0)  # unmatched measurable nozzle = FN

        # angle: tip-matched
        spray_gts = [gi for gi, c in enumerate(gt_classes) if c == SPRAY_IDX and any(v > 0 for v in gt_kpts[gi])]
        self.n_gt_spray += len(spray_gts)
        if spray_gts:
            low = [p for p in decode_predictions(preds, batch_idx=b, obj_thresh=ANGLE_MATCH_OBJ_THRESH) if p["class_id"] == SPRAY_IDX]
            for gi in spray_gts:
                m = tip_match(low, gt_kpts[gi])
                if m is None:
                    continue
                ga, pa = compute_spray_angle(gt_kpts[gi]), compute_spray_angle(m["keypoints"])
                if ga is not None and pa is not None:
                    self.angle_errors.append(abs(ga - pa))
                    self.tip_errors_px.append(tip_distance_px(m["keypoints"], gt_kpts[gi]))

    def compute_summary(self):
        r = {}
        f1s = []
        for cid, name in enumerate(CLASS_NAMES):
            p, rc, f = _prf(self.tp[cid], self.fp[cid], self.fn[cid])
            r[f"det_P_{name}"], r[f"det_R_{name}"], r[f"det_F1_{name}"] = p, rc, f
            f1s.append(f)
        r["cls_f1"] = float(np.mean(f1s))  # macro over 4 classes, instance level

        if self.obs_true:
            yt, yp = np.array(self.obs_true), np.array(self.obs_pred)
            tp = int(((yp == 1) & (yt == 1)).sum())
            fp = int(((yp == 1) & (yt == 0)).sum())
            fn = int(((yp == 0) & (yt == 1)).sum())
            r["obs_precision"], r["obs_recall"], r["obs_f1"] = _prf(tp, fp, fn)
            r["obs_n"] = int(len(yt))
        else:
            r["obs_precision"] = r["obs_recall"] = r["obs_f1"] = float("nan")
            r["obs_n"] = 0

        r["angle_mae_deg"] = float(np.mean(self.angle_errors)) if self.angle_errors else float("nan")
        r["angle_n"] = len(self.angle_errors)
        r["angle_n_gt"] = self.n_gt_spray
        r["spray_tip_match_rate"] = (len(self.angle_errors) / self.n_gt_spray) if self.n_gt_spray else float("nan")
        r["tip_err_px"] = float(np.mean(self.tip_errors_px)) if self.tip_errors_px else float("nan")

        m = compute_map_classaware(self.pred_per_image, self.gt_per_image, self.gt_classes_per_image)
        r["mAP_50"] = m["mAP"]
        for cid, name in enumerate(CLASS_NAMES):
            r[f"AP_50_{name}"] = m["per_class"][cid]
        return r


if __name__ == "__main__":
    print("metrics.py (v10, instance level) loaded")
    a = compute_spray_angle([0.5, 0.5, 0.4, 0.6, 0.6, 0.6])
    print(f"angle of a symmetric test cone (pixel deg): {a:.2f}")
