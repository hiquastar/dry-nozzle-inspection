"""
Auto-ROI generator for the Profile editor.

Flow (per detection):
    1. DL detect bbox on a full-frame snapshot.
    2. Expand AI bbox by 1.5x (give Hough more pixel context).
    3. Run CV NozzleDetector (Hough/contour) inside the expanded crop to find
       the precise head center.
    4. Build a 270x270 square ROI around the refined head.
    5. Discard the AI bbox — only the new square ROI is carried forward.
    6. Greedy reject if the new ROI overlaps any previously accepted ROI.

Output schema matches config/manual_zones.json:
    {'rect': [x, y, w, h], 'angle_deg': float, ...metadata}
The AI bbox is intentionally not stored in the output.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .dl_inference import BIG_IDX, CLASS_NAMES, AFGNetInference
from .nozzle_detector import NozzleDetector

DEFAULT_ROI_SIZE_PX = 270
DEFAULT_PADDING_PX = 8
DEFAULT_OVERLAP_IOU_MAX = 0.0
DEFAULT_BBOX_EXPAND_RATIO = 2.5


def _aabb_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / max(union, 1e-7)


def _clip_rect_to_frame(
    rect: Tuple[int, int, int, int], frame_w: int, frame_h: int, min_size: int = 24
) -> Optional[Tuple[int, int, int, int]]:
    x, y, w, h = rect
    x = max(0, int(x))
    y = max(0, int(y))
    w = int(w)
    h = int(h)
    if x + w > frame_w:
        w = frame_w - x
    if y + h > frame_h:
        h = frame_h - y
    if w < min_size or h < min_size:
        return None
    return (x, y, w, h)


def _square_roi_around_center(
    cx: int, cy: int, size: int, frame_w: int, frame_h: int
) -> Optional[Tuple[int, int, int, int]]:
    half = size // 2
    x = int(cx - half)
    y = int(cy - half)
    w = int(size)
    h = int(size)
    return _clip_rect_to_frame((x, y, w, h), frame_w, frame_h)


def _expand_bbox(
    bbox_xyxy: Tuple[int, int, int, int], ratio: float, frame_w: int, frame_h: int
) -> Tuple[int, int, int, int]:
    """Scale a bbox around its center by `ratio` and clip to frame."""
    x1, y1, x2, y2 = bbox_xyxy
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    new_w = (x2 - x1) * float(ratio)
    new_h = (y2 - y1) * float(ratio)
    nx1 = int(max(0, round(cx - new_w / 2)))
    ny1 = int(max(0, round(cy - new_h / 2)))
    nx2 = int(min(frame_w, round(cx + new_w / 2)))
    ny2 = int(min(frame_h, round(cy + new_h / 2)))
    return (nx1, ny1, nx2, ny2)


def _refine_head_inside_bbox(
    refiner: NozzleDetector,
    frame_bgr: np.ndarray,
    bbox_xyxy: Tuple[int, int, int, int],
    expand_ratio: float = DEFAULT_BBOX_EXPAND_RATIO,
    max_drift_ratio: float = 0.6,
    min_radius_px: int = 8,
) -> Tuple[Optional[Tuple[int, int]], Optional[float], Optional[float]]:
    """Expand AI bbox by `expand_ratio`, run CV Hough head detection inside it.

    Returns (head_xy, radius, orientation_deg) in FULL FRAME coords.
    Returns (None, None, None) if CV head is either:
      - not found at all,
      - radius too small (< min_radius_px) — likely false positive from reflection/noise,
      - head center drifted > max_drift_ratio * AI bbox diagonal from AI bbox center
        (likely picked up a neighboring object instead of the true nozzle).
    """
    h_frame, w_frame = frame_bgr.shape[:2]
    ax1, ay1, ax2, ay2 = bbox_xyxy
    a_cx = 0.5 * (ax1 + ax2)
    a_cy = 0.5 * (ay1 + ay2)
    a_w = max(1, ax2 - ax1)
    a_h = max(1, ay2 - ay1)
    a_diag = (a_w * a_w + a_h * a_h) ** 0.5
    drift_limit = float(max_drift_ratio) * a_diag

    x1, y1, x2, y2 = _expand_bbox(bbox_xyxy, expand_ratio, w_frame, h_frame)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None, None, None

    crop = frame_bgr[y1:y2, x1:x2]
    try:
        info = refiner.detect(crop)
    except Exception:
        return None, None, None

    if not info or not info.get("detected"):
        return None, None, None
    pos = info.get("position")
    if pos is None:
        return None, None, None
    head_x = int(pos[0]) + x1
    head_y = int(pos[1]) + y1
    radius = info.get("radius")

    # Validation 1: radius too small => likely noise/reflection
    if radius is not None and float(radius) < float(min_radius_px):
        return None, None, None

    # Validation 2: head drifted too far from AI bbox center => likely a neighbor
    drift = ((head_x - a_cx) ** 2 + (head_y - a_cy) ** 2) ** 0.5
    if drift > drift_limit:
        return None, None, None

    return (head_x, head_y), radius, info.get("orientation")


def _ai_bbox_center(bbox_xyxy: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def _build_refiner(nozzle_config: Dict[str, Any]) -> NozzleDetector:
    """Build a dedicated NozzleDetector for one-shot refinement.

    Uses same config but disables temporal stabilization to avoid polluting
    state between calls.
    """
    cfg = dict(nozzle_config) if nozzle_config else {}
    cfg["head_stability_enabled"] = False
    cfg["kalman_enabled"] = False
    return NozzleDetector(cfg)


def generate_auto_rois(
    frame_bgr: np.ndarray,
    dl: AFGNetInference,
    nozzle_config: Dict[str, Any],
    existing_rois: Optional[List[Tuple[int, int, int, int]]] = None,
    roi_size_px: int = DEFAULT_ROI_SIZE_PX,
    padding_px: int = DEFAULT_PADDING_PX,
    overlap_iou_max: float = DEFAULT_OVERLAP_IOU_MAX,
    include_big_nozzle: bool = False,
    bbox_expand_ratio: float = DEFAULT_BBOX_EXPAND_RATIO,
) -> List[Dict[str, Any]]:
    """Run the full auto-ROI pipeline on a single frame.

    Args:
        frame_bgr: full BGR snapshot from camera.
        dl: loaded AFGNetInference instance.
        nozzle_config: nozzle detection config used to build a temp CV refiner.
        existing_rois: list of (x, y, w, h) for rois that must be respected
            (used for "append" mode — new auto rois will not overlap with them).
        roi_size_px: square side of the generated ROI (270 default = profile mean).
        padding_px: extra padding on each side beyond the ROI itself when checking
            no-overlap (gives visual breathing room).
        overlap_iou_max: maximum allowed IoU vs already-accepted ROIs (default 0 = no overlap).
        include_big_nozzle: if False, skip class_id==BIG_IDX detections (they are
            not inspection targets per locked design).

    Returns:
        List of dicts compatible with manual_zones.json schema:
            {'rect': [x, y, w, h], 'angle_deg': float,
             'meta': {detection class_id/score/source}}
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return []
    h_frame, w_frame = frame_bgr.shape[:2]
    img_center = (w_frame * 0.5, h_frame * 0.5)

    detections = dl.detect(frame_bgr)
    if not detections:
        return []

    # 1. Filter: drop big nozzle by default
    if not include_big_nozzle:
        detections = [d for d in detections if d["class_id"] != BIG_IDX]
    if not detections:
        return []

    # 2. Refine ALL detections first (Hough on 2x expanded bbox),
    #    then sort by FINAL ROI center distance to image center.
    refiner = _build_refiner(nozzle_config)

    candidates: List[Dict[str, Any]] = []
    for det in detections:
        bbox = det["box"]
        head_xy, head_radius, orientation = _refine_head_inside_bbox(
            refiner, frame_bgr, bbox, expand_ratio=bbox_expand_ratio
        )
        if head_xy is None:
            head_xy = _ai_bbox_center(bbox)
            head_source = "ai_bbox_center"
        else:
            head_source = "cv_refined"

        # AI bbox served its purpose — only carry the head forward.
        del bbox

        roi = _square_roi_around_center(
            head_xy[0], head_xy[1], roi_size_px, w_frame, h_frame
        )
        if roi is None:
            continue

        roi_cx = roi[0] + roi[2] * 0.5
        roi_cy = roi[1] + roi[3] * 0.5
        dist_to_center = (
            (roi_cx - img_center[0]) ** 2 + (roi_cy - img_center[1]) ** 2
        ) ** 0.5

        angle_deg = 0.0
        if orientation is not None and np.isfinite(orientation):
            angle_deg = float(orientation)

        candidates.append(
            {
                "roi": roi,
                "angle_deg": angle_deg,
                "dist_to_center": dist_to_center,
                "det": det,
                "head_source": head_source,
                "head_radius": head_radius,
            }
        )

    # Sort: primary = final-ROI-distance-to-image-center ASC, secondary = DL score DESC
    candidates.sort(key=lambda c: (c["dist_to_center"], -c["det"]["score"]))

    # 3. Greedy accept non-overlapping ROIs in center-priority order
    accepted: List[Dict[str, Any]] = []
    accepted_rects: List[Tuple[int, int, int, int]] = list(existing_rois or [])
    n_initial_existing = len(accepted_rects)

    for cnd in candidates:
        roi = cnd["roi"]
        padded = (
            max(0, roi[0] - padding_px),
            max(0, roi[1] - padding_px),
            roi[2] + 2 * padding_px,
            roi[3] + 2 * padding_px,
        )
        overlap_violation = any(
            _aabb_iou(padded, other) > overlap_iou_max for other in accepted_rects
        )
        if overlap_violation:
            continue

        accepted_rects.append(roi)
        det = cnd["det"]
        accepted.append(
            {
                "rect": [roi[0], roi[1], roi[2], roi[3]],
                "angle_deg": cnd["angle_deg"],
                "meta": {
                    "source": "auto_ai_cv",
                    "class_id": int(det["class_id"]),
                    "class_name": CLASS_NAMES[det["class_id"]],
                    "dl_score": float(det["score"]),
                    "observability": float(det.get("observability", 0.0)),
                    "head_source": cnd["head_source"],
                    "head_radius": (
                        float(cnd["head_radius"])
                        if cnd["head_radius"] is not None
                        else None
                    ),
                    "dist_to_image_center_px": float(cnd["dist_to_center"]),
                },
            }
        )

    # Only return the NEW rois (those added in this call), not the existing ones
    return accepted[n_initial_existing:] if existing_rois else accepted


def visualize_pipeline(
    frame_bgr: np.ndarray,
    dl: AFGNetInference,
    nozzle_config: Dict[str, Any],
    roi_size_px: int = DEFAULT_ROI_SIZE_PX,
    bbox_expand_ratio: float = DEFAULT_BBOX_EXPAND_RATIO,
    include_big_nozzle: bool = True,
) -> np.ndarray:
    """Render a debug overlay showing every step of the auto-ROI pipeline.

    Layers on returned BGR image:
        - YELLOW thin box  : AI original bbox
        - CYAN thin box    : 2.5x expanded bbox (fed to CV Hough)
        - RED circle+cross : CV-refined head center (radius shown if known)
        - WHITE thick box  : final 270 square ROI (centered on CV head)
        - Class label printed above the AI bbox (BIG kept here for context).

    No ROIs are accepted/rejected — this is a pure inspection view.
    """
    import cv2 as _cv2

    if frame_bgr is None or frame_bgr.size == 0:
        return frame_bgr

    canvas = frame_bgr.copy()
    h_frame, w_frame = canvas.shape[:2]

    # Image center crosshair
    _cv2.drawMarker(
        canvas, (w_frame // 2, h_frame // 2), (255, 255, 255), _cv2.MARKER_CROSS, 30, 1
    )

    detections = dl.detect(frame_bgr)
    if not include_big_nozzle:
        detections = [d for d in detections if d["class_id"] != 3]
    if not detections:
        _cv2.putText(
            canvas,
            "AI: no detections",
            (40, 60),
            _cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )
        return canvas

    refiner = _build_refiner(nozzle_config)

    # Legend
    legend = [
        ("YELLOW = AI bbox (+ dot = AI center)", (0, 220, 220)),
        (f"CYAN   = expand {bbox_expand_ratio}x", (220, 220, 0)),
        ("RED    = CV head accepted", (0, 0, 255)),
        ("MAGENTA= CV rejected -> fallback to AI center", (200, 0, 200)),
        ("WHITE  = final 270 ROI", (255, 255, 255)),
    ]
    y0 = 30
    for txt, col in legend:
        _cv2.putText(canvas, txt, (10, y0), _cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
        y0 += 24

    for i, det in enumerate(detections):
        bbox = det["box"]
        ax1, ay1, ax2, ay2 = bbox

        # 1) AI bbox in YELLOW + AI center marker
        _cv2.rectangle(canvas, (ax1, ay1), (ax2, ay2), (0, 220, 220), 1)
        ai_cx, ai_cy = _ai_bbox_center(bbox)
        _cv2.circle(canvas, (ai_cx, ai_cy), 3, (0, 220, 220), -1)

        # 2) Expanded bbox in CYAN
        ex1, ey1, ex2, ey2 = _expand_bbox(bbox, bbox_expand_ratio, w_frame, h_frame)
        _cv2.rectangle(canvas, (ex1, ey1), (ex2, ey2), (220, 220, 0), 1)

        # Class label above AI bbox
        cid = int(det["class_id"])
        from .dl_inference import CLASS_NAMES as _CN

        label = f"#{i+1} {_CN[cid]} {det['score']:.2f}"
        _cv2.putText(
            canvas,
            label,
            (ax1, max(15, ay1 - 5)),
            _cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 220, 220),
            1,
        )

        # 3) Run CV refinement on expanded crop (with validation)
        head_xy, head_radius, _ = _refine_head_inside_bbox(
            refiner, frame_bgr, bbox, expand_ratio=bbox_expand_ratio
        )

        if head_xy is not None:
            hx, hy = head_xy
            head_src = "CV"
        else:
            hx, hy = ai_cx, ai_cy
            head_src = "FALLBACK"  # validation rejected CV head OR Hough found nothing

        # 4) Draw head marker (RED for CV, MAGENTA for fallback so it's visually distinct)
        marker_color = (0, 0, 255) if head_src == "CV" else (200, 0, 200)
        _cv2.drawMarker(canvas, (hx, hy), marker_color, _cv2.MARKER_CROSS, 14, 2)
        if head_radius is not None and np.isfinite(head_radius):
            _cv2.circle(canvas, (hx, hy), int(head_radius), marker_color, 1)
        head_lbl = f"{head_src}"
        if head_radius is not None and np.isfinite(head_radius):
            head_lbl += f" r={int(head_radius)}"
        _cv2.putText(
            canvas,
            head_lbl,
            (hx + 8, hy - 8),
            _cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            marker_color,
            1,
        )

        # 5) Final 270 ROI in WHITE thick
        roi = _square_roi_around_center(hx, hy, roi_size_px, w_frame, h_frame)
        if roi is not None:
            rx, ry, rw, rh = roi
            _cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (255, 255, 255), 2)

    return canvas
