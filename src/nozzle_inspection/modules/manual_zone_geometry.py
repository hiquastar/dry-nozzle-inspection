"""
Manual-zone geometry helpers extracted from run_inspection.py.
"""

import cv2
import numpy as np


def normalize_manual_zone_angle(angle_deg):
    """
    Normalize manual-zone angle.

    Logic:
        Normalizes any angle value into the `[-180, 180)` range and snaps near-zero to `0.0`.

    Args:
        angle_deg: Raw angle value in degrees.

    Returns:
        float: Normalized angle in degrees.
    """
    try:
        angle = float(angle_deg)
    except Exception:
        angle = 0.0
    angle = ((angle + 180.0) % 360.0) - 180.0
    if abs(angle) < 1e-6:
        angle = 0.0
    return float(angle)


def manual_zone_to_dict(system, zone_spec):
    """
    Convert manual-zone spec to normalized dict.

    Logic:
        Accepts dict/list/tuple zone formats and converts them into canonical
        `{'rect': (x, y, w, h), 'angle_deg': a}` form.

    Args:
        system: Runtime system instance used for shared normalization rules.
        zone_spec: Manual-zone specification in dict/list/tuple format.

    Returns:
        dict: Canonical manual-zone dictionary.
    """
    if isinstance(zone_spec, dict):
        rect = zone_spec.get("rect")
        if not isinstance(rect, (list, tuple)) or len(rect) < 4:
            raise ValueError("manual zone dict requires 'rect' with 4 values")
        angle = zone_spec.get("angle_deg", zone_spec.get("angle", 0.0))
        x, y, w, h = [int(v) for v in rect[:4]]
        return {
            "rect": (x, y, w, h),
            "angle_deg": normalize_manual_zone_angle(angle),
        }
    if isinstance(zone_spec, (list, tuple)):
        if len(zone_spec) < 4:
            raise ValueError("manual zone list/tuple requires at least 4 values")
        x, y, w, h = [int(v) for v in zone_spec[:4]]
        angle = zone_spec[4] if len(zone_spec) >= 5 else 0.0
        return {
            "rect": (x, y, w, h),
            "angle_deg": normalize_manual_zone_angle(angle),
        }
    _ = system
    raise ValueError("unsupported manual zone format")


def manual_zone_unpack(system, zone_spec):
    """
    Unpack manual-zone spec.

    Logic:
        Converts zone specification to canonical form and returns unpacked tuple values.

    Args:
        system: Runtime system instance used for shared conversion rules.
        zone_spec: Manual-zone specification.

    Returns:
        tuple[int, int, int, int, float]: `(x, y, w, h, angle_deg)`.
    """
    zone = manual_zone_to_dict(system, zone_spec)
    x, y, w, h = zone["rect"]
    return int(x), int(y), int(w), int(h), float(zone.get("angle_deg", 0.0))


def serialize_manual_zone(system, zone_spec):
    """
    Serialize manual-zone spec for storage.

    Logic:
        Converts zone spec to stable JSON-friendly payload with normalized angle.

    Args:
        system: Runtime system instance used for shared conversion rules.
        zone_spec: Manual-zone specification.

    Returns:
        dict: Serializable zone payload.
    """
    x, y, w, h, angle = manual_zone_unpack(system, zone_spec)
    return {
        "rect": [int(x), int(y), int(w), int(h)],
        "angle_deg": float(normalize_manual_zone_angle(angle)),
    }


def manual_zone_polygon(rect, angle_deg=0.0):
    """
    Build rotated manual-zone polygon.

    Logic:
        Converts `(x, y, w, h, angle)` zone parameters to 4-point polygon for drawing/masking.

    Args:
        rect: Axis-aligned rectangle `(x, y, w, h)`.
        angle_deg: Rotation angle in degrees.

    Returns:
        np.ndarray: `4x2` int polygon points.
    """
    x, y, w, h = [float(v) for v in rect]
    cx = x + (0.5 * w)
    cy = y + (0.5 * h)
    box = cv2.boxPoints(
        ((float(cx), float(cy)), (float(w), float(h)), float(angle_deg))
    )
    return np.round(box).astype(np.int32)


def extract_manual_zone_tile(system, frame, rect, angle_deg=0.0):
    """
    Extract manual-zone tile from frame.

    Logic:
        Crops rectangle ROI and applies rotated mask when angle is non-zero.

    Args:
        system: Runtime system instance used for shared normalization rules.
        frame: Source frame.
        rect: Rectangle `(x, y, w, h)` in frame coordinates.
        angle_deg: ROI rotation angle in degrees.

    Returns:
        np.ndarray: Extracted tile (masked if rotated).
    """
    x, y, w, h = [int(v) for v in rect]
    tile = frame[y : y + h, x : x + w]
    if tile.size == 0:
        return tile
    angle = normalize_manual_zone_angle(angle_deg)
    if abs(angle) < 1e-6:
        return tile
    masked = tile.copy()
    mask = np.zeros((int(h), int(w)), dtype=np.uint8)
    local_poly = manual_zone_polygon((0, 0, int(w), int(h)), angle)
    cv2.fillConvexPoly(mask, local_poly, 255)
    _ = system
    return cv2.bitwise_and(masked, masked, mask=mask)


def sanitize_manual_zone(system, zone_spec, frame_shape):
    """
    Sanitize one manual zone.

    Logic:
        Clamps zone dimensions/position to frame bounds while preserving user-selected size as much as possible.

    Args:
        system: Runtime system instance containing manual size defaults.
        zone_spec: Manual-zone specification.
        frame_shape: Frame shape used for bounds checks.

    Returns:
        dict: Sanitized canonical zone dictionary.
    """
    fh, fw = frame_shape[:2]
    x, y, w, h, angle = manual_zone_unpack(system, zone_spec)
    min_size = 1
    w = max(min_size, int(w))
    h = max(min_size, int(h))
    w = min(w, max(min_size, int(fw)))
    h = min(h, max(min_size, int(fh)))
    x = int(np.clip(x, 0, max(0, fw - w)))
    y = int(np.clip(y, 0, max(0, fh - h)))
    return {
        "rect": (x, y, w, h),
        "angle_deg": float(normalize_manual_zone_angle(angle)),
    }


def sanitize_manual_zones(system, zones, frame_shape):
    """
    Sanitize multiple manual zones.

    Logic:
        Applies single-zone sanitization and drops invalid entries safely.

    Args:
        system: Runtime system instance containing geometry rules.
        zones: Iterable manual-zone specifications.
        frame_shape: Frame shape used for bounds checks.

    Returns:
        list[dict]: Sanitized zone list.
    """
    cleaned = []
    for zone_spec in zones:
        try:
            cleaned.append(sanitize_manual_zone(system, zone_spec, frame_shape))
        except Exception:
            continue
    return cleaned


def build_default_manual_zone(system, frame_shape):
    """
    Build default manual zone.

    Logic:
        Creates a centered default zone sized from frame dimensions and system minimum-size settings.

    Args:
        system: Runtime system instance containing manual size defaults.
        frame_shape: Frame shape used to compute default ROI.

    Returns:
        dict: Sanitized default manual-zone dictionary.
    """
    fh, fw = frame_shape[:2]
    zone_w = max(int(system.manual_min_size_px * 3), int(round(fw * 0.22)))
    zone_h = max(int(system.manual_min_size_px * 3), int(round(fh * 0.22)))
    x = int(round((fw - zone_w) * 0.5))
    y = int(round((fh - zone_h) * 0.5))
    return sanitize_manual_zone(system, (x, y, zone_w, zone_h, 0.0), frame_shape)
