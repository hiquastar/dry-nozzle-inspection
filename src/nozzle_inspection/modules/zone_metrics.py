"""
Per-zone metric computation helpers extracted from run_inspection.py.
"""

import math

from .dl_status_ticker import gate_from_status

from . import manual_zone_geometry


def _get_dl_gate_for_zone(system, zone_id):
    """Read the latest DL-classified status for this zone and translate it
    into a measurement gate. Safe-defaults to "measure both" when unknown."""
    ticker = getattr(system, "_dl_status_ticker", None)
    if ticker is None:
        return gate_from_status(None)
    try:
        status = ticker.get_status_for_zone(int(zone_id))
    except Exception:
        status = None
    return gate_from_status(status)


def _dl_spray_from_keypoints(keypoints):
    """Compute (full_cone_angle_deg, anchor, lines) from DL spray keypoints.

    keypoints: [tip_x, tip_y, left_x, left_y, right_x, right_y] in full-frame
    pixel coords (same space as the CV spray lines). Returns None if invalid.
    The angle is the FULL cone angle (tip->left vs tip->right), directly
    comparable to spray_info['spray_angle_deg'].
    """
    if not keypoints or len(keypoints) < 6:
        return None
    try:
        tip = (float(keypoints[0]), float(keypoints[1]))
        left = (float(keypoints[2]), float(keypoints[3]))
        right = (float(keypoints[4]), float(keypoints[5]))
    except (TypeError, ValueError):
        return None
    v1 = (left[0] - tip[0], left[1] - tip[1])
    v2 = (right[0] - tip[0], right[1] - tip[1])
    n1 = math.hypot(v1[0], v1[1])
    n2 = math.hypot(v2[0], v2[1])
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos_a = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    cos_a = max(-1.0, min(1.0, cos_a))
    angle = math.degrees(math.acos(cos_a))
    anchor = (int(round(tip[0])), int(round(tip[1])))
    lines = [
        (anchor, (int(round(left[0])), int(round(left[1])))),
        (anchor, (int(round(right[0])), int(round(right[1])))),
    ]
    return angle, anchor, lines


def _maybe_apply_dl_spray_override(system, spray_info_out, dl_gate):
    """Let the ONNX model override the 2 CV spray lines + angle when its spray
    cone is wider than the CV one (and DL classified the ROI as spray).

    Controlled by config `dl.spray_override_enabled` (default True). A positive
    `dl.spray_override_min_margin_deg` requires the DL angle to exceed CV by that
    margin before overriding (anti-flicker).
    """
    try:
        if not bool(system.config.get("dl.spray_override_enabled", True)):
            return spray_info_out
    except Exception:
        return spray_info_out

    # Only when DL says SPRAY (class_id == 0).
    try:
        if (
            int(
                dl_gate.get("class_id", -1)
                if dl_gate.get("class_id") is not None
                else -1
            )
            != 0
        ):
            return spray_info_out
    except (TypeError, ValueError):
        return spray_info_out

    parsed = _dl_spray_from_keypoints(dl_gate.get("keypoints"))
    if parsed is None:
        return spray_info_out
    dl_angle, dl_anchor, dl_lines = parsed

    try:
        max_diff = float(
            system.config.get("dl.spray_override_max_cv_diff_deg", 10.0) or 10.0
        )
    except Exception:
        max_diff = 10.0

    # Decision: follow CV when CV and DL roughly AGREE (CV is more precise then).
    # Only trust DL when CV is far off (|CV - DL| > max_diff) or CV found no spray.
    cv_detected = bool(spray_info_out.get("detected", False))
    if cv_detected:
        cv_angle = float(spray_info_out.get("spray_angle_deg", 0.0) or 0.0)
        if abs(cv_angle - dl_angle) <= max_diff:
            return spray_info_out
    else:
        cv_angle = 0.0  # no CV spray -> show DL

    # CV missing or differs too much from DL -> redraw the 2 lines + angle from DL.
    overridden = dict(spray_info_out)
    overridden["detected"] = True
    overridden["spray_angle_deg"] = float(dl_angle)
    overridden["lines"] = dl_lines
    overridden["anchor_point"] = dl_anchor
    overridden["spray_axis_deg"] = None  # force overlay to use raw DL lines
    overridden["spray_half_angle_deg"] = None
    overridden["is_prediction"] = False
    overridden["spray_source"] = "dl_onnx"
    overridden["cv_spray_angle_deg"] = float(cv_angle)
    overridden.pop("dl_gated_off", None)
    return overridden


def compute_all_zone_metrics(system, frame, zone_results):
    """
    Compute all zone metrics.

    Logic:
        For each zone, builds nozzle/buildup/warning metric payloads and normalizes local values
        into frame-global coordinates used by overlays and downstream decisions.

    Args:
        system: Runtime system instance holding detectors/configs/state.
        frame: Detection input frame used for per-zone tile extraction.
        zone_results: Per-zone detection outputs for the current frame.

    Returns:
        list[dict]: Full metrics list for all zones in the current frame.
    """
    metrics = []
    weak_cadence = int(
        max(1, system.buildup_config.get("buildup_cadence_weak_frames", 2))
    )
    confirming_cadence = int(
        max(1, system.buildup_config.get("buildup_cadence_confirming_frames", 2))
    )
    yes_hold_frames = int(
        max(0, system.buildup_config.get("buildup_yes_hold_frames", 6))
    )

    for zone in zone_results:
        zone_id = int(zone["zone_id"])
        x, y, w, h = zone["rect"]
        lock_snap = system.zone_visual_lock_snapshots.get(int(zone_id))
        if bool(system.zone_lock_enabled.get(int(zone_id), False)) and isinstance(
            lock_snap, dict
        ):
            locked_rect = lock_snap.get("zone_rect")
            if isinstance(locked_rect, (tuple, list)) and len(locked_rect) == 4:
                try:
                    x, y, w, h = [int(v) for v in locked_rect]
                except (TypeError, ValueError):
                    x, y, w, h = zone["rect"]
        angle_deg = float(zone.get("angle_deg", 0.0))
        tile = manual_zone_geometry.extract_manual_zone_tile(
            system, frame, (x, y, w, h), angle_deg
        )

        nozzle_detector = system.zone_nozzle_detectors[zone_id]
        buildup_detector = system.zone_buildup_detectors[zone_id]
        warning_system = system.zone_warning_systems[zone_id]

        local_nozzle = zone["local_nozzle_info"]
        effective_nozzle = dict(local_nozzle)
        effective_nozzle["detected"] = bool(zone["detected"])
        confidence_ok = bool(local_nozzle.get("confidence_ok", True))
        detection_strength = (
            str(local_nozzle.get("detection_strength", "none")).strip().lower()
        )

        # DL gate: classify ROI by AFGNet, decide which CV measurements still apply.
        dl_gate = _get_dl_gate_for_zone(system, zone_id)
        zone_detected_after_gate = bool(zone["detected"]) and not bool(
            dl_gate.get("skip_all", False)
        )

        if zone_detected_after_gate and confidence_ok:
            cadence = 1
            if detection_strength == "weak":
                cadence = weak_cadence
            elif detection_strength != "strong":
                cadence = confirming_cadence
            cadence = max(1, int(cadence))
            system._buildup_measure_counter[zone_id] = (
                int(system._buildup_measure_counter.get(zone_id, 0)) + 1
            )
            should_measure = (
                cadence <= 1
                or (system._buildup_measure_counter[zone_id] % cadence) == 0
                or zone_id not in system._buildup_cached_info
            )
            if should_measure:
                local_buildup = buildup_detector.detect(
                    tile,
                    effective_nozzle,
                    zone_id=zone_id,
                    spray_info=zone.get("spray_info", {}),
                )
                system._buildup_cached_info[zone_id] = dict(local_buildup)
            else:
                local_buildup = dict(system._buildup_cached_info.get(zone_id, {}))
                if local_buildup:
                    local_buildup["cadence_skipped"] = True
        else:
            system._buildup_measure_counter[zone_id] = 0
            system._buildup_yes_hold_counter[zone_id] = 0
            local_buildup = {
                "has_buildup": False,
                "buildup_percentage": 0.0,
                "buildup_mask": None,
                "texture_score": 0.0,
                "severity": "none",
                "roi": None,
                "roi_mask": None,
                "calibration_mode": system.buildup_calibration_mode,
                "annulus_inner_offset_scale": float(
                    system.buildup_config.get("annulus_head_inner_offset_scale", 0.0)
                ),
                "annulus_band_scale": float(
                    system.buildup_config.get("annulus_head_band_scale", 0.0)
                ),
                "annulus_auto_score": 0.0,
                "annulus_auto_samples": 0,
                "temporal_enabled": buildup_detector.temporal_enabled,
                "has_baseline": buildup_detector.has_baseline(zone_id),
                "diff_mean": None,
                "ssim": None,
                "ema_buildup_percentage": None,
                "ema_diff_mean": None,
                "temporal_increase": False,
                "temporal_score": None,
                "trend": None,
                "glcm_enabled": buildup_detector.glcm_enabled,
                "glcm_contrast": None,
                "glcm_buildup": False,
                "baseline_locked": False,
                "baseline_lock_enabled": bool(
                    system.zone_lock_enabled.get(zone_id, True)
                ),
                "diff_percentage": 0.0,
                "diff_trigger": False,
                "diff_reliable": False,
                "diff_edge_overlap_ratio": 0.0,
                "effective_area_threshold": float(
                    system.buildup_config.get("buildup_area_threshold", 0.15)
                ),
            }
            system._buildup_cached_info[zone_id] = dict(local_buildup)

        has_yes_now = bool(local_buildup.get("has_buildup", False))
        hold_left = int(system._buildup_yes_hold_counter.get(zone_id, 0))
        if has_yes_now:
            system._buildup_yes_hold_counter[zone_id] = yes_hold_frames
            local_buildup["yes_hold_frames_left"] = int(yes_hold_frames)
        elif hold_left > 0:
            # Keep YES visible for a short tail to reduce flicker.
            local_buildup["has_buildup"] = True
            local_buildup["held_yes"] = True
            local_buildup["yes_hold_frames_left"] = int(hold_left)
            system._buildup_yes_hold_counter[zone_id] = int(max(0, hold_left - 1))
        else:
            local_buildup["yes_hold_frames_left"] = 0

        position_deviation = 0.0
        orientation_deviation = 0.0
        if effective_nozzle.get("detected", False):
            position_deviation = nozzle_detector.get_position_deviation(
                effective_nozzle["position"]
            )
            orientation_deviation = nozzle_detector.get_orientation_deviation(
                effective_nozzle["orientation"]
            )

        if zone["detected"]:
            warning_info = warning_system.evaluate(
                effective_nozzle,
                local_buildup,
                position_deviation,
                orientation_deviation,
            )
            color = warning_system.get_color(
                warning_info["alert_level"], system.display_config
            )
            if bool(local_buildup.get("has_buildup", False)):
                warning_info = dict(warning_info)
                warnings = list(warning_info.get("warnings", []))
                if "Buildup YES" not in warnings:
                    warnings.append("Buildup YES")
                warning_info["warnings"] = warnings
                try:
                    warning_info["alert_level"] = max(
                        int(warning_info.get("alert_level", 0)), 1
                    )
                except (TypeError, ValueError):
                    warning_info["alert_level"] = 1
                color = tuple(system.display_config.get("warning_color", [0, 165, 255]))
        else:
            warning_info = {
                "alert_level": 0,
                "warnings": [],
            }
            status = zone.get("status")
            if isinstance(status, str) and status.startswith("OWNED_BY_"):
                color = (180, 0, 180)
            else:
                color = (0, 220, 220) if status == "CONFIRMING" else (0, 0, 220)

        nozzle_info_global = dict(effective_nozzle)
        if (detection_strength != "strong") and (system.current_mode != "MANUAL"):
            orient = nozzle_info_global.get("orientation")
            if orient is not None and abs(float(orient)) < 1e-3:
                nozzle_info_global["orientation"] = None
        if nozzle_info_global.get("position") is not None:
            px, py = nozzle_info_global["position"]
            nozzle_info_global["position"] = (int(px + x), int(py + y))

        bbox = nozzle_info_global.get("bounding_box")
        if bbox is not None:
            bx, by, bw, bh = bbox
            nozzle_info_global["bounding_box"] = (
                int(bx + x),
                int(by + y),
                int(bw),
                int(bh),
            )

        # If DL says spray-measurement is gated off, drop spray_info so downstream
        # alerts/overlays don't act on it.
        spray_info_out = zone.get("spray_info", {})
        if not bool(dl_gate.get("measure_spray", True)):
            spray_info_out = {"detected": False, "dl_gated_off": True}
        else:
            # When the ONNX model sees a wider spray cone than CV, follow the DL
            # keypoints for the 2 spray lines + displayed angle.
            spray_info_out = _maybe_apply_dl_spray_override(
                system, spray_info_out, dl_gate
            )
        # Write the finalized spray back so the grid overlay (which draws the 2
        # spray lines from zone["spray_info"], overlay_renderer.py:~580) stays in
        # sync with the panel/metric value.
        zone["spray_info"] = spray_info_out

        metrics.append(
            {
                "zone_id": zone_id,
                "io_state": "IN" if bool(zone.get("detected", False)) else "OUT",
                "rect": zone["rect"],
                "nozzle_info": nozzle_info_global,
                "raw_detected": bool(zone.get("raw_detected", False)),
                "buildup_info": local_buildup,
                "warning_info": warning_info,
                "position_deviation": float(position_deviation),
                "orientation_deviation": float(orientation_deviation),
                "color": color,
                "status": zone.get("status", "OUT"),
                "owned_by_zone": zone.get("owned_by_zone"),
                "body_length_px": float(zone.get("body_length_px", 0.0)),
                "votes": int(zone.get("votes", 0)),
                "history_len": int(zone.get("history_len", 0)),
                "spray_info": spray_info_out,
                "detection_strength": detection_strength,
                "detection_method": local_nozzle.get("detection_method")
                or zone.get("method"),
                "reject_reason": local_nozzle.get("reject_reason"),
                "debug_details": local_nozzle.get("debug_details", {}),
                "dl_gate": dict(dl_gate),
            }
        )

    return metrics
