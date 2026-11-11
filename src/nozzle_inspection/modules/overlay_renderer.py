"""
Overlay rendering helpers extracted from run_inspection.py.
"""

import cv2
import numpy as np

from . import manual_zone_geometry


def visualize(system, frame, zone_results, zone_metrics):
    """
    Visualize frame overlays.

    Logic:
        Builds the rendered output by composing grid/mask/panel overlays according to current display flags.

    Args:
        system: Runtime system instance holding configs/state and helper methods.
        frame: Current source frame.
        zone_results: Per-zone detection outputs for the current frame.
        zone_metrics: Per-zone metrics payload already computed for the current frame.

    Returns:
        np.ndarray: Frame with visualization overlays applied.
    """
    output = frame.copy()

    if not system.display_config.get("show_detection_overlay", True):
        return output

    if system.show_focus_zone_overlay:
        draw_zone_grid_overlay(system, output, zone_results, zone_metrics=zone_metrics)
    if system.show_track_trails:
        draw_tip_tracks(system, output, zone_results)
    draw_buildup_overlay(system, output, zone_metrics)
    if bool(getattr(system, "show_status_panel", True)):
        draw_status_panel(system, output, zone_metrics)
    if system.display_config.get("show_metrics", True) and system.show_focus_zone_panel:
        manual_mode = str(getattr(system, "current_mode", "")).upper() == "MANUAL"
        prefer_near_zone = (
            bool(getattr(system, "metric_panel_prefer_near_zone", False)) or manual_mode
        )
        if system.hud_enabled and (not system.strict_input_output_size):
            return compose_hud(system, output, zone_metrics)
        had_focus_overlap = draw_all_zone_metrics(system, output, zone_metrics)
        if (
            had_focus_overlap
            and system.metrics_auto_hud_on_overlap
            and (not prefer_near_zone)
            and (not system.strict_input_output_size)
        ):
            return compose_hud(system, output, zone_metrics)
    if system.strict_input_output_size and output.shape[:2] != frame.shape[:2]:
        return cv2.resize(
            output,
            (int(frame.shape[1]), int(frame.shape[0])),
            interpolation=cv2.INTER_AREA,
        )
    return output


def draw_status_panel(system, frame, zone_metrics=None):
    """
    Draw status panel.

    Logic:
        Renders the left-side feature table and runtime status using current mode/toggle/calibration values.

    Args:
        system: Runtime system instance holding configs/state and helper methods.
        frame: Target frame to draw onto (modified in-place).
        zone_metrics: Optional per-zone metrics used to pull calibration/baseline state.

    Returns:
        None.
    """
    dynamic_enabled = bool(
        getattr(system.zone_orchestrator, "dynamic_zoning_enabled", False)
    )
    mode_text = str(system.current_mode)
    calib_mode = str(system.buildup_calibration_mode).upper()

    calib_offset = float(
        system.buildup_config.get("annulus_head_inner_offset_scale", 0.0)
    )
    calib_band = float(system.buildup_config.get("annulus_head_band_scale", 0.0))
    calib_score = 0.0
    calib_samples = 0
    baseline_state = "UNLOCKED"
    if isinstance(zone_metrics, list):
        for metric in zone_metrics:
            binfo = metric.get("buildup_info", {}) or {}
            if not isinstance(binfo, dict):
                continue
            mode_val = str(binfo.get("calibration_mode", "")).strip().lower()
            if mode_val:
                calib_mode = mode_val.upper()
            if "annulus_inner_offset_scale" in binfo:
                calib_offset = float(
                    binfo.get("annulus_inner_offset_scale", calib_offset)
                )
                calib_band = float(binfo.get("annulus_band_scale", calib_band))
                calib_score = float(binfo.get("annulus_auto_score", calib_score))
                calib_samples = int(binfo.get("annulus_auto_samples", calib_samples))
            has_bl = bool(binfo.get("has_baseline", False))
            locked_bl = bool(binfo.get("baseline_locked", False))
            if locked_bl or has_bl:
                baseline_state = "LOCKED"

    calib_items = [
        ("Calib Mode", "TAB", calib_mode),
        ("Offset", "[ / ]", f"{calib_offset:.3f}"),
        ("Band", "; / '", f"{calib_band:.3f}"),
        (
            "Spray Exclusion",
            "U / I",
            f"{100.0 * float(system.buildup_config.get('spray_exclusion_band_ratio_of_annulus', 0.22)):.1f}%",
        ),
        ("Baseline", "-", baseline_state),
    ]
    if calib_mode == "AUTO":
        calib_items.extend(
            [
                ("Auto Score", "-", f"{calib_score:.4f}"),
                ("Auto Samples", "-", f"{calib_samples:d}"),
            ]
        )

    groups = [
        (
            "CALIBRATION",
            calib_items,
        ),
        (
            "NOZZLE DETECTION",
            [
                (
                    "Nozzle Angle",
                    "0",
                    "ON" if system.show_nozzle_angle_arrow else "OFF",
                ),
                ("Head Circle", "1", "ON" if system.show_nozzle_head_circle else "OFF"),
            ],
        ),
        (
            "SPRAY ANGLE",
            [
                ("Spray Angle", "2", "ON" if system.show_spray_angle_lines else "OFF"),
                ("Spray Zone", "3", "ON" if system.show_spray_zone_overlay else "OFF"),
            ],
        ),
        (
            "BUILDUP MASKS",
            [
                ("Annular Mask", "4", "ON" if system.show_annular_mask else "OFF"),
                (
                    "Spray Exclusion",
                    "5",
                    "ON" if system.show_spray_exclusion_mask else "OFF",
                ),
                ("Analysis Mask", "6", "ON" if system.show_analysis_mask else "OFF"),
            ],
        ),
        (
            "DISPLAY",
            [
                (
                    "Focus Zone Overlay",
                    "7",
                    "ON" if system.show_focus_zone_overlay else "OFF",
                ),
                (
                    "Side Information Panel",
                    "8",
                    "ON" if system.show_focus_zone_panel else "OFF",
                ),
                (
                    "Select Lock Zone",
                    "9",
                    "ON" if system.lock_zone_select_mode else "OFF",
                ),
            ],
        ),
    ]
    rows = [item for _gname, items in groups for item in items]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.41
    thick = 1
    pad_x = 10
    line_type = cv2.LINE_AA
    text_main = (255, 255, 245)
    text_muted = (236, 236, 226)
    text_header = (255, 255, 90)
    left_col = "Feature"
    mid_col = "Key"
    right_col = "Value"
    frame_total = int(getattr(system, "video_total_frames", 0) or 0)
    frame_current = int(getattr(system, "video_frame_index", 0) or 0)
    if frame_total > 0:
        frame_current = int(np.clip(max(1, frame_current), 1, frame_total))
        playback_text = f"Frame: {frame_current}/{frame_total}  Loop: {'ON' if bool(getattr(system, 'video_auto_loop', True)) else 'OFF'}"
    elif frame_current > 0:
        playback_text = f"Frame: {frame_current}  Loop: {'ON' if bool(getattr(system, 'video_auto_loop', True)) else 'OFF'}"
    else:
        playback_text = None
    max_feature_w = 0
    max_value_w = 0
    for feature, _key_label, value_txt in rows:
        (fw, _fh), _ = cv2.getTextSize(feature, font, scale, thick)
        max_feature_w = max(max_feature_w, int(fw))
        (vw, _vh), _ = cv2.getTextSize(str(value_txt), font, scale, thick)
        max_value_w = max(max_value_w, int(vw))
    (w_key, _), _ = cv2.getTextSize(mid_col, font, scale, thick)
    table_w = (pad_x * 2) + max_feature_w + 18 + w_key + 20 + max_value_w + 12
    header_w = 0
    for header_text, header_scale in [
        (f"Mode: {mode_text}", 0.58),
        (f"FPS: {system.current_fps:.1f}", 0.58),
        (playback_text, 0.50),
    ]:
        if header_text:
            (hw, _hh), _ = cv2.getTextSize(header_text, font, header_scale, thick)
            header_w = max(header_w, int(hw) + (2 * pad_x) + 10)
    panel_w = int(np.clip(max(table_w, header_w), 248, 390))
    row_h = 15
    group_h = 14
    header_h = 126 if playback_text else 104
    table_head_h = 28
    group_count = len(groups)
    panel_h = int(
        header_h + table_head_h + (row_h * len(rows)) + (group_h * group_count) + 22
    )
    system.status_panel_rect = (8, 8, panel_w, panel_h)
    overlay = frame.copy()
    spx, spy, spw, sph = [int(v) for v in system.status_panel_rect]
    cv2.rectangle(overlay, (spx, spy), (spx + spw, spy + sph), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.74, frame, 0.26, 0, frame)
    cv2.putText(
        frame,
        f"Mode: {mode_text}",
        (14, 29),
        font,
        0.58,
        (
            (90, 255, 150)
            if dynamic_enabled
            else (text_header if mode_text == "MANUAL" else (170, 255, 255))
        ),
        1,
        line_type,
    )
    cv2.putText(
        frame,
        f"FPS: {system.current_fps:.1f}",
        (14, 52),
        font,
        0.58,
        text_main,
        1,
        line_type,
    )
    if playback_text:
        cv2.putText(
            frame,
            playback_text,
            (14, 75),
            font,
            0.50,
            text_main,
            1,
            line_type,
        )
    if system.video_writer is not None:
        cv2.putText(
            frame,
            "REC",
            (spx + spw - 48, 52),
            font,
            0.58,
            (0, 0, 255),
            2,
            line_type,
        )

    tab_y0 = spy + (86 if playback_text else 62)
    tab_h = 24
    tab_gap = 8
    tab_w = int((spw - (2 * pad_x) - tab_gap) / 2)
    tab_x0 = spx + pad_x
    auto_x0 = tab_x0 + tab_w + tab_gap
    manual_active = system.buildup_calibration_mode == "manual"
    auto_active = system.buildup_calibration_mode == "auto"
    cv2.rectangle(
        frame,
        (tab_x0, tab_y0),
        (tab_x0 + tab_w, tab_y0 + tab_h),
        (45, 145, 45) if manual_active else (75, 75, 75),
        -1,
    )
    cv2.rectangle(
        frame,
        (auto_x0, tab_y0),
        (auto_x0 + tab_w, tab_y0 + tab_h),
        (45, 120, 190) if auto_active else (75, 75, 75),
        -1,
    )
    cv2.putText(
        frame, "MANUAL", (tab_x0 + 14, tab_y0 + 16), font, 0.46, text_main, 1, line_type
    )
    cv2.putText(
        frame, "AUTO", (auto_x0 + 30, tab_y0 + 16), font, 0.46, text_main, 1, line_type
    )
    system.calibration_tab_rects = {
        "manual": (tab_x0, tab_y0, tab_w, tab_h),
        "auto": (auto_x0, tab_y0, tab_w, tab_h),
    }

    col_feature_x = spx + pad_x
    col_key_x = col_feature_x + max_feature_w + 18
    col_state_x = col_key_x + w_key + 20
    y = spy + (140 if playback_text else 116)
    cv2.putText(
        frame, left_col, (col_feature_x, y), font, scale, text_main, 1, line_type
    )
    cv2.putText(frame, mid_col, (col_key_x, y), font, scale, text_main, 1, line_type)
    cv2.putText(
        frame, right_col, (col_state_x, y), font, scale, text_main, 1, line_type
    )
    cv2.line(frame, (spx + 8, y + 4), (spx + spw - 8, y + 4), (110, 110, 110), 1)
    for group_name, group_rows in groups:
        y += group_h + 4
        cv2.putText(
            frame, group_name, (col_feature_x, y), font, 0.38, text_header, 1, line_type
        )
        for feature, key_label, value_txt in group_rows:
            y += row_h
            value_str = str(value_txt)
            value_color = text_muted
            if value_str in {"ON", "OFF"}:
                value_color = (70, 255, 70) if value_str == "ON" else (70, 70, 255)
            if feature == "Calib Mode":
                value_color = (255, 230, 80) if value_str == "AUTO" else (125, 255, 225)
            cv2.putText(
                frame,
                feature,
                (col_feature_x, y),
                font,
                scale,
                text_muted,
                1,
                line_type,
            )
            cv2.putText(
                frame, key_label, (col_key_x, y), font, scale, text_muted, 1, line_type
            )
            cv2.putText(
                frame,
                value_str,
                (col_state_x, y),
                font,
                scale,
                value_color,
                1,
                line_type,
            )


def draw_tip_tracks(system, frame, zone_results):
    """
    Draw nozzle tip track trails.

    Logic:
        Appends latest tip points per zone and draws a short fading polyline history.

    Args:
        system: Runtime system instance holding track-history buffers.
        frame: Target frame to draw onto (modified in-place).
        zone_results: Per-zone detection outputs for the current frame.

    Returns:
        None.
    """
    for zone in zone_results:
        zone_id = int(zone["zone_id"])
        pos = zone.get("position") if zone.get("raw_detected", False) else None
        history = system.tip_track_history[zone_id]
        if pos is not None:
            history.append((int(pos[0]), int(pos[1])))
        if len(history) < 2:
            continue
        pts = list(history)
        for idx in range(1, len(pts)):
            alpha = float(idx) / float(len(pts))
            color = (int(80 * (1.0 - alpha)), int(220 * alpha), 255)
            cv2.line(frame, pts[idx - 1], pts[idx], color, 2)


def draw_buildup_overlay(system, frame, zone_metrics):
    """
    Draw buildup masks and buildup visualization.

    Logic:
        Draws debug masks (annular/spray-exclusion/analysis) and then overlays detector buildup results.

    Args:
        system: Runtime system instance holding locks, flags, and detectors.
        frame: Target frame to draw onto (modified in-place).
        zone_metrics: Per-zone metrics payload for the current frame.

    Returns:
        None.
    """
    overlay_runtime = {
        "debug_enabled": bool(system.buildup_debug_overlay_enabled),
        "show_annular": bool(system.show_annular_mask),
        "show_spray_exclusion": bool(system.show_spray_exclusion_mask),
        "show_analysis": bool(system.show_analysis_mask),
    }
    overlay_runtime["show_any_debug"] = bool(
        overlay_runtime["show_annular"]
        or overlay_runtime["show_spray_exclusion"]
        or overlay_runtime["show_analysis"]
    )

    for metric in zone_metrics:
        buildup_info = metric.get("buildup_info", {})
        nozzle_info = metric.get("nozzle_info", {})
        zone_id = int(metric.get("zone_id", 0))
        lock_enabled = bool(system.zone_lock_enabled.get(zone_id, False))
        lock_snap = (
            system.zone_visual_lock_snapshots.get(zone_id) if lock_enabled else None
        )
        roi_bbox = buildup_info.get("roi_bbox")
        annular = buildup_info.get("annular_mask")
        analysis = buildup_info.get("analysis_mask")
        spray_exc = buildup_info.get("spray_exclusion_mask")
        zone_rect = metric.get("rect", (0, 0, 0, 0))
        if isinstance(lock_snap, dict):
            if lock_snap.get("roi_bbox_local") is not None:
                roi_bbox = lock_snap.get("roi_bbox_local")
            locked_rect = lock_snap.get("zone_rect")
            if isinstance(locked_rect, (tuple, list)) and len(locked_rect) == 4:
                zone_rect = tuple(int(v) for v in locked_rect)
            if isinstance(lock_snap.get("annular_mask"), np.ndarray):
                annular = lock_snap.get("annular_mask")
            if isinstance(lock_snap.get("analysis_mask"), np.ndarray):
                analysis = lock_snap.get("analysis_mask")
            if isinstance(lock_snap.get("spray_exclusion_mask"), np.ndarray):
                spray_exc = lock_snap.get("spray_exclusion_mask")
        if (
            overlay_runtime["debug_enabled"]
            and overlay_runtime["show_any_debug"]
            and roi_bbox is not None
        ):
            zx, zy, _, _ = zone_rect
            rx, ry, rw, rh = roi_bbox
            gx, gy, gw, gh = int(rx + zx), int(ry + zy), int(rw), int(rh)
            if gw > 2 and gh > 2:
                roi_frame = frame[gy : gy + gh, gx : gx + gw]
                roi_overlay = np.zeros((gh, gw, 3), dtype=np.uint8)
                if (
                    annular is not None
                    and annular.shape[:2] == (gh, gw)
                    and overlay_runtime["show_annular"]
                ):
                    roi_overlay[annular > 0] = (0, 165, 255)
                if (
                    spray_exc is not None
                    and spray_exc.shape[:2] == (gh, gw)
                    and overlay_runtime["show_spray_exclusion"]
                ):
                    roi_overlay[spray_exc > 0] = (180, 60, 220)
                if (
                    analysis is not None
                    and analysis.shape[:2] == (gh, gw)
                    and overlay_runtime["show_analysis"]
                ):
                    roi_overlay[analysis > 0] = (90, 135, 110)
                blend_mask = np.any(roi_overlay > 0, axis=2)
                if np.any(blend_mask):
                    blended = cv2.addWeighted(roi_overlay, 0.24, roi_frame, 0.76, 0.0)
                    roi_frame[blend_mask] = blended[blend_mask]

                if (
                    overlay_runtime["show_annular"]
                    and annular is not None
                    and annular.shape[:2] == (gh, gw)
                ):
                    ring_contours, _ = cv2.findContours(
                        annular, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                    )
                    if ring_contours:
                        shifted = []
                        for cnt in ring_contours:
                            cnt_g = cnt.copy()
                            cnt_g[:, 0, 0] = cnt_g[:, 0, 0] + int(gx)
                            cnt_g[:, 0, 1] = cnt_g[:, 0, 1] + int(gy)
                            shifted.append(cnt_g)
                        cv2.drawContours(
                            frame, shifted, -1, (0, 165, 255), 1, cv2.LINE_AA
                        )

        if not buildup_info.get("has_buildup", False):
            continue
        if roi_bbox is not None:
            zx, zy, _, _ = zone_rect
            rx, ry, rw, rh = roi_bbox
            global_bbox = (int(rx + zx), int(ry + zy), int(rw), int(rh))
            nozzle_global = dict(nozzle_info)
            nozzle_global["bounding_box"] = global_bbox
            frame[:] = system.zone_buildup_detectors[zone_id].visualize_buildup(
                frame,
                nozzle_global,
                {
                    **buildup_info,
                    "roi_bbox": global_bbox,
                },
            )


def compose_hud(system, frame, zone_metrics):
    """
    Compose right-side HUD canvas.

    Logic:
        Allocates a side canvas and draws one compact row per zone with key live values.

    Args:
        system: Runtime system instance holding display and metric options.
        frame: Source frame to place on left side of HUD canvas.
        zone_metrics: Per-zone metrics payload for the current frame.

    Returns:
        np.ndarray: New canvas containing source frame + HUD column.
    """
    h, w = frame.shape[:2]
    hud_w = max(260, int(system.hud_width))
    canvas = np.zeros((h, w + hud_w, 3), dtype=np.uint8)
    canvas[:, :w] = frame
    hud = canvas[:, w:]
    hud[:] = (14, 18, 24)
    cv2.putText(
        hud,
        f"FPS {system.current_fps:.1f}",
        (18, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (230, 230, 230),
        2,
    )
    cv2.putText(
        hud, "Nozzle HUD", (18, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 220, 255), 2
    )
    y = 88
    row_h = max(68, int((h - 110) / max(1, len(zone_metrics))))
    for metric in zone_metrics:
        y = draw_hud_row(system, hud, metric, y, row_h, hud_w)
        if y >= h - 20:
            break
    return canvas


def draw_hud_row(system, hud, metric, y, row_h, hud_w):
    """
    Draw one HUD row.

    Logic:
        Renders a single zone summary card in the HUD panel.

    Args:
        system: Runtime system instance (kept for uniform API and future extension).
        hud: HUD image region.
        metric: Single zone metric payload.
        y: Current top coordinate for the row.
        row_h: Height reserved for this row.
        hud_w: HUD width.

    Returns:
        int: Next y cursor after finishing this row.
    """
    _ = system
    zone_id = int(metric["zone_id"])
    status = str(metric.get("status", "OUT"))
    buildup = metric.get("buildup_info", {})
    spray = metric.get("spray_info", {})
    nozzle_info = metric.get("nozzle_info", {})
    color = tuple(int(v) for v in metric.get("color", (0, 220, 0)))
    x0, x1 = 12, hud_w - 12
    y1 = min(hud.shape[0] - 8, y + row_h - 8)
    cv2.rectangle(hud, (x0, y), (x1, y1), (28, 34, 42), -1)
    cv2.rectangle(hud, (x0, y), (x1, y1), color, 1)
    title = f"Arm {zone_id}  {status}"
    cv2.putText(hud, title, (x0 + 10, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)
    pos = nozzle_info.get("position")
    pos_text = "pos: -, -" if pos is None else f"pos: {int(pos[0])}, {int(pos[1])}"
    cv2.putText(
        hud,
        pos_text,
        (x0 + 10, y + 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
    )
    buildup_pct = 100.0 * float(buildup.get("buildup_percentage", 0.0))
    trend = buildup.get("trend") or "-"
    build_text = f"buildup: {buildup_pct:.1f}%  {trend}"
    build_color = (0, 80, 255) if buildup.get("has_buildup", False) else (0, 200, 0)
    cv2.putText(
        hud,
        build_text,
        (x0 + 10, y + 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        build_color,
        1,
    )
    spray_angle = spray.get("spray_angle_deg")
    spray_text = (
        "spray: -" if spray_angle is None else f"spray: {float(spray_angle):.1f}"
    )
    cv2.putText(
        hud,
        spray_text,
        (x0 + 210, y + 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 255, 180),
        1,
    )
    return y + row_h


def draw_zone_grid_overlay(system, frame, zone_results, zone_metrics=None):
    """
    Draw zone grid overlay.

    Logic:
        Draws per-zone boxes/polygons, status colors, and optional head/body/spray debug primitives.

    Args:
        system: Runtime system instance holding configs/state and helper methods.
        frame: Target frame to draw onto (modified in-place).
        zone_results: Per-zone detection outputs for the current frame.

    Returns:
        None.
    """
    dynamic_enabled = bool(
        getattr(system.zone_orchestrator, "dynamic_zoning_enabled", False)
    )
    buildup_yes_by_zone = {}
    if isinstance(zone_metrics, list):
        for metric in zone_metrics:
            try:
                zid = int(metric.get("zone_id", 0))
            except Exception:
                zid = 0
            if zid <= 0:
                continue
            binfo = metric.get("buildup_info", {}) or {}
            buildup_yes_by_zone[zid] = bool(binfo.get("has_buildup", False))
    force_show_zones = system.current_mode == "MANUAL"
    for zone in zone_results:
        zone_id = int(zone.get("zone_id", 0))
        raw_detected = bool(zone.get("raw_detected", False))
        confirmed_detected = bool(zone.get("detected", False))
        if not raw_detected and zone_id > 0:
            system._stabilize_debug_bbox(zone_id, None)
        if (
            (not force_show_zones)
            and system.hide_undetected_zone_boxes
            and not (raw_detected or confirmed_detected)
        ):
            continue

        # DL gate: if AI classified this zone as BIG, render nothing
        # (the X cross is added later by frame_pipeline._draw_skipped_zone_crosses).
        _dl_big_skip = False
        try:
            _ticker = getattr(system, "_dl_status_ticker", None)
            if _ticker is not None:
                _status = _ticker.get_status_for_zone(int(zone_id))
                if _status is not None and int(_status.get("class_id", -1)) == 3:
                    _dl_big_skip = True
        except Exception:
            _dl_big_skip = False
        if _dl_big_skip:
            continue

        x, y, w, h = zone["rect"]
        status = zone.get("status", "OUT")
        is_owned = isinstance(status, str) and status.startswith("OWNED_BY_")
        is_in = status == "IN"
        is_confirming = status == "CONFIRMING"
        zone_color = (
            (180, 0, 180)
            if is_owned
            else (
                (0, 220, 0)
                if is_in
                else ((0, 220, 220) if is_confirming else (0, 0, 220))
            )
        )
        if bool(buildup_yes_by_zone.get(zone_id, False)):
            zone_color = tuple(
                system.display_config.get("warning_color", [0, 165, 255])
            )
        if system.lock_zone_select_mode and (
            zone_id == int(system.selected_lock_zone_id)
        ):
            zone_color = (0, 255, 255)
        zone_angle_deg = float(zone.get("angle_deg", 0.0))
        if abs(zone_angle_deg) > 1e-6:
            zone_poly = manual_zone_geometry.manual_zone_polygon(
                (x, y, w, h), zone_angle_deg
            )
            cv2.polylines(frame, [zone_poly], True, zone_color, 2, cv2.LINE_AA)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), zone_color, 2)

        local_debug = zone.get("local_nozzle_info", {}).get("debug_details", {}) or {}
        if not raw_detected:
            shaft_pt = local_debug.get("best_shaft_endpoint")
            tip_pt = local_debug.get("best_tip_candidate")
            if shaft_pt is not None:
                sp = (int(round(float(shaft_pt[0]))), int(round(float(shaft_pt[1]))))
                cv2.drawMarker(frame, sp, (255, 200, 0), cv2.MARKER_CROSS, 14, 1)
            if tip_pt is not None:
                tp = (int(round(float(tip_pt[0]))), int(round(float(tip_pt[1]))))
                cv2.circle(frame, tp, 5, (0, 180, 255), 1)

        if dynamic_enabled and system.show_dynamic_zone_debug_overlay:
            cx = int(round(x + (w * 0.5)))
            cy = int(round(y + (h * 0.5)))
            cv2.circle(frame, (cx, cy), 4, (255, 255, 0), -1)
            cv2.circle(frame, (cx, cy), 9, (255, 150, 0), 1)

        if raw_detected:
            bbox = zone.get("bounding_box")
            pos = zone.get("position")
            ang = zone.get("orientation")
            body_polygon = zone.get("body_polygon")
            body_dir = zone.get("body_direction")
            spray_info = zone.get("spray_info", {})

            if bbox is not None:
                lock_enabled = bool(system.zone_lock_enabled.get(int(zone_id), False))
                if lock_enabled:
                    draw_bbox = system._stabilize_debug_bbox(zone_id, bbox)
                else:
                    system._stabilize_debug_bbox(zone_id, None)
                    draw_bbox = bbox
                bx, by, bw, bh = draw_bbox if draw_bbox is not None else bbox
                if system.show_nozzle_head_circle:
                    center = (int(round(bx + (0.5 * bw))), int(round(by + (0.5 * bh))))
                    radius = max(2, int(round(0.5 * max(float(bw), float(bh)))))
                    cv2.circle(frame, center, radius, (0, 255, 255), 2)

            if pos is not None:
                cv2.circle(frame, pos, 3, (0, 255, 255), -1)
                if system.show_nozzle_angle_arrow and (ang is not None):
                    try:
                        theta = np.deg2rad(float(ang))
                        arrow_len = max(10.0, float(system.body_direction_arrow_len_px))
                        ex = int(round(float(pos[0]) + (np.cos(theta) * arrow_len)))
                        ey = int(round(float(pos[1]) + (np.sin(theta) * arrow_len)))
                        cv2.arrowedLine(
                            frame,
                            (int(pos[0]), int(pos[1])),
                            (ex, ey),
                            (20, 20, 255),
                            2,
                            tipLength=0.26,
                        )
                    except Exception:
                        pass

            if system.show_body_check_overlay and body_polygon is not None:
                cv2.polylines(
                    frame, [body_polygon.astype(np.int32)], True, (255, 120, 0), 2
                )
                if pos is not None:
                    cv2.circle(
                        frame,
                        pos,
                        int(system.body_check_head_ignore_radius_px),
                        (255, 0, 255),
                        1,
                    )

            if (
                system.show_body_direction_overlay
                and pos is not None
                and body_dir is not None
            ):
                ux, uy = float(body_dir[0]), float(body_dir[1])
                norm = float(np.hypot(ux, uy))
                if norm > 1e-6:
                    ux /= norm
                    uy /= norm
                    arrow_len = max(10.0, float(system.body_direction_arrow_len_px))
                    ex = int(round(float(pos[0]) + (ux * arrow_len)))
                    ey = int(round(float(pos[1]) + (uy * arrow_len)))
                    cv2.arrowedLine(
                        frame,
                        (int(pos[0]), int(pos[1])),
                        (ex, ey),
                        (0, 200, 255),
                        2,
                        tipLength=0.26,
                    )

            # DL gate: hide spray overlays for zones the AI says are not measurable for spray
            # (BIG, NO_SPRAY, FACING). Safe default = show if AI hasn't classified yet.
            _dl_allows_spray = True
            try:
                _ticker = getattr(system, "_dl_status_ticker", None)
                if _ticker is not None:
                    _status = _ticker.get_status_for_zone(int(zone_id))
                    if _status is not None:
                        _cid = int(_status.get("class_id", -1))
                        # Only class 0 (SPRAY) keeps spray visualization
                        _dl_allows_spray = _cid == 0
            except Exception:
                _dl_allows_spray = True

            spray_detected = bool(spray_info.get("detected", False))
            if (
                system.spray_overlay_enabled
                and system.show_spray_angle_lines
                and spray_detected
                and _dl_allows_spray
            ):
                spray_line_color = (0, 255, 0)
                raw_lines = spray_info.get("lines", [])
                anchor = spray_info.get("anchor_point")
                if anchor is None and pos is not None:
                    anchor = (int(pos[0]), int(pos[1]))
                if anchor is None and raw_lines:
                    ax = int(round(float(np.mean([ln[0][0] for ln in raw_lines]))))
                    ay = int(round(float(np.mean([ln[0][1] for ln in raw_lines]))))
                    anchor = (ax, ay)

                rendered_endpoints = []
                axis_deg = spray_info.get("spray_axis_deg")
                half_deg = spray_info.get("spray_half_angle_deg")
                if anchor is not None and axis_deg is not None and half_deg is not None:
                    if raw_lines:
                        ray_len = float(
                            np.mean(
                                [
                                    np.hypot(ln[1][0] - ln[0][0], ln[1][1] - ln[0][1])
                                    for ln in raw_lines
                                ]
                            )
                        )
                    else:
                        ray_len = 24.0
                    ray_len = max(10.0, ray_len)
                    for sign in (+1.0, -1.0):
                        a = np.deg2rad(float(axis_deg) + (sign * float(half_deg)))
                        ex = int(round(float(anchor[0]) + (np.cos(a) * ray_len)))
                        ey = int(round(float(anchor[1]) + (np.sin(a) * ray_len)))
                        rendered_endpoints.append((ex, ey))
                        cv2.line(
                            frame,
                            (int(anchor[0]), int(anchor[1])),
                            (ex, ey),
                            spray_line_color,
                            2,
                            cv2.LINE_AA,
                        )
                else:
                    for p1, p2 in raw_lines:
                        if anchor is None:
                            cv2.line(frame, p1, p2, spray_line_color, 2, cv2.LINE_AA)
                            continue
                        dx = float(p2[0] - p1[0])
                        dy = float(p2[1] - p1[1])
                        norm = float(np.hypot(dx, dy))
                        if norm <= 1e-6:
                            continue
                        ray_len = max(10.0, norm)
                        ex = int(round(float(anchor[0]) + ((dx / norm) * ray_len)))
                        ey = int(round(float(anchor[1]) + ((dy / norm) * ray_len)))
                        rendered_endpoints.append((ex, ey))
                        cv2.line(
                            frame,
                            (int(anchor[0]), int(anchor[1])),
                            (ex, ey),
                            spray_line_color,
                            2,
                            cv2.LINE_AA,
                        )

                if anchor is not None and len(rendered_endpoints) >= 2:
                    a0 = float(
                        np.degrees(
                            np.arctan2(
                                rendered_endpoints[0][1] - anchor[1],
                                rendered_endpoints[0][0] - anchor[0],
                            )
                        )
                    )
                    a1 = float(
                        np.degrees(
                            np.arctan2(
                                rendered_endpoints[1][1] - anchor[1],
                                rendered_endpoints[1][0] - anchor[0],
                            )
                        )
                    )
                    delta = ((a1 - a0 + 540.0) % 360.0) - 180.0
                    start = a0
                    end = a0 + delta
                    arc_r = max(
                        6,
                        int(
                            round(
                                0.30
                                * min(
                                    np.hypot(
                                        rendered_endpoints[0][0] - anchor[0],
                                        rendered_endpoints[0][1] - anchor[1],
                                    ),
                                    np.hypot(
                                        rendered_endpoints[1][0] - anchor[0],
                                        rendered_endpoints[1][1] - anchor[1],
                                    ),
                                )
                            )
                        ),
                    )
                    cv2.ellipse(
                        frame,
                        (int(anchor[0]), int(anchor[1])),
                        (arc_r, arc_r),
                        0.0,
                        float(start),
                        float(end),
                        spray_line_color,
                        1,
                        cv2.LINE_AA,
                    )

            if system.show_spray_zone_overlay and spray_detected and _dl_allows_spray:
                roi_poly = spray_info.get("roi_polygon")
                if roi_poly is not None:
                    zone_x, zone_y, _, _ = (x, y, w, h)
                    roi_global = roi_poly.astype(np.int32).copy()
                    roi_global[:, 0] += int(zone_x)
                    roi_global[:, 1] += int(zone_y)
                    cv2.polylines(frame, [roi_global], True, (255, 120, 0), 2)


def draw_all_zone_metrics(system, frame, zone_metrics):
    """
    Draw all zone metric panels.

    Logic:
        Draws one panel per visible zone and tracks overlap to decide if HUD fallback should be used.

    Args:
        system: Runtime system instance holding display state and helper methods.
        frame: Target frame to draw onto (modified in-place).
        zone_metrics: Per-zone metrics payload for the current frame.

    Returns:
        bool: True when any panel overlaps focus regions, else False.
    """
    force_show_metrics = system.current_mode == "MANUAL"
    visible_metrics = []
    for metric in zone_metrics:
        if (
            (not force_show_metrics)
            and system.hide_undetected_zone_boxes
            and not bool(metric.get("raw_detected", False))
        ):
            continue
        # DL gate: skip side panel for BIG-classified zones (X cross is enough).
        _gate = metric.get("dl_gate") if isinstance(metric, dict) else None
        if _gate and bool(_gate.get("skip_all", False)):
            continue
        visible_metrics.append(metric)
    focus_rects = []
    for metric in visible_metrics:
        if bool(metric.get("raw_detected", False)) or bool(
            metric.get("nozzle_info", {}).get("detected", False)
        ):
            zx, zy, zw, zh = metric["rect"]
            focus_rects.append((int(zx), int(zy), int(zw), int(zh)))
    if not focus_rects:
        for metric in visible_metrics:
            zx, zy, zw, zh = metric["rect"]
            focus_rects.append((int(zx), int(zy), int(zw), int(zh)))
    used_panel_rects = []
    had_focus_overlap = False
    for metric in visible_metrics:
        overlap_focus = draw_zone_metrics_panel(
            system, frame, metric, focus_rects, used_panel_rects
        )
        had_focus_overlap = had_focus_overlap or (int(overlap_focus) > 0)
    return had_focus_overlap


def draw_zone_metrics_panel(
    system, frame, metric, focus_rects=None, used_panel_rects=None
):
    """
    Draw one zone metrics panel.

    Logic:
        Builds textual summary for one nozzle and places the panel using overlap-aware anchor selection.

    Args:
        system: Runtime system instance holding style flags and anchor state.
        frame: Target frame to draw onto (modified in-place).
        metric: Single zone metric payload.
        focus_rects: Optional focus rectangles to avoid when placing panel.
        used_panel_rects: Optional list tracking already-used panel rectangles.

    Returns:
        int: Focus overlap area for this panel.
    """
    nozzle_info = metric["nozzle_info"]
    buildup_info = metric["buildup_info"]
    spray_info = metric.get("spray_info", {})
    detection_strength = str(metric.get("detection_strength", "none")).strip().lower()
    zone_id = metric["zone_id"]
    io_state = str(metric.get("io_state", "OUT")).strip().upper()

    font = cv2.FONT_HERSHEY_SIMPLEX
    line_height = 22
    pad_x = 6
    pad_y = 8

    detected = nozzle_info.get("detected", False)
    io_color = (0, 220, 0) if io_state == "IN" else (0, 0, 255)

    # AI classification status (replaces the Detected: WEAK/STRONG line).
    _DL_GATE_COLORS = {
        0: (0, 220, 0),  # SPRAY -> green
        1: (0, 165, 255),  # NO_SPRAY -> orange
        2: (50, 50, 255),  # FACING -> red
        3: (255, 100, 50),  # BIG -> blue
    }
    _DL_GATE_SHORT = {0: "SPRAY", 1: "NO_SPRAY", 2: "FACING", 3: "BIG"}
    dl_gate = metric.get("dl_gate") if isinstance(metric, dict) else None
    dl_cid = dl_gate.get("class_id") if dl_gate else None
    if dl_cid is None:
        ai_text = "AI: ..."
        ai_color = (140, 140, 140)
    else:
        ai_name = _DL_GATE_SHORT.get(int(dl_cid), str(dl_cid))
        ai_score = float(dl_gate.get("score", 0.0)) if dl_gate else 0.0
        ai_text = f"AI: {ai_name} {ai_score:.2f}"
        ai_color = _DL_GATE_COLORS.get(int(dl_cid), (200, 200, 200))

    nozzle_angle = nozzle_info.get("orientation")
    nozzle_angle_text = "N/A" if nozzle_angle is None else f"{float(nozzle_angle):.1f}"
    spray_angle = spray_info.get("spray_angle_deg")
    spray_text = "N/A" if spray_angle is None else f"{float(spray_angle):.1f}"
    buildup_flag = bool(buildup_info.get("has_buildup", False))
    buildup_pct = float(buildup_info.get("buildup_percentage", 0.0)) * 100.0
    buildup_text = "YES" if buildup_flag else "NO"
    # YES (>= threshold) -> warning orange; NO (below threshold) -> safe green, so a
    # dim neutral text over the warm nozzle panel is never mistaken for an alert.
    buildup_color = (0, 80, 255) if buildup_flag else (0, 200, 0)
    lock_enabled = bool(system.zone_lock_enabled.get(int(zone_id), True))
    lock_txt = "ON" if lock_enabled else "OFF"
    lock_color = (80, 225, 80) if lock_enabled else (80, 80, 255)
    reject_reason_raw = str(metric.get("reject_reason") or "").strip()
    detect_method_raw = str(metric.get("detection_method") or "").strip()
    reject_reason = reject_reason_raw if reject_reason_raw else "nominal_tracking"
    detect_method = detect_method_raw if detect_method_raw else "adaptive_fusion"
    detect_method_label = detect_method.replace("_", " ").replace("-", " ").title()
    reject_reason_label = reject_reason.replace("_", " ").replace("-", " ").title()
    reason_color = (0, 0, 255) if io_state == "OUT" else (150, 150, 150)
    compact_reason = reject_reason_label
    compact_method = detect_method_label.replace(" ", "")
    if compact_reason.lower() == "nominal tracking":
        compact_reason = "Nominal"
    if len(compact_reason) > 16:
        compact_reason = compact_reason[:13] + "..."
    if len(compact_method) > 16:
        compact_method = compact_method[:13] + "..."
    reason_line = f"Status: {compact_reason}"
    engine_line = f"Method: {compact_method}"

    lines = [
        (f"Nozzle: {zone_id}", 0.52, (255, 255, 255), 1),
        (f"FPS: {system.current_fps:.1f}", 0.47, (255, 255, 255), 1),
        (f"State: {io_state}", 0.52, io_color, 1),
        (ai_text, 0.52, ai_color, 1),
        (f"Nozzle angle: {nozzle_angle_text}", 0.47, (255, 255, 255), 1),
        (f"Spray angle: {spray_text}", 0.47, (255, 255, 255), 1),
        (f"Buildup: {buildup_text} ({buildup_pct:.1f}%)", 0.47, buildup_color, 1),
    ]
    if system.lock_debug_visible:
        lines.append((f"Lock: {lock_txt}", 0.47, lock_color, 1))
    lines.append((engine_line, 0.40, (135, 190, 255), 1))

    max_text_w = 0
    for text, scale, _draw_color, thick in lines:
        (tw, _th), _base = cv2.getTextSize(text, font, scale, thick)
        max_text_w = max(max_text_w, int(tw))
    panel_w = int(np.clip(max_text_w + (2 * pad_x) + 10, 176, 248))
    panel_h = int((pad_y + 8) + (line_height * len(lines)) + pad_y)
    if str(getattr(system, "current_mode", "")).upper() == "FIX":
        zx, zy, zw, zh = [int(v) for v in metric.get("rect", (0, 0, panel_w, panel_h))]
        max_panel_w = max(1, int(zw - 4))
        max_panel_h = max(1, int(zh - 4))
        panel_w = int(min(panel_w, max_panel_w))
        panel_h = int(min(panel_h, max_panel_h))
        max_lines_fit = max(
            1, int((panel_h - (pad_y + 8 + pad_y)) // max(1, line_height))
        )
        if len(lines) > max_lines_fit:
            if max_lines_fit >= 2:
                lines = lines[: max_lines_fit - 1] + [("...", 0.40, (170, 170, 170), 1)]
            else:
                lines = [lines[0]]
    panel_x0, panel_y0, overlap_focus = compute_metric_panel_anchor(
        system,
        frame.shape,
        metric,
        panel_w,
        panel_h,
        focus_rects=focus_rects,
        used_panel_rects=used_panel_rects,
    )
    panel_x1 = min(frame.shape[1] - 1, panel_x0 + panel_w)
    panel_y1 = min(frame.shape[0] - 1, panel_y0 + panel_h)
    if isinstance(used_panel_rects, list):
        used_panel_rects.append(
            (int(panel_x0), int(panel_y0), int(panel_w), int(panel_h))
        )

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x0, panel_y0), (panel_x1, panel_y1), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)

    tx = panel_x0 + pad_x
    y_offset = panel_y0 + pad_y + 8
    for text, scale, draw_color, thick in lines:
        cv2.putText(frame, text, (tx, y_offset), font, scale, draw_color, thick)
        y_offset += line_height
    return int(overlap_focus)


def rect_overlap_area(a, b):
    """Intersection area in pixels of two (x, y, w, h) rectangles."""
    ax, ay, aw, ah = [int(v) for v in a]
    bx, by, bw, bh = [int(v) for v in b]
    iw = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0, min(ay + ah, by + bh) - max(ay, by))
    return iw * ih


def compute_metric_panel_anchor(
    system,
    frame_shape,
    metric,
    panel_w,
    panel_h,
    focus_rects=None,
    used_panel_rects=None,
):
    """
    Compute metric panel anchor.

    Logic:
        Scores candidate panel anchors to minimize overlap and jitter, then stores the chosen anchor per zone.

    Args:
        system: Runtime system instance holding anchor state and placement options.
        frame_shape: Frame shape tuple.
        metric: Single zone metric payload.
        panel_w: Panel width in pixels.
        panel_h: Panel height in pixels.
        focus_rects: Optional focus rectangles to avoid.
        used_panel_rects: Optional rectangles already used by other panels.

    Returns:
        tuple[int, int, int]: `(x, y, overlap_focus)` chosen anchor and focus-overlap amount.
    """
    frame_h, frame_w = frame_shape[:2]
    margin = 6
    gap = 8
    mode_txt = str(getattr(system, "current_mode", "")).upper()
    fixed_mode = mode_txt == "FIX"
    manual_mode = mode_txt == "MANUAL"
    prefer_near_zone = (
        bool(getattr(system, "metric_panel_prefer_near_zone", False)) or manual_mode
    )
    zone_id = max(1, int(metric.get("zone_id", 1)))
    zx, zy, zw, zh = [int(v) for v in metric.get("rect", (0, 0, panel_w, panel_h))]
    zone_cx = float(zx + (0.5 * zw))
    zone_cy = float(zy + (0.5 * zh))
    own_rect = (int(zx), int(zy), int(zw), int(zh))
    frame_cx = 0.5 * float(frame_w)
    frame_cy = 0.5 * float(frame_h)

    rows_per_col = max(
        1, int((frame_h - (2 * margin) + gap) // max(1, (panel_h + gap)))
    )
    cols_fit = max(1, int((frame_w - (2 * margin) + gap) // max(1, (panel_w + gap))))
    max_cols = max(1, min(cols_fit, max(2, system.total_zones)))

    candidates = []
    if fixed_mode:
        # FIX mode: panel must stay inside its own zone at zone corners.
        zone_margin = max(2, int(gap // 2))
        candidates.extend(
            [
                (int(zx + zone_margin), int(zy + zone_margin)),  # top-left
                (
                    int(zx + zw - panel_w - zone_margin),
                    int(zy + zone_margin),
                ),  # top-right
                (
                    int(zx + zone_margin),
                    int(zy + zh - panel_h - zone_margin),
                ),  # bottom-left
                (
                    int(zx + zw - panel_w - zone_margin),
                    int(zy + zh - panel_h - zone_margin),
                ),  # bottom-right
            ]
        )
    else:
        if prefer_near_zone:
            candidates.extend(
                [
                    (int(zx + zw + gap), int(zy - 2)),
                    (int(zx + zw + gap), int(zy + zh - panel_h + 2)),
                    (int(zx + zw + gap), int(zy + (0.5 * zh) - (0.5 * panel_h))),
                    (int(zx - panel_w - gap), int(zy - 2)),
                    (int(zx - panel_w - gap), int(zy + zh - panel_h + 2)),
                    (int(zx - panel_w - gap), int(zy + (0.5 * zh) - (0.5 * panel_h))),
                    (int(zx + 2), int(zy - panel_h - gap)),
                    (int(zx + zw - panel_w - 2), int(zy - panel_h - gap)),
                    (int(zx + (0.5 * zw) - (0.5 * panel_w)), int(zy - panel_h - gap)),
                    (int(zx + 2), int(zy + zh + gap)),
                    (int(zx + zw - panel_w - 2), int(zy + zh + gap)),
                    (int(zx + (0.5 * zw) - (0.5 * panel_w)), int(zy + zh + gap)),
                ]
            )

        for col in range(max_cols):
            for row in range(rows_per_col):
                candidates.append(
                    (
                        int(frame_w - margin - panel_w - (col * (panel_w + gap))),
                        int(margin + (row * (panel_h + gap))),
                    )
                )
        for col in range(max_cols):
            for row in range(rows_per_col):
                candidates.append(
                    (
                        int(margin + (col * (panel_w + gap))),
                        int(margin + (row * (panel_h + gap))),
                    )
                )
        for col in range(max_cols):
            candidates.append((int(margin + (col * (panel_w + gap))), int(margin)))
            candidates.append(
                (int(margin + (col * (panel_w + gap))), int(frame_h - margin - panel_h))
            )

    focus_rects = list(focus_rects or [])
    used_panel_rects = list(used_panel_rects or [])
    reserved_rects = [tuple(int(v) for v in system.status_panel_rect)]
    prev = system.metric_panel_anchor_state.get(zone_id)

    best = None
    best_score = None
    best_overlap_focus = 0
    best_strict = None
    best_strict_score = None
    best_relaxed = None
    best_relaxed_score = None
    for px, py in candidates:
        if fixed_mode:
            min_x = int(zx + 2)
            max_x = int(max(min_x, (zx + zw) - panel_w - 2))
            min_y = int(zy + 2)
            max_y = int(max(min_y, (zy + zh) - panel_h - 2))
            px = int(np.clip(px, min_x, max_x))
            py = int(np.clip(py, min_y, max_y))
        else:
            px = int(np.clip(px, 4, max(4, frame_w - panel_w - 4)))
            py = int(np.clip(py, 4, max(4, frame_h - panel_h - 4)))
        rect = (px, py, int(panel_w), int(panel_h))

        overlap_focus = 0
        for frect in focus_rects:
            if fixed_mode and tuple(int(v) for v in frect) == own_rect:
                continue
            overlap_focus += rect_overlap_area(rect, frect)
        overlap_own = 0 if fixed_mode else rect_overlap_area(rect, own_rect)

        overlap_others = 0
        for orect in used_panel_rects:
            overlap_others += rect_overlap_area(rect, orect)
        for rrect in reserved_rects:
            overlap_others += rect_overlap_area(rect, rrect)

        stability_cost = 0.0
        if prev is not None:
            dx = float(px - prev[0])
            dy = float(py - prev[1])
            stability_cost = float(np.hypot(dx, dy))

        panel_cx = float(px + (0.5 * panel_w))
        panel_cy = float(py + (0.5 * panel_h))
        zone_distance = float(np.hypot(panel_cx - zone_cx, panel_cy - zone_cy))
        center_distance = float(np.hypot(panel_cx - frame_cx, panel_cy - frame_cy))

        score = (
            (1000.0 * float(overlap_focus))
            + (5000.0 * float(overlap_own))
            + (1200.0 * float(overlap_others))
            + (0.22 * stability_cost)
        )
        if prefer_near_zone:
            score += 0.40 * zone_distance
        score -= 0.12 * center_distance

        strict_ok = (int(overlap_focus) == 0) and (int(overlap_others) == 0)
        if strict_ok:
            if best_strict_score is None or score < best_strict_score:
                best_strict = (px, py)
                best_strict_score = score
        else:
            relaxed_score = (
                (float(overlap_own) * 10000.0)
                + (float(overlap_others) * 10000.0)
                + score
            )
            if best_relaxed_score is None or relaxed_score < best_relaxed_score:
                best_relaxed = (px, py)
                best_relaxed_score = relaxed_score

        if best_score is None or score < best_score:
            best = (px, py)
            best_score = score
            best_overlap_focus = int(overlap_focus)

    if best is None:
        best = (4, 4)
        best_overlap_focus = int(panel_w * panel_h)
    elif best_strict is not None:
        best = best_strict
        best_overlap_focus = 0
    elif best_relaxed is not None:
        best = best_relaxed
    system.metric_panel_anchor_state[zone_id] = (int(best[0]), int(best[1]))
    return int(best[0]), int(best[1]), int(best_overlap_focus)
