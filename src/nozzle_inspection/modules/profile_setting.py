"""
Profile Setting (MANUAL mode) helpers extracted from run_inspection.py.
"""

import cv2
import numpy as np

from . import auto_roi, manual_zone_geometry, profile_store, ui_dialogs
from .dl_inference import CLASS_NAMES

# BGR colors per AFGNet class — match afgnet_onnx.py palette
AI_CLASS_COLORS = {
    0: (0, 220, 0),  # SPRAY  -> green
    1: (0, 165, 255),  # NO_SPRAY -> orange
    2: (50, 50, 255),  # FACING -> red
    3: (255, 100, 50),  # BIG (filtered, never drawn) -> blue
}
AI_CLASS_SHORT = {
    0: "SPRAY",
    1: "NO_SPRAY",
    2: "FACING",
    3: "BIG",
}


def _ai_classify_existing_zones(system, frame, zones):
    """One-shot AI classification of existing profile zones (no ROI generation).

    Runs DL detect on the current frame, maps detections onto the given zones
    by IoU/containment, and populates `system._ai_meta_by_idx` so the editor
    can render class colors + labels immediately.

    Returns the same `zones` list unchanged (only metadata is populated).
    """
    dl = system.get_dl_inference() if hasattr(system, "get_dl_inference") else None
    if dl is None or not zones:
        return zones

    from .dl_inference import CLASS_NAMES, map_detections_to_rois

    rois = []
    for z in zones:
        try:
            x, y, w, h, _ = manual_zone_geometry.manual_zone_unpack(system, z)
            rois.append((int(x), int(y), int(w), int(h)))
        except Exception:
            rois.append((0, 0, 0, 0))

    try:
        detections = dl.detect(frame)
    except Exception as exc:
        print(f"[AI] classify failed: {exc}")
        return zones

    mapping = map_detections_to_rois(detections, rois)

    meta_list = [None] * len(zones)
    for roi_idx, det in mapping.items():
        if 0 <= roi_idx < len(meta_list):
            cid = int(det.get("class_id", -1))
            meta_list[roi_idx] = {
                "source": "auto_ai_classify_existing",
                "class_id": cid,
                "class_name": CLASS_NAMES[cid] if 0 <= cid < len(CLASS_NAMES) else None,
                "dl_score": float(det.get("score", 0.0)),
                "observability": float(det.get("observability", 0.0)),
            }
    system._ai_meta_by_idx = meta_list
    classified = sum(1 for m in meta_list if m is not None)
    print(f"[AI] Re-classified existing profile: {classified}/{len(zones)} zone(s).")
    return zones


def _show_ai_pipeline_debug(system, frame):
    """Render the AI pipeline (bbox -> expanded -> CV head -> ROI) in a debug window."""
    dl = system.get_dl_inference() if hasattr(system, "get_dl_inference") else None
    if dl is None:
        print("\n[AI debug] Model not available.")
        return
    nozzle_cfg = (
        dict(system.nozzle_config) if isinstance(system.nozzle_config, dict) else {}
    )
    roi_size = int(
        system.nozzle_config.get("auto_roi_size_px", auto_roi.DEFAULT_ROI_SIZE_PX)
    )
    expand_ratio = float(
        system.nozzle_config.get(
            "auto_roi_bbox_expand", auto_roi.DEFAULT_BBOX_EXPAND_RATIO
        )
    )

    main_window = system.display_config.get("window_name", "Nozzle Inspection")
    busy = frame.copy()
    cv2.putText(
        busy,
        "AI pipeline debug...",
        (40, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 220, 255),
        2,
    )
    cv2.imshow(main_window, system._scale_display_frame(busy))
    cv2.waitKey(1)

    debug_img = auto_roi.visualize_pipeline(
        frame_bgr=frame,
        dl=dl,
        nozzle_config=nozzle_cfg,
        roi_size_px=roi_size,
        bbox_expand_ratio=expand_ratio,
        include_big_nozzle=True,
    )

    debug_window = "AI Pipeline Debug (press any key to close)"
    cv2.imshow(debug_window, debug_img)
    cv2.waitKey(0)
    try:
        cv2.destroyWindow(debug_window)
    except Exception:
        pass


def _print_ai_summary(new_meta_list, total_dropped_big):
    """Console summary of AI detection breakdown."""
    if not new_meta_list:
        return
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for m in new_meta_list:
        cid = int(m.get("class_id", -1))
        if cid in counts:
            counts[cid] += 1
    parts = []
    for cid in (0, 1, 2, 3):
        if counts[cid] > 0:
            parts.append(f"{counts[cid]} {AI_CLASS_SHORT[cid]}")
    msg = "[AI] Breakdown: " + ", ".join(parts) if parts else "[AI] Breakdown: none"
    if total_dropped_big > 0:
        msg += f" (filtered: {total_dropped_big} BIG)"
    print(msg)


def _run_ai_auto_detect(system, frame, current_zones, frame_shape):
    """AI auto-detect ROIs and merge with current zones based on user choice.

    Behaviour:
        - If no DL model available: print error, return current_zones unchanged.
        - If no current_zones: returns the AI-generated zones directly.
        - If current_zones already populated: opens a 3-way dialog asking
          Replace / Append / Cancel and returns merged zones accordingly.

    Returns the resulting zones list (sanitized).
    """
    dl = system.get_dl_inference() if hasattr(system, "get_dl_inference") else None
    if dl is None:
        print(
            "\n[AI] AFGNet model not available. Check 'dl_model_path' in config "
            "or 'onnxruntime' installation."
        )
        return current_zones

    window_name = system.display_config.get("window_name", "Nozzle Inspection")

    # Inform user during what may be a slow first-call inference (~6ms after warmup,
    # but cold-load + first detect can take ~1-2s).
    busy = frame.copy()
    cv2.putText(
        busy,
        "AI detecting nozzles...",
        (40, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 220, 255),
        2,
    )
    cv2.imshow(window_name, system._scale_display_frame(busy))
    cv2.waitKey(1)

    nozzle_cfg = (
        dict(system.nozzle_config) if isinstance(system.nozzle_config, dict) else {}
    )
    roi_size = int(
        system.nozzle_config.get("auto_roi_size_px", auto_roi.DEFAULT_ROI_SIZE_PX)
    )
    expand_ratio = float(
        system.nozzle_config.get(
            "auto_roi_bbox_expand", auto_roi.DEFAULT_BBOX_EXPAND_RATIO
        )
    )

    existing_rects = []
    for z in current_zones or []:
        try:
            x, y, w, h, _ = manual_zone_geometry.manual_zone_unpack(system, z)
            existing_rects.append((int(x), int(y), int(w), int(h)))
        except Exception:
            continue

    has_existing = bool(current_zones)

    # If there are existing zones, ask user how to merge BEFORE running expensive
    # inference would also be valid, but running it first means the user sees a
    # consistent dialog regardless of inference outcome. We do it AFTER so
    # cancellation is cheap, but inference itself is fast (~tens of ms).
    new_zones_raw = auto_roi.generate_auto_rois(
        frame_bgr=frame,
        dl=dl,
        nozzle_config=nozzle_cfg,
        existing_rois=existing_rects if has_existing else None,
        roi_size_px=roi_size,
        bbox_expand_ratio=expand_ratio,
    )

    if not new_zones_raw:
        print(
            "\n[AI] No new nozzles detected (or all detections overlapped existing ROIs)."
        )
        return current_zones

    # Convert auto_roi schema -> editor zone spec tuples
    def _to_zone_spec(item):
        rect = item.get("rect") or [0, 0, 0, 0]
        ang = float(item.get("angle_deg", 0.0))
        return manual_zone_geometry.sanitize_manual_zone(
            system,
            (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]), ang),
            frame_shape,
        )

    new_zone_specs = [_to_zone_spec(it) for it in new_zones_raw]
    new_meta = [dict(it.get("meta") or {}) for it in new_zones_raw]

    # Also count BIG nozzles that were filtered out before ROI generation
    raw_dets = dl.detect(frame)
    n_big_filtered = sum(1 for d in raw_dets if int(d.get("class_id", -1)) == 3)

    if not has_existing:
        print(f"\n[AI] Auto-detected {len(new_zone_specs)} ROI(s).")
        _print_ai_summary(new_meta, n_big_filtered)
        system._ai_meta_by_idx = list(new_meta)
        return new_zone_specs

    # Existing zones present -> ask Replace / Append / Cancel
    choice = ui_dialogs.three_way_dialog(
        window_name=window_name,
        frame=frame,
        title=f"AI found {len(new_zone_specs)} new ROI(s). Profile has {len(current_zones)}.",
        line1="R: REPLACE existing with AI result",
        line2="A: APPEND AI result to existing",
        line3="C / Q: CANCEL (keep current)",
    )
    if choice == "r":
        print(
            f"\n[AI] Replaced {len(current_zones)} zone(s) with {len(new_zone_specs)} AI zone(s)."
        )
        _print_ai_summary(new_meta, n_big_filtered)
        system._ai_meta_by_idx = list(new_meta)
        return new_zone_specs
    if choice == "a":
        merged = list(current_zones) + new_zone_specs
        print(f"\n[AI] Appended {len(new_zone_specs)} zone(s). Total: {len(merged)}.")
        _print_ai_summary(new_meta, n_big_filtered)
        # Existing zones have no meta (unknown class); pad with None to keep index alignment
        existing_meta = getattr(system, "_ai_meta_by_idx", []) or []
        if len(existing_meta) < len(current_zones):
            existing_meta = list(existing_meta) + [None] * (
                len(current_zones) - len(existing_meta)
            )
        else:
            existing_meta = existing_meta[: len(current_zones)]
        system._ai_meta_by_idx = list(existing_meta) + list(new_meta)
        return manual_zone_geometry.sanitize_manual_zones(system, merged, frame_shape)
    print("\n[AI] Canceled.")
    return current_zones


def manual_setup_zones(system, frame):
    """
    Manual setup zones.

    Logic:
        Implements a focused helper step used by the surrounding pipeline/workflow.

    Args:
        system: Input used by this step.
        frame: Input used by this step.

    Returns:
        Any: Result value or structured output for the caller.
    """
    if frame is None or frame.size == 0:
        print("\nManual setup failed: no frame available")
        return []
    frame_shape = frame.shape
    preloaded = system._load_manual_zones(frame_shape)
    target_count = int(system.nozzle_config.get("manual_zone_count", 0))
    target_count = max(0, target_count)
    if preloaded:
        zones = list(preloaded)
        print(
            f"\nLoaded {len(zones)} zones from profile '{system.manual_zone_profile}' ({system.manual_zone_path})"
        )
        # Auto AI-classify existing zones so editor shows class colors immediately.
        zones = _ai_classify_existing_zones(system, frame, zones)
    else:
        print(
            f"\nProfile '{system.manual_zone_profile}' has no zones. Trying AI auto-detect..."
        )
        ai_zones = _run_ai_auto_detect(
            system, frame, current_zones=[], frame_shape=frame_shape
        )
        if ai_zones:
            zones = list(ai_zones)
            print(
                f"AI auto-detect populated {len(zones)} zone(s). Press I to re-run, ENTER to save."
            )
        else:
            zones = [
                manual_zone_geometry.build_default_manual_zone(system, frame_shape)
            ]
            print(
                "AI auto-detect found nothing or is unavailable. Started with 1 default zone."
            )

    if target_count > 0 and len(zones) != target_count:
        print(
            f"Configured manual_zone_count={target_count}, selected {len(zones)} zones."
        )
    return manual_adjust_zones(system, frame, zones)


def reload_manual_editor_zones(system, frame_shape):
    """
    Reload manual editor zones.

    Logic:
        Implements a focused helper step used by the surrounding pipeline/workflow.

    Args:
        system: Input used by this step.
        frame_shape: Input used by this step.

    Returns:
        Any: Result value or structured output for the caller.
    """
    reloaded = system._load_manual_zones(frame_shape)
    return (
        list(reloaded)
        if reloaded
        else [manual_zone_geometry.build_default_manual_zone(system, frame_shape)]
    )


def save_editor_zones_and_reload(system, zones, frame_shape, profile_action=None):
    """
    Save editor zones and reload.

    Logic:
        Serializes current data/state and writes it to the configured destination.

    Args:
        system: Input used by this step.
        zones: Input used by this step.
        frame_shape: Input used by this step.
        profile_action: Input used by this step.

    Returns:
        Any: Result value or structured output for the caller.
    """
    system.manual_zones = manual_zone_geometry.sanitize_manual_zones(
        system, zones, frame_shape
    )
    system._save_manual_zones()
    if callable(profile_action):
        profile_action()
    zones = reload_manual_editor_zones(system, frame_shape)
    system.manual_selected_zone_idx = 0
    return zones


def select_manual_roi(system, frame, roi_window_name, zone_label):
    """
    Select manual roi.

    Logic:
        Implements a focused helper step used by the surrounding pipeline/workflow.

    Args:
        system: Input used by this step.
        frame: Input used by this step.
        roi_window_name: Input used by this step.
        zone_label: Input used by this step.

    Returns:
        Any: Result value or structured output for the caller.
    """
    new_roi = select_roi_with_prompt(roi_window_name, frame, zone_label, system=system)
    cv2.destroyWindow(roi_window_name)
    if new_roi is None or len(new_roi) != 4:
        return None
    rx, ry, rw, rh = [int(v) for v in tuple(map(int, new_roi))]
    if rw <= 0 or rh <= 0:
        return None
    return (rx, ry, rw, rh)


def manual_adjust_zones(system, frame, zones):
    """
    Manual adjust zones.

    Logic:
        Implements a focused helper step used by the surrounding pipeline/workflow.

    Args:
        system: Input used by this step.
        frame: Input used by this step.
        zones: Input used by this step.

    Returns:
        Any: Result value or structured output for the caller.
    """
    if not zones:
        return []

    frame_shape = frame.shape
    zones = manual_zone_geometry.sanitize_manual_zones(system, zones, frame_shape)
    system.manual_selected_zone_idx = int(
        np.clip(system.manual_selected_zone_idx, 0, len(zones) - 1)
    )
    window_name = system.display_config.get("window_name", "Nozzle Inspection")
    new_profile_draft_active = False
    new_profile_draft_name = None
    new_profile_prev_name = None
    new_profile_prev_zones = None

    while True:
        display = frame.copy()
        ai_meta_list = getattr(system, "_ai_meta_by_idx", None) or []
        for idx, zone_spec in enumerate(zones):
            x, y, w, h, angle_deg = manual_zone_geometry.manual_zone_unpack(
                system, zone_spec
            )
            selected = idx == system.manual_selected_zone_idx
            ai_meta = ai_meta_list[idx] if idx < len(ai_meta_list) else None
            if ai_meta and isinstance(ai_meta, dict):
                cid = int(ai_meta.get("class_id", -1))
                ai_color = AI_CLASS_COLORS.get(cid, (200, 200, 200))
                ai_label = AI_CLASS_SHORT.get(cid, "?")
                ai_score = float(ai_meta.get("dl_score", 0.0))
            else:
                ai_color = None
                ai_label = None
                ai_score = None
            if selected:
                color = (0, 255, 0)
            elif ai_color is not None:
                color = ai_color
            else:
                color = (255, 140, 0)
            thickness = 3 if selected else 2
            zone_poly = manual_zone_geometry.manual_zone_polygon(
                (x, y, w, h), angle_deg
            )
            cv2.polylines(display, [zone_poly], True, color, thickness, cv2.LINE_AA)
            min_y = int(np.min(zone_poly[:, 1])) if zone_poly.size > 0 else int(y)
            if ai_label is not None:
                label = f"Z{idx + 1} {ai_label} {ai_score:.2f} ({angle_deg:+.0f}d)"
            else:
                label = f"Z{idx + 1} ({angle_deg:+.0f}d)"
            cv2.putText(
                display,
                label,
                (int(x + 4), max(16, min_y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
            )

        help_items = [
            ("Profile Setting: ON", (120, 255, 210)),
            (f"Profile: {system.manual_zone_profile}", (0, 220, 255)),
            ("P: switch profile", (210, 210, 210)),
            ("N: new empty profile (draft)", (210, 210, 210)),
            ("SPACE/TAB: next zone", (210, 210, 210)),
            ("DEL: delete zone", (210, 210, 210)),
            ("A: add zone", (210, 210, 210)),
            ("R: redraw zone", (210, 210, 210)),
            ("I: AI auto-detect (R/A/C)", (0, 220, 255)),
            ("V: visualize AI pipeline", (180, 220, 220)),
            ("^ v < >: move", (210, 210, 210)),
            ("+/-: resize", (210, 210, 210)),
            (
                f"[ / ]: rotate -/+{int(round(abs(float(system.manual_rotate_step_deg))))}deg",
                (210, 210, 210),
            ),
            ("ENTER: save", (90, 235, 90)),
            ("Q: exit (cancel)", (90, 170, 255)),
        ]
        if new_profile_draft_active and new_profile_draft_name:
            help_items.insert(
                1, (f"DRAFT: {new_profile_draft_name} (empty)", (0, 220, 255))
            )
            help_items.insert(2, ("ENTER: save draft profile", (90, 235, 90)))
            help_items.insert(3, ("Q: exit draft without saving", (90, 170, 255)))
        panel_font = cv2.FONT_HERSHEY_SIMPLEX
        panel_scale = 0.5
        panel_thickness = 1
        panel_pad = 10
        line_gap = 6
        text_sizes = [
            cv2.getTextSize(text, panel_font, panel_scale, panel_thickness)[0]
            for text, _ in help_items
        ]
        line_h = max(14, max(h for _, h in text_sizes))
        panel_w = min(
            display.shape[1] - 20,
            max((w for w, _ in text_sizes), default=120) + (panel_pad * 2),
        )
        panel_h = (
            (panel_pad * 2)
            + (line_h * len(help_items))
            + (line_gap * max(0, len(help_items) - 1))
        )
        px0 = 10
        py0 = 10
        cv2.rectangle(
            display, (px0, py0), (px0 + panel_w, py0 + panel_h), (0, 0, 0), -1
        )
        y_cursor = py0 + panel_pad + line_h
        for line, line_color in help_items:
            cv2.putText(
                display,
                line,
                (px0 + panel_pad, y_cursor),
                panel_font,
                panel_scale,
                line_color,
                panel_thickness,
            )
            y_cursor += line_h + line_gap

        cv2.imshow(window_name, system._scale_display_frame(display))

        key_code = cv2.waitKeyEx(0)
        full_code = int(key_code)
        low_code = int(key_code & 0xFF)

        if low_code == 13:  # Enter only = save
            if new_profile_draft_active and new_profile_draft_name:
                payload = profile_store.load_manual_zone_payload(
                    system.manual_zone_path
                )
                if not isinstance(payload, dict):
                    payload = {}
                profiles = profile_store.extract_profiles_from_payload(payload)
                profiles[str(new_profile_draft_name)] = [
                    manual_zone_geometry.serialize_manual_zone(system, z)
                    for z in manual_zone_geometry.sanitize_manual_zones(
                        system, zones, frame_shape
                    )
                ]
                payload = {
                    "active_profile": str(new_profile_draft_name),
                    "profiles": profiles,
                }
                profile_store.save_manual_zone_payload(system.manual_zone_path, payload)
                system.manual_zone_profile = str(new_profile_draft_name)
                print(
                    f"\nSaved new profile '{system.manual_zone_profile}' with {len(zones)} zone(s)."
                )
                new_profile_draft_active = False
                new_profile_draft_name = None
                new_profile_prev_name = None
                new_profile_prev_zones = None
            break
        if low_code in (ord("q"), ord("Q")):  # Q cancel
            if new_profile_draft_active:
                if isinstance(new_profile_prev_zones, list):
                    zones = manual_zone_geometry.sanitize_manual_zones(
                        system, new_profile_prev_zones, frame_shape
                    )
                if isinstance(new_profile_prev_name, str) and new_profile_prev_name:
                    system.manual_zone_profile = new_profile_prev_name
                system.manual_selected_zone_idx = 0
                new_profile_draft_active = False
                new_profile_draft_name = None
                new_profile_prev_name = None
                new_profile_prev_zones = None
                print("\nCanceled new profile draft.")
                continue
            return []
        if low_code in (9, 32):  # TAB / SPACE = select next zone
            if len(zones) <= 0:
                continue
            system.manual_selected_zone_idx = (
                system.manual_selected_zone_idx + 1
            ) % len(zones)
            continue
        if low_code in (ord("p"), ord("P")):
            if new_profile_draft_active:
                print(
                    "\nP disabled while new-profile draft is active. Save (ENTER) or cancel (Q)."
                )
                continue
            zones = save_editor_zones_and_reload(
                system=system,
                zones=zones,
                frame_shape=frame_shape,
                profile_action=system._cycle_manual_zone_profile,
            )
            continue
        if low_code in (ord("n"), ord("N")):
            if new_profile_draft_active:
                print("\nN draft already active. Save with ENTER or cancel with Q.")
                continue
            draft_name = profile_store.suggest_new_manual_profile_name(
                system.manual_zone_path, base="profile"
            )
            confirm = ui_dialogs.confirm_action_dialog(
                window_name,
                frame,
                f"Create a new empty profile '{draft_name}'?",
                "Press SPACE or Y to confirm. Press Q or N to cancel.",
                alpha=0.66,
            )
            if not confirm:
                continue
            new_profile_draft_active = True
            new_profile_draft_name = str(draft_name)
            new_profile_prev_name = str(system.manual_zone_profile)
            new_profile_prev_zones = manual_zone_geometry.sanitize_manual_zones(
                system, zones, frame_shape
            )
            zones = []
            system.manual_selected_zone_idx = 0
            print(
                f"\nDraft profile '{new_profile_draft_name}' created (empty). Press ENTER to save, Q to cancel."
            )
            continue
        if low_code in (ord("i"), ord("I")):
            zones = _run_ai_auto_detect(
                system, frame, current_zones=zones, frame_shape=frame_shape
            )
            zones = manual_zone_geometry.sanitize_manual_zones(
                system, zones or [], frame_shape
            )
            if zones:
                system.manual_selected_zone_idx = int(
                    np.clip(system.manual_selected_zone_idx, 0, len(zones) - 1)
                )
            else:
                system.manual_selected_zone_idx = 0
            continue
        if low_code in (ord("v"), ord("V")):
            _show_ai_pipeline_debug(system, frame)
            continue

        if len(zones) <= 0:
            if low_code in (ord("a"), ord("A")):
                add_window = "Manual Focus Zones - Add ROI"
                roi = select_manual_roi(
                    system=system,
                    frame=frame,
                    roi_window_name=add_window,
                    zone_label="Add first zone to draft profile",
                )
                if roi is not None:
                    rx, ry, rw, rh = roi
                    zones.append(
                        manual_zone_geometry.sanitize_manual_zone(
                            system, (rx, ry, rw, rh, 0.0), frame_shape
                        )
                    )
                    system.manual_selected_zone_idx = 0
            continue

        idx = system.manual_selected_zone_idx
        x, y, w, h, angle_deg = manual_zone_geometry.manual_zone_unpack(
            system, zones[idx]
        )
        move_step = max(1, int(system.manual_move_step_px))
        resize_step = max(1, int(system.manual_resize_step_px))
        rotate_step = max(1.0, abs(float(system.manual_rotate_step_deg)))
        if system._is_delete_key(full_code, low_code):
            if len(zones) <= 1:
                confirm = ui_dialogs.confirm_action_dialog(
                    window_name,
                    frame,
                    f"Delete last zone in profile '{system.manual_zone_profile}'?",
                    "This will delete the profile. Press SPACE or Y to confirm. N/Q to cancel.",
                    alpha=0.60,
                )
                if not confirm:
                    continue
                deleted = system._delete_manual_zone_profile(system.manual_zone_profile)
                if not deleted:
                    print("\nDelete failed: profile was not removed.")
                    continue
                zones = reload_manual_editor_zones(system, frame_shape)
                system.manual_selected_zone_idx = 0
                continue
            zones.pop(idx)
            _meta = getattr(system, "_ai_meta_by_idx", None)
            if isinstance(_meta, list) and idx < len(_meta):
                _meta.pop(idx)
            system.manual_selected_zone_idx = int(
                np.clip(system.manual_selected_zone_idx, 0, len(zones) - 1)
            )
            continue
        if low_code in (ord("a"), ord("A")):
            add_window = "Manual Focus Zones - Add ROI"
            roi = select_manual_roi(
                system=system,
                frame=frame,
                roi_window_name=add_window,
                zone_label=f"Add new zone (next: Z{len(zones) + 1})",
            )
            if roi is not None:
                rx, ry, rw, rh = roi
                zones.append(
                    manual_zone_geometry.sanitize_manual_zone(
                        system, (rx, ry, rw, rh, 0.0), frame_shape
                    )
                )
                _meta = getattr(system, "_ai_meta_by_idx", None)
                if isinstance(_meta, list):
                    _meta.append(None)
                system.manual_selected_zone_idx = len(zones) - 1
            continue
        if low_code in (ord("r"), ord("R")):
            redraw_window = f"Manual Focus Zones - Redraw Z{idx + 1}"
            roi = select_manual_roi(
                system=system,
                frame=frame,
                roi_window_name=redraw_window,
                zone_label=f"Redraw zone Z{idx + 1}",
            )
            if roi is not None:
                rx, ry, rw, rh = roi
                zones[idx] = manual_zone_geometry.sanitize_manual_zone(
                    system, (rx, ry, rw, rh, angle_deg), frame_shape
                )
                _meta = getattr(system, "_ai_meta_by_idx", None)
                if isinstance(_meta, list) and idx < len(_meta):
                    _meta[idx] = (
                        None  # User redrew, AI class no longer guaranteed accurate
                    )
            continue

        if full_code in (82, 2490368):  # Up
            y -= move_step
        elif full_code in (84, 2621440):  # Down
            y += move_step
        elif full_code in (81, 2424832):  # Left
            x -= move_step
        elif full_code in (83, 2555904):  # Right
            x += move_step
        elif low_code in (ord("+"), ord("=")):
            x -= resize_step // 2
            y -= resize_step // 2
            w += resize_step
            h += resize_step
        elif low_code in (ord("-"), ord("_")):
            x += resize_step // 2
            y += resize_step // 2
            w -= resize_step
            h -= resize_step
        elif low_code in (ord("["), ord("{")):
            angle_deg -= rotate_step
        elif low_code in (ord("]"), ord("}")):
            angle_deg += rotate_step
        else:
            continue

        zones[idx] = manual_zone_geometry.sanitize_manual_zone(
            system, (x, y, w, h, angle_deg), frame_shape
        )

    return manual_zone_geometry.sanitize_manual_zones(system, zones, frame_shape)


def select_roi_with_prompt(window_name, frame, zone_label, system=None):
    """
    Select roi with prompt.

    Logic:
        Implements a focused helper step used by the surrounding pipeline/workflow.

    Args:
        window_name: Input used by this step.
        frame: Input used by this step.
        zone_label: Input used by this step.

    Returns:
        Any: Result value or structured output for the caller.
    """
    if frame is None or frame.size == 0:
        return None
    print(f"\n[ROI] {zone_label}: drag box -> SPACE/ENTER confirm | Q cancel.")

    base = frame.copy()
    h, w = base.shape[:2]
    state = {
        "dragging": False,
        "p0": None,
        "p1": None,
        "rect": None,
    }

    def _clamp_point(px, py):
        """
        Clamp point.

        Logic:
            Implements a focused helper step used by the surrounding pipeline/workflow.

        Args:
            px: Input used by this step.
            py: Input used by this step.

        Returns:
            Any: Result value or structured output for the caller.
        """
        cx = int(np.clip(int(px), 0, max(0, w - 1)))
        cy = int(np.clip(int(py), 0, max(0, h - 1)))
        return cx, cy

    def _build_rect(pa, pb):
        """
        Build rect.

        Logic:
            Implements a focused helper step used by the surrounding pipeline/workflow.

        Args:
            pa: Input used by this step.
            pb: Input used by this step.

        Returns:
            Any: Result value or structured output for the caller.
        """
        if pa is None or pb is None:
            return None
        x0, y0 = int(pa[0]), int(pa[1])
        x1, y1 = int(pb[0]), int(pb[1])
        left = min(x0, x1)
        top = min(y0, y1)
        right = max(x0, x1)
        bottom = max(y0, y1)
        return (
            int(left),
            int(top),
            int(max(0, right - left)),
            int(max(0, bottom - top)),
        )

    def _on_mouse(event, mx, my, _flags, _param):
        """
        On mouse.

        Logic:
            Implements a focused helper step used by the surrounding pipeline/workflow.

        Args:
            event: Input used by this step.
            mx: Input used by this step.
            my: Input used by this step.
            _flags: Input used by this step.
            _param: Input used by this step.

        Returns:
            None.
        """
        if system is not None:
            mx, my = system._map_display_xy(mx, my)
        x, y = _clamp_point(mx, my)
        if event == cv2.EVENT_LBUTTONDOWN:
            state["dragging"] = True
            state["p0"] = (x, y)
            state["p1"] = (x, y)
            state["rect"] = (x, y, 0, 0)
            return
        if (
            event == cv2.EVENT_MOUSEMOVE
            and state["dragging"]
            and state["p0"] is not None
        ):
            state["p1"] = (x, y)
            state["rect"] = _build_rect(state["p0"], state["p1"])
            return
        if event == cv2.EVENT_LBUTTONUP and state["p0"] is not None:
            state["dragging"] = False
            state["p1"] = (x, y)
            state["rect"] = _build_rect(state["p0"], state["p1"])

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, _on_mouse)
    try:
        while True:
            display = base.copy()

            rect = state.get("rect")
            if rect is not None:
                rx, ry, rw, rh = [int(v) for v in rect]
                if rw > 0 and rh > 0:
                    cv2.rectangle(display, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
                    cv2.putText(
                        display,
                        f"{rw}x{rh}",
                        (rx + 4, max(16, ry - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )

            tip_lines = [
                str(zone_label),
                "Drag to draw ROI",
                "SPACE/ENTER: confirm",
                "Q: cancel | R: reset",
            ]
            tip_font = cv2.FONT_HERSHEY_SIMPLEX
            tip_scale = 0.5
            tip_thickness = 1
            tip_pad = 8
            tip_gap = 5
            tip_sizes = [
                cv2.getTextSize(line, tip_font, tip_scale, tip_thickness)[0]
                for line in tip_lines
            ]
            tip_h = max(14, max(hh for _, hh in tip_sizes))
            box_w = min(
                max(180, max((ww for ww, _ in tip_sizes), default=140) + (tip_pad * 2)),
                max(180, display.shape[1] - 20),
            )
            box_h = (
                (tip_pad * 2)
                + (tip_h * len(tip_lines))
                + (tip_gap * max(0, len(tip_lines) - 1))
            )
            bx0, by0 = 10, 10
            overlay = display.copy()
            cv2.rectangle(
                overlay, (bx0, by0), (bx0 + box_w, by0 + box_h), (0, 0, 0), -1
            )
            cv2.addWeighted(overlay, 0.68, display, 0.32, 0, display)
            ty = by0 + tip_pad + tip_h
            for i, line in enumerate(tip_lines):
                color = (
                    (0, 220, 255)
                    if i == 0
                    else (
                        (90, 235, 90)
                        if i == 2
                        else ((90, 170, 255) if i == 3 else (220, 220, 220))
                    )
                )
                cv2.putText(
                    display,
                    line,
                    (bx0 + tip_pad, ty),
                    tip_font,
                    tip_scale,
                    color,
                    tip_thickness,
                )
                ty += tip_h + tip_gap

            cv2.imshow(
                window_name,
                system._scale_display_frame(display) if system is not None else display,
            )
            key_code = cv2.waitKeyEx(20)
            if int(key_code) == -1:
                try:
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        return None
                except Exception:
                    pass
                continue

            low_code = int(key_code & 0xFF)
            if low_code in (ord("q"), ord("Q"), ord("c"), ord("C"), 27):
                return None
            if low_code in (ord("r"), ord("R")):
                state["dragging"] = False
                state["p0"] = None
                state["p1"] = None
                state["rect"] = None
                continue
            if low_code in (13, 32):
                rect = state.get("rect")
                if rect is None:
                    continue
                rx, ry, rw, rh = [int(v) for v in rect]
                if rw > 0 and rh > 0:
                    return (int(rx), int(ry), int(rw), int(rh))
    finally:
        try:
            cv2.setMouseCallback(window_name, lambda *_args: None)
        except Exception:
            pass


def enter_manual_mode(system):
    """
    Enter manual mode.

    Logic:
        Initializes state required to enter this mode/workflow and returns enter status.

    Args:
        system: Input used by this step.

    Returns:
        Any: Result value or structured output for the caller.
    """
    if not bool(system.mode_enabled.get("MANUAL", True)):
        print("\nManual mode is disabled in config.")
        return False
    setup_frame = (
        system.last_input_frame
        if system.last_input_frame is not None
        else system.last_result_frame
    )
    if setup_frame is None:
        print("\nManual mode unavailable: no frame received yet")
        return False
    new_zones = manual_setup_zones(system, setup_frame.copy())
    if not new_zones:
        return False
    system.previous_mode_before_manual = (
        system.current_mode
        if system.current_mode in {"DYN", "FIX", "VIDEO"}
        else system._resolve_runtime_mode_fallback("FIX")
    )
    system.manual_zones = list(new_zones)
    system.current_mode = "MANUAL"
    system._manual_zone_reconfigure_runtime()
    system._save_manual_zones()
    preview = system.process_frame(setup_frame.copy())
    system.last_result_frame = preview.copy()
    if system.display_config.get("show_video", True):
        cv2.imshow(
            system.display_config.get("window_name", "Nozzle Inspection"),
            system._scale_display_frame(preview),
        )
    system._flush_pending_keys()
    print(f"\nZone mode: MANUAL ({len(system.manual_zones)} zones)")
    return True


def reload_manual_profile_live(system):
    """
    Reload manual profile live.

    Logic:
        Implements a focused helper step used by the surrounding pipeline/workflow.

    Args:
        system: Input used by this step.

    Returns:
        Any: Result value or structured output for the caller.
    """
    frame_ref = (
        system.last_input_frame
        if system.last_input_frame is not None
        else system.last_result_frame
    )
    if frame_ref is None:
        return False
    frame_shape = frame_ref.shape
    zones = system._load_manual_zones(frame_shape)
    if not zones:
        print(f"\nProfile '{system.manual_zone_profile}' has no zones.")
        return False
    system.manual_zones = list(zones)
    system._manual_zone_reconfigure_runtime()
    print(
        f"\nMANUAL profile '{system.manual_zone_profile}' loaded ({len(system.manual_zones)} zones)"
    )
    return True


def exit_manual_mode(system):
    """
    Exit manual mode.

    Logic:
        Restores fallback state and finalizes cleanup when leaving this mode/workflow.

    Args:
        system: Input used by this step.

    Returns:
        None.
    """
    fallback_mode_raw = str(system.previous_mode_before_manual or "").strip().upper()
    if fallback_mode_raw == "VIDEO":
        system.current_mode = "VIDEO"
    else:
        fallback_mode = system._resolve_runtime_mode_fallback(fallback_mode_raw)
        system._apply_runtime_mode(fallback_mode)
    system._flush_pending_keys()
    print(f"\nZone mode: {system.current_mode}")
