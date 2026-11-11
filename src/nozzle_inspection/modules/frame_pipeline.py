"""
Per-frame processing: input -> detection -> analysis -> lock snapshots -> presentation.
"""

import cv2

from . import overlay_renderer, zone_metrics

# BGR colors per DL class (match profile editor palette)
_DL_GATE_COLORS = {
    0: (0, 220, 0),  # SPRAY -> green
    1: (0, 165, 255),  # NO_SPRAY -> orange
    2: (50, 50, 255),  # FACING -> red
    3: (255, 100, 50),  # BIG -> blue
}


def _draw_skipped_zone_crosses(output_frame, metrics):
    """Mark zones the DL gate skipped (BIG) with an X; the AI label itself is in the side panel."""
    if output_frame is None or not metrics:
        return
    for m in metrics:
        gate = m.get("dl_gate") if isinstance(m, dict) else None
        if not gate or not bool(gate.get("skip_all", False)):
            continue
        rect = m.get("rect")
        if not rect or len(rect) < 4:
            continue
        x, y, w, h = [int(v) for v in rect[:4]]
        color = _DL_GATE_COLORS.get(int(gate.get("class_id", 3)), (255, 100, 50))
        cv2.line(output_frame, (x, y), (x + w, y + h), color, 2)
        cv2.line(output_frame, (x + w, y), (x, y + h), color, 2)


def process_frame(system, frame):
    """Run one frame through the pipeline and return the rendered output frame."""
    if str(getattr(system, "current_mode", "")).upper() == "VIDEO":
        system.last_zone_metrics = []
        return frame.copy()

    analysis_frame = system._prepare_detection_input(frame)
    if system.current_mode == "MANUAL":
        zone_results = system._detect_nozzles_manual(analysis_frame)
    else:
        zone_results = system.zone_orchestrator.detect_nozzles_by_grid(analysis_frame)
    metrics = zone_metrics.compute_all_zone_metrics(
        system, analysis_frame, zone_results
    )

    # Capture a lock snapshot for every locked zone that does not have one yet.
    for metric in list(metrics or []):
        try:
            zid = int(metric.get("zone_id", -1))
        except Exception:
            continue
        if bool(system.zone_lock_enabled.get(zid, False)) and (
            zid not in system.zone_visual_lock_snapshots
        ):
            system._capture_zone_visual_lock_snapshot(zid)

    system.last_zone_metrics = list(metrics)
    output = overlay_renderer.visualize(system, frame, zone_results, metrics)
    _draw_skipped_zone_crosses(output, metrics)
    return output
