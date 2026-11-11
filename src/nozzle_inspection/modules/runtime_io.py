"""
Runtime I/O helpers extracted from run_inspection.py.
"""

import os
import time
from datetime import datetime
from typing import cast

import cv2

from . import runtime_keys


def _publish_to_dl_ticker(system, frame_bgr):
    """Push latest frame + zone rois to the DL status ticker (Phase 2 runtime).

    Non-fatal: any error is swallowed so the main loop is never blocked.
    The ticker itself is lazy-started here on first call.
    """
    try:
        ticker = (
            system.get_dl_status_ticker()
            if hasattr(system, "get_dl_status_ticker")
            else None
        )
        if ticker is None:
            return
        zones_metrics = getattr(system, "last_zone_metrics", None) or []
        rois = []
        zone_ids = []
        for m in zones_metrics:
            if not isinstance(m, dict):
                continue
            rect = m.get("rect")
            zid = m.get("zone_id")
            if rect is None or zid is None or len(rect) < 4:
                continue
            try:
                rois.append((int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])))
                zone_ids.append(int(zid))
            except (TypeError, ValueError):
                continue
        ticker.publish_frame(frame_bgr, rois, zone_ids)
    except Exception:
        pass


def _normalize_codec_list(raw_value):
    """
    Normalize codec config into a non-empty ordered list.
    """
    if isinstance(raw_value, str):
        items = [raw_value]
    elif isinstance(raw_value, (list, tuple)):
        items = list(raw_value)
    else:
        items = []
    out = []
    for item in items:
        token = str(item or "").strip()
        if len(token) == 4:
            out.append(token)
    if not out:
        out = ["avc1", "mp4v", "XVID", "MJPG"]
    return out


def _ext_for_codec(codec: str) -> str:
    token = str(codec or "").strip().upper()
    if token in {"MJPG", "XVID", "DIVX"}:
        return ".avi"
    return ".mp4"


def _try_open_video_writer(path: str, codec: str, fps: float, frame_size):
    fourcc_fn = getattr(cv2, "VideoWriter_fourcc", None)
    if callable(fourcc_fn):
        fourcc = cast(int, fourcc_fn(*str(codec)))
    else:
        fourcc = cast(int, cv2.VideoWriter.fourcc(*str(codec)))
    return cv2.VideoWriter(path, fourcc, float(fps), frame_size)


def _prepare_frame_for_writer(frame, target_size):
    """
    Normalize output frame before writing (size/channels) to avoid silent writer failures.
    """
    if frame is None:
        return None
    out = frame
    if len(out.shape) == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    elif len(out.shape) == 3 and int(out.shape[2]) == 4:
        out = cv2.cvtColor(out, cv2.COLOR_BGRA2BGR)
    tw, th = int(target_size[0]), int(target_size[1])
    if out.shape[1] != tw or out.shape[0] != th:
        out = cv2.resize(out, (tw, th), interpolation=cv2.INTER_LINEAR)
    return out


def _resolve_file_playback_fps(system) -> float:
    """
    Resolve playback FPS for file inputs.

    Logic:
        Chooses a stable file playback FPS using explicit override first, then source metadata.

    Args:
        system: Runtime system instance holding config and camera.

    Returns:
        float: Effective playback FPS.
    """
    override_fps = float(
        system.config.get("camera.file_playback_fps_override", 0.0) or 0.0
    )
    min_fps = float(system.config.get("camera.file_playback_fps_min", 1.0) or 1.0)
    max_fps = float(system.config.get("camera.file_playback_fps_max", 120.0) or 120.0)
    if max_fps < min_fps:
        max_fps = min_fps

    if min_fps <= override_fps <= max_fps:
        return override_fps

    source_fps = float(system.camera.get_fps() or 0.0)
    if min_fps <= source_fps <= max_fps:
        return source_fps

    configured_fps = float(getattr(system.camera, "fps", 0.0) or 0.0)
    if min_fps <= configured_fps <= max_fps:
        return configured_fps

    return 25.0


def _read_video_total_frames(system) -> int:
    """
    Read total frame count for file playback, returning 0 when unavailable.
    """
    cap = getattr(getattr(system, "camera", None), "cap", None)
    if cap is None:
        return 0
    try:
        total = int(round(float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)))
    except Exception:
        total = 0
    return max(0, total)


def _read_video_frame_index(system) -> int:
    """
    Read current 1-based frame position from OpenCV after a successful read.
    """
    cap = getattr(getattr(system, "camera", None), "cap", None)
    if cap is None:
        return int(getattr(system, "video_frame_index", 0) or 0)
    try:
        pos = int(round(float(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0.0)))
    except Exception:
        pos = int(getattr(system, "video_frame_index", 0) or 0)
    return max(0, pos)


def draw_playback_status(system, frame):
    """
    Draw file playback frame counter and loop state on the output frame.
    """
    if frame is None:
        return frame
    if str(getattr(system, "current_mode", "")).upper() != "VIDEO" and bool(
        getattr(system, "show_status_panel", True)
    ):
        return frame
    total = int(getattr(system, "video_total_frames", 0) or 0)
    current = int(getattr(system, "video_frame_index", 0) or 0)
    if total <= 0 and current <= 0:
        return frame

    if total > 0:
        current = int(max(1, min(current, total)))
        text = f"Frame: {current}/{total}"
    else:
        text = f"Frame: {current}"
    text += (
        f"  Loop: {'ON' if bool(getattr(system, 'video_auto_loop', True)) else 'OFF'}"
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.56
    thick = 1
    pad_x = 8
    pad_y = 6
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    h, w = frame.shape[:2]
    x1 = max(6, int(w - tw - (2 * pad_x) - 10))
    y1 = 8
    x2 = min(w - 6, int(x1 + tw + (2 * pad_x)))
    y2 = int(y1 + th + base + (2 * pad_y))
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.50, frame, 0.50, 0.0, frame)
    cv2.putText(
        frame,
        text,
        (x1 + pad_x, y2 - pad_y - 1),
        font,
        scale,
        (235, 235, 235),
        thick,
        cv2.LINE_AA,
    )
    return frame


def _scale_for_display(system, frame):
    """Resize ``frame`` by the shared display scale (aspect preserved, never
    distorted). Delegates to the system helper so the main view and the manual
    editor use the exact same scaling."""
    return system._scale_display_frame(frame)


def _handle_display_scale_key(system, key, window_name):
    """Adjust the display scale with '-'/'_' (smaller) and '='/'+' (larger).

    Returns True when the key was a scale key (and was handled), so the caller
    skips the normal key dispatch.
    """
    if key in (ord("-"), ord("_")):
        delta = -10
    elif key in (ord("="), ord("+")):
        delta = 10
    else:
        return False
    sp = int(
        max(
            10,
            min(200, int(getattr(system, "display_scale_percent", 100) or 100) + delta),
        )
    )
    system.display_scale_percent = sp
    print(f"\nWindow scale: {sp}%")
    # Re-show the last frame immediately so the change is visible even when paused.
    if system.display_config.get("show_video", True):
        last = getattr(system, "last_result_frame", None)
        if last is not None:
            cv2.imshow(window_name, _scale_for_display(system, last))
    return True


def run(system):
    """
    Run main inspection loop.

    Logic:
        Opens camera, drives per-frame processing, handles keyboard input, and performs runtime cleanup.

    Args:
        system: Runtime system instance holding camera/config/state and processing methods.

    Returns:
        None.
    """
    print("=" * 60)
    print("Nozzle Inspection System")
    print("=" * 60)
    print("Starting system...")

    if not system.camera.open():
        print("Error: Failed to open camera")
        return

    print("Camera opened successfully")
    print(f"Frame size: {system.camera.get_frame_size()}")
    print(f"Camera FPS: {system.camera.get_fps()}")
    print(f"Zone profile: {system.manual_zone_profile}")
    print(
        f"Input mode: {'processed_output' if system.processed_output_input_mode else 'raw'}"
    )
    if str(getattr(system, "current_mode", "")).upper() == "VIDEO":
        print("Startup mode: VIDEO preview only (press D/F/M to pick mode)")
    print("\nKeyboard shortcuts:")
    print("  [System]")
    print("    q : quit")
    print("    space : pause/resume playback")
    print("    o : toggle video auto-loop at end")
    print("    s : start/stop video recording")
    print("    r : reset trackers/warnings")
    print("  [Mode / Zones]")
    enabled_runtime_modes = "/".join(system._enabled_runtime_modes())
    print(f"    d : enter/switch runtime mode ({enabled_runtime_modes})")
    print("    f : force FIX mode")
    if bool(system.mode_enabled.get("MANUAL", True)):
        print("    m : edit manual zones (draw/adjust ROIs)")
    else:
        print("    m : disabled (MANUAL mode OFF in config)")
    print("    p : switch zone profile (saved in manual_zones.json)")
    print("    n : MANUAL editor only (new empty profile draft)")
    print("  [Calibration]")
    print("    tab : switch MANUAL/AUTO (or cycle lock zone when lock-select is ON)")
    print("    [ / ] : manual annulus offset -/+")
    print("    ; / ' : manual annulus band -/+")
    print("    U / I : spray exclusion band (%) -/+")
    print("  [Lock]")
    print("    9 : toggle lock-zone select mode (ON/OFF)")
    print("    l : toggle lock for selected zone")
    print("  [Overlay]")
    print("    1 : toggle nozzle head circle")
    print("    0 : toggle nozzle angle arrow")
    print("    2 : toggle spray angle lines")
    print("    3 : toggle spray zone polygon")
    print("    4 : toggle annular mask (orange)")
    print("    5 : toggle spray exclusion mask (purple)")
    print("    6 : toggle analysis mask (gray-green)")
    print("    7 : toggle focus zone overlay")
    print("    8 : toggle focus zone side panel")
    print("=" * 60)
    paused = False
    window_name = system.display_config.get("window_name", "Nozzle Inspection")
    # Display scale (percent). The frame is uniformly resized and shown in an
    # AUTOSIZE window, so the window always fits the scaled image exactly and the
    # video is NEVER distorted. Adjust live with '-' (smaller) and '=' (larger).
    system.display_scale_percent = int(
        max(
            10,
            min(
                200, int(system.display_config.get("window_scale_percent", 100) or 100)
            ),
        )
    )
    system._display_transform = (1.0, 0, 0)
    if system.display_config.get("show_video", True):
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, system._on_main_window_mouse)
        print("    - / = : shrink / grow video window  (10%% steps, no distortion)")
        if system.display_scale_percent != 100:
            print(f"Window scale: {system.display_scale_percent}%")

    source_obj = getattr(system.camera, "source", None)
    is_file_source = isinstance(source_obj, str)
    source_fps = (
        _resolve_file_playback_fps(system)
        if is_file_source
        else float(system.camera.get_fps() or 0.0)
    )
    system.video_total_frames = (
        _read_video_total_frames(system) if is_file_source else 0
    )
    system.video_frame_index = 0
    if source_fps <= 1.0:
        source_fps = 25.0
    frame_interval_sec = 1.0 / source_fps
    next_frame_deadline = time.perf_counter()

    try:
        while True:
            wait_ms = 1
            if not paused:
                ret, frame = system.camera.read_frame()

                if not ret or frame is None:
                    source = getattr(system.camera, "source", None)
                    if isinstance(source, str) and bool(
                        getattr(system, "video_auto_loop", True)
                    ):
                        system.camera.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        system.frame_id = 0
                        system.video_frame_index = 0
                        ret, frame = system.camera.read_frame()
                        next_frame_deadline = time.perf_counter()
                    if not ret or frame is None:
                        if isinstance(source, str):
                            print("Video playback ended")
                        else:
                            print("Error: Failed to read frame")
                        break
                if is_file_source:
                    system.video_frame_index = _read_video_frame_index(system)
                system.last_input_frame = frame
                result_frame = system.process_frame(frame)
                _publish_to_dl_ticker(system, frame)
                if is_file_source:
                    draw_playback_status(system, result_frame)
                system.last_result_frame = result_frame
                maybe_auto_start_recording(system)
                try:
                    system.frame_id = int(getattr(system, "frame_id", 0)) + 1
                except Exception:
                    system.frame_id = 0

                write_output_frame(system, result_frame)

                if system.display_config.get("show_video", True):
                    cv2.imshow(window_name, _scale_for_display(system, result_frame))

                update_fps(system)
                if is_file_source:
                    next_frame_deadline += frame_interval_sec
                    now = time.perf_counter()
                    lag_limit = 3.0 * frame_interval_sec
                    if (now - next_frame_deadline) > lag_limit:
                        next_frame_deadline = now
                    remain = next_frame_deadline - now
                    wait_ms = max(1, int(round(remain * 1000.0))) if remain > 0.0 else 1
            else:
                wait_ms = 30

            was_paused = paused
            key = cv2.waitKey(wait_ms) & 0xFF
            if _handle_display_scale_key(system, key, window_name):
                action = "handled"
            else:
                action, paused = runtime_keys.handle_runtime_key(system, key, paused)
            if was_paused and (not paused) and is_file_source:
                next_frame_deadline = time.perf_counter()
            if action == "quit":
                break

    finally:
        close_video_writer(system)
        try:
            if hasattr(system, "stop_dl_status_ticker"):
                system.stop_dl_status_ticker()
        except Exception:
            pass
        system.camera.release()
        cv2.destroyAllWindows()
        print("System shutdown complete")


def toggle_video_recording(system):
    """
    Toggle video recording state.

    Logic:
        Starts recording using current output-frame geometry or stops and closes active writer.

    Args:
        system: Runtime system instance holding video writer state.

    Returns:
        None.
    """
    if system.video_writer is not None:
        path = system.video_output_path
        close_video_writer(system)
        print(f"\nRecording stopped: {path}")
        return
    start_video_recording(system)


def maybe_auto_start_recording(system):
    """
    Auto-start recording once on the first processed frame when enabled.

    Logic:
        When `recording.auto_record` is on, begins recording the processed output
        a single time per session so a manual stop (key 's') is not overridden.

    Args:
        system: Runtime system instance holding writer/auto-record state.

    Returns:
        None.
    """
    if not bool(system.config.get("recording.auto_record", False)):
        return
    if getattr(system, "auto_record_started", False):
        return
    if system.video_writer is not None:
        return
    if system.last_result_frame is None:
        return
    if start_video_recording(system):
        system.auto_record_started = True
        print("[Recording] Auto-record started on launch")


def start_video_recording(system) -> bool:
    """
    Start video recording using current output-frame geometry.

    Logic:
        Opens a VideoWriter for the latest processed frame, choosing the first
        supported codec/FPS, and writes the first frame.

    Args:
        system: Runtime system instance holding video writer state.

    Returns:
        bool: True when a writer was opened successfully, False otherwise.
    """
    frame = system.last_result_frame
    if frame is None:
        print("\nRecording not started: no frame available yet")
        return False

    h, w = frame.shape[:2]
    source_obj = getattr(system.camera, "source", None)
    is_file_source = isinstance(source_obj, str)
    source_fps = (
        _resolve_file_playback_fps(system)
        if is_file_source
        else float(system.camera.get_fps() or 0.0)
    )
    runtime_fps = float(system.current_fps or 0.0)
    if is_file_source and source_fps > 1.0:
        # Preserve original video timing when recording processed output from file sources.
        fps = source_fps
    elif runtime_fps > 1.0 and source_fps > 1.0:
        fps = min(source_fps, runtime_fps)
    elif runtime_fps > 1.0:
        fps = runtime_fps
    elif source_fps > 1.0:
        fps = source_fps
    else:
        fps = 25.0
    output_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "docs", "media", "recordings"
    )
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Optional explicit FPS override for recording output.
    record_fps_override = float(
        system.config.get("recording.output_fps_override", 0.0) or 0.0
    )
    if record_fps_override > 1.0:
        fps = float(record_fps_override)

    preferred_codecs = _normalize_codec_list(
        system.config.get(
            "recording.codec_preference", ["avc1", "mp4v", "XVID", "MJPG"]
        )
    )
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = None
    chosen_codec = None
    chosen_path = None
    for codec in preferred_codecs:
        ext = _ext_for_codec(codec)
        candidate_path = os.path.join(
            output_dir, f"inspection-{timestamp}-{str(codec).lower()}{ext}"
        )
        candidate_writer = _try_open_video_writer(candidate_path, codec, fps, (w, h))
        if candidate_writer is not None and candidate_writer.isOpened():
            writer = candidate_writer
            chosen_codec = str(codec)
            chosen_path = candidate_path
            break
        if candidate_writer is not None:
            candidate_writer.release()
        print(f"Recording codec unavailable: {codec}")

    if writer is None or chosen_path is None or chosen_codec is None:
        # Legacy safety fallback (old behavior).
        fallback_codec = "mp4v"
        fallback_path = os.path.join(output_dir, f"inspection-{timestamp}.mp4")
        fallback_writer = _try_open_video_writer(
            fallback_path, fallback_codec, fps, (w, h)
        )
        if fallback_writer is not None and fallback_writer.isOpened():
            writer = fallback_writer
            chosen_codec = fallback_codec
            chosen_path = fallback_path
        else:
            if fallback_writer is not None:
                fallback_writer.release()
            system.video_writer = None
            system.video_output_path = None
            print("\nRecording failed: no supported codec from preference list")
            return False

    system.video_output_path = chosen_path
    system.video_writer = writer
    system.video_writer_frame_size = (int(w), int(h))
    frame_to_write = _prepare_frame_for_writer(frame, system.video_writer_frame_size)
    if frame_to_write is not None:
        system.video_writer.write(frame_to_write)
    print(
        f"\nRecording started: {system.video_output_path} (codec={chosen_codec}, fps={fps:.2f})"
    )
    return True


def write_output_frame(system, frame):
    """
    Write one output frame.

    Logic:
        Pushes one processed frame into the active writer when recording is enabled.

    Args:
        system: Runtime system instance holding writer state.
        frame: Processed frame to write.

    Returns:
        None.
    """
    if system.video_writer is None or frame is None:
        return
    size = getattr(system, "video_writer_frame_size", None)
    if size is None:
        h, w = frame.shape[:2]
        size = (int(w), int(h))
        system.video_writer_frame_size = size
    frame_to_write = _prepare_frame_for_writer(frame, size)
    if frame_to_write is not None:
        system.video_writer.write(frame_to_write)


def close_video_writer(system):
    """
    Close active video writer.

    Logic:
        Flushes/releases writer resources and clears output path state.

    Args:
        system: Runtime system instance holding writer state.

    Returns:
        None.
    """
    if system.video_writer is not None:
        system.video_writer.release()
        system.video_writer = None
    system.video_output_path = None
    if hasattr(system, "video_writer_frame_size"):
        system.video_writer_frame_size = None


def update_fps(system):
    """
    Update runtime FPS estimate.

    Logic:
        Accumulates frame count and periodically refreshes `current_fps` using elapsed wall time.

    Args:
        system: Runtime system instance holding FPS counters.

    Returns:
        None.
    """
    system.fps_counter += 1

    if system.fps_counter >= 10:
        current_time = time.time()
        elapsed = current_time - system.fps_start_time
        if elapsed > 0:
            system.current_fps = system.fps_counter / elapsed
        system.fps_counter = 0
        system.fps_start_time = current_time
