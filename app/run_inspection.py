"""
Main inspection system - Example script for running nozzle inspection.

This script demonstrates how to use the nozzle inspection system to perform
real-time monitoring of dry nozzles.
"""

import copy
import os
import sys
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nozzle_inspection import (
    BuildupDetector,
    CameraCapture,
    ConfigLoader,
    GridZoneOrchestrator,
    NozzleDetector,
    SprayAngleDetector,
    WarningSystem,
)
from nozzle_inspection.modules import (
    frame_pipeline,
    manual_zone_geometry,
    profile_store,
    runtime_io,
)


class NozzleInspectionSystem:
    """
    Main nozzle inspection system orchestrator.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        source_override: Any = None,
        source_mode: str = "raw",
    ):
        """
        Initialize the inspection system.

        Args:
            config_path: Path to configuration file (optional)
            source_override: Camera index or video path to override config camera.source
            source_mode: 'raw' or 'processed_output'
        """
        # Load configuration
        self.config = ConfigLoader(config_path)

        # Initialize components
        camera_config = self.config.get_section("camera")
        camera_source = camera_config.get("source", 0)
        if source_override is not None:
            camera_source = source_override

        self.camera = CameraCapture(
            source=camera_source,
            width=camera_config.get("width", 640),
            height=camera_config.get("height", 480),
            fps=camera_config.get("fps", 30),
        )

        nozzle_config = self.config.get_section("nozzle_detection")
        self.nozzle_config = dict(nozzle_config)
        self.base_nozzle_config = dict(self.nozzle_config)
        self.nozzle_detector = NozzleDetector(nozzle_config)

        buildup_config = self.config.get_section("buildup_detection")
        self.buildup_config = dict(buildup_config)
        self.buildup_detector = BuildupDetector(buildup_config)
        self.buildup_debug_overlay_enabled = bool(
            self.buildup_config.get("debug_overlay_enabled", False)
        )

        warning_config = self.config.get_section("warning")
        self.warning_config = dict(warning_config)

        self.display_config = self.config.get_section("display")
        self.source_mode = str(source_mode or "raw").strip().lower()
        self.processed_output_input_mode = self.source_mode in {
            "processed_output",
            "processed",
            "output",
        }
        self.strict_input_output_size = bool(
            self.display_config.get("strict_input_output_size", True)
        )
        self.lock_window_size = bool(self.display_config.get("lock_window_size", True))
        self.fixed_grid_rows = max(1, int(self.base_nozzle_config.get("grid_rows", 2)))
        self.fixed_grid_cols = max(1, int(self.base_nozzle_config.get("grid_cols", 3)))
        self.runtime_modes_config = self.config.get_section("runtime_modes")
        self.mode_enabled = {
            "DYN": bool(self.runtime_modes_config.get("enable_dyn", True)),
            "FIX": bool(self.runtime_modes_config.get("enable_fix", True)),
            "MANUAL": bool(self.runtime_modes_config.get("enable_manual", True)),
        }
        if (not self.mode_enabled["DYN"]) and (not self.mode_enabled["FIX"]):
            # Safety fallback: runtime detection needs at least one non-manual mode.
            self.mode_enabled["FIX"] = True
        startup_mode = (
            str(self.runtime_modes_config.get("startup_mode", "FIX")).strip().upper()
        )
        self.startup_runtime_mode = self._resolve_runtime_mode_fallback(startup_mode)
        self.start_in_video_only = bool(
            self.runtime_modes_config.get("start_in_video_only", True)
        )
        self.current_mode = (
            "VIDEO" if self.start_in_video_only else self.startup_runtime_mode
        )

        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.grid_rows = self.fixed_grid_rows
        self.grid_cols = self.fixed_grid_cols
        self.total_zones = self.grid_rows * self.grid_cols
        self.zone_nozzle_detectors: Dict[int, NozzleDetector] = {}
        self.zone_buildup_detectors: Dict[int, BuildupDetector] = {}
        self.zone_warning_systems: Dict[int, WarningSystem] = {}
        self.spray_angle_detector: SprayAngleDetector = SprayAngleDetector(
            self.nozzle_config, self.total_zones
        )
        self.zone_orchestrator: Optional[GridZoneOrchestrator] = None

        self.confirmation_window = int(self.nozzle_config.get("confirmation_window", 6))
        self.enter_confirmations = int(self.nozzle_config.get("enter_confirmations", 4))
        self.exit_rejections = int(self.nozzle_config.get("exit_rejections", 3))
        self.max_position_jump_px = float(
            self.nozzle_config.get("max_position_jump_px", 45.0)
        )
        self.body_check_half_width_px = float(
            self.nozzle_config.get("body_check_half_width_px", 22.0)
        )
        self.body_check_start_offset_px = float(
            self.nozzle_config.get("body_check_start_offset_px", 10.0)
        )
        self.body_check_head_ignore_radius_px = float(
            self.nozzle_config.get("body_check_head_ignore_radius_px", 18.0)
        )
        self.body_check_edge_margin_px = float(
            self.nozzle_config.get("body_check_edge_margin_px", 3.0)
        )
        self.body_check_min_length_to_edge_px = float(
            self.nozzle_config.get("body_check_min_length_to_edge_px", 18.0)
        )
        self.body_check_overlap_area_threshold = float(
            self.nozzle_config.get("body_check_overlap_area_threshold", 90.0)
        )
        self.show_body_check_overlay = bool(
            self.nozzle_config.get("show_body_check_overlay", True)
        )
        self.show_body_direction_overlay = bool(
            self.nozzle_config.get("show_body_direction_overlay", True)
        )
        self.body_direction_arrow_len_px = float(
            self.nozzle_config.get("body_direction_arrow_len_px", 42.0)
        )
        self.rescue_detection_enabled = bool(
            self.nozzle_config.get("rescue_detection_enabled", True)
        )
        self.rescue_min_aspect_ratio = float(
            self.nozzle_config.get("rescue_min_aspect_ratio", 0.45)
        )
        self.rescue_max_aspect_ratio = float(
            self.nozzle_config.get("rescue_max_aspect_ratio", 2.40)
        )
        self.rescue_center_x_tolerance_ratio = float(
            self.nozzle_config.get("rescue_center_x_tolerance_ratio", 0.40)
        )
        self.rescue_max_y_ratio_no_history = float(
            self.nozzle_config.get("rescue_max_y_ratio_no_history", 0.93)
        )
        self.spray_overlay_enabled = bool(
            self.nozzle_config.get("spray_overlay_enabled", True)
        )
        self.show_nozzle_head_circle = False
        self.show_nozzle_angle_arrow = False
        self.show_spray_angle_lines = False
        self.show_annular_mask = False
        self.show_spray_exclusion_mask = False
        self.show_analysis_mask = False
        self.show_spray_zone_overlay = False
        self.show_focus_zone_overlay = True
        self.show_focus_zone_panel = True
        self.show_status_panel = bool(
            self.display_config.get("show_status_panel", True)
        )
        self.show_body_check_overlay_dynamic = bool(self.show_body_check_overlay)
        self.show_body_direction_overlay_dynamic = bool(
            self.show_body_direction_overlay
        )
        self.show_focus_zone_overlay_dynamic = bool(self.show_focus_zone_overlay)
        self.show_status_panel_dynamic = bool(self.show_status_panel)
        self.status_panel_rect = (8, 8, 236, 186)
        self.debug_bbox_stabilization_enabled = bool(
            self.nozzle_config.get("debug_bbox_stabilization_enabled", True)
        )
        self.debug_bbox_ema_alpha = float(
            self.nozzle_config.get("debug_bbox_ema_alpha", 0.30)
        )
        self.show_dynamic_zone_debug_overlay = bool(
            self.nozzle_config.get("show_dynamic_zone_debug_overlay", True)
        )
        self.hide_undetected_zone_boxes = bool(
            self.nozzle_config.get("hide_undetected_zone_boxes", True)
        )
        self.hud_enabled = bool(self.display_config.get("hud_enabled", False))
        self.hud_width = int(self.display_config.get("hud_width", 420))
        self.metrics_auto_hud_on_overlap = bool(
            self.display_config.get("metrics_auto_hud_on_overlap", True)
        )
        self.metric_panel_prefer_near_zone_dynamic = bool(
            self.display_config.get("metric_panel_prefer_near_zone", True)
        )
        self.metric_panel_prefer_near_zone = False
        self.show_track_trails = bool(
            self.nozzle_config.get("show_track_trails", False)
        )
        self.track_history_length = int(
            self.nozzle_config.get("track_history_length", 24)
        )
        self.tip_track_history: Dict[int, Deque[Tuple[int, int]]] = {}
        self.zone_confirmation_history: Dict[int, Deque[bool]] = {}
        self.zone_confirmed_state: Dict[int, bool] = {}
        self.zone_ever_confirmed: Dict[int, bool] = {}
        self.zone_last_confirmed_position: Dict[int, Optional[Tuple[int, int]]] = {}
        self.zone_last_confirmed_global_position: Dict[
            int, Optional[Tuple[int, int]]
        ] = {}
        self.debug_bbox_state: Dict[int, Optional[np.ndarray]] = {}
        self.metric_panel_anchor_state: Dict[int, Optional[Tuple[int, int]]] = {}
        self._buildup_measure_counter: Dict[int, int] = {}
        self._buildup_cached_info: Dict[int, Dict] = {}
        self._buildup_yes_hold_counter: Dict[int, int] = {}
        self.video_writer = None
        self.video_writer_frame_size = None
        self.video_output_path = None
        self.auto_record_started = False
        # Display scale (percent) shared by ALL windows (main view + manual editor
        # + sub-tools) so the size stays in sync. Adjust live with '-' / '='.
        self.display_scale_percent = 100
        # (scale, off_x, off_y) mapping displayed window coords -> frame coords.
        self._display_transform = (1.0, 0, 0)
        self.last_result_frame = None
        self.last_input_frame = None
        self.last_zone_metrics: List[Dict[str, Any]] = []
        self.frame_id = 0
        self.video_frame_index = 0
        self.video_total_frames = 0
        self.video_auto_loop = bool(self.config.get("camera.video_auto_loop", True))
        self.last_mode_toggle_time = 0.0
        self.manual_zones = []
        self.manual_selected_zone_idx = 0
        self.previous_mode_before_manual = "FIX"
        self.manual_zone_path = self._resolve_repo_path(
            self.nozzle_config.get("manual_zones_path"), "config/manual_zones.json"
        )
        self.manual_move_step_px = int(self.nozzle_config.get("manual_move_step_px", 5))
        self.manual_resize_step_px = int(
            self.nozzle_config.get("manual_resize_step_px", 10)
        )
        self.manual_rotate_step_deg = float(
            self.nozzle_config.get("manual_rotate_step_deg", 15.0)
        )
        self.manual_min_size_px = int(self.nozzle_config.get("manual_min_size_px", 24))
        self.manual_zone_profile = (
            str(self.nozzle_config.get("manual_zone_profile", "default"))
            .strip()
            .lower()
        )
        if not self.manual_zone_profile:
            self.manual_zone_profile = "default"
        self.buildup_calibration_mode = (
            str(self.buildup_config.get("calibration_mode", "manual")).strip().lower()
        )
        if self.buildup_calibration_mode not in {"manual", "auto"}:
            self.buildup_calibration_mode = "manual"
        self.calib_manual_offset_step = float(
            self.buildup_config.get("calib_manual_offset_step", 0.01)
        )
        self.calib_manual_band_step = float(
            self.buildup_config.get("calib_manual_band_step", 0.02)
        )
        self.calib_spray_ring_step_px = int(
            self.buildup_config.get("calib_spray_ring_step_px", 1)
        )
        self.calib_spray_band_ratio_step = float(
            self.buildup_config.get("calib_spray_band_ratio_step", 0.01)
        )
        self.calibration_tab_rects = {}
        self.zone_lock_enabled = {}
        self.selected_lock_zone_id = 1
        self.lock_zone_select_mode = False
        self.lock_debug_visible = False
        self.zone_visual_lock_snapshots: Dict[int, Dict[str, Any]] = {}
        self.zone_mask_param_lock: Dict[int, Dict[str, Any]] = {}
        self._dl_inference = None
        self._dl_status_ticker = None
        self._ai_meta_by_idx = []
        self._dl_path_resolved = self._resolve_repo_path(
            self.nozzle_config.get("dl_model_path"),
            "deployment/pi5_final",
        )
        self._apply_runtime_mode(self.startup_runtime_mode)
        if self.start_in_video_only:
            self.current_mode = "VIDEO"
        self._apply_buildup_calibration_mode(self.buildup_calibration_mode, quiet=True)
        self._adjust_spray_exclusion_param(
            ring_delta=0, band_ratio_delta=0.0, quiet=True
        )

    def _enabled_runtime_modes(self):
        """
        Return enabled runtime modes excluding MANUAL.

        Logic:
            Produces ordered runtime mode list used by key-driven mode switching.

        Args:
            None.

        Returns:
            list[str]: Enabled runtime modes from {'DYN', 'FIX'}.
        """
        ordered = []
        for mode_name in ("DYN", "FIX"):
            if bool(self.mode_enabled.get(mode_name, False)):
                ordered.append(mode_name)
        if not ordered:
            ordered.append("FIX")
        return ordered

    def _resolve_runtime_mode_fallback(self, desired_mode):
        """
        Resolve runtime mode to an enabled fallback.

        Logic:
            Validates a desired mode and falls back to the first enabled runtime mode when needed.

        Args:
            desired_mode: Requested mode label.

        Returns:
            str: Enabled runtime mode in {'DYN', 'FIX'}.
        """
        candidate = str(desired_mode or "").strip().upper()
        enabled = self._enabled_runtime_modes()
        if candidate in enabled:
            return candidate
        return str(enabled[0])

    def _apply_runtime_mode(self, mode_label: str):
        mode = self._resolve_runtime_mode_fallback(mode_label)
        previous_mode = str(getattr(self, "current_mode", "")).strip().upper()

        self.current_mode = mode
        preset = copy.deepcopy(self.base_nozzle_config)
        preset["dynamic_zoning_enabled"] = mode == "DYN"
        self.nozzle_config = preset
        self.metric_panel_prefer_near_zone = (
            bool(self.metric_panel_prefer_near_zone_dynamic) if mode == "DYN" else False
        )
        if mode == "FIX":
            self.show_body_check_overlay = False
            self.show_body_direction_overlay = False
            self.show_focus_zone_overlay = True
            self.show_status_panel = True
        else:
            self.show_body_check_overlay = bool(self.show_body_check_overlay_dynamic)
            self.show_body_direction_overlay = bool(
                self.show_body_direction_overlay_dynamic
            )
            self.show_focus_zone_overlay = bool(self.show_focus_zone_overlay_dynamic)
            self.show_status_panel = bool(self.show_status_panel_dynamic)

        self.debug_bbox_stabilization_enabled = bool(
            self.nozzle_config.get("debug_bbox_stabilization_enabled", True)
        )
        self.debug_bbox_ema_alpha = float(
            self.nozzle_config.get("debug_bbox_ema_alpha", 0.30)
        )

        self.grid_rows = self.fixed_grid_rows
        self.grid_cols = self.fixed_grid_cols
        self.total_zones = int(self.grid_rows * self.grid_cols)

        self._rebuild_zone_state()
        self.zone_orchestrator = GridZoneOrchestrator(
            config=self.nozzle_config,
            zone_nozzle_detectors=self.zone_nozzle_detectors,
            spray_angle_detector=self.spray_angle_detector,
            grid_rows=self.grid_rows,
            grid_cols=self.grid_cols,
        )
        if mode == "DYN" and previous_mode == "FIX":
            self._seed_dynamic_zones_from_last_fix_metrics()
        self._reapply_zone_params()

    def _rebuild_zone_state(self):
        """Create fresh per-zone detectors and tracking state for self.total_zones zones."""
        zone_ids = range(1, self.total_zones + 1)
        self.zone_nozzle_detectors = {
            zid: NozzleDetector(self.nozzle_config) for zid in zone_ids
        }
        self.zone_buildup_detectors = {
            zid: BuildupDetector(self.buildup_config) for zid in zone_ids
        }
        self.zone_warning_systems = {
            zid: WarningSystem(self.warning_config) for zid in zone_ids
        }
        self.spray_angle_detector = SprayAngleDetector(
            self.nozzle_config, self.total_zones
        )
        self.tip_track_history = {
            zid: deque(maxlen=max(2, self.track_history_length)) for zid in zone_ids
        }
        self.zone_confirmation_history = {
            zid: deque(maxlen=self.confirmation_window) for zid in zone_ids
        }
        self.zone_confirmed_state = dict.fromkeys(zone_ids, False)
        self.zone_ever_confirmed = dict.fromkeys(zone_ids, False)
        self.zone_last_confirmed_position = dict.fromkeys(zone_ids)
        self.zone_last_confirmed_global_position = dict.fromkeys(zone_ids)
        self.debug_bbox_state = dict.fromkeys(zone_ids)
        self.metric_panel_anchor_state = dict.fromkeys(zone_ids)

    def _reapply_zone_params(self):
        """Re-apply lock states and calibration offsets after the zones were rebuilt."""
        self._apply_zone_lock_states(preserve_previous=True)
        self._apply_buildup_calibration_mode(self.buildup_calibration_mode, quiet=True)
        self._adjust_manual_annulus_param(offset_delta=0.0, band_delta=0.0, quiet=True)
        self._adjust_spray_exclusion_param(
            ring_delta=0, band_ratio_delta=0.0, quiet=True
        )

    def _seed_dynamic_zones_from_last_fix_metrics(self):
        orchestrator = self.zone_orchestrator
        if orchestrator is None or (
            not bool(getattr(orchestrator, "dynamic_zoning_enabled", False))
        ):
            return

        metrics = list(self.last_zone_metrics or [])
        if not metrics:
            return

        frame_ref = (
            self.last_input_frame
            if self.last_input_frame is not None
            else self.last_result_frame
        )
        if frame_ref is None:
            return

        defaults = orchestrator._default_grid_centers(frame_ref.shape)
        detected_by_zone: Dict[int, Tuple[float, float]] = {}
        detected_points: List[Tuple[float, float]] = []
        for metric in metrics:
            try:
                zone_id = int(metric.get("zone_id", 0))
            except Exception:
                continue
            if zone_id < 1 or zone_id > int(self.total_zones):
                continue
            nozzle_info = metric.get("nozzle_info", {}) or {}
            pos = nozzle_info.get("position")
            if pos is None:
                continue
            raw_detected = bool(metric.get("raw_detected", False))
            io_state = str(metric.get("io_state", "")).strip().upper()
            if (not raw_detected) and io_state != "IN":
                continue
            cx = float(pos[0])
            cy = float(pos[1])
            detected_by_zone[zone_id] = (cx, cy)
            detected_points.append((cx, cy))

        if not detected_points:
            return

        seeded_centers = {}
        for zone_id in range(1, int(self.total_zones) + 1):
            if zone_id in detected_by_zone:
                seeded_centers[zone_id] = detected_by_zone[zone_id]
                continue
            base = (
                defaults[zone_id - 1] if (zone_id - 1) < len(defaults) else defaults[-1]
            )
            nearest = min(
                detected_points,
                key=lambda p: float(
                    np.hypot(float(p[0]) - float(base[0]), float(p[1]) - float(base[1]))
                ),
            )
            seeded_centers[zone_id] = (float(nearest[0]), float(nearest[1]))

        orchestrator.dynamic_zone_centers = {
            zone_id: seeded_centers.get(zone_id)
            for zone_id in range(1, int(self.total_zones) + 1)
        }
        if bool(getattr(orchestrator, "dynamic_lock_then_track_enabled", False)):
            orchestrator.dynamic_locked_centers = {
                zone_id: seeded_centers.get(zone_id)
                for zone_id in range(1, int(self.total_zones) + 1)
            }
        orchestrator.dynamic_zone_miss_streak = {
            zone_id: 0 for zone_id in range(1, int(self.total_zones) + 1)
        }

    def _apply_zone_lock_states(self, preserve_previous=True):
        prev = (
            dict(self.zone_lock_enabled)
            if (preserve_previous and isinstance(self.zone_lock_enabled, dict))
            else {}
        )
        default_enabled = bool(
            self.buildup_config.get("baseline_lock_user_enabled", True)
        )
        lock_map = {}
        for zone_id in range(1, self.total_zones + 1):
            enabled = bool(prev.get(zone_id, default_enabled))
            lock_map[zone_id] = enabled
            detector = self.zone_buildup_detectors.get(zone_id)
            if detector is not None:
                detector.set_user_lock_enabled(
                    enabled, zone_id=zone_id, clear_baseline_on_disable=True
                )
            nozzle_detector = self.zone_nozzle_detectors.get(zone_id)
            if nozzle_detector is not None:
                nozzle_detector.set_user_geometry_lock(enabled, clear_on_disable=True)
            if enabled:
                if zone_id not in self.zone_mask_param_lock:
                    self._capture_zone_mask_param_lock(zone_id)
                self._apply_zone_mask_param_lock(zone_id)
            else:
                self.zone_visual_lock_snapshots.pop(zone_id, None)
                self.zone_mask_param_lock.pop(zone_id, None)
        self.zone_lock_enabled = lock_map
        self.selected_lock_zone_id = int(
            np.clip(int(self.selected_lock_zone_id), 1, max(1, self.total_zones))
        )

    def _toggle_zone_lock(self, zone_id=None, quiet=False):
        zid = int(zone_id if zone_id is not None else self.selected_lock_zone_id)
        zid = int(np.clip(zid, 1, max(1, self.total_zones)))
        enabled_now = bool(self.zone_lock_enabled.get(zid, True))
        enabled_next = not enabled_now
        self.zone_lock_enabled[zid] = enabled_next
        detector = self.zone_buildup_detectors.get(zid)
        if detector is not None:
            detector.set_user_lock_enabled(
                enabled_next,
                zone_id=zid,
                clear_baseline_on_disable=True,
            )
            if enabled_next:
                detector.request_force_baseline_capture(zone_id=zid)
            else:
                detector.set_calibration_mode(self.buildup_calibration_mode)
                detector.set_annulus_manual_params(
                    inner_offset_scale=self.buildup_config.get(
                        "annulus_head_inner_offset_scale"
                    ),
                    band_scale=self.buildup_config.get("annulus_head_band_scale"),
                )
                detector.set_spray_exclusion_params(
                    ring_dilate_px=self.buildup_config.get("analysis_ring_dilate_px"),
                    band_ratio=self.buildup_config.get(
                        "spray_exclusion_band_ratio_of_annulus"
                    ),
                )
        nozzle_detector = self.zone_nozzle_detectors.get(zid)
        if nozzle_detector is not None:
            nozzle_detector.set_user_geometry_lock(
                enabled_next,
                clear_on_disable=True,
            )
            if not enabled_next:
                nozzle_detector.reset_reference()
        if not enabled_next:
            self._stabilize_debug_bbox(zid, None)
        if enabled_next:
            self._try_capture_zone_lock_references(zid)
            self._capture_zone_mask_param_lock(zid)
            self._apply_zone_mask_param_lock(zid)
            self._capture_zone_visual_lock_snapshot(zid)
        else:
            self.zone_visual_lock_snapshots.pop(zid, None)
            self.zone_mask_param_lock.pop(zid, None)
        if not quiet:
            state_txt = "ON" if enabled_next else "OFF"
            print(f"\nZone {zid} lock: {state_txt}")

    def _cycle_selected_zone_lock(self):
        if self.total_zones <= 0:
            self.selected_lock_zone_id = 1
            return
        self.selected_lock_zone_id = 1 + (
            int(self.selected_lock_zone_id) % int(self.total_zones)
        )
        print(f"\nSelected lock zone: {int(self.selected_lock_zone_id)}")

    def _try_capture_zone_lock_references(self, zone_id):
        zid = int(zone_id)
        metric = None
        for item in list(self.last_zone_metrics or []):
            try:
                if int(item.get("zone_id", -1)) == zid:
                    metric = item
                    break
            except Exception:
                continue
        if metric is None:
            return
        nozzle_detector = self.zone_nozzle_detectors.get(zid)
        if nozzle_detector is not None:
            info = dict(metric.get("nozzle_info", {}) or {})
            pos = info.get("position")
            if pos is not None:
                try:
                    zx, zy, _zw, _zh = metric.get("rect", (0, 0, 0, 0))
                    info["position"] = (int(pos[0]) - int(zx), int(pos[1]) - int(zy))
                except Exception:
                    pass
            bbox = info.get("bounding_box")
            if bbox is not None:
                try:
                    zx, zy, _zw, _zh = metric.get("rect", (0, 0, 0, 0))
                    bx, by, bw, bh = bbox
                    info["bounding_box"] = (
                        int(bx) - int(zx),
                        int(by) - int(zy),
                        int(bw),
                        int(bh),
                    )
                except Exception:
                    pass
            nozzle_detector.request_user_geometry_lock_capture(info, force=False)

    def _capture_zone_visual_lock_snapshot(self, zone_id):
        zid = int(zone_id)
        for metric in list(self.last_zone_metrics or []):
            try:
                if int(metric.get("zone_id", -1)) != zid:
                    continue
            except Exception:
                continue
            binfo = metric.get("buildup_info", {}) or {}
            snap = {
                "zone_rect": tuple(metric.get("rect", (0, 0, 0, 0))),
                "roi_bbox_local": binfo.get("roi_bbox"),
                "annular_mask": None,
                "spray_exclusion_mask": None,
                "analysis_mask": None,
            }
            for key in ("annular_mask", "spray_exclusion_mask", "analysis_mask"):
                arr = binfo.get(key)
                if isinstance(arr, np.ndarray):
                    snap[key] = arr.copy()
            self.zone_visual_lock_snapshots[zid] = snap
            return

    def _capture_zone_mask_param_lock(self, zone_id):
        zid = int(zone_id)
        detector = self.zone_buildup_detectors.get(zid)
        if detector is None:
            return
        snap = detector.get_calibration_snapshot(zone_id=zid)
        offset_raw = snap.get(
            "active_offset_scale",
            snap.get(
                "manual_offset_scale",
                self.buildup_config.get("annulus_head_inner_offset_scale", 0.0),
            ),
        )
        band_raw = snap.get(
            "active_band_scale",
            snap.get(
                "manual_band_scale",
                self.buildup_config.get("annulus_head_band_scale", 0.0),
            ),
        )
        spray_band_raw = getattr(
            detector,
            "spray_exclusion_band_ratio_of_annulus",
            self.buildup_config.get("spray_exclusion_band_ratio_of_annulus", 0.22),
        )
        ring_dilate_raw = getattr(
            detector,
            "analysis_ring_dilate_px",
            self.buildup_config.get("analysis_ring_dilate_px", 7),
        )
        if isinstance(offset_raw, (int, float, np.integer, np.floating, str)):
            offset_val = float(offset_raw)
        else:
            offset_val = 0.0
        if isinstance(band_raw, (int, float, np.integer, np.floating, str)):
            band_val = float(band_raw)
        else:
            band_val = 0.0
        if isinstance(spray_band_raw, (int, float, np.integer, np.floating, str)):
            spray_band_val = float(spray_band_raw)
        else:
            spray_band_val = 0.22
        if isinstance(ring_dilate_raw, (int, float, np.integer, np.floating, str)):
            ring_dilate_val = int(float(ring_dilate_raw))
        else:
            ring_dilate_val = 7

        self.zone_mask_param_lock[zid] = {
            "calibration_mode": "manual",
            "offset": offset_val,
            "band": band_val,
            "spray_band_ratio": spray_band_val,
            "ring_dilate_px": ring_dilate_val,
        }

    def _apply_zone_mask_param_lock(self, zone_id):
        zid = int(zone_id)
        detector = self.zone_buildup_detectors.get(zid)
        snap = self.zone_mask_param_lock.get(zid)
        if detector is None or not isinstance(snap, dict):
            return
        detector.set_calibration_mode(str(snap.get("calibration_mode", "manual")))
        detector.set_annulus_manual_params(
            inner_offset_scale=snap.get("offset"),
            band_scale=snap.get("band"),
        )
        detector.set_spray_exclusion_params(
            ring_dilate_px=snap.get("ring_dilate_px"),
            band_ratio=snap.get("spray_band_ratio"),
        )

    def _prepare_detection_input(self, frame):
        if frame is None:
            return frame
        if not self.processed_output_input_mode:
            return frame
        if frame.ndim != 3 or frame.shape[2] != 3:
            return frame

        # Processed-output replay mode: suppress colored overlays before detection.
        b = frame[:, :, 0].astype(np.int16)
        g = frame[:, :, 1].astype(np.int16)
        r = frame[:, :, 2].astype(np.int16)
        chroma = np.maximum.reduce((np.abs(r - g), np.abs(g - b), np.abs(r - b)))
        overlay_mask = (chroma >= 18).astype(np.uint8) * 255
        if int(np.count_nonzero(overlay_mask)) <= 0:
            return frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_fill = cv2.GaussianBlur(gray, (5, 5), 0)
        cleaned = frame.copy()
        sel = overlay_mask > 0
        cleaned[sel] = gray_fill[sel][:, None]
        return cleaned

    def _persist_manual_zone_active_profile(self):
        active = profile_store.persist_manual_zone_active_profile(
            self.manual_zone_path,
            self.manual_zone_profile,
        )
        self.manual_zone_profile = str(active or "default").strip().lower() or "default"

    def _cycle_manual_zone_profile(self):
        names = profile_store.list_manual_zone_profiles(self.manual_zone_path)
        current = str(self.manual_zone_profile or "default").strip().lower()
        if current not in names:
            current = names[0]
        next_name = names[(names.index(current) + 1) % len(names)]
        self.manual_zone_profile = next_name
        self._persist_manual_zone_active_profile()
        print(f"\nZone profile: {self.manual_zone_profile}")

    def _delete_manual_zone_profile(self, profile_name):
        ok, next_name = profile_store.delete_manual_zone_profile(
            self.manual_zone_path, profile_name
        )
        if not ok:
            return False
        self.manual_zone_profile = (
            str(next_name or "default").strip().lower() or "default"
        )
        target = str(profile_name or "").strip().lower()
        print(
            f"\nDeleted profile '{target}'. Active profile -> '{self.manual_zone_profile}'"
        )
        return True

    @staticmethod
    def _resolve_repo_path(path_value, default_rel):
        """Resolve a config path relative to the repo root; empty means default_rel."""
        raw = str(path_value or "").strip() or default_rel
        if os.path.isabs(raw):
            return raw
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        return os.path.abspath(os.path.join(root_dir, raw))

    def get_dl_inference(self):
        """Lazy-load AFGNet ONNX inference. Returns None if model file missing or onnxruntime unavailable."""
        if self._dl_inference is not None:
            return self._dl_inference
        from nozzle_inspection.modules.dl_inference import load_afgnet

        inst = load_afgnet(
            self._dl_path_resolved,
            n_threads=int(self.nozzle_config.get("dl_threads", 4)),
            obj_threshold=float(self.nozzle_config.get("dl_obj_threshold", 0.10)),
            seed=int(self.nozzle_config.get("dl_seed", 42)),
        )
        if not inst.available:
            print(f"\n[AI] Model not available: {self._dl_path_resolved}")
            return None
        print(f"\n[AI] Loaded {type(inst).__name__} from {self._dl_path_resolved}")
        self._dl_inference = inst
        return inst

    def get_dl_status_ticker(self):
        """Lazy-construct + start background DL status ticker. Returns None if DL unavailable."""
        if getattr(self, "_dl_status_ticker", None) is not None:
            return self._dl_status_ticker
        dl = self.get_dl_inference()
        if dl is None:
            return None
        from nozzle_inspection.modules.dl_status_ticker import DLStatusTicker

        interval = float(self.nozzle_config.get("dl_status_interval_sec", 0.5))
        ticker = DLStatusTicker(dl=dl, interval_sec=interval)
        ticker.start()
        self._dl_status_ticker = ticker
        return ticker

    def stop_dl_status_ticker(self):
        ticker = getattr(self, "_dl_status_ticker", None)
        if ticker is not None:
            try:
                ticker.stop()
            except Exception:
                pass
            self._dl_status_ticker = None

    @staticmethod
    def _is_delete_key(full_code, low_code):
        return int(low_code) in (8, 127) or int(  # Backspace / Delete (ASCII)
            full_code
        ) in (
            3014656,
        )  # Delete on Windows waitKeyEx

    def _save_manual_zones(self):
        if not self.manual_zones:
            return
        payload = profile_store.load_manual_zone_payload(self.manual_zone_path)
        if not isinstance(payload, dict):
            payload = {}
        profiles = payload.get("profiles")
        if not isinstance(profiles, dict):
            # Upgrade legacy format.
            legacy_zones = (
                payload.get("zones") if isinstance(payload.get("zones"), list) else []
            )
            profiles = {"default": legacy_zones}
        profile_name = str(self.manual_zone_profile or "default").strip().lower()
        if not profile_name:
            profile_name = "default"
        profiles[profile_name] = [
            manual_zone_geometry.serialize_manual_zone(self, zone)
            for zone in self.manual_zones
        ]
        payload = {
            "active_profile": profile_name,
            "profiles": profiles,
        }
        profile_store.save_manual_zone_payload(self.manual_zone_path, payload)

    def _load_manual_zones(self, frame_shape):
        payload = profile_store.load_manual_zone_payload(self.manual_zone_path)
        if not isinstance(payload, dict):
            return []
        profiles = profile_store.extract_profiles_from_payload(payload)
        # If current profile is missing, fallback to persisted active then first available.
        requested = str(self.manual_zone_profile or "").strip().lower()
        if requested not in profiles:
            persisted_active = str(payload.get("active_profile", "")).strip().lower()
            if persisted_active in profiles:
                self.manual_zone_profile = persisted_active
            else:
                self.manual_zone_profile = sorted(profiles.keys())[0]
        zone_list = profile_store.extract_zones_from_payload(
            payload, self.manual_zone_profile
        )
        return manual_zone_geometry.sanitize_manual_zones(self, zone_list, frame_shape)

    def _manual_zone_reconfigure_runtime(self):
        self.total_zones = len(self.manual_zones)
        self.grid_rows = 1
        self.grid_cols = max(1, self.total_zones)
        self._rebuild_zone_state()
        self.zone_orchestrator = None
        self._reapply_zone_params()

    def _stabilize_debug_bbox(self, zone_id, bbox):
        zid = int(zone_id)
        if bbox is None:
            self.debug_bbox_state[zid] = None
            return None

        bx, by, bw, bh = [float(v) for v in bbox]
        current = np.array([bx, by, bw, bh], dtype=np.float32)
        if not self.debug_bbox_stabilization_enabled:
            self.debug_bbox_state[zid] = current.copy()
            return (
                int(round(bx)),
                int(round(by)),
                max(1, int(round(bw))),
                max(1, int(round(bh))),
            )

        prev = self.debug_bbox_state.get(zid)
        alpha = float(np.clip(self.debug_bbox_ema_alpha, 0.05, 1.0))
        if prev is None:
            smoothed = current
        else:
            smoothed = ((1.0 - alpha) * prev) + (alpha * current)
        self.debug_bbox_state[zid] = smoothed.copy()

        sx = int(round(float(smoothed[0])))
        sy = int(round(float(smoothed[1])))
        sw = max(1, int(round(float(smoothed[2]))))
        sh = max(1, int(round(float(smoothed[3]))))
        return (sx, sy, sw, sh)

    @staticmethod
    def _flush_pending_keys(max_reads=8):
        for _ in range(max(1, int(max_reads))):
            key = cv2.waitKey(1)
            if key == -1:
                break

    def _update_manual_zone_state(self, zone_id, candidate_ok, tile_info):
        history = self.zone_confirmation_history[zone_id]
        history.append(bool(candidate_ok))
        votes = int(sum(history))
        history_len = len(history)
        rejections = int(history_len - votes)

        current_state = bool(self.zone_confirmed_state[zone_id])
        if not current_state and votes >= self.enter_confirmations:
            current_state = True
        elif current_state and rejections >= self.exit_rejections:
            current_state = False

        self.zone_confirmed_state[zone_id] = current_state
        if current_state:
            self.zone_ever_confirmed[zone_id] = True
        if current_state and tile_info.get("position") is not None:
            self.zone_last_confirmed_position[zone_id] = tile_info.get("position")
        return current_state, votes, history_len

    def _detect_nozzles_manual(self, frame):
        zone_results = []
        frame_shape = frame.shape
        self.manual_zones = manual_zone_geometry.sanitize_manual_zones(
            self, self.manual_zones, frame_shape
        )
        for zone_id, zone_spec in enumerate(self.manual_zones, start=1):
            x, y, w, h, angle_deg = manual_zone_geometry.manual_zone_unpack(
                self, zone_spec
            )
            tile = manual_zone_geometry.extract_manual_zone_tile(
                self, frame, (x, y, w, h), angle_deg
            )
            tile_info = self.zone_nozzle_detectors[zone_id].detect(tile)
            raw_detected = bool(tile_info.get("detected", False))
            local_pos = tile_info.get("position")

            motion_ok = True
            if raw_detected and local_pos is not None:
                gx = float(local_pos[0] + x)
                gy = float(local_pos[1] + y)
                last_pos = self.zone_last_confirmed_global_position.get(zone_id)
                if last_pos is not None:
                    jump = float(
                        np.hypot(gx - float(last_pos[0]), gy - float(last_pos[1]))
                    )
                    motion_ok = jump <= float(self.max_position_jump_px)

            confirmed, votes, history_len = self._update_manual_zone_state(
                zone_id=zone_id,
                candidate_ok=(raw_detected and motion_ok),
                tile_info=tile_info,
            )

            zone_result = {
                "zone_id": zone_id,
                "row": 0,
                "col": zone_id - 1,
                "rect": (x, y, w, h),
                "angle_deg": float(angle_deg),
                "detected": bool(confirmed),
                "raw_detected": raw_detected,
                "status": (
                    "IN" if confirmed else ("CONFIRMING" if raw_detected else "OUT")
                ),
                "position": None,
                "bounding_box": None,
                "orientation": tile_info.get("orientation"),
                "radius": tile_info.get("radius"),
                "method": tile_info.get("detection_method", "manual_zone"),
                "motion_ok": bool(motion_ok),
                "owned_by_zone": zone_id if raw_detected else None,
                "body_length_px": 0.0,
                "body_polygon": None,
                "body_direction": None,
                "votes": int(votes),
                "history_len": int(history_len),
                "spray_info": dict(self.spray_angle_detector.empty_result()),
                "local_nozzle_info": tile_info,
            }

            if raw_detected and local_pos is not None:
                gx = int(local_pos[0] + x)
                gy = int(local_pos[1] + y)
                zone_result["position"] = (gx, gy)
                orient = tile_info.get("orientation")
                try:
                    orient_val = None if orient is None else float(orient)
                except Exception:
                    orient_val = None
                if orient_val is None or abs(float(orient_val)) < 1e-3:
                    fh, fw = frame.shape[:2]
                    cx = 0.5 * float(fw)
                    cy = 0.5 * float(fh)
                    # Fallback orientation in MANUAL mode: radial direction from frame center to nozzle.
                    # This keeps nozzle angle populated and responsive even when circle-method returns 0.
                    dx = float(gx) - cx
                    dy = float(gy) - cy
                    if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                        fallback_ang = float(np.degrees(np.arctan2(dy, dx)))
                        zone_result["orientation"] = fallback_ang
                        tile_info["orientation"] = fallback_ang
                if confirmed:
                    self.zone_last_confirmed_global_position[zone_id] = (gx, gy)
            local_bbox = tile_info.get("bounding_box")
            if raw_detected and local_bbox is not None:
                bx, by, bw, bh = local_bbox
                zone_result["bounding_box"] = (
                    int(bx + x),
                    int(by + y),
                    int(bw),
                    int(bh),
                )

            can_measure_spray = bool(zone_result["detected"]) and bool(
                zone_result["owned_by_zone"] == zone_id
            )
            if can_measure_spray:
                zone_result["spray_info"] = self.spray_angle_detector.estimate(
                    frame, zone_result
                )
            else:
                self.spray_angle_detector.reset_zone(zone_id)
                zone_result["spray_info"] = self.spray_angle_detector.empty_result()
            zone_results.append(zone_result)
        return zone_results

    def _apply_buildup_calibration_mode(self, mode, quiet=False):
        mode_txt = str(mode or "").strip().lower()
        if mode_txt not in {"manual", "auto"}:
            mode_txt = "manual"
        self.buildup_calibration_mode = mode_txt
        for zone_id, detector in self.zone_buildup_detectors.items():
            if bool(self.zone_lock_enabled.get(int(zone_id), False)):
                self._apply_zone_mask_param_lock(int(zone_id))
            else:
                detector.set_calibration_mode(mode_txt)
        self.buildup_config["calibration_mode"] = mode_txt
        if not quiet:
            print(f"\nBuildup calibration mode: {mode_txt.upper()}")

    def _toggle_buildup_calibration_mode(self):
        next_mode = "auto" if self.buildup_calibration_mode == "manual" else "manual"
        self._apply_buildup_calibration_mode(next_mode)

    def _adjust_manual_annulus_param(
        self, offset_delta=0.0, band_delta=0.0, quiet=False
    ):
        step_off = max(0.001, float(self.calib_manual_offset_step))
        step_band = max(0.002, float(self.calib_manual_band_step))
        off_min = float(
            self.buildup_config.get("annulus_head_inner_offset_min_scale", -0.35)
        )
        off_max = float(
            self.buildup_config.get("annulus_head_inner_offset_max_scale", 0.50)
        )
        if off_max < off_min:
            off_max = off_min
        if abs(float(offset_delta)) > 1e-9:
            self.buildup_config["annulus_head_inner_offset_scale"] = float(
                np.clip(
                    float(
                        self.buildup_config.get("annulus_head_inner_offset_scale", 0.06)
                    )
                    + (float(offset_delta) * step_off),
                    off_min,
                    off_max,
                )
            )
        if abs(float(band_delta)) > 1e-9:
            self.buildup_config["annulus_head_band_scale"] = float(
                np.clip(
                    float(self.buildup_config.get("annulus_head_band_scale", 0.58))
                    + (float(band_delta) * step_band),
                    0.05,
                    1.40,
                )
            )
        for zone_id, detector in self.zone_buildup_detectors.items():
            zid = int(zone_id)
            if bool(self.zone_lock_enabled.get(zid, False)):
                self._apply_zone_mask_param_lock(zid)
                continue
            detector.set_annulus_manual_params(
                inner_offset_scale=self.buildup_config.get(
                    "annulus_head_inner_offset_scale"
                ),
                band_scale=self.buildup_config.get("annulus_head_band_scale"),
            )
        if not quiet:
            print(
                "\nManual annulus params: "
                f"offset={float(self.buildup_config.get('annulus_head_inner_offset_scale', 0.0)):.3f}, "
                f"band={float(self.buildup_config.get('annulus_head_band_scale', 0.0)):.3f}"
            )

    def _scale_display_frame(self, frame):
        """Resize `frame` by self.display_scale_percent (aspect preserved, never
        distorted) and record the transform for mouse mapping. Used by every
        window so the main view and the manual editor stay the same size."""
        if frame is None:
            self._display_transform = (1.0, 0, 0)
            return frame
        sp = int(getattr(self, "display_scale_percent", 100) or 100)
        if sp == 100:
            self._display_transform = (1.0, 0, 0)
            return frame
        fh, fw = frame.shape[:2]
        factor = sp / 100.0
        nw = max(1, int(round(fw * factor)))
        nh = max(1, int(round(fh * factor)))
        interp = cv2.INTER_AREA if factor < 1.0 else cv2.INTER_LINEAR
        self._display_transform = (factor, 0, 0)
        return cv2.resize(frame, (nw, nh), interpolation=interp)

    def _map_display_xy(self, x, y):
        """Map displayed-window coords back to original frame coords."""
        tr = getattr(self, "_display_transform", None)
        if tr is None:
            return int(x), int(y)
        scale, off_x, off_y = tr
        if scale and scale > 1e-6:
            return int(round((x - off_x) / scale)), int(round((y - off_y) / scale))
        return int(x), int(y)

    def _on_main_window_mouse(self, event, x, y, flags, param):
        # Map window coords back to frame coords (when the view is scaled).
        x, y = self._map_display_xy(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            tabs = dict(self.calibration_tab_rects or {})
            for mode_name, rect in tabs.items():
                rx, ry, rw, rh = [int(v) for v in rect]
                if rx <= int(x) <= (rx + rw) and ry <= int(y) <= (ry + rh):
                    self._apply_buildup_calibration_mode(str(mode_name).lower())
                    return
            return

    def _adjust_spray_exclusion_param(
        self, line_delta=0, ring_delta=0, band_ratio_delta=0.0, quiet=False
    ):
        _ = line_delta  # spray-line control deprecated
        ring_step = max(1, int(self.calib_spray_ring_step_px))
        ratio_step = max(0.001, float(self.calib_spray_band_ratio_step))
        if int(ring_delta) != 0:
            self.buildup_config["analysis_ring_dilate_px"] = int(
                np.clip(
                    int(self.buildup_config.get("analysis_ring_dilate_px", 7))
                    + (int(ring_delta) * ring_step),
                    1,
                    80,
                )
            )
        if abs(float(band_ratio_delta)) > 1e-9:
            self.buildup_config["spray_exclusion_band_ratio_of_annulus"] = float(
                np.clip(
                    float(
                        self.buildup_config.get(
                            "spray_exclusion_band_ratio_of_annulus", 0.22
                        )
                    )
                    + (float(band_ratio_delta) * ratio_step),
                    0.01,
                    1.0,
                )
            )
        for zone_id, detector in self.zone_buildup_detectors.items():
            zid = int(zone_id)
            if bool(self.zone_lock_enabled.get(zid, False)):
                self._apply_zone_mask_param_lock(zid)
                continue
            detector.set_spray_exclusion_params(
                ring_dilate_px=self.buildup_config.get("analysis_ring_dilate_px"),
                band_ratio=self.buildup_config.get(
                    "spray_exclusion_band_ratio_of_annulus"
                ),
            )
        if not quiet:
            print(
                "\nSpray exclusion params: "
                f"band={100.0 * float(self.buildup_config.get('spray_exclusion_band_ratio_of_annulus', 0.0)):.1f}%, "
                f"ring={int(self.buildup_config.get('analysis_ring_dilate_px', 0))} px"
            )

    def run(self):
        """Delegate runtime main loop to runtime_io module."""
        runtime_io.run(self)

    def process_frame(self, frame):
        """
        Process a single frame through the inspection pipeline.

        Args:
            frame: Input frame

        Returns:
            Processed frame with visualization overlays
        """
        return frame_pipeline.process_frame(self, frame)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Nozzle Inspection System")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to configuration file"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Camera index (e.g. 0) or video file path",
    )
    parser.add_argument(
        "--source-mode",
        type=str,
        default="raw",
        choices=["raw", "processed_output"],
        help="Input stream type: raw camera/video or processed output replay",
    )

    args = parser.parse_args()

    source_override = None
    if args.source is not None:
        source_arg = args.source.strip()
        if source_arg.isdigit():
            source_override = int(source_arg)
        else:
            source_override = source_arg

    # Create and run system
    system = NozzleInspectionSystem(
        args.config, source_override=source_override, source_mode=args.source_mode
    )
    system.run()


if __name__ == "__main__":
    main()
