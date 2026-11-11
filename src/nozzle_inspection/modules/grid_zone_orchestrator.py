"""
Grid-zone orchestration module for per-zone nozzle detection workflow.
"""

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


class GridZoneOrchestrator:
    """
    Orchestrates per-zone detection lifecycle:
    split grid, detect, rescue, ownership resolution, confirmation state, spray estimate.
    """

    def __init__(
        self,
        config: Dict,
        zone_nozzle_detectors: Dict[int, Any],
        spray_angle_detector: Any,
        grid_rows: int = 2,
        grid_cols: int = 3,
    ):
        """Initialize object."""
        self.config = config
        self.zone_nozzle_detectors = zone_nozzle_detectors
        self.spray_angle_detector = spray_angle_detector
        self.grid_rows = int(grid_rows)
        self.grid_cols = int(grid_cols)
        self.total_zones = int(self.grid_rows * self.grid_cols)
        self.fullframe_min_contour_area = float(
            config.get(
                "fullframe_center_min_contour_area",
                config.get("min_contour_area", 500.0),
            )
        )
        self.fullframe_max_contour_area = float(
            config.get(
                "fullframe_center_max_contour_area",
                config.get("max_contour_area", 120000.0),
            )
        )

        self.dynamic_zoning_enabled = bool(config.get("dynamic_zoning_enabled", False))
        self.dynamic_tracking_mode = (
            str(config.get("dynamic_tracking_mode", "adaptive")).strip().lower()
        )
        self.dynamic_lock_then_track_enabled = (
            self.dynamic_tracking_mode == "lock_then_track"
        )
        self.dynamic_zone_margin_px = float(config.get("dynamic_zone_margin_px", 28.0))
        self.dynamic_zone_ema_alpha = float(config.get("dynamic_zone_ema_alpha", 0.45))
        self.dynamic_zone_min_size_px = int(config.get("dynamic_zone_min_size_px", 72))
        self.dynamic_zone_max_size_ratio = float(
            config.get("dynamic_zone_max_size_ratio", 0.55)
        )
        self.dynamic_zone_box_scale = float(config.get("dynamic_zone_box_scale", 0.72))
        self.dynamic_zone_kalman_enabled = bool(
            config.get("dynamic_zone_kalman_enabled", True)
        )
        self.dynamic_zone_kalman_process_noise = float(
            config.get("dynamic_zone_kalman_process_noise", 0.05)
        )
        self.dynamic_zone_kalman_measurement_noise = float(
            config.get("dynamic_zone_kalman_measurement_noise", 6.0)
        )
        self.dynamic_zone_max_center_jump_px = float(
            config.get("dynamic_zone_max_center_jump_px", 42.0)
        )
        self.dynamic_zone_anti_overlap_enabled = bool(
            config.get("dynamic_zone_anti_overlap_enabled", True)
        )
        self.dynamic_zone_min_center_spacing_ratio = float(
            config.get("dynamic_zone_min_center_spacing_ratio", 0.56)
        )
        self.zone_duplicate_claim_distance_px = float(
            config.get("zone_duplicate_claim_distance_px", 64.0)
        )
        self.dynamic_zone_lost_hold_frames = int(
            config.get("dynamic_zone_lost_hold_frames", 8)
        )
        self.dynamic_zone_reacquire_max_dist_px = float(
            config.get("dynamic_zone_reacquire_max_dist_px", 180.0)
        )
        self.dynamic_zone_bootstrap_search_scale = float(
            config.get("dynamic_zone_bootstrap_search_scale", 0.72)
        )
        self.dynamic_zone_center_deadband_px = float(
            config.get("dynamic_zone_center_deadband_px", 8.0)
        )
        self.dynamic_zone_use_raw_for_reacquire_only = bool(
            config.get("dynamic_zone_use_raw_for_reacquire_only", True)
        )
        self.dynamic_zone_center_smooth_alpha = float(
            config.get("dynamic_zone_center_smooth_alpha", 0.18)
        )
        self.dynamic_zone_update_interval_frames = int(
            config.get("dynamic_zone_update_interval_frames", 2)
        )
        self.dynamic_lock_min_votes = int(config.get("dynamic_lock_min_votes", 6))
        self.dynamic_lock_hold_frames = int(config.get("dynamic_lock_hold_frames", 18))
        self.dynamic_lock_update_alpha = float(
            config.get("dynamic_lock_update_alpha", 0.08)
        )
        self.dynamic_lock_roi_scale = float(config.get("dynamic_lock_roi_scale", 0.72))
        self.fullframe_rescue_enabled = bool(
            config.get("fullframe_rescue_enabled", True)
        )
        self.fullframe_rescue_min_fail_streak = int(
            config.get("fullframe_rescue_min_fail_streak", 3)
        )
        self.fullframe_rescue_max_assign_distance_px = float(
            config.get("fullframe_rescue_max_assign_distance_px", 260.0)
        )
        self.fullframe_rescue_claim_radius_px = float(
            config.get("fullframe_rescue_claim_radius_px", 70.0)
        )
        self.ownership_mode = (
            str(config.get("ownership_mode", "hybrid")).strip().lower()
        )
        self.ownership_center_max_dist_px = float(
            config.get("ownership_center_max_dist_px", 220.0)
        )
        self.fullframe_center_topk = int(
            config.get("fullframe_center_topk", max(12, self.total_zones * 4))
        )
        self.fullframe_center_diverse_min_dist_px = float(
            config.get("fullframe_center_diverse_min_dist_px", 150.0)
        )
        self.fullframe_center_var_threshold = float(
            config.get("fullframe_center_var_threshold", 5000.0)
        )
        self.fullframe_center_canny_low_default = int(
            config.get("fullframe_center_canny_low_default", 50)
        )
        self.fullframe_center_canny_low_noisy = int(
            config.get("fullframe_center_canny_low_noisy", 30)
        )
        self.fullframe_center_canny_high = int(
            config.get("fullframe_center_canny_high", 150)
        )
        self.fullframe_center_min_aspect = float(
            config.get("fullframe_center_min_aspect", 0.08)
        )
        self.fullframe_center_max_aspect = float(
            config.get("fullframe_center_max_aspect", 0.60)
        )
        self.fullframe_center_dual_path_enabled = bool(
            config.get("fullframe_center_dual_path_enabled", True)
        )
        self.fullframe_center_relaxed_min_aspect = float(
            config.get("fullframe_center_relaxed_min_aspect", 0.06)
        )
        self.fullframe_center_relaxed_max_aspect = float(
            config.get("fullframe_center_relaxed_max_aspect", 0.85)
        )
        self.fullframe_center_relax_center_ratio = float(
            config.get("fullframe_center_relax_center_ratio", 0.22)
        )
        self.fullframe_center_relax_max_aspect = float(
            config.get("fullframe_center_relax_max_aspect", 0.85)
        )
        self.dynamic_bootstrap_gate_enabled = bool(
            config.get("dynamic_bootstrap_gate_enabled", True)
        )
        self.dynamic_bootstrap_min_bbox_area_ratio = float(
            config.get("dynamic_bootstrap_min_bbox_area_ratio", 0.0012)
        )
        self.dynamic_bootstrap_border_margin_ratio = float(
            config.get("dynamic_bootstrap_border_margin_ratio", 0.05)
        )

        self.confirmation_window = int(config.get("confirmation_window", 6))
        self.enter_confirmations = int(config.get("enter_confirmations", 4))
        self.exit_rejections = int(config.get("exit_rejections", 3))
        self.max_position_jump_px = float(config.get("max_position_jump_px", 45.0))
        self.confirm_enter_trusted_only = bool(
            config.get("confirm_enter_trusted_only", True)
        )
        self.confirm_enter_require_kalman = bool(
            config.get("confirm_enter_require_kalman", True)
        )
        allowed_bases = config.get(
            "confirm_enter_allowed_base_methods", ["circle", "contour"]
        )
        if isinstance(allowed_bases, (list, tuple)):
            self.confirm_enter_allowed_base_methods = {
                str(item).strip().lower() for item in allowed_bases if str(item).strip()
            }
        else:
            self.confirm_enter_allowed_base_methods = {"circle", "contour"}
        if not self.confirm_enter_allowed_base_methods:
            self.confirm_enter_allowed_base_methods = {"circle", "contour"}

        self.body_check_enabled = bool(config.get("body_check_enabled", True))
        self.body_check_length_px = float(config.get("body_check_length_px", 140.0))
        self.body_check_half_width_px = float(
            config.get("body_check_half_width_px", 22.0)
        )
        self.body_check_start_offset_px = float(
            config.get("body_check_start_offset_px", 10.0)
        )
        self.body_check_head_ignore_radius_px = float(
            config.get("body_check_head_ignore_radius_px", 18.0)
        )
        self.body_check_edge_margin_px = float(
            config.get("body_check_edge_margin_px", 3.0)
        )
        self.body_check_min_length_to_edge_px = float(
            config.get("body_check_min_length_to_edge_px", 18.0)
        )
        self.body_check_overlap_area_threshold = float(
            config.get("body_check_overlap_area_threshold", 90.0)
        )
        self.body_direction_center_blend = float(
            config.get("body_direction_center_blend", 0.22)
        )

        self.rescue_detection_enabled = bool(
            config.get("rescue_detection_enabled", True)
        )
        self.rescue_center_only = bool(config.get("rescue_center_only", True))
        self.rescue_min_area_ratio = float(config.get("rescue_min_area_ratio", 0.0015))
        self.rescue_max_area_ratio = float(config.get("rescue_max_area_ratio", 0.72))
        self.rescue_binary_bias = float(config.get("rescue_binary_bias", 8.0))
        self.rescue_max_tip_jump_px = float(config.get("rescue_max_tip_jump_px", 80.0))
        self.rescue_min_circularity = float(config.get("rescue_min_circularity", 0.45))
        self.rescue_min_extent = float(config.get("rescue_min_extent", 0.35))
        self.rescue_max_extent = float(config.get("rescue_max_extent", 0.92))
        self.rescue_min_aspect_ratio = float(
            config.get("rescue_min_aspect_ratio", 0.45)
        )
        self.rescue_max_aspect_ratio = float(
            config.get("rescue_max_aspect_ratio", 2.40)
        )
        self.rescue_min_fail_streak = int(config.get("rescue_min_fail_streak", 2))
        self.rescue_confirm_frames = int(config.get("rescue_confirm_frames", 2))
        self.rescue_center_x_tolerance_ratio = float(
            config.get("rescue_center_x_tolerance_ratio", 0.40)
        )
        self.rescue_max_y_ratio_no_history = float(
            config.get("rescue_max_y_ratio_no_history", 0.93)
        )
        self.spray_cadence_weak_frames = int(
            max(1, config.get("spray_cadence_weak_frames", 2))
        )
        self.spray_cadence_confirming_frames = int(
            max(1, config.get("spray_cadence_confirming_frames", 2))
        )

        self.zone_confirmation_history: Dict[int, deque] = {}
        self.zone_confirmed_state: Dict[int, bool] = {}
        self.zone_ever_confirmed: Dict[int, bool] = {}
        self.zone_last_confirmed_position: Dict[int, Optional[Tuple[int, int]]] = {}
        self.zone_last_confirmed_global_position: Dict[
            int, Optional[Tuple[int, int]]
        ] = {}
        self.zone_rescue_fail_streak: Dict[int, int] = {}
        self.zone_rescue_pending_hits: Dict[int, int] = {}
        self.dynamic_zone_centers: Dict[int, Optional[Tuple[float, float]]] = {}
        self.dynamic_zone_kalman_filters: Dict[int, Optional[Any]] = {}
        self.dynamic_zone_miss_streak: Dict[int, int] = {}
        self.dynamic_locked_centers: Dict[int, Optional[Tuple[float, float]]] = {}
        self.dynamic_lock_miss_streak: Dict[int, int] = {}
        self.frame_index = 0
        self.spray_measure_counter: Dict[int, int] = {}
        self.spray_cached_info: Dict[int, Dict[str, Any]] = {}
        self.reset()

    def reset(self):
        """Clear all per-zone tracking state (confirmation history, dynamic centers, Kalman filters, locks, spray cache)."""
        self.zone_confirmation_history = {
            zone_id: deque(maxlen=self.confirmation_window)
            for zone_id in range(1, self.total_zones + 1)
        }
        self.zone_confirmed_state = {
            zone_id: False for zone_id in range(1, self.total_zones + 1)
        }
        self.zone_ever_confirmed = {
            zone_id: False for zone_id in range(1, self.total_zones + 1)
        }
        self.zone_last_confirmed_position = {
            zone_id: None for zone_id in range(1, self.total_zones + 1)
        }
        self.zone_last_confirmed_global_position = {
            zone_id: None for zone_id in range(1, self.total_zones + 1)
        }
        self.zone_rescue_fail_streak = {
            zone_id: 0 for zone_id in range(1, self.total_zones + 1)
        }
        self.zone_rescue_pending_hits = {
            zone_id: 0 for zone_id in range(1, self.total_zones + 1)
        }
        self.dynamic_zone_centers = {
            zone_id: None for zone_id in range(1, self.total_zones + 1)
        }
        self.dynamic_zone_kalman_filters = {
            zone_id: None for zone_id in range(1, self.total_zones + 1)
        }
        self.dynamic_zone_miss_streak = {
            zone_id: 0 for zone_id in range(1, self.total_zones + 1)
        }
        self.dynamic_locked_centers = {
            zone_id: None for zone_id in range(1, self.total_zones + 1)
        }
        self.dynamic_lock_miss_streak = {
            zone_id: 0 for zone_id in range(1, self.total_zones + 1)
        }
        self.frame_index = 0
        self.spray_measure_counter = {
            zone_id: 0 for zone_id in range(1, self.total_zones + 1)
        }
        self.spray_cached_info = {
            zone_id: dict(self.spray_angle_detector.empty_result())
            for zone_id in range(1, self.total_zones + 1)
        }

    @staticmethod
    def _clip_jump(prev_pt, new_pt, max_jump):
        """Limit a point's displacement from prev_pt to max_jump pixels, scaling the step back along its direction if exceeded."""
        if prev_pt is None or max_jump <= 0.0:
            return (float(new_pt[0]), float(new_pt[1]))
        dx = float(new_pt[0]) - float(prev_pt[0])
        dy = float(new_pt[1]) - float(prev_pt[1])
        dist = float(np.hypot(dx, dy))
        if dist <= max_jump or dist <= 1e-6:
            return (float(new_pt[0]), float(new_pt[1]))
        scale = float(max_jump / dist)
        return (float(prev_pt[0]) + (dx * scale), float(prev_pt[1]) + (dy * scale))

    def _init_center_kalman(self, zone_id, center):
        """Create and store a constant-velocity Kalman filter for the zone, seeded at the given center with zero velocity."""
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            np.float32,
        )
        proc = max(1e-6, float(self.dynamic_zone_kalman_process_noise))
        meas = max(1e-6, float(self.dynamic_zone_kalman_measurement_noise))
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * proc
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * meas
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        cx, cy = float(center[0]), float(center[1])
        kf.statePost = np.array([[cx], [cy], [0.0], [0.0]], np.float32)
        self.dynamic_zone_kalman_filters[int(zone_id)] = kf
        return kf

    def _update_center_track(self, zone_id, measured_center, default_center):
        """Smooth a zone's tracked center via deadband, jump-clip, then Kalman correction or EMA, returning the filtered center."""
        prev_center = self.dynamic_zone_centers.get(int(zone_id))
        if measured_center is None:
            measured_center = prev_center if prev_center is not None else default_center

        # Reject tiny center oscillations to avoid jitter on visually static nozzles.
        if prev_center is not None:
            deadband = max(0.0, float(self.dynamic_zone_center_deadband_px))
            if deadband > 0.0:
                d_small = float(
                    np.hypot(
                        float(measured_center[0]) - float(prev_center[0]),
                        float(measured_center[1]) - float(prev_center[1]),
                    )
                )
                if d_small <= deadband:
                    measured_center = prev_center

        measured_center = self._clip_jump(
            prev_center, measured_center, self.dynamic_zone_max_center_jump_px
        )

        if not self.dynamic_zone_kalman_enabled:
            if prev_center is None:
                return (float(measured_center[0]), float(measured_center[1]))
            alpha = float(np.clip(self.dynamic_zone_ema_alpha, 0.0, 1.0))
            sx = ((1.0 - alpha) * float(prev_center[0])) + (
                alpha * float(measured_center[0])
            )
            sy = ((1.0 - alpha) * float(prev_center[1])) + (
                alpha * float(measured_center[1])
            )
            return (float(sx), float(sy))

        kf = self.dynamic_zone_kalman_filters.get(int(zone_id))
        if kf is None:
            kf = self._init_center_kalman(zone_id, measured_center)

        pred = kf.predict()
        pred_center = (float(pred[0, 0]), float(pred[1, 0]))
        clipped_measure = self._clip_jump(
            pred_center, measured_center, self.dynamic_zone_max_center_jump_px
        )
        meas = np.array(
            [[float(clipped_measure[0])], [float(clipped_measure[1])]], dtype=np.float32
        )
        corr = kf.correct(meas)
        out = (float(corr[0, 0]), float(corr[1, 0]))
        if prev_center is not None:
            a = float(np.clip(self.dynamic_zone_center_smooth_alpha, 0.01, 1.0))
            out = (
                ((1.0 - a) * float(prev_center[0])) + (a * float(out[0])),
                ((1.0 - a) * float(prev_center[1])) + (a * float(out[1])),
            )
        return out

    @staticmethod
    def _separate_centers(centers_by_zone, min_spacing, frame_w, frame_h):
        """Iteratively push overlapping zone centers apart to enforce min_spacing, clamped within frame bounds."""
        if len(centers_by_zone) <= 1 or min_spacing <= 0.0:
            return centers_by_zone
        zone_ids = sorted(centers_by_zone.keys())
        centers = {}
        for z in zone_ids:
            c = centers_by_zone.get(z)
            if c is None:
                continue
            centers[z] = (float(c[0]), float(c[1]))
        if len(centers) <= 1:
            return centers_by_zone

        active_zone_ids = sorted(centers.keys())
        for _ in range(4):
            moved = False
            for i, zi in enumerate(active_zone_ids):
                for zj in active_zone_ids[i + 1 :]:
                    xi, yi = centers[zi]
                    xj, yj = centers[zj]
                    dx = xj - xi
                    dy = yj - yi
                    dist = float(np.hypot(dx, dy))
                    if dist >= min_spacing:
                        continue
                    if dist <= 1e-6:
                        dx, dy, dist = 1.0, 0.0, 1.0
                    push = 0.5 * (min_spacing - dist)
                    ux = dx / dist
                    uy = dy / dist
                    xi2 = float(np.clip(xi - (ux * push), 0.0, float(frame_w - 1)))
                    yi2 = float(np.clip(yi - (uy * push), 0.0, float(frame_h - 1)))
                    xj2 = float(np.clip(xj + (ux * push), 0.0, float(frame_w - 1)))
                    yj2 = float(np.clip(yj + (uy * push), 0.0, float(frame_h - 1)))
                    centers[zi] = (xi2, yi2)
                    centers[zj] = (xj2, yj2)
                    moved = True
            if not moved:
                break
        out = dict(centers_by_zone)
        for z, c in centers.items():
            out[z] = c
        return out

    def _is_rescue_zone(self, row, col):
        """Return whether rescue detection applies to this grid cell (all cells, or only the center column when center-only)."""
        if not self.rescue_detection_enabled:
            return False
        if not self.rescue_center_only:
            return True
        return int(col) == int(self.grid_cols // 2)

    def _default_grid_centers(self, frame_shape) -> List[Tuple[float, float]]:
        """Compute the geometric center of each fixed grid tile in row-major order for the given frame size."""
        h, w = frame_shape[:2]
        tile_w = max(1, w // self.grid_cols)
        tile_h = max(1, h // self.grid_rows)
        centers = []
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                x = col * tile_w
                y = row * tile_h
                tw = (w - x) if col == self.grid_cols - 1 else tile_w
                th = (h - y) if row == self.grid_rows - 1 else tile_h
                centers.append((float(x + (tw * 0.5)), float(y + (th * 0.5))))
        return centers

    def _full_frame_detect_candidates(self, frame):
        """Find slender nozzle-like blobs across the whole frame via adaptive Canny + contour filtering, ranked by area (top-k)."""
        if frame is None or frame.size == 0:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        frame_var = float(np.var(blurred))
        canny_low = (
            self.fullframe_center_canny_low_noisy
            if frame_var > self.fullframe_center_var_threshold
            else self.fullframe_center_canny_low_default
        )
        canny_high = int(max(int(canny_low) + 1, self.fullframe_center_canny_high))
        edges = cv2.Canny(blurred, int(max(1, canny_low)), canny_high)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        strict_candidates = []
        relaxed_candidates = []
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if (
                area <= self.fullframe_min_contour_area
                or area >= self.fullframe_max_contour_area
            ):
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w <= 2 or h <= 2:
                continue

            # Orientation-invariant slenderness ratio in [0, 1].
            ratio = float(min(float(w) / float(max(1, h)), float(h) / float(max(1, w))))
            # Relax aspect for candidates near frame center to avoid missing middle nozzle.
            cx_box = float(x + (0.5 * w))
            cy_box = float(y + (0.5 * h))
            frame_center_x = 0.5 * float(frame.shape[1])
            frame_center_y = 0.5 * float(frame.shape[0])
            center_dist = float(
                np.hypot(cx_box - frame_center_x, cy_box - frame_center_y)
            )
            center_radius = float(min(frame.shape[0], frame.shape[1])) * float(
                np.clip(self.fullframe_center_relax_center_ratio, 0.05, 0.45)
            )
            max_aspect = (
                self.fullframe_center_relax_max_aspect
                if center_dist <= center_radius
                else self.fullframe_center_max_aspect
            )
            m = cv2.moments(cnt)
            if abs(float(m.get("m00", 0.0))) > 1e-6:
                cx = float(m["m10"] / m["m00"])
                cy = float(m["m01"] / m["m00"])
            else:
                cx = float(x + (0.5 * w))
                cy = float(y + (0.5 * h))
            item = (cx, cy, area, cnt, (int(x), int(y), int(w), int(h)))
            if ratio >= self.fullframe_center_min_aspect and ratio <= max_aspect:
                strict_candidates.append(item)
            elif (
                ratio >= self.fullframe_center_relaxed_min_aspect
                and ratio <= self.fullframe_center_relaxed_max_aspect
            ):
                relaxed_candidates.append(item)

        strict_candidates.sort(key=lambda c: float(c[2]), reverse=True)
        relaxed_candidates.sort(key=lambda c: float(c[2]), reverse=True)
        topk = max(1, int(self.fullframe_center_topk))
        if not self.fullframe_center_dual_path_enabled:
            return strict_candidates[:topk]

        out = list(strict_candidates[:topk])
        seen = {(int(c[0]), int(c[1])) for c in out}
        for item in relaxed_candidates:
            key = (int(item[0]), int(item[1]))
            if key in seen:
                continue
            out.append(item)
            seen.add(key)
            if len(out) >= topk:
                break
        return out[:topk]

    def _passes_dynamic_bootstrap_gate(self, tile_info, tile_w, tile_h, ever_in):
        """Gate a first-time zone detection on minimum bbox area ratio and non-border placement to avoid spurious bootstraps."""
        if not self.dynamic_bootstrap_gate_enabled:
            return True
        if ever_in:
            return True
        bbox = tile_info.get("bounding_box")
        if bbox is None:
            return True
        bx, by, bw, bh = [int(v) for v in bbox]
        if bw <= 0 or bh <= 0 or tile_w <= 0 or tile_h <= 0:
            return False
        area_ratio = float(bw * bh) / float(max(1, tile_w * tile_h))
        if area_ratio < float(self.dynamic_bootstrap_min_bbox_area_ratio):
            return False
        margin = float(
            np.clip(self.dynamic_bootstrap_border_margin_ratio, 0.0, 0.3)
        ) * float(min(tile_w, tile_h))
        touches_border = (
            float(bx) <= margin
            or float(by) <= margin
            or float(bx + bw) >= (float(tile_w) - margin)
            or float(by + bh) >= (float(tile_h) - margin)
        )
        if touches_border:
            return False
        return True

    def _full_frame_detect_nozzles(self, frame) -> List[Dict]:
        """Build full-frame nozzle dicts (tip, center, bbox, contour) from candidates, spatially diversified and capped at top-k."""
        if frame is None or frame.size == 0:
            return []
        frame_h, frame_w = frame.shape[:2]
        raw_candidates = self._full_frame_detect_candidates(frame)
        full_nozzles = []
        for cx, cy, area, cnt, bbox in raw_candidates:
            center = (float(cx), float(cy))
            tip = self._estimate_global_tip_from_contour(
                cnt, center, (frame_h, frame_w)
            )
            x, y, w, h = bbox
            full_nozzles.append(
                {
                    "position": tip,
                    "ellipse_center": center,
                    "bbox": (int(x), int(y), int(w), int(h)),
                    "contour": cnt,
                    "area": float(area),
                }
            )

        full_nozzles.sort(key=lambda n: float(n.get("area", 0.0)), reverse=True)
        topk = max(1, int(self.fullframe_center_topk))
        diverse = []
        min_dist = max(0.0, float(self.fullframe_center_diverse_min_dist_px))
        for item in full_nozzles:
            p = item.get("position")
            if p is None:
                continue
            px, py = float(p[0]), float(p[1])
            if diverse:
                nearest = min(
                    float(
                        np.hypot(
                            px - float(d["position"][0]), py - float(d["position"][1])
                        )
                    )
                    for d in diverse
                )
                if nearest < min_dist:
                    continue
            diverse.append(item)
            if len(diverse) >= topk:
                break

        # Keep robust start-up: if diversity filter is too strict, fallback to top area list.
        if len(diverse) == 0:
            diverse = full_nozzles[:topk]
        return diverse[:topk]

    @staticmethod
    def _compute_center_unit(pos, frame_center):
        """Return the unit vector pointing from the frame center toward pos, or None if they nearly coincide."""
        dx = float(pos[0] - frame_center[0])
        dy = float(pos[1] - frame_center[1])
        norm = float(np.hypot(dx, dy))
        if norm < 1e-6:
            return None
        return (dx / norm, dy / norm)

    def _estimate_global_tip_from_contour(self, contour, fallback_center, frame_hw):
        """Pick the contour point furthest inward (toward frame center) as the nozzle tip, falling back to the center."""
        if contour is None:
            return (float(fallback_center[0]), float(fallback_center[1]))
        try:
            pts = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
        except Exception:
            return (float(fallback_center[0]), float(fallback_center[1]))
        if pts.shape[0] < 4:
            return (float(fallback_center[0]), float(fallback_center[1]))

        cx = float(fallback_center[0])
        cy = float(fallback_center[1])
        frame_h, frame_w = frame_hw
        frame_center = (0.5 * float(frame_w), 0.5 * float(frame_h))
        center_unit = self._compute_center_unit((cx, cy), frame_center)
        if center_unit is None:
            return (cx, cy)

        ux, uy = float(center_unit[0]), float(center_unit[1])
        rel_x = pts[:, 0] - cx
        rel_y = pts[:, 1] - cy
        proj = (rel_x * ux) + (rel_y * uy)
        if proj.size == 0:
            return (cx, cy)

        # Use inward tip (toward frame center) as nozzle head position.
        inward_idx = int(np.argmin(proj))
        tip = (float(pts[inward_idx, 0]), float(pts[inward_idx, 1]))
        return tip

    @staticmethod
    def _zone_rect_center(rect):
        """Return the center (x, y) of a zone rectangle given as (x, y, w, h)."""
        x, y, w, h = rect
        return (float(x) + (0.5 * float(w)), float(y) + (0.5 * float(h)))

    def _apply_fullframe_rescue(self, zone_results, full_nozzles):
        """Assign unclaimed full-frame nozzles to undetected high-fail-streak zones (nearest within range), filling their results in place."""
        if not self.fullframe_rescue_enabled or not full_nozzles:
            return

        claimed = set()
        claimed_positions = []
        for idx, nozzle in enumerate(full_nozzles):
            nx, ny = nozzle["position"]
            keep = True
            for zone in zone_results:
                if not zone.get("raw_detected", False):
                    continue
                zpos = zone.get("position")
                if zpos is None:
                    continue
                if (
                    float(np.hypot(float(zpos[0]) - nx, float(zpos[1]) - ny))
                    <= self.fullframe_rescue_claim_radius_px
                ):
                    claimed.add(idx)
                    keep = False
                    break
            if keep:
                claimed_positions.append((idx, nozzle["position"]))

        candidates = []
        for zone in zone_results:
            if zone.get("raw_detected", False):
                continue
            zone_id = int(zone["zone_id"])
            fail_streak = int(self.zone_rescue_fail_streak.get(zone_id, 0))
            if fail_streak < max(1, self.fullframe_rescue_min_fail_streak):
                continue
            candidates.append((fail_streak, zone))

        candidates.sort(key=lambda item: -item[0])
        for _fail_streak, zone in candidates:
            zc = self._zone_rect_center(zone["rect"])
            best_idx = None
            best_dist = float("inf")
            for idx, pos in claimed_positions:
                if idx in claimed:
                    continue
                dist = float(np.hypot(pos[0] - zc[0], pos[1] - zc[1]))
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            if (
                best_idx is None
                or best_dist > self.fullframe_rescue_max_assign_distance_px
            ):
                continue

            chosen = full_nozzles[best_idx]
            nx, ny = chosen["position"]
            x, y, w, h = zone["rect"]
            local_x = int(round(nx - float(x)))
            local_y = int(round(ny - float(y)))
            if local_x < 0 or local_x >= w or local_y < 0 or local_y >= h:
                continue

            bx, by, bw, bh = chosen["bbox"]
            clipped_x0 = max(int(bx), int(x))
            clipped_y0 = max(int(by), int(y))
            clipped_x1 = min(int(bx + bw), int(x + w))
            clipped_y1 = min(int(by + bh), int(y + h))
            if clipped_x1 <= clipped_x0 or clipped_y1 <= clipped_y0:
                continue

            zone["raw_detected"] = True
            zone["position"] = (int(round(nx)), int(round(ny)))
            zone["bounding_box"] = (
                int(clipped_x0),
                int(clipped_y0),
                int(clipped_x1 - clipped_x0),
                int(clipped_y1 - clipped_y0),
            )
            zone["method"] = "fullframe_rescue"
            zone["local_nozzle_info"] = {
                "detected": True,
                "position": (int(local_x), int(local_y)),
                "bounding_box": (
                    int(clipped_x0 - x),
                    int(clipped_y0 - y),
                    int(clipped_x1 - clipped_x0),
                    int(clipped_y1 - clipped_y0),
                ),
                "orientation": None,
                "radius": int(
                    max(
                        3,
                        round(
                            0.5 * max(clipped_x1 - clipped_x0, clipped_y1 - clipped_y0)
                        ),
                    )
                ),
                "contour": chosen["contour"],
                "detection_method": "fullframe_rescue",
            }

            last_pos = self.zone_last_confirmed_position.get(int(zone["zone_id"]))
            if last_pos is not None:
                jump = float(
                    np.hypot(
                        zone["position"][0] - last_pos[0],
                        zone["position"][1] - last_pos[1],
                    )
                )
                zone["motion_ok"] = jump <= self.max_position_jump_px
            else:
                zone["motion_ok"] = True

            claimed.add(best_idx)

    def _build_zone_specs(
        self, frame
    ) -> List[Tuple[int, int, int, Tuple[int, int, int, int]]]:
        """Build per-zone search rectangles: fixed grid, or dynamic boxes sized/centered on locked or tracked centers (bootstrap if none)."""
        h, w = frame.shape[:2]
        if not self.dynamic_zoning_enabled:
            return self._build_fixed_grid_specs(frame)

        fixed_specs = self._build_fixed_grid_specs(frame)
        base_w = int(
            min(
                w * self.dynamic_zone_max_size_ratio,
                (w / max(1, self.grid_cols)) + (2.0 * self.dynamic_zone_margin_px),
            )
        )
        base_h = int(
            min(
                h * self.dynamic_zone_max_size_ratio,
                (h / max(1, self.grid_rows)) + (2.0 * self.dynamic_zone_margin_px),
            )
        )
        zone_w = max(
            int(self.dynamic_zone_min_size_px),
            int(round(base_w * float(np.clip(self.dynamic_zone_box_scale, 0.45, 1.0)))),
        )
        zone_h = max(
            int(self.dynamic_zone_min_size_px),
            int(round(base_h * float(np.clip(self.dynamic_zone_box_scale, 0.45, 1.0)))),
        )
        lock_w = max(
            int(self.dynamic_zone_min_size_px),
            int(
                round(
                    float(zone_w)
                    * float(np.clip(self.dynamic_lock_roi_scale, 0.45, 1.0))
                )
            ),
        )
        lock_h = max(
            int(self.dynamic_zone_min_size_px),
            int(
                round(
                    float(zone_h)
                    * float(np.clip(self.dynamic_lock_roi_scale, 0.45, 1.0))
                )
            ),
        )

        specs = []
        for zone_id, row, col, fixed_rect in fixed_specs:
            lock_center = (
                self.dynamic_locked_centers.get(int(zone_id))
                if self.dynamic_lock_then_track_enabled
                else None
            )
            if lock_center is not None:
                cx, cy = lock_center
                x0 = int(round(cx - (lock_w * 0.5)))
                y0 = int(round(cy - (lock_h * 0.5)))
                x0 = int(np.clip(x0, 0, max(0, w - lock_w)))
                y0 = int(np.clip(y0, 0, max(0, h - lock_h)))
                tw = int(min(lock_w, w - x0))
                th = int(min(lock_h, h - y0))
                specs.append((zone_id, row, col, (x0, y0, tw, th)))
                continue

            center = self.dynamic_zone_centers.get(int(zone_id))
            # No tracked nozzle for this zone yet -> keep fixed search area for detection bootstrap.
            if center is None:
                fx, fy, fw, fh = fixed_rect
                bs = float(np.clip(self.dynamic_zone_bootstrap_search_scale, 0.45, 1.0))
                bsw = max(
                    int(self.dynamic_zone_min_size_px), int(round(float(fw) * bs))
                )
                bsh = max(
                    int(self.dynamic_zone_min_size_px), int(round(float(fh) * bs))
                )
                cx = float(fx) + (0.5 * float(fw))
                cy = float(fy) + (0.5 * float(fh))
                bx = int(round(cx - (0.5 * float(bsw))))
                by = int(round(cy - (0.5 * float(bsh))))
                bx = int(np.clip(bx, 0, max(0, w - bsw)))
                by = int(np.clip(by, 0, max(0, h - bsh)))
                specs.append(
                    (
                        zone_id,
                        row,
                        col,
                        (bx, by, int(min(bsw, w - bx)), int(min(bsh, h - by))),
                    )
                )
                continue
            cx, cy = center
            x0 = int(round(cx - (zone_w * 0.5)))
            y0 = int(round(cy - (zone_h * 0.5)))
            x0 = int(np.clip(x0, 0, max(0, w - zone_w)))
            y0 = int(np.clip(y0, 0, max(0, h - zone_h)))
            tw = int(min(zone_w, w - x0))
            th = int(min(zone_h, h - y0))
            specs.append((zone_id, row, col, (x0, y0, tw, th)))
        return specs

    def _update_dynamic_zone_centers_from_results(self, zone_results, frame_shape):
        """Advance each dynamic zone center from detections, handling lock-and-track, miss-streak holds, and anti-overlap spacing."""
        if not self.dynamic_zoning_enabled:
            return
        default_centers = self._default_grid_centers(frame_shape)
        updated = dict(self.dynamic_zone_centers)
        for zone in zone_results:
            zone_id = int(zone["zone_id"])
            pos = zone.get("position")
            zone_confirmed = bool(zone.get("detected", False))
            zone_raw = bool(zone.get("raw_detected", False))
            zone_motion_ok = bool(zone.get("motion_ok", False))
            owner_zone = zone.get("owned_by_zone")
            zone_owns_itself = owner_zone in (None, zone_id)
            miss_before = int(self.dynamic_zone_miss_streak.get(zone_id, 0))
            if self.dynamic_zone_use_raw_for_reacquire_only:
                use_raw = zone_raw and zone_motion_ok and (miss_before > 0)
            else:
                use_raw = zone_raw and zone_motion_ok
            reliable_detect = zone_confirmed or use_raw

            if self.dynamic_lock_then_track_enabled:
                lock_center = self.dynamic_locked_centers.get(zone_id)
                if lock_center is not None:
                    if reliable_detect and zone_owns_itself and pos is not None:
                        self.dynamic_zone_miss_streak[zone_id] = 0
                        self.dynamic_lock_miss_streak[zone_id] = 0
                        if zone_confirmed:
                            a = float(np.clip(self.dynamic_lock_update_alpha, 0.0, 1.0))
                            lock_center = (
                                ((1.0 - a) * float(lock_center[0]))
                                + (a * float(pos[0])),
                                ((1.0 - a) * float(lock_center[1]))
                                + (a * float(pos[1])),
                            )
                        updated[zone_id] = lock_center
                        self.dynamic_locked_centers[zone_id] = lock_center
                    else:
                        self.dynamic_zone_miss_streak[zone_id] = (
                            int(self.dynamic_zone_miss_streak.get(zone_id, 0)) + 1
                        )
                        self.dynamic_lock_miss_streak[zone_id] = (
                            int(self.dynamic_lock_miss_streak.get(zone_id, 0)) + 1
                        )
                        if int(self.dynamic_lock_miss_streak[zone_id]) <= max(
                            1, int(self.dynamic_lock_hold_frames)
                        ):
                            updated[zone_id] = lock_center
                        else:
                            self.dynamic_locked_centers[zone_id] = None
                            self.dynamic_lock_miss_streak[zone_id] = 0
                            updated[zone_id] = None
                            self.dynamic_zone_kalman_filters[zone_id] = None
                    continue

            if reliable_detect and zone_owns_itself and pos is not None:
                self.dynamic_zone_miss_streak[zone_id] = 0
                prev = updated.get(zone_id)
                interval = max(1, int(self.dynamic_zone_update_interval_frames))
                should_update_now = (int(self.frame_index) % interval) == 0
                if use_raw:
                    should_update_now = True
                if prev is not None and not should_update_now:
                    updated[zone_id] = prev
                else:
                    updated[zone_id] = self._update_center_track(
                        zone_id,
                        (float(pos[0]), float(pos[1])),
                        default_centers[zone_id - 1],
                    )
                if (
                    self.dynamic_lock_then_track_enabled
                    and zone_confirmed
                    and int(zone.get("votes", 0))
                    >= max(1, int(self.dynamic_lock_min_votes))
                ):
                    self.dynamic_locked_centers[zone_id] = updated[zone_id]
                    self.dynamic_lock_miss_streak[zone_id] = 0
            else:
                self.dynamic_zone_miss_streak[zone_id] = (
                    int(self.dynamic_zone_miss_streak.get(zone_id, 0)) + 1
                )
                miss = int(self.dynamic_zone_miss_streak[zone_id])
                prev = updated.get(zone_id)
                if prev is not None and miss <= max(
                    1, int(self.dynamic_zone_lost_hold_frames)
                ):
                    # Freeze center while briefly missing to avoid time-drift wobble.
                    updated[zone_id] = prev
                else:
                    updated[zone_id] = None
                    self.dynamic_zone_kalman_filters[zone_id] = None

        if self.dynamic_zone_anti_overlap_enabled:
            if self.dynamic_lock_then_track_enabled:
                self.dynamic_zone_centers = updated
                return
            h, w = frame_shape[:2]
            tmp_w = float(
                max(
                    1,
                    int(
                        min(
                            w * self.dynamic_zone_max_size_ratio,
                            (w / max(1, self.grid_cols))
                            + (2.0 * self.dynamic_zone_margin_px),
                        )
                    ),
                )
            )
            tmp_h = float(
                max(
                    1,
                    int(
                        min(
                            h * self.dynamic_zone_max_size_ratio,
                            (h / max(1, self.grid_rows))
                            + (2.0 * self.dynamic_zone_margin_px),
                        )
                    ),
                )
            )
            min_spacing = float(min(tmp_w, tmp_h)) * float(
                np.clip(self.dynamic_zone_min_center_spacing_ratio, 0.25, 1.2)
            )
            updated = self._separate_centers(updated, min_spacing, w, h)

        self.dynamic_zone_centers = updated

    def _build_fixed_grid_specs(
        self, frame
    ) -> List[Tuple[int, int, int, Tuple[int, int, int, int]]]:
        """Tile the frame into a grid_rows x grid_cols layout, returning (zone_id, row, col, rect) for each fixed cell."""
        h, w = frame.shape[:2]
        tile_w = max(1, w // self.grid_cols)
        tile_h = max(1, h // self.grid_rows)
        specs = []
        zone_index = 1
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                x = col * tile_w
                y = row * tile_h
                tw = (w - x) if col == self.grid_cols - 1 else tile_w
                th = (h - y) if row == self.grid_rows - 1 else tile_h
                specs.append((zone_index, row, col, (int(x), int(y), int(tw), int(th))))
                zone_index += 1
        return specs

    @staticmethod
    def _calculate_contour_orientation(contour):
        """Estimate a contour's orientation angle in degrees via fitted ellipse, falling back to image moments."""
        try:
            if contour is not None and len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                return float(ellipse[2])
        except cv2.error:
            pass
        if contour is None:
            return 0.0
        moments = cv2.moments(contour)
        if moments.get("mu20", 0.0) != moments.get("mu02", 0.0):
            angle = 0.5 * np.arctan2(
                2.0 * float(moments.get("mu11", 0.0)),
                float(moments.get("mu20", 0.0)) - float(moments.get("mu02", 0.0)),
            )
            return float(np.degrees(angle))
        return 0.0

    def _detect_nozzle_rescue(self, tile, zone_id):
        """Relaxed CLAHE+Otsu contour fallback within a tile, scoring shape/area/proximity to recover a missed nozzle."""
        if tile is None or tile.size == 0:
            return None
        th, tw = tile.shape[:2]
        if th < 6 or tw < 6:
            return None

        gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        otsu_thr, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        relaxed_thr = float(
            np.clip(float(otsu_thr) - self.rescue_binary_bias, 0.0, 255.0)
        )
        _, bw = cv2.threshold(blur, relaxed_thr, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        tile_area = float(max(1, tw * th))
        min_area = self.rescue_min_area_ratio * tile_area
        max_area = self.rescue_max_area_ratio * tile_area
        last_pos = self.zone_last_confirmed_position.get(int(zone_id))

        best = None
        best_score = -1e9
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < min_area or area > max_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w <= 1 or h <= 1:
                continue
            cx = int(x + (w // 2))
            cy = int(y + (h // 2))
            aspect_ratio = float(w) / float(max(1, h))
            if (
                aspect_ratio < self.rescue_min_aspect_ratio
                or aspect_ratio > self.rescue_max_aspect_ratio
            ):
                continue
            fill_ratio = area / float(max(1, w * h))
            if (
                fill_ratio < self.rescue_min_extent
                or fill_ratio > self.rescue_max_extent
            ):
                continue
            perimeter = float(cv2.arcLength(cnt, True))
            if perimeter <= 1e-6:
                continue
            circularity = float((4.0 * np.pi * area) / (perimeter * perimeter))
            if circularity < self.rescue_min_circularity:
                continue
            border_margin = int(round(0.03 * float(min(tw, th))))
            touches_border = (
                x <= border_margin
                or y <= border_margin
                or (x + w) >= (tw - border_margin)
                or (y + h) >= (th - border_margin)
            )
            if touches_border:
                continue

            area_ratio = area / tile_area
            score = (1.8 * area_ratio) + (1.2 * fill_ratio) + (1.0 * circularity)
            if last_pos is not None:
                dist = float(
                    np.hypot(
                        float(cx) - float(last_pos[0]), float(cy) - float(last_pos[1])
                    )
                )
                if dist > self.rescue_max_tip_jump_px:
                    continue
                score -= 0.006 * dist
            else:
                if cy > (self.rescue_max_y_ratio_no_history * float(th)):
                    continue
                if self.rescue_center_only:
                    zone_col = (int(zone_id) - 1) % self.grid_cols
                    if zone_col == (self.grid_cols // 2):
                        if abs(float(cx) - (0.5 * float(tw))) > (
                            self.rescue_center_x_tolerance_ratio * float(tw)
                        ):
                            continue

            if score > best_score:
                best_score = score
                best = (cnt, x, y, w, h, cx, cy)

        if best is None:
            return None
        cnt, x, y, w, h, cx, cy = best
        orientation = self._calculate_contour_orientation(cnt)
        radius = int(max(3, round(0.5 * max(w, h))))
        return {
            "detected": True,
            "position": (int(cx), int(cy)),
            "orientation": float(orientation),
            "radius": radius,
            "contour": cnt,
            "bounding_box": (int(x), int(y), int(w), int(h)),
            "detection_method": "rescue_contour",
        }

    def _estimate_local_tip_position(
        self, tile_info, zone_id, zone_origin, frame_shape
    ):
        """Choose the inward/outward contour extreme as the tip in tile-local coords, preferring continuity with the last confirmed position."""
        fallback = tile_info.get("position")
        if fallback is None:
            return None
        contour = tile_info.get("contour")
        if contour is None:
            return fallback
        try:
            pts = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
        except Exception:
            return fallback
        if pts.shape[0] < 4:
            return fallback

        cx = float(fallback[0])
        cy = float(fallback[1])
        gx = float(zone_origin[0]) + cx
        gy = float(zone_origin[1]) + cy
        frame_center = (0.5 * float(frame_shape[1]), 0.5 * float(frame_shape[0]))
        center_unit = self._compute_center_unit((gx, gy), frame_center)
        if center_unit is None:
            return fallback

        ux, uy = float(center_unit[0]), float(center_unit[1])
        rel_x = pts[:, 0] - cx
        rel_y = pts[:, 1] - cy
        proj = (rel_x * ux) + (rel_y * uy)
        if proj.size == 0:
            return fallback

        inward_idx = int(np.argmin(proj))
        outward_idx = int(np.argmax(proj))
        inward_tip = (float(pts[inward_idx, 0]), float(pts[inward_idx, 1]))
        outward_tip = (float(pts[outward_idx, 0]), float(pts[outward_idx, 1]))

        prev = self.zone_last_confirmed_position.get(int(zone_id))
        chosen = inward_tip
        if prev is not None:
            d_in = float(
                np.hypot(inward_tip[0] - float(prev[0]), inward_tip[1] - float(prev[1]))
            )
            d_out = float(
                np.hypot(
                    outward_tip[0] - float(prev[0]), outward_tip[1] - float(prev[1])
                )
            )
            chosen = inward_tip if d_in <= d_out else outward_tip
        return (int(round(chosen[0])), int(round(chosen[1])))

    def _is_trusted_enter_candidate(self, tile_info: Dict) -> bool:
        """Return whether a detection is trustworthy enough to vote a zone IN: real (non-predicted) detection via an allowed base method."""
        info = tile_info or {}
        if not bool(info.get("detected", False)):
            return False
        method_txt = str(info.get("detection_method") or "").strip().lower()
        if not method_txt:
            return False
        base_method = method_txt.split("+")[0].strip()
        if base_method not in self.confirm_enter_allowed_base_methods:
            return False
        if bool(info.get("is_prediction", False)) or (base_method == "kalman_predict"):
            return False
        if self.confirm_enter_require_kalman and ("+kalman" not in method_txt):
            return False
        return True

    def _update_zone_state(self, zone_id, candidate_ok, tile_info):
        """Push a vote into the zone's confirmation window and flip its IN/OUT state via enter/exit hysteresis; returns state, votes, window len."""
        current_state = bool(self.zone_confirmed_state[zone_id])
        vote_ok = bool(candidate_ok)
        if vote_ok and self.confirm_enter_trusted_only and (not current_state):
            vote_ok = self._is_trusted_enter_candidate(tile_info)

        history = self.zone_confirmation_history[zone_id]
        history.append(bool(vote_ok))
        votes = int(sum(history))
        history_len = len(history)
        rejections = history_len - votes

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

    def _build_body_polygon_for_length(
        self, nozzle_pos, frame_center, body_length_px, direction=None
    ):
        """Build a rotated rectangle (box points) spanning the nozzle body from its tip along the given/outward direction for body_length_px."""
        if body_length_px <= 1.0:
            return None
        if direction is None:
            dx = float(nozzle_pos[0] - frame_center[0])
            dy = float(nozzle_pos[1] - frame_center[1])
            norm = float(np.hypot(dx, dy))
            if norm < 1e-6:
                return None
            ux = dx / norm
            uy = dy / norm
        else:
            ux = float(direction[0])
            uy = float(direction[1])
            norm = float(np.hypot(ux, uy))
            if norm < 1e-6:
                return None
            ux /= norm
            uy /= norm

        start_x = float(nozzle_pos[0]) + (ux * self.body_check_start_offset_px)
        start_y = float(nozzle_pos[1]) + (uy * self.body_check_start_offset_px)
        center_x = start_x + (ux * body_length_px * 0.5)
        center_y = start_y + (uy * body_length_px * 0.5)
        angle_deg = float(np.degrees(np.arctan2(uy, ux)))
        rect = (
            (center_x, center_y),
            (body_length_px, self.body_check_half_width_px * 2.0),
            angle_deg,
        )
        return cv2.boxPoints(rect).astype(np.int32)

    def _estimate_body_direction(self, zone, frame_center):
        """Estimate the nozzle body's outward unit direction from contour line-fit/orientation, sign-aligned and blended toward the radial vector."""
        pos = zone.get("position")
        if pos is None:
            return None
        center_unit = self._compute_center_unit(pos, frame_center)
        if center_unit is None:
            return None

        local_info = zone.get("local_nozzle_info", {})
        method = (
            str(local_info.get("detection_method") or zone.get("method") or "")
            .strip()
            .lower()
        )
        candidates = []
        contour = local_info.get("contour")
        if contour is not None:
            try:
                cnt = np.asarray(contour, dtype=np.float32).reshape(-1, 1, 2)
                if cnt.shape[0] >= 5:
                    line = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
                    candidates.append((float(line[0]), float(line[1])))
            except Exception:
                pass

        if method in {"contour", "rescue_contour", "center_object_contour"}:
            ang = zone.get("orientation")
            if ang is not None:
                theta = np.deg2rad(float(ang))
                candidates.append((float(np.cos(theta)), float(np.sin(theta))))

        chosen = None
        for cand in candidates:
            vx, vy = float(cand[0]), float(cand[1])
            vnorm = float(np.hypot(vx, vy))
            if vnorm <= 1e-6:
                continue
            vx /= vnorm
            vy /= vnorm
            if (vx * center_unit[0] + vy * center_unit[1]) < 0.0:
                vx = -vx
                vy = -vy
            chosen = (vx, vy)
            break
        if chosen is None:
            return center_unit

        blend = float(np.clip(self.body_direction_center_blend, 0.0, 1.0))
        bx = ((1.0 - blend) * chosen[0]) + (blend * center_unit[0])
        by = ((1.0 - blend) * chosen[1]) + (blend * center_unit[1])
        bnorm = float(np.hypot(bx, by))
        if bnorm <= 1e-6:
            return chosen
        return (bx / bnorm, by / bnorm)

    @staticmethod
    def _distance_to_zone_boundary_along_direction(position, direction, zone_rect):
        """Ray-cast from position along direction and return the distance to the nearest zone-rectangle edge intersection."""
        zx, zy, zw, zh = zone_rect
        px, py = float(position[0]), float(position[1])
        ux, uy = float(direction[0]), float(direction[1])
        candidates = []

        if abs(ux) > 1e-6:
            t = (float(zx) - px) / ux
            if t > 0:
                y_at_t = py + (uy * t)
                if float(zy) <= y_at_t <= float(zy + zh):
                    candidates.append(t)
            t = (float(zx + zw) - px) / ux
            if t > 0:
                y_at_t = py + (uy * t)
                if float(zy) <= y_at_t <= float(zy + zh):
                    candidates.append(t)

        if abs(uy) > 1e-6:
            t = (float(zy) - py) / uy
            if t > 0:
                x_at_t = px + (ux * t)
                if float(zx) <= x_at_t <= float(zx + zw):
                    candidates.append(t)
            t = (float(zy + zh) - py) / uy
            if t > 0:
                x_at_t = px + (ux * t)
                if float(zx) <= x_at_t <= float(zx + zw):
                    candidates.append(t)

        if not candidates:
            return 0.0
        return float(min(candidates))

    def _compute_body_length_to_zone_edge(self, zone, frame_center, direction=None):
        """Compute usable body length: distance from tip to zone edge minus start/edge offsets, capped at the configured max."""
        pos = zone.get("position")
        if pos is None:
            return 0.0
        if direction is None:
            unit = self._compute_center_unit(pos, frame_center)
        else:
            ux = float(direction[0])
            uy = float(direction[1])
            dnorm = float(np.hypot(ux, uy))
            unit = None if dnorm < 1e-6 else (ux / dnorm, uy / dnorm)
        if unit is None:
            return 0.0

        boundary_distance = self._distance_to_zone_boundary_along_direction(
            pos, unit, zone["rect"]
        )
        usable_length = (
            boundary_distance
            - self.body_check_start_offset_px
            - self.body_check_edge_margin_px
        )
        usable_length = max(0.0, usable_length)
        return float(min(self.body_check_length_px, usable_length))

    def _resolve_body_ownership(self, zone_results, frame_w, frame_h):
        """Reassign a detection's owning zone when its tip falls inside a longer neighbor's body polygon (overlap/short-body test)."""
        if not self.body_check_enabled:
            return
        frame_center = (frame_w * 0.5, frame_h * 0.5)
        body_info = {}

        for zone in zone_results:
            zone["body_polygon"] = None
            zone["body_direction"] = None
            if not zone.get("raw_detected", False):
                continue
            zone_id = int(zone["zone_id"])
            pos = zone.get("position")
            if pos is None:
                continue

            body_dir = self._estimate_body_direction(zone, frame_center)
            zone["body_direction"] = body_dir
            if body_dir is not None:
                ang_deg = float(
                    np.degrees(np.arctan2(float(body_dir[1]), float(body_dir[0])))
                )
                if ang_deg < 0.0:
                    ang_deg += 360.0
                zone["orientation"] = ang_deg
                local_info = zone.get("local_nozzle_info")
                if isinstance(local_info, dict):
                    local_info["orientation"] = ang_deg

            body_length_px = self._compute_body_length_to_zone_edge(
                zone, frame_center, direction=body_dir
            )
            zone["body_length_px"] = body_length_px
            polygon = self._build_body_polygon_for_length(
                pos, frame_center, body_length_px, direction=body_dir
            )
            if polygon is not None:
                zone["body_polygon"] = polygon
                body_info[zone_id] = {
                    "polygon": polygon,
                    "position": pos,
                    "length": body_length_px,
                }

        for zone in zone_results:
            if not zone.get("raw_detected", False):
                continue
            if bool(zone.get("_ownership_lock", False)):
                continue
            zone_id = int(zone["zone_id"])
            pos = zone.get("position")
            if pos is None:
                continue

            owner_zone = zone_id
            best_owner_length = body_info.get(zone_id, {}).get(
                "length", zone.get("body_length_px", 0.0)
            )
            for other_zone_id, other_info in body_info.items():
                if other_zone_id == zone_id:
                    continue
                other_polygon = other_info["polygon"]
                other_tip = other_info["position"]
                tip_distance = float(
                    np.hypot(pos[0] - other_tip[0], pos[1] - other_tip[1])
                )
                if tip_distance <= self.body_check_head_ignore_radius_px:
                    continue
                inside = cv2.pointPolygonTest(
                    other_polygon.astype(np.float32),
                    (float(pos[0]), float(pos[1])),
                    False,
                )
                if inside < 0:
                    continue

                own_polygon = body_info.get(zone_id, {}).get("polygon")
                overlap_area = 0.0
                if own_polygon is not None:
                    overlap_area, _ = cv2.intersectConvexConvex(
                        own_polygon.astype(np.float32),
                        other_polygon.astype(np.float32),
                    )
                    overlap_area = float(max(0.0, overlap_area))

                short_body = (
                    float(zone.get("body_length_px", 0.0))
                    < self.body_check_min_length_to_edge_px
                )
                has_overlap = overlap_area >= self.body_check_overlap_area_threshold
                if not has_overlap and not short_body:
                    continue
                other_length = float(other_info["length"])
                if other_length > best_owner_length:
                    owner_zone = int(other_zone_id)
                    best_owner_length = other_length

            zone["owned_by_zone"] = owner_zone

    def _resolve_center_ownership(self, zone_results, full_nozzles):
        """Group zones snapping to the same full-frame nozzle and award ownership to the nearest, locking the rest to that winner."""
        max_dist = max(1.0, float(self.ownership_center_max_dist_px))
        for zone in zone_results:
            zone["_ownership_lock"] = False
            if bool(zone.get("raw_detected", False)):
                zone["owned_by_zone"] = int(zone["zone_id"])

        if not full_nozzles:
            return

        candidates = []
        for zone in zone_results:
            if not bool(zone.get("raw_detected", False)):
                continue
            zpos = zone.get("position")
            if zpos is None:
                continue
            best_idx = None
            best_dist = float("inf")
            for idx, noz in enumerate(full_nozzles):
                npos = noz.get("position")
                if npos is None:
                    continue
                d = float(
                    np.hypot(
                        float(zpos[0]) - float(npos[0]), float(zpos[1]) - float(npos[1])
                    )
                )
                if d < best_dist:
                    best_dist = d
                    best_idx = idx
            if best_idx is None or best_dist > max_dist:
                continue
            candidates.append((best_idx, best_dist, zone))

        groups = {}
        for noz_idx, dist, zone in candidates:
            groups.setdefault(int(noz_idx), []).append((float(dist), zone))

        for _noz_idx, lst in groups.items():
            lst.sort(key=lambda x: x[0])
            winner_zone = lst[0][1]
            winner_id = int(winner_zone["zone_id"])
            winner_zone["owned_by_zone"] = winner_id
            winner_zone["_ownership_lock"] = True
            for _dist, zone in lst[1:]:
                zone["owned_by_zone"] = winner_id
                zone["_ownership_lock"] = True

    def _suppress_duplicate_zone_claims(self, zone_results):
        """Drop the worse-centered of any two zones detecting near-identical positions, clearing the loser's detection fields."""
        claim_dist = max(1.0, float(self.zone_duplicate_claim_distance_px))
        detections = []
        for zone in zone_results:
            if not zone.get("raw_detected", False):
                continue
            pos = zone.get("position")
            if pos is None:
                continue
            zx, zy, zw, zh = zone["rect"]
            zc = (float(zx) + (0.5 * float(zw)), float(zy) + (0.5 * float(zh)))
            d = float(np.hypot(float(pos[0]) - zc[0], float(pos[1]) - zc[1]))
            detections.append((zone, d))

        for i in range(len(detections)):
            zi, di = detections[i]
            pi = zi.get("position")
            if pi is None or not zi.get("raw_detected", False):
                continue
            for j in range(i + 1, len(detections)):
                zj, dj = detections[j]
                pj = zj.get("position")
                if pj is None or not zj.get("raw_detected", False):
                    continue
                dist = float(
                    np.hypot(float(pi[0]) - float(pj[0]), float(pi[1]) - float(pj[1]))
                )
                if dist > claim_dist:
                    continue
                if di <= dj:
                    loser = zj
                    loser_d = dj
                else:
                    loser = zi
                    loser_d = di
                loser["raw_detected"] = False
                loser["position"] = None
                loser["bounding_box"] = None
                loser["orientation"] = None
                loser["radius"] = None
                loser["motion_ok"] = False
                loser["method"] = "duplicate_suppressed"
                loser["local_nozzle_info"] = {
                    "detected": False,
                    "position": None,
                    "bounding_box": None,
                    "orientation": None,
                    "radius": None,
                    "contour": None,
                    "detection_method": "duplicate_suppressed",
                    "duplicate_distance": float(loser_d),
                }

    def detect_nozzles_by_grid(self, frame):
        """Run the full per-frame pipeline: build zone specs, detect/rescue per tile, resolve ownership, confirm states, and return zone results."""
        self.frame_index = int(self.frame_index) + 1
        zone_specs = self._build_zone_specs(frame)
        full_nozzles = self._full_frame_detect_nozzles(frame)
        zone_results = []
        for zone_index, row, col, rect in zone_specs:
            x, y, tw, th = rect
            tile = frame[y : y + th, x : x + tw]
            tile_info = self.zone_nozzle_detectors[zone_index].detect(tile)
            method = str(tile_info.get("detection_method") or "").strip().lower()
            is_prediction = bool(tile_info.get("is_prediction", False)) or (
                method == "kalman_predict"
            )
            ever_in = bool(self.zone_ever_confirmed.get(zone_index, False))
            allow_prediction_vote = is_prediction and ever_in
            main_detected = bool(tile_info.get("detected", False)) and (
                not is_prediction or allow_prediction_vote
            )

            if main_detected:
                self.zone_rescue_fail_streak[zone_index] = 0
                self.zone_rescue_pending_hits[zone_index] = 0
            else:
                self.zone_rescue_fail_streak[zone_index] = (
                    int(self.zone_rescue_fail_streak.get(zone_index, 0)) + 1
                )
                self.zone_rescue_pending_hits[zone_index] = int(
                    self.zone_rescue_pending_hits.get(zone_index, 0)
                )
                allow_rescue = self._is_rescue_zone(row, col) and int(
                    self.zone_rescue_fail_streak[zone_index]
                ) >= max(1, self.rescue_min_fail_streak)
                if allow_rescue:
                    rescued = self._detect_nozzle_rescue(tile, zone_index)
                    if rescued is not None and bool(rescued.get("detected", False)):
                        self.zone_rescue_pending_hits[zone_index] += 1
                        if int(self.zone_rescue_pending_hits[zone_index]) >= max(
                            1, self.rescue_confirm_frames
                        ):
                            tile_info = rescued
                            self.zone_rescue_fail_streak[zone_index] = 0
                    else:
                        self.zone_rescue_pending_hits[zone_index] = 0

            method = str(tile_info.get("detection_method") or "").strip().lower()
            is_prediction = bool(tile_info.get("is_prediction", False)) or (
                method == "kalman_predict"
            )
            ever_in = bool(self.zone_ever_confirmed.get(zone_index, False))
            allow_prediction_vote = is_prediction and ever_in
            raw_detected = bool(tile_info.get("detected", False)) and (
                not is_prediction or allow_prediction_vote
            )

            if self.dynamic_zoning_enabled and raw_detected:
                if not self._passes_dynamic_bootstrap_gate(tile_info, tw, th, ever_in):
                    raw_detected = False
                    tile_info = {
                        "detected": False,
                        "position": None,
                        "bounding_box": None,
                        "orientation": None,
                        "radius": None,
                        "contour": None,
                        "detection_method": "dynamic_bootstrap_gate_reject",
                    }

            local_pos = tile_info.get("position")
            if raw_detected and local_pos is not None:
                local_tip = self._estimate_local_tip_position(
                    tile_info, zone_index, (x, y), frame.shape
                )
                if local_tip is not None:
                    local_pos = local_tip
                    tile_info["position"] = local_pos

            # Reacquire gate for dynamic mode: avoid zone hopping to a far nozzle.
            if self.dynamic_zoning_enabled and raw_detected and local_pos is not None:
                prev_center = self.dynamic_zone_centers.get(int(zone_index))
                zone_was_confirmed = bool(
                    self.zone_confirmed_state.get(int(zone_index), False)
                )
                if prev_center is not None and zone_was_confirmed:
                    miss_streak = int(
                        self.dynamic_zone_miss_streak.get(int(zone_index), 0)
                    )
                    relax = 1.0 + (0.25 * float(min(4, miss_streak)))
                    gate_dist = float(self.dynamic_zone_reacquire_max_dist_px) * relax
                    gx = float(local_pos[0] + x)
                    gy = float(local_pos[1] + y)
                    jump = float(
                        np.hypot(gx - float(prev_center[0]), gy - float(prev_center[1]))
                    )
                    if jump > gate_dist:
                        raw_detected = False
                        tile_info = {
                            "detected": False,
                            "position": None,
                            "bounding_box": None,
                            "orientation": None,
                            "radius": None,
                            "contour": None,
                            "detection_method": "dynamic_reacquire_reject",
                        }
                        local_pos = None

            motion_ok = True
            if raw_detected and local_pos is not None:
                last_pos_global = self.zone_last_confirmed_global_position.get(
                    zone_index
                )
                if last_pos_global is not None:
                    gx = float(local_pos[0] + x)
                    gy = float(local_pos[1] + y)
                    jump = float(
                        np.hypot(
                            gx - float(last_pos_global[0]),
                            gy - float(last_pos_global[1]),
                        )
                    )
                    motion_ok = jump <= self.max_position_jump_px

            zone_result = {
                "zone_id": zone_index,
                "row": row,
                "col": col,
                "rect": (x, y, tw, th),
                "detected": False,
                "raw_detected": raw_detected,
                "status": "OUT",
                "position": None,
                "bounding_box": None,
                "orientation": None,
                "radius": None,
                "method": tile_info.get("detection_method"),
                "motion_ok": motion_ok,
                "owned_by_zone": None,
                "body_length_px": 0.0,
                "body_polygon": None,
                "body_direction": None,
                "votes": 0,
                "history_len": 0,
                "spray_info": dict(self.spray_angle_detector.empty_result()),
                "local_nozzle_info": tile_info,
            }

            if raw_detected:
                local_bbox = tile_info.get("bounding_box")
                if local_pos is not None:
                    zone_result["position"] = (
                        int(local_pos[0] + x),
                        int(local_pos[1] + y),
                    )
                    orient = tile_info.get("orientation")
                    try:
                        orient_val = None if orient is None else float(orient)
                    except Exception:
                        orient_val = None
                    if orient_val is None or abs(float(orient_val)) < 1e-3:
                        fh, fw = frame.shape[:2]
                        cx = 0.5 * float(fw)
                        cy = 0.5 * float(fh)
                        gx = float(local_pos[0] + x)
                        gy = float(local_pos[1] + y)
                        dx = gx - cx
                        dy = gy - cy
                        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                            fallback_ang = float(np.degrees(np.arctan2(dy, dx)))
                            tile_info["orientation"] = fallback_ang
                if local_bbox is not None:
                    bx, by, bw, bh = local_bbox
                    zone_result["bounding_box"] = (
                        int(bx + x),
                        int(by + y),
                        int(bw),
                        int(bh),
                    )
                zone_result["orientation"] = tile_info.get("orientation")
                zone_result["radius"] = tile_info.get("radius")

            zone_results.append(zone_result)

        self._apply_fullframe_rescue(zone_results, full_nozzles)
        if self.ownership_mode in {"center", "hybrid"}:
            self._resolve_center_ownership(zone_results, full_nozzles)
        if self.ownership_mode in {"body", "hybrid"}:
            self._resolve_body_ownership(zone_results, frame.shape[1], frame.shape[0])
        for zone in zone_results:
            if (
                bool(zone.get("raw_detected", False))
                and zone.get("owned_by_zone") is None
            ):
                zone["owned_by_zone"] = int(zone["zone_id"])
        self._suppress_duplicate_zone_claims(zone_results)

        for zone in zone_results:
            zone_id = int(zone["zone_id"])
            tile_info = zone["local_nozzle_info"]
            raw_detected = bool(zone.get("raw_detected", False))
            motion_ok = bool(zone.get("motion_ok", False))
            owner_zone = zone.get("owned_by_zone")
            is_owned_by_other = owner_zone is not None and int(owner_zone) != zone_id
            candidate_ok = raw_detected and motion_ok and not is_owned_by_other
            confirmed, votes, history_len = self._update_zone_state(
                zone_id, candidate_ok, tile_info
            )

            zone["detected"] = bool(confirmed)
            zone["votes"] = int(votes)
            zone["history_len"] = int(history_len)
            if bool(confirmed) and zone.get("position") is not None:
                self.zone_last_confirmed_global_position[zone_id] = zone.get("position")
            if is_owned_by_other:
                zone["status"] = f"OWNED_BY_{int(owner_zone)}"
            else:
                zone["status"] = (
                    "IN" if confirmed else ("CONFIRMING" if raw_detected else "OUT")
                )

        for zone in zone_results:
            zone_id = int(zone["zone_id"])
            local_info = zone.get("local_nozzle_info") or {}
            can_measure_spray = bool(zone.get("detected", False)) and zone.get(
                "owned_by_zone"
            ) in (None, zone_id)
            if can_measure_spray:
                strength = (
                    str(local_info.get("detection_strength", "none")).strip().lower()
                )
                cadence = 1
                if strength == "weak":
                    cadence = int(self.spray_cadence_weak_frames)
                elif strength != "strong":
                    cadence = int(self.spray_cadence_confirming_frames)
                cadence = max(1, cadence)
                self.spray_measure_counter[zone_id] = (
                    int(self.spray_measure_counter.get(zone_id, 0)) + 1
                )
                should_measure = (
                    cadence <= 1
                    or (self.spray_measure_counter[zone_id] % cadence) == 0
                    or not bool(
                        self.spray_cached_info.get(zone_id, {}).get("detected", False)
                    )
                )
                if should_measure:
                    measured = self.spray_angle_detector.estimate(frame, zone)
                    self.spray_cached_info[zone_id] = dict(measured)
                    zone["spray_info"] = measured
                else:
                    _cached = self.spray_cached_info.get(zone_id)
                    reused = dict(
                        _cached
                        if _cached is not None
                        else self.spray_angle_detector.empty_result()
                    )
                    reused["is_prediction"] = True
                    zone["spray_info"] = reused
            else:
                self.spray_angle_detector.reset_zone(zone_id)
                self.spray_measure_counter[zone_id] = 0
                self.spray_cached_info[zone_id] = dict(
                    self.spray_angle_detector.empty_result()
                )
                zone["spray_info"] = self.spray_angle_detector.empty_result()

        self._update_dynamic_zone_centers_from_results(zone_results, frame.shape)
        for zone in zone_results:
            if "_ownership_lock" in zone:
                del zone["_ownership_lock"]
        return zone_results
