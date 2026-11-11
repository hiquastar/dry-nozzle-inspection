"""
Spray angle detection module for estimating nozzle spray cone angle.
"""

from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np


class SprayAngleDetector:
    """
    Detect spray boundary lines and estimate spray angle per zone.
    Includes short-term Kalman prediction when measurements are missing.
    """

    def __init__(self, config: Dict, total_zones: int):
        """Load spray-angle config, then allocate per-zone angle history, Kalman, polar-energy EMA and overlay state."""
        self.enabled = bool(config.get("spray_angle_enabled", True))
        self.roi_width_ratio = float(config.get("spray_roi_width_ratio", 0.55))
        self.roi_length_ratio = float(config.get("spray_roi_length_ratio", 0.70))
        self.start_offset_px = float(config.get("spray_start_offset_px", 8.0))
        self.tip_ignore_radius_px = float(
            config.get("spray_tip_ignore_radius_px", 10.0)
        )
        self.blur_kernel = int(config.get("spray_blur_kernel", 5))
        self.canny_low = int(config.get("spray_canny_low", 50))
        self.canny_high = int(config.get("spray_canny_high", 150))
        self.auto_canny_enabled = bool(config.get("spray_auto_canny_enabled", True))
        self.auto_canny_sigma = float(config.get("spray_auto_canny_sigma", 0.33))
        self.clahe_enabled = bool(config.get("spray_clahe_enabled", True))
        self.clahe_clip_limit = float(config.get("spray_clahe_clip_limit", 2.0))
        self.clahe_tile_grid = int(config.get("spray_clahe_tile_grid", 8))
        self.hough_threshold = int(config.get("spray_hough_threshold", 20))
        self.hough_min_line_length = int(config.get("spray_hough_min_line_length", 12))
        self.hough_max_line_gap = int(config.get("spray_hough_max_line_gap", 10))
        self.line_near_tip_px = float(config.get("spray_line_near_tip_px", 45.0))
        self.line_min_span_px = float(config.get("spray_line_min_span_px", 16.0))
        self.min_half_angle_deg = float(config.get("spray_min_half_angle_deg", 3.0))
        self.max_half_angle_deg = float(config.get("spray_max_half_angle_deg", 65.0))
        self.smoothing_window = int(config.get("spray_smoothing_window", 5))
        self.temporal_filter_mode = (
            str(config.get("spray_temporal_filter_mode", "median")).strip().lower()
        )
        self.temporal_filter_window = int(
            config.get("spray_temporal_filter_window", 30)
        )
        self.overlay_stabilization_enabled = bool(
            config.get("spray_overlay_stabilization_enabled", True)
        )
        self.overlay_line_ema_alpha = float(
            config.get("spray_overlay_line_ema_alpha", 0.25)
        )

        self.kalman_enabled = bool(config.get("spray_kalman_enabled", True))
        self.kalman_predict_on_miss = bool(
            config.get("spray_kalman_predict_on_miss", True)
        )
        self.kalman_max_predict_frames = int(
            config.get("spray_kalman_max_predict_frames", 8)
        )
        self.kalman_process_noise = float(
            config.get("spray_kalman_process_noise", 0.06)
        )
        self.kalman_measurement_noise = float(
            config.get("spray_kalman_measurement_noise", 9.0)
        )
        self.kalman_error_cov = float(config.get("spray_kalman_error_cov", 1.0))
        self.outlier_gate_enabled = bool(config.get("spray_outlier_gate_enabled", True))
        self.outlier_max_jump_deg = float(
            config.get("spray_outlier_max_jump_deg", 18.0)
        )
        self.deadband_enabled = bool(config.get("spray_deadband_enabled", True))
        self.deadband_deg = float(config.get("spray_deadband_deg", 1.0))
        self.rate_limit_enabled = bool(config.get("spray_rate_limit_enabled", True))
        self.rate_limit_deg_per_frame = float(
            config.get("spray_rate_limit_deg_per_frame", 2.5)
        )
        self.confidence_weight_enabled = bool(
            config.get("spray_confidence_weight_enabled", True)
        )
        self.confidence_predict_threshold = float(
            config.get("spray_confidence_predict_threshold", 0.18)
        )
        self.confidence_full_threshold = float(
            config.get("spray_confidence_full_threshold", 0.60)
        )
        self.confidence_min_weight = float(
            config.get("spray_confidence_min_weight", 0.20)
        )
        self.weak_confidence_cap = float(config.get("spray_weak_confidence_cap", 0.45))
        self.polar_enabled = bool(config.get("spray_polar_enabled", True))
        self.polar_theta_bins = int(config.get("spray_polar_theta_bins", 181))
        self.polar_min_radius_px = float(config.get("spray_polar_min_radius_px", 10.0))
        self.polar_max_radius_ratio = float(
            config.get("spray_polar_max_radius_ratio", 0.95)
        )
        self.polar_peak_half_window_deg = float(
            config.get("spray_polar_peak_half_window_deg", 4.0)
        )
        self.polar_ema_alpha = float(config.get("spray_polar_ema_alpha", 0.30))
        self.polar_min_peak_energy = float(
            config.get("spray_polar_min_peak_energy", 80.0)
        )
        self.polar_min_side_points = int(config.get("spray_polar_min_side_points", 16))
        self.polar_confidence_min = float(
            config.get("spray_polar_confidence_min", 0.22)
        )
        self.polar_smooth_window = int(config.get("spray_polar_smooth_window", 7))
        self.spray_angle_mode = (
            str(config.get("spray_angle_mode", "peak")).strip().lower()
        )
        self.envelope_left_percentile = float(
            config.get("spray_envelope_left_percentile", 98.5)
        )
        self.envelope_right_percentile = float(
            config.get("spray_envelope_right_percentile", 1.5)
        )
        self.envelope_band_half_width_deg = float(
            config.get("spray_envelope_band_half_width_deg", 2.0)
        )
        self.envelope_min_side_points = int(
            config.get("spray_envelope_min_side_points", 20)
        )
        self.envelope_side_balance_ratio_max = float(
            config.get("spray_envelope_side_balance_ratio_max", 3.2)
        )
        self.envelope_outward_expand_deg = float(
            config.get("spray_envelope_outward_expand_deg", 1.2)
        )
        self.envelope_side_balance_hard_mult = float(
            config.get("spray_envelope_side_balance_hard_mult", 2.2)
        )
        self.direction_stabilization_enabled = bool(
            config.get("spray_direction_stabilization_enabled", True)
        )
        self.direction_ema_alpha = float(config.get("spray_direction_ema_alpha", 0.22))
        self.direction_ema_alpha_weak = float(
            config.get("spray_direction_ema_alpha_weak", 0.08)
        )
        self.direction_max_turn_deg = float(
            config.get("spray_direction_max_turn_deg", 8.0)
        )
        self.direction_weak_freeze_turn_deg = float(
            config.get("spray_direction_weak_freeze_turn_deg", 24.0)
        )

        self.angle_history: Dict[int, Deque[float]] = {
            zone_id: deque(maxlen=max(1, self.smoothing_window))
            for zone_id in range(1, int(total_zones) + 1)
        }
        self.kalman_state: Dict[int, Dict[str, Any]] = {
            zone_id: self._create_kalman_state()
            for zone_id in range(1, int(total_zones) + 1)
        }
        self.polar_energy_ema: Dict[int, Optional[np.ndarray]] = {
            zone_id: None for zone_id in range(1, int(total_zones) + 1)
        }
        self.temporal_angle_history: Dict[int, Deque[float]] = {
            zone_id: deque(maxlen=max(1, self.temporal_filter_window))
            for zone_id in range(1, int(total_zones) + 1)
        }
        self.overlay_line_state: Dict[int, Optional[List[np.ndarray]]] = {
            zone_id: None for zone_id in range(1, int(total_zones) + 1)
        }
        self.direction_state: Dict[int, Optional[Tuple[float, float]]] = {
            zone_id: None for zone_id in range(1, int(total_zones) + 1)
        }
        self.debug_axis_state: Dict[int, Optional[Tuple[float, float]]] = {
            zone_id: None for zone_id in range(1, int(total_zones) + 1)
        }

    @staticmethod
    def empty_result() -> Dict:
        """Return a blank spray-angle result dict (no detection, all angle/line/confidence fields nulled)."""
        return {
            "detected": False,
            "spray_angle_deg": None,
            "raw_spray_angle_deg": None,
            "spray_axis_deg": None,
            "spray_half_angle_deg": None,
            "left_angle_deg": None,
            "right_angle_deg": None,
            "is_prediction": False,
            "lines": [],
            "anchor_point": None,
            "roi_polygon": None,
            "confidence": None,
        }

    def reset_zone(self, zone_id: int):
        """Clear a zone's angle history, temporal buffer, overlay/direction/axis state and reinit its Kalman filter."""
        if zone_id in self.angle_history:
            self.angle_history[zone_id].clear()
        if zone_id in self.temporal_angle_history:
            self.temporal_angle_history[zone_id].clear()
        if zone_id in self.overlay_line_state:
            self.overlay_line_state[zone_id] = None
        if zone_id in self.direction_state:
            self.direction_state[zone_id] = None
        if zone_id in self.debug_axis_state:
            self.debug_axis_state[zone_id] = None
        self.kalman_state[zone_id] = self._create_kalman_state()
        if zone_id in self.polar_energy_ema:
            self.polar_energy_ema[zone_id] = None

    def _apply_temporal_filter(self, zone_id: int, angle_deg: float) -> float:
        """Push the angle into the zone's time window and return its rolling median (or mean) for temporal smoothing."""
        mode = self.temporal_filter_mode
        if mode in {"off", "none"} or self.temporal_filter_window <= 1:
            return float(angle_deg)

        history = self.temporal_angle_history[zone_id]
        history.append(float(angle_deg))
        values = np.asarray(history, dtype=np.float32)
        if values.size == 0:
            return float(angle_deg)
        if mode == "median":
            return float(np.median(values))
        return float(np.mean(values))

    def _stabilize_overlay_lines(self, zone_id: int, global_lines: list) -> list:
        """EMA-smooth the two overlay boundary lines across frames (with endpoint swap-matching) and round to pixels."""
        if not global_lines:
            return []

        current = []
        for p1, p2 in global_lines:
            current.append(
                np.array(
                    [float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1])],
                    dtype=np.float32,
                )
            )

        if not self.overlay_stabilization_enabled:
            self.overlay_line_state[zone_id] = [arr.copy() for arr in current]
            return [
                (
                    (int(round(arr[0])), int(round(arr[1]))),
                    (int(round(arr[2])), int(round(arr[3]))),
                )
                for arr in current
            ]

        prev = self.overlay_line_state.get(zone_id)
        alpha = float(np.clip(self.overlay_line_ema_alpha, 0.05, 1.0))
        if prev is None or len(prev) != len(current):
            smoothed = [arr.copy() for arr in current]
        else:
            if len(current) == 2:
                direct_cost = float(
                    np.sum(np.abs(prev[0] - current[0]))
                    + np.sum(np.abs(prev[1] - current[1]))
                )
                swap_cost = float(
                    np.sum(np.abs(prev[0] - current[1]))
                    + np.sum(np.abs(prev[1] - current[0]))
                )
                if swap_cost < direct_cost:
                    current = [current[1], current[0]]
            smoothed = []
            for prev_arr, cur_arr in zip(prev, current):
                smoothed.append(((1.0 - alpha) * prev_arr) + (alpha * cur_arr))

        self.overlay_line_state[zone_id] = [arr.copy() for arr in smoothed]
        return [
            (
                (int(round(arr[0])), int(round(arr[1]))),
                (int(round(arr[2])), int(round(arr[3]))),
            )
            for arr in smoothed
        ]

    def _create_kalman_state(self):
        """Build a fresh constant-velocity (angle, rate) Kalman filter and its per-zone init/miss-count tracking dict."""
        if not self.kalman_enabled:
            return {"kalman": None, "initialized": False, "miss_count": 0}

        kalman = cv2.KalmanFilter(2, 1)
        kalman.measurementMatrix = np.array([[1.0, 0.0]], dtype=np.float32)
        kalman.transitionMatrix = np.array(
            [[1.0, 1.0], [0.0, 1.0]],
            dtype=np.float32,
        )
        kalman.processNoiseCov = np.eye(2, dtype=np.float32) * np.float32(
            self.kalman_process_noise
        )
        kalman.measurementNoiseCov = np.array(
            [[self.kalman_measurement_noise]], dtype=np.float32
        )
        kalman.errorCovPost = np.eye(2, dtype=np.float32) * np.float32(
            self.kalman_error_cov
        )
        kalman.statePost = np.zeros((2, 1), dtype=np.float32)
        kalman.statePre = np.zeros((2, 1), dtype=np.float32)
        return {"kalman": kalman, "initialized": False, "miss_count": 0}

    @staticmethod
    def _normalize_vector(vec: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """Return the unit-length version of a 2D vector, or None if its magnitude is effectively zero."""
        vx = float(vec[0])
        vy = float(vec[1])
        norm = float(np.hypot(vx, vy))
        if norm <= 1e-6:
            return None
        return (vx / norm, vy / norm)

    @staticmethod
    def _signed_angle_deg(
        base_vec: Tuple[float, float], target_vec: Tuple[float, float]
    ) -> float:
        """Return the signed angle in degrees from base_vec to target_vec via atan2(cross, dot)."""
        bx, by = float(base_vec[0]), float(base_vec[1])
        tx, ty = float(target_vec[0]), float(target_vec[1])
        cross = (bx * ty) - (by * tx)
        dot = (bx * tx) + (by * ty)
        return float(np.degrees(np.arctan2(cross, dot)))

    @staticmethod
    def _rotate_vector(
        vec: Tuple[float, float], angle_deg: float
    ) -> Tuple[float, float]:
        """Rotate a 2D vector counter-clockwise by angle_deg and return the rotated components."""
        vx = float(vec[0])
        vy = float(vec[1])
        rad = float(np.radians(angle_deg))
        c = float(np.cos(rad))
        s = float(np.sin(rad))
        return ((vx * c) - (vy * s), (vx * s) + (vy * c))

    def _build_debug_rays(
        self,
        tip_local: Tuple[float, float],
        spray_dir: Tuple[float, float],
        spray_angle_deg: float,
        ray_len: float,
    ) -> Optional[list]:
        """Build the two debug boundary rays from the tip by rotating the spray axis by +/- half the cone angle."""
        axis = self._normalize_vector(spray_dir)
        if axis is None:
            return None
        half = float(
            np.clip(0.5 * float(spray_angle_deg), 0.0, float(self.max_half_angle_deg))
        )
        left_dir = self._normalize_vector(self._rotate_vector(axis, +half))
        right_dir = self._normalize_vector(self._rotate_vector(axis, -half))
        if left_dir is None or right_dir is None:
            return None
        tip_ix = int(round(float(tip_local[0])))
        tip_iy = int(round(float(tip_local[1])))
        ray_len = float(max(10.0, ray_len))
        return [
            (
                tip_ix,
                tip_iy,
                int(round(tip_ix + (left_dir[0] * ray_len))),
                int(round(tip_iy + (left_dir[1] * ray_len))),
            ),
            (
                tip_ix,
                tip_iy,
                int(round(tip_ix + (right_dir[0] * ray_len))),
                int(round(tip_iy + (right_dir[1] * ray_len))),
            ),
        ]

    def _axis_from_selected_lines(
        self,
        tip_local: Tuple[float, float],
        selected_lines: list,
        fallback_dir: Tuple[float, float],
    ) -> Tuple[float, float]:
        """Estimate the spray axis by averaging the selected boundary lines' tip-outward directions (flips kept forward)."""
        dirs = []
        tx = float(tip_local[0])
        ty = float(tip_local[1])
        fallback = self._normalize_vector(fallback_dir)
        for ln in selected_lines or []:
            if ln is None or len(ln) != 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in ln]
            d1 = float(np.hypot(x1 - tx, y1 - ty))
            d2 = float(np.hypot(x2 - tx, y2 - ty))
            if d2 >= d1:
                cand = self._normalize_vector((x2 - tx, y2 - ty))
            else:
                cand = self._normalize_vector((x1 - tx, y1 - ty))
            if cand is None:
                continue
            # Keep line direction consistent with spray forward direction to avoid 180-deg flips.
            if (
                fallback is not None
                and ((cand[0] * fallback[0]) + (cand[1] * fallback[1])) < 0.0
            ):
                cand = (-cand[0], -cand[1])
            dirs.append(cand)

        if len(dirs) == 1:
            one = dirs[0]
            return (
                one
                if one is not None
                else (fallback if fallback is not None else (1.0, 0.0))
            )
        if len(dirs) < 2:
            return fallback if fallback is not None else (1.0, 0.0)

        mix = self._normalize_vector((dirs[0][0] + dirs[1][0], dirs[0][1] + dirs[1][1]))
        if mix is None:
            return fallback if fallback is not None else dirs[0]
        if (
            fallback is not None
            and ((mix[0] * fallback[0]) + (mix[1] * fallback[1])) < 0.0
        ):
            mix = (-mix[0], -mix[1])
        return mix

    def _stabilize_debug_axis(
        self,
        zone_id: int,
        axis_raw: Tuple[float, float],
        weak_measurement: bool,
    ) -> Tuple[float, float]:
        """Smooth the debug axis per zone via turn-rate clamping plus EMA blend, tightened for weak measurements."""
        raw = self._normalize_vector(axis_raw)
        if raw is None:
            prev = self.debug_axis_state.get(zone_id)
            prev = self._normalize_vector(prev) if prev is not None else None
            return prev if prev is not None else (1.0, 0.0)

        prev = self.debug_axis_state.get(zone_id)
        prev = self._normalize_vector(prev) if prev is not None else None
        if prev is None:
            self.debug_axis_state[zone_id] = raw
            return raw

        turn = float(self._signed_angle_deg(prev, raw))
        # Keep debug axis stable: only slightly looser than direction state.
        max_turn = max(0.8, float(self.direction_max_turn_deg) * 1.1)
        if weak_measurement:
            max_turn *= 0.45
        if abs(turn) > max_turn:
            limited = self._normalize_vector(
                self._rotate_vector(prev, np.sign(turn) * max_turn)
            )
            if limited is None:
                limited = raw
        else:
            limited = raw

        alpha = float(self.overlay_line_ema_alpha)
        if weak_measurement:
            alpha = min(alpha, float(self.direction_ema_alpha_weak))
        alpha = float(np.clip(alpha, 0.03, 0.35))
        mixed = (
            ((1.0 - alpha) * prev[0]) + (alpha * limited[0]),
            ((1.0 - alpha) * prev[1]) + (alpha * limited[1]),
        )
        out = self._normalize_vector(mixed)
        if out is None:
            out = limited
        self.debug_axis_state[zone_id] = out
        return out

    def _stabilize_spray_direction(
        self,
        zone_id: int,
        spray_dir: Tuple[float, float],
        weak_measurement: bool,
    ) -> Tuple[float, float]:
        """Smooth the spray direction per zone via turn-rate clamping and EMA, freezing on large weak-measurement jumps."""
        raw = self._normalize_vector(spray_dir)
        if raw is None:
            return spray_dir
        if not self.direction_stabilization_enabled:
            self.direction_state[zone_id] = raw
            return raw

        prev = self.direction_state.get(zone_id)
        if prev is None:
            self.direction_state[zone_id] = raw
            return raw

        prev_norm = self._normalize_vector(prev)
        if prev_norm is None:
            self.direction_state[zone_id] = raw
            return raw

        turn = float(self._signed_angle_deg(prev_norm, raw))
        abs_turn = abs(turn)
        if weak_measurement and abs_turn > max(
            1.0, float(self.direction_weak_freeze_turn_deg)
        ):
            return prev_norm

        limited = raw
        max_turn = max(0.5, float(self.direction_max_turn_deg))
        if abs_turn > max_turn:
            limited = self._normalize_vector(
                self._rotate_vector(prev_norm, np.sign(turn) * max_turn)
            )
            if limited is None:
                limited = raw

        alpha = float(
            self.direction_ema_alpha_weak
            if weak_measurement
            else self.direction_ema_alpha
        )
        alpha = float(np.clip(alpha, 0.01, 1.0))
        blended = (
            ((1.0 - alpha) * prev_norm[0]) + (alpha * limited[0]),
            ((1.0 - alpha) * prev_norm[1]) + (alpha * limited[1]),
        )
        final_dir = self._normalize_vector(blended)
        if final_dir is None:
            final_dir = limited
        self.direction_state[zone_id] = final_dir
        return final_dir

    def _compute_spray_direction(
        self, zone: Dict, frame_shape
    ) -> Optional[Tuple[float, float]]:
        """Derive the forward spray direction from the zone's reversed body axis, falling back to a tip-to-frame-center vector."""
        body_dir = zone.get("body_direction")
        if body_dir is not None:
            spray_dir = self._normalize_vector(
                (-float(body_dir[0]), -float(body_dir[1]))
            )
            if spray_dir is not None:
                return spray_dir

        pos = zone.get("position")
        if pos is None:
            return None
        center = (0.5 * float(frame_shape[1]), 0.5 * float(frame_shape[0]))
        return self._normalize_vector(
            (center[0] - float(pos[0]), center[1] - float(pos[1]))
        )

    def _compute_canny_thresholds(
        self, gray: np.ndarray, mask: Optional[np.ndarray]
    ) -> Tuple[int, int]:
        """Pick Canny low/high thresholds, auto-deriving them from the ROI-masked gray median and sigma when enabled."""
        if not self.auto_canny_enabled:
            return int(self.canny_low), int(self.canny_high)

        median_val = None
        if mask is not None:
            roi_values = gray[mask > 0]
            if roi_values.size >= 32:
                median_val = float(np.median(roi_values))
        if median_val is None:
            median_val = float(np.median(gray))

        sigma = float(np.clip(self.auto_canny_sigma, 0.05, 0.95))
        low = int(max(0.0, (1.0 - sigma) * median_val))
        high = int(min(255.0, (1.0 + sigma) * median_val))
        if high <= low:
            high = min(255, low + 8)
        return low, high

    def _is_outlier_measurement(self, zone_id: int, raw_angle_deg: float) -> bool:
        """Flag the raw angle as an outlier if it jumps past the (miss-count-widened) gate from the Kalman/history reference."""
        if not self.outlier_gate_enabled:
            return False

        max_jump = float(self.outlier_max_jump_deg)
        if max_jump <= 0.0:
            return False

        reference = None
        dynamic_jump = max_jump
        if self.kalman_enabled:
            state = self.kalman_state.get(zone_id, {})
            kalman = state.get("kalman")
            initialized = bool(state.get("initialized", False))
            if kalman is not None and initialized:
                reference = float(kalman.statePost[0][0])
                miss_count = int(state.get("miss_count", 0))
                dynamic_jump += float(min(12, max(0, miss_count) * 2))

        if reference is None:
            history = self.angle_history.get(zone_id)
            if history and len(history) > 0:
                reference = float(history[-1])

        if reference is None:
            return False

        return abs(float(raw_angle_deg) - reference) > dynamic_jump

    def _get_reference_angle(self, zone_id: int) -> Optional[float]:
        """Return the zone's current reference angle from the Kalman state if initialized, else the last history value."""
        if self.kalman_enabled:
            state = self.kalman_state.get(zone_id, {})
            kalman = state.get("kalman")
            if kalman is not None and bool(state.get("initialized", False)):
                return float(kalman.statePost[0][0])
        history = self.angle_history.get(zone_id)
        if history and len(history) > 0:
            return float(history[-1])
        return None

    def _stabilize_measurement(
        self,
        zone_id: int,
        measured_angle: float,
        confidence: Optional[float],
        weak_measurement: bool,
    ) -> Tuple[bool, float]:
        """Blend the measured angle toward the reference by confidence weight, then apply rate-limit and deadband gating."""
        reference = self._get_reference_angle(zone_id)
        if reference is None:
            return False, float(measured_angle)

        stabilized = float(measured_angle)
        conf = 1.0 if confidence is None else float(np.clip(confidence, 0.0, 1.0))
        if weak_measurement:
            conf = min(conf, max(0.0, float(self.weak_confidence_cap)))

        if self.confidence_weight_enabled:
            predict_th = float(max(0.0, min(1.0, self.confidence_predict_threshold)))
            full_th = float(
                max(predict_th + 1e-6, min(1.0, self.confidence_full_threshold))
            )
            min_w = float(np.clip(self.confidence_min_weight, 0.0, 1.0))
            if conf < predict_th:
                return True, reference
            if conf >= full_th:
                weight = 1.0
            else:
                t = (conf - predict_th) / max(1e-6, (full_th - predict_th))
                weight = min_w + ((1.0 - min_w) * float(t))
            stabilized = reference + ((stabilized - reference) * weight)

        if self.rate_limit_enabled:
            max_step = max(0.05, float(self.rate_limit_deg_per_frame))
            delta = float(np.clip(stabilized - reference, -max_step, max_step))
            stabilized = reference + delta

        if self.deadband_enabled:
            deadband = max(0.0, float(self.deadband_deg))
            if abs(stabilized - reference) < deadband:
                stabilized = reference

        return False, float(stabilized)

    def _apply_angle_filter(self, zone_id: int, measured_angle: Optional[float]):
        """Feed the angle through the Kalman filter (predict on miss, correct on hit), or moving-average if Kalman is off."""
        if measured_angle is None:
            if not self.kalman_enabled:
                return None, False
            state = self.kalman_state[zone_id]
            kalman = state.get("kalman")
            if kalman is None or not state.get("initialized", False):
                return None, False
            if not self.kalman_predict_on_miss:
                return None, False
            if int(state.get("miss_count", 0)) >= max(
                0, self.kalman_max_predict_frames
            ):
                return None, False

            predicted = kalman.predict()
            state["miss_count"] = int(state.get("miss_count", 0)) + 1
            return float(predicted[0][0]), True

        measured = float(measured_angle)
        if self.kalman_enabled:
            state = self.kalman_state[zone_id]
            kalman = state.get("kalman")
            if kalman is None:
                return measured, False

            if not state.get("initialized", False):
                kalman.statePost = np.array([[measured], [0.0]], dtype=np.float32)
                kalman.statePre = kalman.statePost.copy()
                state["initialized"] = True
                state["miss_count"] = 0
                return measured, False

            kalman.predict()
            corrected = kalman.correct(
                np.array([[np.float32(measured)]], dtype=np.float32)
            )
            state["miss_count"] = 0
            return float(corrected[0][0]), False

        history = self.angle_history[zone_id]
        history.append(measured)
        smooth_angle = float(sum(history) / float(len(history)))
        return smooth_angle, False

    @staticmethod
    def _smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
        """Box-filter a 1D array with a forced-odd window via same-length convolution (used on polar energy bins)."""
        if values is None or values.size == 0:
            return values
        k = max(1, int(window))
        if k % 2 == 0:
            k += 1
        if k <= 1:
            return values
        kernel = np.ones((k,), dtype=np.float32) / float(k)
        return np.convolve(values.astype(np.float32), kernel, mode="same")

    def _estimate_with_hough(
        self,
        edges: np.ndarray,
        tip_local: Tuple[float, float],
        axis_x: float,
        axis_y: float,
        roi_length: float,
    ) -> Optional[Tuple[float, float, float, list, float]]:
        """Estimate the cone via HoughLinesP near-tip boundary lines, falling back to edge-pixel angle percentiles."""
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap,
        )
        if lines is None:
            return None

        left_candidates = []
        right_candidates = []
        for seg in lines:
            x1, y1, x2, y2 = seg[0]
            p1 = np.array([float(x1), float(y1)], dtype=np.float32)
            p2 = np.array([float(x2), float(y2)], dtype=np.float32)
            d1 = float(np.hypot(p1[0] - tip_local[0], p1[1] - tip_local[1]))
            d2 = float(np.hypot(p2[0] - tip_local[0], p2[1] - tip_local[1]))
            near_pt, far_pt = (p1, p2) if d1 <= d2 else (p2, p1)

            if min(d1, d2) > self.line_near_tip_px:
                continue

            seg_vec = (float(far_pt[0] - near_pt[0]), float(far_pt[1] - near_pt[1]))
            seg_dir = self._normalize_vector(seg_vec)
            if seg_dir is None:
                continue

            seg_len = float(np.hypot(seg_vec[0], seg_vec[1]))
            if seg_len < self.line_min_span_px:
                continue

            axis_dot = (seg_dir[0] * axis_x) + (seg_dir[1] * axis_y)
            if axis_dot <= 0.10:
                continue

            rel = self._signed_angle_deg((axis_x, axis_y), seg_dir)
            abs_rel = abs(rel)
            if abs_rel < self.min_half_angle_deg or abs_rel > self.max_half_angle_deg:
                continue

            candidate = {
                "rel": float(rel),
                "len": seg_len,
                "line_local": (int(x1), int(y1), int(x2), int(y2)),
            }
            if rel >= 0.0:
                left_candidates.append(candidate)
            else:
                right_candidates.append(candidate)

        selected_lines = []
        if left_candidates and right_candidates:
            left_best = max(left_candidates, key=lambda c: c["len"])
            right_best = max(right_candidates, key=lambda c: c["len"])
            left_angle = float(left_best["rel"])
            right_angle = float(right_best["rel"])
            raw_angle = float(abs(left_angle - right_angle))
            selected_lines = [left_best["line_local"], right_best["line_local"]]
            return left_angle, right_angle, raw_angle, selected_lines, 0.45

        ys, xs = np.where(edges > 0)
        if xs.size < 32:
            return None

        dx = xs.astype(np.float32) - np.float32(tip_local[0])
        dy = ys.astype(np.float32) - np.float32(tip_local[1])
        proj = (dx * np.float32(axis_x)) + (dy * np.float32(axis_y))
        perp = (-dx * np.float32(axis_y)) + (dy * np.float32(axis_x))
        rel = np.degrees(np.arctan2(perp, proj))

        valid = (
            (proj > max(4.0, 0.5 * self.start_offset_px))
            & (proj < (roi_length + 8.0))
            & (np.abs(rel) >= self.min_half_angle_deg)
            & (np.abs(rel) <= self.max_half_angle_deg)
        )
        if not np.any(valid):
            return None

        rel_valid = rel[valid]
        left_vals = rel_valid[rel_valid > 0.0]
        right_vals = rel_valid[rel_valid < 0.0]
        if left_vals.size < 12 or right_vals.size < 12:
            return None

        left_angle = float(np.percentile(left_vals, 78))
        right_angle = float(np.percentile(right_vals, 22))
        raw_angle = float(abs(left_angle - right_angle))

        ray_len = float(max(22.0, min(roi_length, 0.85 * roi_length)))
        tip_ix = int(round(tip_local[0]))
        tip_iy = int(round(tip_local[1]))

        def dir_from_angle(rel_deg):
            """Rotate the spray axis by rel_deg and return the resulting unit boundary direction (None if degenerate)."""
            rad = np.radians(rel_deg)
            dxr = (axis_x * np.cos(rad)) - (axis_y * np.sin(rad))
            dyr = (axis_x * np.sin(rad)) + (axis_y * np.cos(rad))
            norm = float(np.hypot(dxr, dyr))
            if norm <= 1e-6:
                return None
            return (dxr / norm, dyr / norm)

        ldir = dir_from_angle(left_angle)
        rdir = dir_from_angle(right_angle)
        if ldir is None or rdir is None:
            return None

        selected_lines = [
            (
                tip_ix,
                tip_iy,
                int(round(tip_ix + (ldir[0] * ray_len))),
                int(round(tip_iy + (ldir[1] * ray_len))),
            ),
            (
                tip_ix,
                tip_iy,
                int(round(tip_ix + (rdir[0] * ray_len))),
                int(round(tip_iy + (rdir[1] * ray_len))),
            ),
        ]
        return left_angle, right_angle, raw_angle, selected_lines, 0.30

    def _estimate_with_polar(
        self,
        gray: np.ndarray,
        edges: np.ndarray,
        roi_mask: np.ndarray,
        tip_local: Tuple[float, float],
        axis_x: float,
        axis_y: float,
        roi_length: float,
        zone_id: int,
    ) -> Optional[Tuple[float, float, float, list, float]]:
        """Estimate the cone by binning gradient energy over polar angle, picking EMA-smoothed peaks and line-fitting sides."""
        if not self.polar_enabled:
            return None

        bins = max(91, int(self.polar_theta_bins))
        if bins % 2 == 0:
            bins += 1
        theta_axis = np.linspace(
            -self.max_half_angle_deg, self.max_half_angle_deg, bins, dtype=np.float32
        )

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(gx, gy)

        ys, xs = np.where(roi_mask > 0)
        if xs.size < 64:
            return None

        dx = xs.astype(np.float32) - np.float32(tip_local[0])
        dy = ys.astype(np.float32) - np.float32(tip_local[1])
        proj = (dx * np.float32(axis_x)) + (dy * np.float32(axis_y))
        perp = (-dx * np.float32(axis_y)) + (dy * np.float32(axis_x))
        radius = np.hypot(dx, dy)
        rel = np.degrees(np.arctan2(perp, proj))

        min_r = max(
            3.0, float(self.polar_min_radius_px), float(self.tip_ignore_radius_px)
        )
        max_r = max(min_r + 6.0, float(roi_length) * float(self.polar_max_radius_ratio))
        valid = (
            (proj > max(3.0, 0.35 * float(self.start_offset_px)))
            & (radius >= min_r)
            & (radius <= max_r)
            & (np.abs(rel) >= self.min_half_angle_deg)
            & (np.abs(rel) <= self.max_half_angle_deg)
        )
        if not np.any(valid):
            return None

        xs_v = xs[valid]
        ys_v = ys[valid]
        rel_v = rel[valid]
        energy_v = grad_mag[ys_v, xs_v].astype(np.float32)
        if energy_v.size < 32:
            return None

        energy_bins, _ = np.histogram(
            rel_v,
            bins=bins,
            range=(-self.max_half_angle_deg, self.max_half_angle_deg),
            weights=energy_v,
        )
        energy_bins = energy_bins.astype(np.float32)
        energy_bins = self._smooth_1d(energy_bins, self.polar_smooth_window)

        prev = self.polar_energy_ema.get(zone_id)
        if prev is None or prev.shape[0] != bins:
            ema = energy_bins
        else:
            alpha = float(np.clip(self.polar_ema_alpha, 0.05, 1.0))
            ema = (alpha * energy_bins) + ((1.0 - alpha) * prev)
        self.polar_energy_ema[zone_id] = ema.copy()

        left_mask = theta_axis > 0.0
        right_mask = theta_axis < 0.0
        if not np.any(left_mask) or not np.any(right_mask):
            return None

        left_energies = ema[left_mask]
        right_energies = ema[right_mask]
        li = int(np.argmax(left_energies))
        ri = int(np.argmax(right_energies))
        left_angle = float(theta_axis[left_mask][li])
        right_angle = float(theta_axis[right_mask][ri])
        left_peak = float(left_energies[li])
        right_peak = float(right_energies[ri])

        if (
            left_peak < self.polar_min_peak_energy
            or right_peak < self.polar_min_peak_energy
        ):
            return None

        raw_angle = float(abs(left_angle - right_angle))
        if raw_angle < (2.0 * self.min_half_angle_deg) or raw_angle > (
            2.0 * self.max_half_angle_deg
        ):
            return None

        peak_half_window = max(1.0, float(self.polar_peak_half_window_deg))
        select_left = np.abs(rel_v - left_angle) <= peak_half_window
        select_right = np.abs(rel_v - right_angle) <= peak_half_window
        if int(np.count_nonzero(select_left)) < self.polar_min_side_points:
            return None
        if int(np.count_nonzero(select_right)) < self.polar_min_side_points:
            return None

        edge_strength = edges[ys_v, xs_v] > 0
        select_left = select_left & edge_strength
        select_right = select_right & edge_strength
        if int(np.count_nonzero(select_left)) < self.polar_min_side_points:
            return None
        if int(np.count_nonzero(select_right)) < self.polar_min_side_points:
            return None

        def fit_side(select_mask):
            """Fit a least-squares line to one side's selected edge points and return its forward-oriented relative angle."""
            px = xs_v[select_mask].astype(np.float32)
            py = ys_v[select_mask].astype(np.float32)
            pts = np.stack([px, py], axis=1).reshape(-1, 1, 2)
            vx, vy, _x0, _y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
            d = self._normalize_vector((float(vx[0]), float(vy[0])))
            if d is None:
                return None
            if (d[0] * axis_x + d[1] * axis_y) < 0.0:
                d = (-d[0], -d[1])
            rel_fit = self._signed_angle_deg((axis_x, axis_y), d)
            return rel_fit, d

        left_fit = fit_side(select_left)
        right_fit = fit_side(select_right)
        if left_fit is None or right_fit is None:
            return None

        left_angle = float(left_fit[0])
        right_angle = float(right_fit[0])
        raw_angle = float(abs(left_angle - right_angle))
        if raw_angle < (2.0 * self.min_half_angle_deg) or raw_angle > (
            2.0 * self.max_half_angle_deg
        ):
            return None

        ray_len = float(max(24.0, min(roi_length, 0.90 * roi_length)))
        tip_ix = int(round(tip_local[0]))
        tip_iy = int(round(tip_local[1]))
        left_dir = left_fit[1]
        right_dir = right_fit[1]
        selected_lines = [
            (
                tip_ix,
                tip_iy,
                int(round(tip_ix + (left_dir[0] * ray_len))),
                int(round(tip_iy + (left_dir[1] * ray_len))),
            ),
            (
                tip_ix,
                tip_iy,
                int(round(tip_ix + (right_dir[0] * ray_len))),
                int(round(tip_iy + (right_dir[1] * ray_len))),
            ),
        ]

        base_floor = float(np.percentile(ema, 55))
        left_conf = (left_peak - base_floor) / max(1e-6, left_peak)
        right_conf = (right_peak - base_floor) / max(1e-6, right_peak)
        sep_conf = np.clip(
            raw_angle / max(1.0, (2.0 * self.max_half_angle_deg)), 0.0, 1.0
        )
        confidence = float(
            np.clip(
                0.5 * (left_conf + right_conf) * (0.65 + (0.35 * sep_conf)), 0.0, 1.0
            )
        )
        if confidence < self.polar_confidence_min:
            return None

        return left_angle, right_angle, raw_angle, selected_lines, confidence

    def _estimate_with_polar_envelope(
        self,
        gray: np.ndarray,
        edges: np.ndarray,
        roi_mask: np.ndarray,
        tip_local: Tuple[float, float],
        axis_x: float,
        axis_y: float,
        roi_length: float,
    ) -> Optional[Tuple[float, float, float, list, float]]:
        """Estimate the cone from outer edge-angle percentiles per side (with outward bias and side-balance gating)."""
        ys, xs = np.where(roi_mask > 0)
        if xs.size < 64:
            return None

        dx = xs.astype(np.float32) - np.float32(tip_local[0])
        dy = ys.astype(np.float32) - np.float32(tip_local[1])
        proj = (dx * np.float32(axis_x)) + (dy * np.float32(axis_y))
        perp = (-dx * np.float32(axis_y)) + (dy * np.float32(axis_x))
        radius = np.hypot(dx, dy)
        rel = np.degrees(np.arctan2(perp, proj))

        min_r = max(
            3.0, float(self.polar_min_radius_px), float(self.tip_ignore_radius_px)
        )
        max_r = max(min_r + 6.0, float(roi_length) * float(self.polar_max_radius_ratio))
        valid = (
            (proj > max(3.0, 0.35 * float(self.start_offset_px)))
            & (radius >= min_r)
            & (radius <= max_r)
            & (np.abs(rel) >= self.min_half_angle_deg)
            & (np.abs(rel) <= self.max_half_angle_deg)
        )
        if not np.any(valid):
            return None

        xs_v = xs[valid]
        ys_v = ys[valid]
        rel_v = rel[valid]
        edge_strength = edges[ys_v, xs_v] > 0
        if int(np.count_nonzero(edge_strength)) < max(
            24, int(self.envelope_min_side_points)
        ):
            return None
        rel_edge = rel_v[edge_strength]
        if rel_edge.size < max(24, int(self.envelope_min_side_points)):
            return None

        left_vals = rel_edge[rel_edge > 0.0]
        right_vals = rel_edge[rel_edge < 0.0]
        min_side_points = max(8, int(self.envelope_min_side_points))
        if left_vals.size < min_side_points or right_vals.size < min_side_points:
            return None
        side_ratio = float(
            max(left_vals.size, right_vals.size)
            / max(1.0, float(min(left_vals.size, right_vals.size)))
        )
        ratio_soft = max(1.2, float(self.envelope_side_balance_ratio_max))
        ratio_hard = max(
            ratio_soft + 1e-6,
            ratio_soft * max(1.0, float(self.envelope_side_balance_hard_mult)),
        )
        if side_ratio > ratio_hard:
            return None

        left_pct = float(np.clip(self.envelope_left_percentile, 50.5, 99.95))
        right_pct = float(np.clip(self.envelope_right_percentile, 0.05, 49.5))
        if right_pct >= left_pct:
            right_pct = min(49.5, left_pct - 1.0)

        left_angle = float(np.percentile(left_vals, left_pct))
        right_angle = float(np.percentile(right_vals, right_pct))
        # Bias slightly outward so debug rays hug the outer visible spray boundary.
        expand = float(np.clip(self.envelope_outward_expand_deg, 0.0, 8.0))
        left_angle = float(
            np.clip(
                left_angle + expand, self.min_half_angle_deg, self.max_half_angle_deg
            )
        )
        right_angle = float(
            np.clip(
                right_angle - expand, -self.max_half_angle_deg, -self.min_half_angle_deg
            )
        )
        raw_angle = float(abs(left_angle - right_angle))
        if raw_angle < (2.0 * self.min_half_angle_deg) or raw_angle > (
            2.0 * self.max_half_angle_deg
        ):
            return None

        band_half = max(0.5, float(self.envelope_band_half_width_deg))
        left_mask = np.abs(rel_edge - left_angle) <= band_half
        right_mask = np.abs(rel_edge - right_angle) <= band_half
        if (
            int(np.count_nonzero(left_mask)) < min_side_points
            or int(np.count_nonzero(right_mask)) < min_side_points
        ):
            return None

        ray_len = float(max(24.0, min(roi_length, 0.95 * roi_length)))
        tip_ix = int(round(tip_local[0]))
        tip_iy = int(round(tip_local[1]))

        def dir_from_rel(rel_deg):
            """Rotate the spray axis by rel_deg and return the resulting unit boundary direction (None if degenerate)."""
            rad = np.radians(rel_deg)
            dxr = (axis_x * np.cos(rad)) - (axis_y * np.sin(rad))
            dyr = (axis_x * np.sin(rad)) + (axis_y * np.cos(rad))
            nrm = float(np.hypot(dxr, dyr))
            if nrm <= 1e-6:
                return None
            return (float(dxr / nrm), float(dyr / nrm))

        left_dir = dir_from_rel(left_angle)
        right_dir = dir_from_rel(right_angle)
        if left_dir is None or right_dir is None:
            return None

        selected_lines = [
            (
                tip_ix,
                tip_iy,
                int(round(tip_ix + (left_dir[0] * ray_len))),
                int(round(tip_iy + (left_dir[1] * ray_len))),
            ),
            (
                tip_ix,
                tip_iy,
                int(round(tip_ix + (right_dir[0] * ray_len))),
                int(round(tip_iy + (right_dir[1] * ray_len))),
            ),
        ]

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(gx, gy)
        energy = grad_mag[ys_v, xs_v].astype(np.float32)
        if energy.size == 0:
            return None
        energy_edge = energy[edge_strength]
        left_energy = (
            float(np.mean(energy_edge[left_mask]))
            if int(np.count_nonzero(left_mask)) > 0
            else 0.0
        )
        right_energy = (
            float(np.mean(energy_edge[right_mask]))
            if int(np.count_nonzero(right_mask)) > 0
            else 0.0
        )
        base = float(np.percentile(energy_edge, 50))
        left_conf = (left_energy - base) / max(1e-6, left_energy + 1e-6)
        right_conf = (right_energy - base) / max(1e-6, right_energy + 1e-6)
        sep_conf = np.clip(
            raw_angle / max(1.0, (2.0 * self.max_half_angle_deg)), 0.0, 1.0
        )
        density_conf = np.clip(
            (left_vals.size + right_vals.size) / max(60.0, float(rel_edge.size)),
            0.0,
            1.0,
        )
        balance_penalty = float(
            np.clip(ratio_soft / max(ratio_soft, side_ratio), 0.45, 1.0)
        )
        confidence = float(
            np.clip(
                (
                    (0.45 * (left_conf + right_conf))
                    + (0.35 * sep_conf)
                    + (0.20 * density_conf)
                )
                * balance_penalty,
                0.0,
                1.0,
            )
        )
        if confidence < float(self.polar_confidence_min):
            return None
        return left_angle, right_angle, raw_angle, selected_lines, confidence

    def estimate(self, frame, zone: Dict) -> Dict:
        """Estimate a zone's spray cone angle end to end: build ROI, run polar/Hough, then filter, stabilize and emit overlay."""
        result = self.empty_result()
        if not self.enabled:
            return result

        zone_id = int(zone["zone_id"])

        def predict_only():
            """Emit a Kalman-predicted spray result reusing the prior axis and overlay lines when no measurement is available."""
            predicted_angle, is_pred = self._apply_angle_filter(zone_id, None)
            if predicted_angle is None:
                return result
            final_angle = self._apply_temporal_filter(zone_id, float(predicted_angle))
            out = dict(result)
            out["detected"] = True
            out["spray_angle_deg"] = float(final_angle)
            out["is_prediction"] = bool(is_pred)
            dir_prev = self.debug_axis_state.get(zone_id)
            if dir_prev is None:
                dir_prev = self.direction_state.get(zone_id)
            dir_prev = (
                self._normalize_vector(dir_prev) if dir_prev is not None else None
            )
            if dir_prev is not None:
                out["spray_axis_deg"] = float(
                    np.degrees(np.arctan2(dir_prev[1], dir_prev[0]))
                )
                out["spray_half_angle_deg"] = float(max(0.0, 0.5 * float(final_angle)))
            prev_lines = self.overlay_line_state.get(zone_id)
            if prev_lines:
                out["lines"] = [
                    (
                        (int(round(arr[0])), int(round(arr[1]))),
                        (int(round(arr[2])), int(round(arr[3]))),
                    )
                    for arr in prev_lines
                ]
            return out

        rect = zone.get("rect")
        local_info = zone.get("local_nozzle_info") or {}
        local_tip = local_info.get("position")
        if rect is None or local_tip is None:
            return predict_only()

        x, y, w, h = rect
        if w < 8 or h < 8:
            return predict_only()
        tile = frame[y : y + h, x : x + w]
        if tile is None or tile.size == 0:
            return predict_only()

        weak_measurement = (
            str(local_info.get("detection_strength", "")).strip().lower() == "weak"
        )
        spray_dir = self._compute_spray_direction(zone, frame.shape)
        if spray_dir is None:
            return predict_only()
        spray_dir = self._stabilize_spray_direction(
            zone_id, spray_dir, weak_measurement
        )

        axis_x, axis_y = spray_dir
        tip_local = (float(local_tip[0]), float(local_tip[1]))
        roi_length = max(12.0, float(h) * self.roi_length_ratio)
        roi_width = max(10.0, float(w) * self.roi_width_ratio)

        start_x = tip_local[0] + (axis_x * self.start_offset_px)
        start_y = tip_local[1] + (axis_y * self.start_offset_px)
        center_x = start_x + (axis_x * roi_length * 0.5)
        center_y = start_y + (axis_y * roi_length * 0.5)
        roi_angle_deg = float(np.degrees(np.arctan2(axis_y, axis_x)))
        roi_rect = ((center_x, center_y), (roi_length, roi_width), roi_angle_deg)
        roi_poly = cv2.boxPoints(roi_rect).astype(np.int32)
        result["roi_polygon"] = roi_poly

        gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(roi_mask, roi_poly, 255)
        if self.tip_ignore_radius_px > 0:
            cv2.circle(
                roi_mask,
                (int(round(tip_local[0])), int(round(tip_local[1]))),
                int(round(self.tip_ignore_radius_px)),
                0,
                -1,
            )

        if self.clahe_enabled:
            tile_grid = max(2, int(self.clahe_tile_grid))
            clahe = cv2.createCLAHE(
                clipLimit=max(0.1, float(self.clahe_clip_limit)),
                tileGridSize=(tile_grid, tile_grid),
            )
            gray = clahe.apply(gray)

        blur_k = max(3, int(self.blur_kernel))
        if blur_k % 2 == 0:
            blur_k += 1
        gray = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
        canny_low, canny_high = self._compute_canny_thresholds(gray, roi_mask)
        edges = cv2.Canny(gray, canny_low, canny_high)
        edges = cv2.bitwise_and(edges, edges, mask=roi_mask)

        estimate = None
        if self.spray_angle_mode == "envelope_max":
            estimate = self._estimate_with_polar_envelope(
                gray=gray,
                edges=edges,
                roi_mask=roi_mask,
                tip_local=tip_local,
                axis_x=axis_x,
                axis_y=axis_y,
                roi_length=roi_length,
            )
        if estimate is None:
            estimate = self._estimate_with_polar(
                gray=gray,
                edges=edges,
                roi_mask=roi_mask,
                tip_local=tip_local,
                axis_x=axis_x,
                axis_y=axis_y,
                roi_length=roi_length,
                zone_id=zone_id,
            )
        if estimate is None:
            estimate = self._estimate_with_hough(
                edges=edges,
                tip_local=tip_local,
                axis_x=axis_x,
                axis_y=axis_y,
                roi_length=roi_length,
            )
        if estimate is None:
            return predict_only()
        left_angle, right_angle, raw_angle, selected_lines, confidence = estimate

        should_predict, stabilized_raw_angle = self._stabilize_measurement(
            zone_id=zone_id,
            measured_angle=raw_angle,
            confidence=confidence,
            weak_measurement=weak_measurement,
        )
        if should_predict:
            return predict_only()
        raw_angle = float(stabilized_raw_angle)

        if self._is_outlier_measurement(zone_id, raw_angle):
            return predict_only()

        filtered_angle, _ = self._apply_angle_filter(zone_id, raw_angle)
        if filtered_angle is None:
            return predict_only()
        final_angle = self._apply_temporal_filter(zone_id, float(filtered_angle))

        axis_for_debug = self._axis_from_selected_lines(
            tip_local=tip_local,
            selected_lines=selected_lines,
            fallback_dir=spray_dir,
        )
        axis_for_debug = self._stabilize_debug_axis(
            zone_id=zone_id,
            axis_raw=axis_for_debug,
            weak_measurement=weak_measurement,
        )
        debug_local_lines = self._build_debug_rays(
            tip_local=tip_local,
            spray_dir=axis_for_debug,
            spray_angle_deg=final_angle,
            ray_len=float(max(22.0, min(roi_length, 0.85 * roi_length))),
        )
        if not debug_local_lines:
            debug_local_lines = selected_lines

        global_lines = []
        for lx1, ly1, lx2, ly2 in debug_local_lines:
            global_lines.append(
                (
                    (int(lx1 + x), int(ly1 + y)),
                    (int(lx2 + x), int(ly2 + y)),
                )
            )
        global_lines = self._stabilize_overlay_lines(zone_id, global_lines)

        result["detected"] = True
        result["spray_angle_deg"] = float(final_angle)
        result["raw_spray_angle_deg"] = raw_angle
        result["spray_axis_deg"] = float(
            np.degrees(np.arctan2(float(axis_for_debug[1]), float(axis_for_debug[0])))
        )
        result["spray_half_angle_deg"] = float(max(0.0, 0.5 * float(final_angle)))
        result["left_angle_deg"] = left_angle
        result["right_angle_deg"] = right_angle
        result["is_prediction"] = False
        result["lines"] = global_lines
        result["anchor_point"] = (
            int(round(float(tip_local[0]) + x)),
            int(round(float(tip_local[1]) + y)),
        )
        result["confidence"] = float(confidence)
        return result
