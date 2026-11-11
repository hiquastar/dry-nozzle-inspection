"""
Buildup detection module for identifying material accumulation on nozzles.
"""

from collections import deque
from typing import Deque, Dict, Optional, Tuple

import cv2
import numpy as np

try:
    from skimage.feature import graycomatrix, graycoprops  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    graycomatrix = None
    graycoprops = None


class BuildupDetector:
    """
    Detects material buildup on nozzles using edge detection and texture analysis.
    """

    def __init__(self, config: Dict):
        """Initialize object."""
        self.edge_threshold_low = config.get("edge_threshold_low", 50)
        self.edge_threshold_high = config.get("edge_threshold_high", 150)
        self.kernel_size = config.get("kernel_size", 5)
        self.morph_iterations = config.get("morph_iterations", 2)
        self.buildup_area_threshold = config.get("buildup_area_threshold", 0.15)
        # Hard floor: when True, buildup is never reported YES below buildup_area_threshold,
        # even if texture (GLCM) / temporal / diff cues fire. Keeps "report at >= N%" intuitive.
        self.buildup_area_hard_floor = bool(config.get("buildup_area_hard_floor", True))
        self.buildup_require_strong_detection = bool(
            config.get("buildup_require_strong_detection", True)
        )
        self.buildup_weak_override_ratio = float(
            np.clip(config.get("buildup_weak_override_ratio", 0.30), 0.0, 1.0)
        )
        self.buildup_weak_threshold_scale = float(
            max(1.0, config.get("buildup_weak_threshold_scale", 1.8))
        )

        # Phase 1: detector state for future temporal tracking.
        self.temporal_enabled = bool(config.get("temporal_enabled", True))
        self.max_history = int(config.get("temporal_history_size", 90))
        self.ema_alpha = float(config.get("temporal_ema_alpha", 0.08))
        self.temporal_min_frames = int(config.get("temporal_min_frames", 20))
        self.temporal_diff_threshold = float(
            config.get("temporal_diff_threshold", 0.08)
        )
        self.anomaly_threshold = float(config.get("anomaly_threshold", 0.05))
        self.glcm_enabled = (
            bool(config.get("glcm_enabled", True))
            and graycomatrix is not None
            and graycoprops is not None
        )
        self.glcm_levels = int(config.get("glcm_levels", 32))
        self.glcm_contrast_ratio = float(config.get("glcm_contrast_ratio", 1.20))
        self.glcm_min_buildup_ratio = float(config.get("glcm_min_buildup_ratio", 0.01))
        self.temporal_min_buildup_ratio = float(
            config.get("temporal_min_buildup_ratio", 0.015)
        )

        # Phase 2: localized tip ROI controls.
        self.tip_roi_scale_x = float(config.get("tip_roi_scale_x", 0.65))
        self.tip_roi_scale_y = float(config.get("tip_roi_scale_y", 0.65))
        self.tip_roi_min_size_px = int(config.get("tip_roi_min_size_px", 20))
        self.tip_roi_radius_padding_scale = float(
            config.get("tip_roi_radius_padding_scale", 2.8)
        )
        self.tip_roi_offset_ratio = float(config.get("tip_roi_offset_ratio", 0.0))
        self.tip_roi_bottom_ignore_ratio = float(
            config.get("tip_roi_bottom_ignore_ratio", 0.30)
        )
        self.annulus_inner_ratio = float(config.get("annulus_inner_ratio", 0.14))
        self.annulus_outer_ratio = float(config.get("annulus_outer_ratio", 0.32))
        self.annulus_use_head_circle = bool(config.get("annulus_use_head_circle", True))
        self.annulus_head_outer_scale = float(
            config.get("annulus_head_outer_scale", 1.45)
        )
        self.annulus_head_inner_scale = float(
            config.get("annulus_head_inner_scale", 0.82)
        )
        self.annulus_head_outer_only = bool(config.get("annulus_head_outer_only", True))
        self.annulus_head_inner_offset_min_scale = float(
            np.clip(
                config.get("annulus_head_inner_offset_min_scale", -0.35), -0.95, 0.95
            )
        )
        self.annulus_head_inner_offset_max_scale = float(
            np.clip(
                config.get("annulus_head_inner_offset_max_scale", 0.50), -0.95, 1.40
            )
        )
        if (
            self.annulus_head_inner_offset_max_scale
            < self.annulus_head_inner_offset_min_scale
        ):
            self.annulus_head_inner_offset_max_scale = (
                self.annulus_head_inner_offset_min_scale
            )
        self.annulus_head_inner_offset_scale = float(
            config.get("annulus_head_inner_offset_scale", 0.04)
        )
        self.annulus_head_band_scale = float(
            config.get("annulus_head_band_scale", 0.62)
        )
        self.annulus_notch_enabled = bool(config.get("annulus_notch_enabled", True))
        self.annulus_notch_fraction = float(config.get("annulus_notch_fraction", 0.25))
        self.annulus_notch_fallback_angle_deg = float(
            config.get("annulus_notch_fallback_angle_deg", 90.0)
        )
        self.annulus_notch_reverse = bool(config.get("annulus_notch_reverse", False))
        self.annulus_notch_angle_ema_alpha = float(
            config.get("annulus_notch_angle_ema_alpha", 0.30)
        )
        self.annulus_notch_max_turn_deg = float(
            config.get("annulus_notch_max_turn_deg", 20.0)
        )
        self.calibration_mode = (
            str(config.get("calibration_mode", "manual")).strip().lower()
        )
        if self.calibration_mode not in {"manual", "auto"}:
            self.calibration_mode = "manual"
        self.auto_calibration_enabled = bool(
            config.get("auto_calibration_enabled", True)
        )
        self.auto_offset_min = float(config.get("auto_offset_min", 0.02))
        self.auto_offset_max = float(config.get("auto_offset_max", 0.24))
        self.auto_offset_step = float(config.get("auto_offset_step", 0.02))
        self.auto_band_min = float(config.get("auto_band_min", 0.30))
        self.auto_band_max = float(config.get("auto_band_max", 0.95))
        self.auto_band_step = float(config.get("auto_band_step", 0.05))
        self.auto_tune_ema_alpha = float(config.get("auto_tune_ema_alpha", 0.22))
        self.auto_tune_prior_weight = float(config.get("auto_tune_prior_weight", 0.18))
        self.auto_tune_edge_floor = float(config.get("auto_tune_edge_floor", 0.002))
        self.auto_tune_max_jump_offset = float(
            config.get("auto_tune_max_jump_offset", 0.06)
        )
        self.auto_tune_max_jump_band = float(
            config.get("auto_tune_max_jump_band", 0.12)
        )
        self.auto_tune_adaptive_enabled = bool(
            config.get("auto_tune_adaptive_enabled", True)
        )
        self.auto_tune_confidence_margin = float(
            config.get("auto_tune_confidence_margin", 0.0015)
        )
        self.auto_tune_confidence_scale = float(
            max(1e-6, config.get("auto_tune_confidence_scale", 0.010))
        )
        self.auto_tune_alpha_max = float(
            np.clip(config.get("auto_tune_alpha_max", 0.65), 0.05, 1.0)
        )
        self.auto_tune_jump_boost = float(
            np.clip(config.get("auto_tune_jump_boost", 2.2), 1.0, 6.0)
        )
        self.auto_tune_min_confidence = float(
            np.clip(config.get("auto_tune_min_confidence", 0.20), 0.0, 1.0)
        )
        self.auto_tune_hold_low_conf_frames = int(
            max(1, config.get("auto_tune_hold_low_conf_frames", 4))
        )
        self.auto_tune_history_window = int(
            max(3, config.get("auto_tune_history_window", 11))
        )
        self.auto_tune_contrast_weight = float(
            max(0.0, config.get("auto_tune_contrast_weight", 0.35))
        )
        self.auto_tune_temporal_weight = float(
            max(0.0, config.get("auto_tune_temporal_weight", 0.30))
        )
        self.auto_tune_flicker_penalty = float(
            max(0.0, config.get("auto_tune_flicker_penalty", 0.10))
        )
        self.auto_tune_coarse_refine_enabled = bool(
            config.get("auto_tune_coarse_refine_enabled", True)
        )
        self.auto_tune_coarse_stride_offset = int(
            max(1, config.get("auto_tune_coarse_stride_offset", 2))
        )
        self.auto_tune_coarse_stride_band = int(
            max(1, config.get("auto_tune_coarse_stride_band", 2))
        )
        self.analysis_mask_min_pixels = int(config.get("analysis_mask_min_pixels", 25))
        self.analysis_ring_dilate_px = int(config.get("analysis_ring_dilate_px", 7))
        # Legacy line-thickness control is intentionally deprecated; band ratio is used instead.
        self.spray_line_thickness_px = int(config.get("spray_line_thickness_px", 10))
        self.spray_exclusion_from_inner_circle = bool(
            config.get("spray_exclusion_from_inner_circle", True)
        )
        self.spray_exclusion_cover_analysis = bool(
            config.get("spray_exclusion_cover_analysis", True)
        )
        self.spray_exclusion_inner_erode_px = int(
            config.get("spray_exclusion_inner_erode_px", 3)
        )
        self.spray_exclusion_blur_ksize = int(
            config.get("spray_exclusion_blur_ksize", 5)
        )
        self.spray_exclusion_morph_iterations = int(
            config.get("spray_exclusion_morph_iterations", 2)
        )
        self.spray_exclusion_band_ratio_of_annulus = float(
            np.clip(
                config.get("spray_exclusion_band_ratio_of_annulus", 0.22), 0.01, 1.0
            )
        )
        self.spray_exclusion_min_band_px = int(
            max(1, config.get("spray_exclusion_min_band_px", 1))
        )
        self.spray_exclusion_max_band_px = int(
            max(1, config.get("spray_exclusion_max_band_px", 24))
        )
        # Analysis-zone stabilization (per-zone EMA + jump gating)
        self.analysis_stabilization_enabled = bool(
            config.get("analysis_stabilization_enabled", True)
        )
        self.analysis_center_ema_alpha = float(
            config.get("analysis_center_ema_alpha", 0.22)
        )
        self.analysis_radius_ema_alpha = float(
            config.get("analysis_radius_ema_alpha", 0.26)
        )
        self.analysis_max_center_jump_px = float(
            config.get("analysis_max_center_jump_px", 28.0)
        )
        self.analysis_max_radius_delta_ratio = float(
            config.get("analysis_max_radius_delta_ratio", 0.40)
        )
        self.analysis_reacquire_frames = int(config.get("analysis_reacquire_frames", 8))

        # Baseline lock-and-track (capture only when nozzle is stable and buildup is near zero).
        self.baseline_lock_enabled = bool(config.get("baseline_lock_enabled", True))
        self.baseline_lock_user_enabled = bool(
            config.get("baseline_lock_user_enabled", True)
        )
        self.baseline_lock_max_buildup_ratio = float(
            np.clip(config.get("baseline_lock_max_buildup_ratio", 0.003), 0.0, 0.05)
        )
        self.baseline_lock_required_stable_frames = int(
            max(1, config.get("baseline_lock_required_stable_frames", 35))
        )
        self.baseline_lock_max_center_jitter_px = float(
            max(0.0, config.get("baseline_lock_max_center_jitter_px", 2.0))
        )
        self.baseline_lock_max_radius_jitter_ratio = float(
            max(0.0, config.get("baseline_lock_max_radius_jitter_ratio", 0.06))
        )
        self.baseline_lock_allow_refresh = bool(
            config.get("baseline_lock_allow_refresh", False)
        )
        self.baseline_lock_require_real_detection = bool(
            config.get("baseline_lock_require_real_detection", True)
        )
        self.baseline_lock_require_kalman_smoothed = bool(
            config.get("baseline_lock_require_kalman_smoothed", True)
        )
        self.baseline_lock_require_strong_detection = bool(
            config.get("baseline_lock_require_strong_detection", True)
        )
        self.baseline_lock_allow_relaxed_methods = bool(
            config.get("baseline_lock_allow_relaxed_methods", False)
        )
        self.baseline_lock_allow_blob_method = bool(
            config.get("baseline_lock_allow_blob_method", False)
        )
        self.baseline_lock_allow_rescue_methods = bool(
            config.get("baseline_lock_allow_rescue_methods", False)
        )
        self.baseline_lock_radius_anchor_enabled = bool(
            config.get("baseline_lock_radius_anchor_enabled", True)
        )
        self.baseline_lock_radius_anchor_min_ratio = float(
            np.clip(
                config.get("baseline_lock_radius_anchor_min_ratio", 0.95), 0.50, 1.20
            )
        )
        self.baseline_lock_radius_anchor_max_ratio = float(
            np.clip(
                config.get("baseline_lock_radius_anchor_max_ratio", 1.30), 0.55, 2.50
            )
        )
        if (
            self.baseline_lock_radius_anchor_max_ratio
            < self.baseline_lock_radius_anchor_min_ratio
        ):
            self.baseline_lock_radius_anchor_max_ratio = (
                self.baseline_lock_radius_anchor_min_ratio
            )
        self.baseline_lock_radius_anchor_on_weak_only = bool(
            config.get("baseline_lock_radius_anchor_on_weak_only", True)
        )
        self.buildup_diff_enabled = bool(config.get("buildup_diff_enabled", False))
        self.buildup_diff_threshold = int(
            np.clip(config.get("buildup_diff_threshold", 18), 1, 255)
        )
        self.buildup_diff_blur_kernel = int(
            max(1, config.get("buildup_diff_blur_kernel", 3))
        )
        self.buildup_diff_min_ratio = float(
            np.clip(config.get("buildup_diff_min_ratio", 0.015), 0.0, 1.0)
        )
        self.buildup_diff_use_for_mask = bool(
            config.get("buildup_diff_use_for_mask", False)
        )
        self.buildup_diff_requires_strong_detection = bool(
            config.get("buildup_diff_requires_strong_detection", True)
        )
        self.buildup_diff_allow_circle_relaxed = bool(
            config.get("buildup_diff_allow_circle_relaxed", False)
        )
        self.buildup_diff_allow_circle_blob = bool(
            config.get("buildup_diff_allow_circle_blob", False)
        )
        self.buildup_diff_allow_kalman_predict = bool(
            config.get("buildup_diff_allow_kalman_predict", False)
        )
        self.buildup_diff_edge_overlap_required = bool(
            config.get("buildup_diff_edge_overlap_required", True)
        )
        self.buildup_diff_edge_overlap_kernel = int(
            max(1, config.get("buildup_diff_edge_overlap_kernel", 3))
        )
        self.buildup_diff_edge_overlap_min_ratio = float(
            np.clip(config.get("buildup_diff_edge_overlap_min_ratio", 0.35), 0.0, 1.0)
        )
        self.buildup_component_filter_enabled = bool(
            config.get("buildup_component_filter_enabled", True)
        )
        self.buildup_component_min_area_px = int(
            max(1, config.get("buildup_component_min_area_px", 12))
        )
        self.buildup_component_max_area_ratio = float(
            np.clip(config.get("buildup_component_max_area_ratio", 1.0), 0.05, 1.0)
        )
        self.buildup_component_inner_anchor_required = bool(
            config.get("buildup_component_inner_anchor_required", True)
        )
        self.buildup_component_inner_anchor_band_ratio = float(
            np.clip(
                config.get("buildup_component_inner_anchor_band_ratio", 0.30), 0.01, 1.0
            )
        )
        self.buildup_component_inner_anchor_min_overlap_ratio = float(
            np.clip(
                config.get("buildup_component_inner_anchor_min_overlap_ratio", 0.08),
                0.0,
                1.0,
            )
        )

        # Temporal state is stored per zone. Baseline compare is added later.
        self.baselines: Dict[int, Dict] = {}
        self.history: Dict[int, Deque[Dict[str, float]]] = {}
        self.ema_percentage: Dict[int, float] = {}
        self.ema_diff: Dict[int, float] = {}
        self.analysis_geometry_state: Dict[int, Dict[str, float]] = {}
        self.annulus_tune_state: Dict[int, Dict[str, float]] = {}
        self.annulus_param_history: Dict[int, Deque[Tuple[float, float, float]]] = {}
        self.annulus_notch_angle_state: Dict[int, float] = {}
        self.baseline_lock_state: Dict[int, Dict[str, Optional[float]]] = {}
        self.force_baseline_capture_pending: Dict[int, bool] = {}
        self._index_grid_cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._kernel_cache: Dict[int, np.ndarray] = {}

        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.kernel_size, self.kernel_size),
        )

    def _get_index_grid(self, shape: Tuple[int, int]) -> np.ndarray:
        """Get cached index grid."""
        h, w = int(shape[0]), int(shape[1])
        key = (h, w)
        grid = self._index_grid_cache.get(key)
        if grid is None:
            if len(self._index_grid_cache) >= 24:
                self._index_grid_cache.clear()
            grid = np.indices((h, w), dtype=np.float32)
            self._index_grid_cache[key] = grid
        return grid

    def _get_ellipse_kernel(self, size: int) -> np.ndarray:
        """Get cached elliptical kernel."""
        k = max(1, int(size))
        if k % 2 == 0:
            k += 1
        kernel = self._kernel_cache.get(k)
        if kernel is None:
            if len(self._kernel_cache) >= 32:
                self._kernel_cache.clear()
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            self._kernel_cache[k] = kernel
        return kernel

    def set_calibration_mode(self, mode: str) -> str:
        """Set the annulus calibration mode to 'manual' or 'auto' (invalid input falls back to 'manual') and return it."""
        mode_txt = str(mode or "").strip().lower()
        if mode_txt not in {"manual", "auto"}:
            mode_txt = "manual"
        self.calibration_mode = mode_txt
        return self.calibration_mode

    def set_annulus_manual_params(
        self,
        inner_offset_scale: Optional[float] = None,
        band_scale: Optional[float] = None,
    ) -> Dict[str, float]:
        """Clamp and store the manual annulus inner-offset and band scales, returning the applied values."""
        if inner_offset_scale is not None:
            self.annulus_head_inner_offset_scale = float(
                np.clip(
                    float(inner_offset_scale),
                    float(self.annulus_head_inner_offset_min_scale),
                    float(self.annulus_head_inner_offset_max_scale),
                )
            )
        if band_scale is not None:
            self.annulus_head_band_scale = float(np.clip(float(band_scale), 0.05, 1.40))
        return {
            "inner_offset_scale": float(self.annulus_head_inner_offset_scale),
            "band_scale": float(self.annulus_head_band_scale),
        }

    def get_calibration_snapshot(
        self, zone_id: Optional[int] = None
    ) -> Dict[str, object]:
        """Return the current calibration mode plus manual and active (auto-tuned) offset/band scales and auto score for a zone."""
        zone_key = int(zone_id) if zone_id is not None else 0
        state = self.annulus_tune_state.get(zone_key, {})
        return {
            "mode": str(self.calibration_mode),
            "manual_offset_scale": float(self.annulus_head_inner_offset_scale),
            "manual_band_scale": float(self.annulus_head_band_scale),
            "active_offset_scale": float(
                state.get("offset", self.annulus_head_inner_offset_scale)
            ),
            "active_band_scale": float(state.get("band", self.annulus_head_band_scale)),
            "auto_score": float(state.get("score", 0.0)),
            "auto_samples": int(state.get("samples", 0)),
        }

    def set_spray_exclusion_params(
        self,
        line_thickness_px: Optional[float] = None,
        ring_dilate_px: Optional[float] = None,
        band_ratio: Optional[float] = None,
    ) -> Dict[str, float]:
        # Deprecated: line_thickness_px kept for backward compatibility only.
        """Update the spray-exclusion ring-dilate px and band ratio (line_thickness_px ignored), returning the applied values."""
        if line_thickness_px is not None:
            _ = line_thickness_px
        if ring_dilate_px is not None:
            self.analysis_ring_dilate_px = int(
                np.clip(int(round(float(ring_dilate_px))), 1, 80)
            )
        if band_ratio is not None:
            self.spray_exclusion_band_ratio_of_annulus = float(
                np.clip(float(band_ratio), 0.01, 1.0)
            )
        return {
            "spray_line_thickness_px": float(self.spray_line_thickness_px),
            "analysis_ring_dilate_px": float(self.analysis_ring_dilate_px),
            "spray_exclusion_band_ratio_of_annulus": float(
                self.spray_exclusion_band_ratio_of_annulus
            ),
        }

    def set_baseline(
        self, zone_id: int, roi_gray: np.ndarray, metadata: Optional[Dict] = None
    ) -> bool:
        """Store the given ROI grayscale (with metadata and its GLCM contrast) as the locked baseline for a zone; return True on success."""
        if zone_id is None or roi_gray is None or roi_gray.size == 0:
            return False

        self.baselines[int(zone_id)] = {
            "roi_gray": roi_gray.copy(),
            "metadata": dict(metadata or {}),
            "glcm_contrast": self._compute_glcm_contrast(roi_gray),
        }
        return True

    def set_user_lock_enabled(
        self,
        enabled: bool,
        zone_id: Optional[int] = None,
        clear_baseline_on_disable: bool = True,
    ) -> Dict[str, bool]:
        """Toggle the user baseline-lock flag, optionally clearing the zone's baseline when disabled; return lock and baseline status."""
        self.baseline_lock_user_enabled = bool(enabled)
        if (not self.baseline_lock_user_enabled) and clear_baseline_on_disable:
            self.clear_baseline(zone_id)
            if zone_id is None:
                self.force_baseline_capture_pending.clear()
            else:
                self.force_baseline_capture_pending.pop(int(zone_id), None)
        return {
            "user_lock_enabled": bool(self.baseline_lock_user_enabled),
            "has_baseline": bool(self.has_baseline(zone_id)),
        }

    def request_force_baseline_capture(self, zone_id: Optional[int]) -> None:
        """Flag a zone to force-capture its baseline on the next eligible detect frame (no-op if zone_id is None)."""
        if zone_id is None:
            return
        self.force_baseline_capture_pending[int(zone_id)] = True

    def clear_baseline(self, zone_id: Optional[int] = None) -> None:
        """Drop baseline, history, EMA, geometry, notch and lock state for one zone, or all zones when zone_id is None."""
        if zone_id is None:
            self.baselines.clear()
            self.history.clear()
            self.ema_percentage.clear()
            self.ema_diff.clear()
            self.analysis_geometry_state.clear()
            self.annulus_notch_angle_state.clear()
            self.baseline_lock_state.clear()
            self.force_baseline_capture_pending.clear()
            return

        zone_key = int(zone_id)
        self.baselines.pop(zone_key, None)
        self.history.pop(zone_key, None)
        self.ema_percentage.pop(zone_key, None)
        self.ema_diff.pop(zone_key, None)
        self.analysis_geometry_state.pop(zone_key, None)
        self.annulus_notch_angle_state.pop(zone_key, None)
        self.baseline_lock_state.pop(zone_key, None)
        self.force_baseline_capture_pending.pop(zone_key, None)

    def has_baseline(self, zone_id: Optional[int]) -> bool:
        """Return True if a baseline has been captured for the given zone (False when zone_id is None)."""
        if zone_id is None:
            return False
        return int(zone_id) in self.baselines

    def get_temporal_state(self, zone_id: Optional[int]) -> Dict:
        """Return a zone's temporal snapshot: baseline presence, history length, EMA buildup/diff, and baseline GLCM contrast."""
        if zone_id is None:
            return {
                "has_baseline": False,
                "history_length": 0,
                "ema_buildup_percentage": None,
                "ema_diff_mean": None,
                "baseline_glcm_contrast": None,
            }

        zone_key = int(zone_id)
        return {
            "has_baseline": zone_key in self.baselines,
            "history_length": len(self.history.get(zone_key, [])),
            "ema_buildup_percentage": self.ema_percentage.get(zone_key),
            "ema_diff_mean": self.ema_diff.get(zone_key),
            "baseline_glcm_contrast": self.baselines.get(zone_key, {}).get(
                "glcm_contrast"
            ),
        }

    def detect(
        self,
        frame: np.ndarray,
        nozzle_info: Dict,
        zone_id: Optional[int] = None,
        spray_info: Optional[Dict] = None,
    ) -> Dict:
        """Run full buildup detection on the nozzle tip ROI and return a result dict (mask, percentage, severity, triggers, temporal/GLCM/diff cues)."""
        if frame is None or not nozzle_info.get("detected", False):
            return self._empty_result()

        (
            roi,
            roi_mask,
            roi_bbox,
            analysis_mask,
            annular_mask,
            spray_exclusion_mask,
            calib_info,
        ) = self._extract_tip_roi(
            frame,
            nozzle_info,
            zone_id=zone_id,
            spray_info=spray_info,
        )
        if calib_info is None:
            calib_info = {}

        if roi is None or roi_mask is None:
            return self._empty_result()

        edge_mask = self._detect_edges(roi)
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Difference tracking must follow analysis area only (green mask),
        # explicitly ignoring spray-exclusion area (purple mask).
        diff_mask_roi = roi_mask
        diff_reliable = self._is_diff_tracking_reliable(nozzle_info)
        diff_mask = (
            self._compute_baseline_diff_mask(
                zone_id=zone_id, roi_gray=roi_gray, roi_mask=diff_mask_roi
            )
            if diff_reliable
            else None
        )
        diff_overlap_ratio = 0.0
        if diff_mask is not None and self.buildup_diff_edge_overlap_required:
            k = int(max(1, self.buildup_diff_edge_overlap_kernel))
            if k % 2 == 0:
                k += 1
            edge_for_overlap = edge_mask
            if k >= 3:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                edge_for_overlap = cv2.dilate(edge_for_overlap, kernel, iterations=1)
            overlap = cv2.bitwise_and(diff_mask, edge_for_overlap)
            diff_pixels = int(np.count_nonzero(diff_mask))
            overlap_pixels = int(np.count_nonzero(overlap))
            diff_overlap_ratio = (
                (float(overlap_pixels) / float(max(1, diff_pixels)))
                if diff_pixels > 0
                else 0.0
            )
            if diff_overlap_ratio < float(self.buildup_diff_edge_overlap_min_ratio):
                diff_mask = None
            else:
                diff_mask = overlap
        texture_score = self._analyze_texture(roi, roi_mask)
        buildup_mask = self._detect_buildup_regions(
            edge_mask, roi_mask, annulus_meta=calib_info
        )
        if self.buildup_diff_use_for_mask and diff_mask is not None:
            buildup_mask = cv2.bitwise_or(buildup_mask, diff_mask)
        buildup_percentage = self._calculate_buildup_percentage(buildup_mask, roi_mask)
        diff_percentage = (
            self._calculate_buildup_percentage(diff_mask, roi_mask)
            if diff_mask is not None
            else 0.0
        )
        force_locked = self._maybe_force_capture_locked_baseline(
            zone_id=zone_id,
            nozzle_info=nozzle_info,
            roi_gray=roi_gray,
        )
        baseline_locked = force_locked or self._maybe_update_locked_baseline(
            zone_id=zone_id,
            nozzle_info=nozzle_info,
            roi_gray=roi_gray,
            buildup_percentage=buildup_percentage,
        )
        ema_buildup_percentage = self._update_temporal_ema(
            zone_id, buildup_percentage, self.ema_percentage
        )
        diff_mean = self._compute_temporal_diff(zone_id, roi_gray)
        ema_diff_mean = (
            self._update_temporal_ema(zone_id, diff_mean, self.ema_diff)
            if diff_mean is not None
            else None
        )
        glcm_contrast = self._compute_glcm_contrast(roi_gray)
        glcm_buildup = self._detect_glcm_anomaly(zone_id, glcm_contrast)
        temporal_increase, temporal_score, trend = self._compute_temporal_trend(zone_id)
        temporal_state = self.get_temporal_state(zone_id)
        severity = self._determine_severity(buildup_percentage)
        detection_strength = (
            str((nozzle_info or {}).get("detection_strength", "")).strip().lower()
        )
        weak_detection = detection_strength == "weak"
        area_gate_scale = (
            float(self.buildup_weak_threshold_scale) if weak_detection else 1.0
        )
        effective_area_threshold = float(
            np.clip(float(self.buildup_area_threshold) * area_gate_scale, 0.0, 1.0)
        )
        direct_trigger = bool(buildup_percentage > effective_area_threshold)
        if weak_detection and self.buildup_require_strong_detection:
            weak_override = float(np.clip(self.buildup_weak_override_ratio, 0.0, 1.0))
            direct_trigger = bool(
                buildup_percentage > max(effective_area_threshold, weak_override)
            )
        glcm_gate_base = float(self.glcm_min_buildup_ratio) * area_gate_scale
        glcm_gate = max(0.0, min(effective_area_threshold, glcm_gate_base))
        glcm_trigger = bool(glcm_buildup and (buildup_percentage > glcm_gate))
        if weak_detection and self.buildup_require_strong_detection:
            glcm_trigger = False
        temporal_gate_base = float(self.temporal_min_buildup_ratio) * area_gate_scale
        temporal_gate = max(0.0, min(effective_area_threshold, temporal_gate_base))
        temporal_trigger = bool(
            temporal_increase
            and (ema_diff_mean is not None)
            and (ema_diff_mean > self.temporal_diff_threshold)
            and (buildup_percentage > temporal_gate)
        )
        if weak_detection and self.buildup_require_strong_detection:
            temporal_trigger = False
        diff_trigger = bool(
            diff_mask is not None
            and (diff_percentage > float(self.buildup_diff_min_ratio))
        )
        has_buildup = bool(
            direct_trigger or glcm_trigger or temporal_trigger or diff_trigger
        )
        # Enforce the area threshold as a hard floor so secondary cues (GLCM/temporal/diff)
        # cannot raise YES while the measured buildup area is still below the threshold.
        if self.buildup_area_hard_floor and (
            buildup_percentage < effective_area_threshold
        ):
            has_buildup = False

        if zone_id is not None:
            zone_key = int(zone_id)
            history = self.history.setdefault(zone_key, deque(maxlen=self.max_history))
            if history:
                history[-1]["diff_mean"] = (
                    float(diff_mean) if diff_mean is not None else 0.0
                )
                history[-1]["glcm_contrast"] = (
                    float(glcm_contrast) if glcm_contrast is not None else 0.0
                )

        annulus_offset_raw = calib_info.get(
            "offset", self.annulus_head_inner_offset_scale
        )
        annulus_band_raw = calib_info.get("band", self.annulus_head_band_scale)
        annulus_score_raw = calib_info.get("score", 0.0)
        annulus_samples_raw = calib_info.get("samples", 0)
        if isinstance(annulus_offset_raw, (int, float, np.integer, np.floating, str)):
            annulus_offset = float(annulus_offset_raw)
        else:
            annulus_offset = float(self.annulus_head_inner_offset_scale)
        if isinstance(annulus_band_raw, (int, float, np.integer, np.floating, str)):
            annulus_band = float(annulus_band_raw)
        else:
            annulus_band = float(self.annulus_head_band_scale)
        if isinstance(annulus_score_raw, (int, float, np.integer, np.floating, str)):
            annulus_score = float(annulus_score_raw)
        else:
            annulus_score = 0.0
        if isinstance(annulus_samples_raw, (int, float, np.integer, np.floating, str)):
            annulus_samples = int(float(annulus_samples_raw))
        else:
            annulus_samples = 0

        return {
            "has_buildup": has_buildup,
            "buildup_percentage": buildup_percentage,
            "buildup_mask": buildup_mask,
            "texture_score": texture_score,
            "severity": severity,
            "roi": roi,
            "roi_mask": roi_mask,
            "roi_bbox": roi_bbox,
            "analysis_mask": analysis_mask,
            "annular_mask": annular_mask,
            "spray_exclusion_mask": spray_exclusion_mask,
            "calibration_mode": str(calib_info.get("mode", self.calibration_mode)),
            "annulus_inner_offset_scale": annulus_offset,
            "annulus_band_scale": annulus_band,
            "annulus_auto_score": annulus_score,
            "annulus_auto_samples": annulus_samples,
            "temporal_enabled": self.temporal_enabled,
            "has_baseline": temporal_state["has_baseline"],
            "diff_mean": diff_mean,
            "ssim": None,
            "ema_buildup_percentage": ema_buildup_percentage,
            "ema_diff_mean": ema_diff_mean,
            "temporal_increase": temporal_increase,
            "temporal_score": temporal_score,
            "trend": trend,
            "glcm_enabled": self.glcm_enabled,
            "glcm_contrast": glcm_contrast,
            "glcm_buildup": glcm_buildup,
            "baseline_locked": baseline_locked,
            "baseline_lock_enabled": bool(self.baseline_lock_user_enabled),
            "diff_percentage": float(diff_percentage),
            "diff_trigger": diff_trigger,
            "diff_reliable": bool(diff_reliable),
            "diff_edge_overlap_ratio": float(diff_overlap_ratio),
            "effective_area_threshold": float(effective_area_threshold),
        }

    def _baseline_lock_state_for(
        self, zone_id: Optional[int]
    ) -> Dict[str, Optional[float]]:
        """Return (creating if absent) the mutable baseline-lock tracking state dict (stable_count, last x/y/r, locked) for a zone."""
        zone_key = int(zone_id) if zone_id is not None else -1
        state = self.baseline_lock_state.get(zone_key)
        if state is None:
            state = {
                "stable_count": 0.0,
                "last_x": None,
                "last_y": None,
                "last_r": None,
                "locked": 0.0,
            }
            self.baseline_lock_state[zone_key] = state
        return state

    @staticmethod
    def _base_detection_method(nozzle_info: Dict) -> str:
        """Return the primary detection method (the part before any '+') from nozzle_info, lowercased and stripped."""
        method = str((nozzle_info or {}).get("detection_method", "")).strip().lower()
        if not method:
            return ""
        return method.split("+")[0].strip()

    def _is_diff_tracking_reliable(self, nozzle_info: Dict) -> bool:
        """Return True if baseline-diff tracking is trustworthy for this detection (enabled, strong, and a permitted method)."""
        if not self.buildup_diff_enabled:
            return False
        if self.buildup_diff_requires_strong_detection:
            strength = (
                str((nozzle_info or {}).get("detection_strength", "")).strip().lower()
            )
            if strength == "weak":
                return False
        method = self._base_detection_method(nozzle_info)
        if method == "kalman_predict" and (not self.buildup_diff_allow_kalman_predict):
            return False
        if method == "circle_relaxed" and (not self.buildup_diff_allow_circle_relaxed):
            return False
        if method == "circle_blob" and (not self.buildup_diff_allow_circle_blob):
            return False
        if method in {"circle", "contour"}:
            return True
        return False

    def _maybe_update_locked_baseline(
        self,
        zone_id: Optional[int],
        nozzle_info: Dict,
        roi_gray: np.ndarray,
        buildup_percentage: float,
    ) -> bool:
        """Capture the baseline once the nozzle stays stable with near-zero buildup for enough frames; return whether a baseline exists."""
        if zone_id is None or roi_gray is None or roi_gray.size == 0:
            return False
        if not self.baseline_lock_user_enabled:
            return False

        zone_key = int(zone_id)
        has_baseline = zone_key in self.baselines
        if has_baseline and (not self.baseline_lock_allow_refresh):
            state = self._baseline_lock_state_for(zone_key)
            state["locked"] = 1.0
            return True

        if not self.baseline_lock_enabled:
            if not has_baseline:
                self.set_baseline(zone_key, roi_gray, metadata={"mode": "legacy_auto"})
            return zone_key in self.baselines

        state = self._baseline_lock_state_for(zone_key)
        if not self._is_baseline_lock_detection_eligible(nozzle_info):
            state["stable_count"] = 0.0
            return has_baseline

        pos = nozzle_info.get("position")
        radius = nozzle_info.get("radius")
        if pos is None or radius is None:
            return has_baseline

        try:
            px, py = float(pos[0]), float(pos[1])
            pr = max(1.0, float(radius))
        except Exception:
            return has_baseline

        last_x = state.get("last_x")
        last_y = state.get("last_y")
        last_r = state.get("last_r")
        state["last_x"] = px
        state["last_y"] = py
        state["last_r"] = pr

        if last_x is None or last_y is None or last_r is None:
            state["stable_count"] = 0.0
            return has_baseline

        center_jitter = float(np.hypot(px - float(last_x), py - float(last_y)))
        radius_jitter_ratio = float(abs(pr - float(last_r)) / max(float(last_r), 1e-6))
        low_buildup = float(buildup_percentage) <= float(
            self.baseline_lock_max_buildup_ratio
        )
        stable_now = (
            low_buildup
            and center_jitter <= float(self.baseline_lock_max_center_jitter_px)
            and radius_jitter_ratio <= float(self.baseline_lock_max_radius_jitter_ratio)
        )

        if stable_now:
            stable_count = state.get("stable_count", 0.0)
            stable_count_f = float(0.0 if stable_count is None else stable_count)
            state["stable_count"] = stable_count_f + 1.0
        else:
            state["stable_count"] = 0.0

        need_frames = max(1, int(self.baseline_lock_required_stable_frames))
        stable_count_now = state.get("stable_count", 0.0)
        stable_count_now_f = float(
            0.0 if stable_count_now is None else stable_count_now
        )
        if int(round(stable_count_now_f)) >= need_frames:
            self.set_baseline(
                zone_key,
                roi_gray,
                metadata={
                    "mode": "lock_and_track_auto",
                    "buildup_percentage": float(buildup_percentage),
                    "stable_frames": int(need_frames),
                    "head_radius_px": float(pr),
                    "detection_method": str(nozzle_info.get("detection_method") or ""),
                    "detection_strength": str(
                        nozzle_info.get("detection_strength") or ""
                    ),
                },
            )
            state["locked"] = 1.0
            return True

        return has_baseline

    def _maybe_force_capture_locked_baseline(
        self,
        zone_id: Optional[int],
        nozzle_info: Dict,
        roi_gray: np.ndarray,
    ) -> bool:
        """Honor a pending force-capture request by snapshotting the baseline for a real (non-predicted) detection; return True if captured."""
        if zone_id is None or roi_gray is None or roi_gray.size == 0:
            return False
        zone_key = int(zone_id)
        if not bool(self.force_baseline_capture_pending.get(zone_key, False)):
            return False
        if not self.baseline_lock_user_enabled:
            return False
        if not bool((nozzle_info or {}).get("detected", False)):
            return False
        if bool((nozzle_info or {}).get("is_prediction", False)):
            return False

        pos = (nozzle_info or {}).get("position")
        radius = (nozzle_info or {}).get("radius")
        method = str((nozzle_info or {}).get("detection_method") or "").strip().lower()
        strength = (
            str((nozzle_info or {}).get("detection_strength") or "").strip().lower()
        )
        metadata = {
            "mode": "user_lock_snapshot",
            "forced": True,
            "detection_method": method,
            "detection_strength": strength,
        }
        try:
            if radius is not None:
                metadata["head_radius_px"] = float(max(1.0, float(radius)))
        except Exception:
            pass
        if pos is not None:
            try:
                metadata["position_x"] = float(pos[0])
                metadata["position_y"] = float(pos[1])
            except Exception:
                pass

        captured = bool(self.set_baseline(zone_key, roi_gray, metadata=metadata))
        if captured:
            self.force_baseline_capture_pending.pop(zone_key, None)
            state = self._baseline_lock_state_for(zone_key)
            state["stable_count"] = 0.0
            state["last_x"] = None
            state["last_y"] = None
            state["last_r"] = None
            state["locked"] = 1.0
            return True
        return False

    def _is_baseline_lock_detection_eligible(self, nozzle_info: Dict) -> bool:
        """Return True if this detection's method/strength/prediction state meets the configured gates for auto baseline locking."""
        info = nozzle_info or {}
        method_text = str(info.get("detection_method", "")).strip().lower()
        base_method = self._base_detection_method(info)
        is_prediction = bool(info.get("is_prediction", False)) or (
            base_method == "kalman_predict"
        )
        detection_strength = str(info.get("detection_strength", "")).strip().lower()

        if self.baseline_lock_require_real_detection and is_prediction:
            return False
        if (
            self.baseline_lock_require_strong_detection
            and detection_strength
            and detection_strength != "strong"
        ):
            return False
        if self.baseline_lock_require_kalman_smoothed and (
            "+kalman" not in method_text
        ):
            return False
        if (not self.baseline_lock_allow_relaxed_methods) and (
            base_method == "circle_relaxed"
        ):
            return False
        if (not self.baseline_lock_allow_blob_method) and (
            base_method == "circle_blob"
        ):
            return False
        if (not self.baseline_lock_allow_rescue_methods) and (
            base_method in {"rescue_contour", "fullframe_rescue"}
        ):
            return False
        return base_method in {
            "circle",
            "contour",
            "circle_relaxed",
            "circle_blob",
            "rescue_contour",
            "fullframe_rescue",
        }

    def _baseline_head_radius_px(self, zone_id: Optional[int]) -> Optional[float]:
        """Return the head radius (px) stored in the zone's baseline metadata, or None if absent or non-positive."""
        if zone_id is None:
            return None
        baseline = self.baselines.get(int(zone_id))
        if not isinstance(baseline, dict):
            return None
        metadata = baseline.get("metadata")
        if not isinstance(metadata, dict):
            return None
        value = metadata.get("head_radius_px")
        if value is None:
            return None
        try:
            radius_px = float(value)
        except Exception:
            return None
        if radius_px <= 0.0:
            return None
        return radius_px

    def _is_weak_geometry_signal(self, nozzle_info: Dict) -> bool:
        """Return True when the detection geometry is unreliable (prediction, weak strength, fallback method, or not Kalman-smoothed)."""
        info = nozzle_info or {}
        method_text = str(info.get("detection_method", "")).strip().lower()
        base_method = self._base_detection_method(info)
        is_prediction = bool(info.get("is_prediction", False)) or (
            base_method == "kalman_predict"
        )
        strength = str(info.get("detection_strength", "")).strip().lower()
        weak_strength = strength == "weak"
        weak_method = base_method in {
            "circle_relaxed",
            "circle_blob",
            "kalman_predict",
            "rescue_contour",
            "fullframe_rescue",
        }
        not_kalman_smoothed = "+kalman" not in method_text
        return bool(
            is_prediction or weak_strength or weak_method or not_kalman_smoothed
        )

    def _apply_baseline_radius_anchor(
        self,
        zone_id: Optional[int],
        measured_radius: float,
        nozzle_info: Dict,
    ) -> float:
        """Clamp the measured head radius to a ratio band around the baseline radius (only on weak signals) to stabilize the ROI."""
        radius_now = max(1.0, float(measured_radius))
        if not self.baseline_lock_user_enabled:
            return radius_now
        if (not self.baseline_lock_radius_anchor_enabled) or zone_id is None:
            return radius_now
        baseline_radius = self._baseline_head_radius_px(zone_id)
        if baseline_radius is None:
            return radius_now
        if self.baseline_lock_radius_anchor_on_weak_only and (
            not self._is_weak_geometry_signal(nozzle_info)
        ):
            return radius_now

        min_ratio = float(
            np.clip(self.baseline_lock_radius_anchor_min_ratio, 0.50, 1.20)
        )
        max_ratio = float(
            np.clip(self.baseline_lock_radius_anchor_max_ratio, min_ratio, 2.50)
        )
        floor_r = float(baseline_radius) * min_ratio
        ceil_r = float(baseline_radius) * max_ratio
        return float(np.clip(max(radius_now, floor_r), 1.0, max(ceil_r, floor_r + 1.0)))

    def _compute_baseline_diff_mask(
        self,
        zone_id: Optional[int],
        roi_gray: np.ndarray,
        roi_mask: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """Build a binary mask of pixels differing from the zone baseline beyond an adaptive threshold; None if disabled/empty."""
        if (
            (not self.buildup_diff_enabled)
            or zone_id is None
            or roi_gray is None
            or roi_gray.size == 0
        ):
            return None
        baseline = self.baselines.get(int(zone_id))
        if not baseline:
            return None
        baseline_roi = baseline.get("roi_gray")
        if baseline_roi is None or baseline_roi.size == 0:
            return None
        if baseline_roi.shape != roi_gray.shape:
            baseline_roi = cv2.resize(
                baseline_roi, (roi_gray.shape[1], roi_gray.shape[0])
            )

        diff = cv2.absdiff(roi_gray, baseline_roi)
        k = int(max(1, self.buildup_diff_blur_kernel))
        if k % 2 == 0:
            k += 1
        if k >= 3:
            diff = cv2.GaussianBlur(diff, (k, k), 0)
        # Adaptive threshold: keep sensitivity, but suppress global flicker/noise.
        base_thr = float(self.buildup_diff_threshold)
        if roi_mask is not None and roi_mask.size > 0:
            valid = diff[roi_mask > 0]
        else:
            valid = diff.reshape(-1)
        if valid.size > 0:
            mu = float(np.mean(valid))
            sigma = float(np.std(valid))
            adaptive_thr = float(mu + (2.2 * sigma))
            thr = int(max(base_thr, adaptive_thr))
        else:
            thr = int(base_thr)
        thr = int(np.clip(thr, 1, 255))
        _, mask = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)
        if roi_mask is not None and roi_mask.size > 0:
            mask = cv2.bitwise_and(mask, roi_mask.astype(np.uint8))
        # Remove tiny sparkle regions that create false YES.
        open_kernel = self._get_ellipse_kernel(3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
        if int(np.count_nonzero(mask)) <= 0:
            return None
        return mask

    def _compute_temporal_diff(
        self, zone_id: Optional[int], roi_gray: np.ndarray
    ) -> Optional[float]:
        """Return the normalized (0-1) mean absolute difference between the current ROI gray and the zone baseline, or None."""
        if zone_id is None or roi_gray is None or roi_gray.size == 0:
            return None

        zone_key = int(zone_id)
        if zone_key not in self.baselines:
            if not self.baseline_lock_enabled:
                self.set_baseline(zone_key, roi_gray)
                return 0.0
            return None

        baseline_roi = self.baselines[zone_key].get("roi_gray")
        if baseline_roi is None or baseline_roi.size == 0:
            if not self.baseline_lock_enabled:
                self.set_baseline(zone_key, roi_gray)
                return 0.0
            return None

        if baseline_roi.shape != roi_gray.shape:
            baseline_roi = cv2.resize(
                baseline_roi, (roi_gray.shape[1], roi_gray.shape[0])
            )

        diff = cv2.absdiff(roi_gray, baseline_roi)
        return float(np.mean(diff) / 255.0)

    def _compute_glcm_contrast(self, roi_gray: Optional[np.ndarray]) -> Optional[float]:
        """Compute the mean GLCM texture contrast (over 4 angles) of a grayscale ROI; None if GLCM is disabled/unavailable."""
        if (
            (not self.glcm_enabled)
            or roi_gray is None
            or roi_gray.size == 0
            or graycomatrix is None
            or graycoprops is None
        ):
            return None

        levels = max(8, int(self.glcm_levels))
        scaled = roi_gray.astype(np.float32)
        scaled = np.floor((scaled / 256.0) * float(levels)).astype(np.uint8)
        scaled = np.clip(scaled, 0, levels - 1)
        glcm = graycomatrix(
            scaled,
            distances=[1],
            angles=[0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0],
            levels=levels,
            symmetric=True,
            normed=True,
        )
        return float(graycoprops(glcm, "contrast").mean())

    def _detect_glcm_anomaly(
        self, zone_id: Optional[int], glcm_contrast: Optional[float]
    ) -> bool:
        """Return True if the current GLCM contrast exceeds the zone baseline's contrast by the configured ratio."""
        if zone_id is None or glcm_contrast is None:
            return False
        baseline = self.baselines.get(int(zone_id))
        if not baseline:
            return False
        baseline_contrast = baseline.get("glcm_contrast")
        if baseline_contrast is None or baseline_contrast <= 1e-6:
            return False
        return float(glcm_contrast) > float(baseline_contrast) * float(
            self.glcm_contrast_ratio
        )

    def _compute_temporal_trend(
        self, zone_id: Optional[int]
    ) -> Tuple[bool, Optional[float], Optional[str]]:
        """Compare recent vs. past buildup history to return (is_rising, delta, trend label) once enough frames exist."""
        if zone_id is None:
            return False, None, None

        history = list(self.history.get(int(zone_id), []))
        if len(history) < max(3, self.temporal_min_frames):
            return False, None, None

        values = np.asarray(
            [float(item.get("buildup_percentage", 0.0)) for item in history],
            dtype=np.float32,
        )
        split = max(1, min(values.size - 1, values.size // 3))
        past = values[:-split]
        recent = values[-split:]
        if past.size == 0 or recent.size == 0:
            return False, None, None

        delta = float(recent.mean() - past.mean())
        if delta > self.anomaly_threshold:
            return True, delta, "rising"
        if delta < -self.anomaly_threshold:
            return False, delta, "falling"
        return False, delta, "stable"

    def _extract_tip_roi(
        self,
        frame: np.ndarray,
        nozzle_info: Dict,
        zone_id: Optional[int] = None,
        spray_info: Optional[Dict] = None,
    ) -> Tuple[
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[Tuple[int, int, int, int]],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[Dict[str, object]],
    ]:
        """Crop the stabilized tip ROI and build its annular, analysis, and spray-exclusion masks; return them with calibration info."""
        bbox = nozzle_info.get("bounding_box")
        position = nozzle_info.get("position")

        if bbox is None or position is None:
            return None, None, None, None, None, None, None

        _, _, bbox_w, bbox_h = bbox
        bbox_w = max(1, int(bbox_w))
        bbox_h = max(1, int(bbox_h))
        roi_w = max(self.tip_roi_min_size_px, int(round(bbox_w * self.tip_roi_scale_x)))
        roi_h = max(self.tip_roi_min_size_px, int(round(bbox_h * self.tip_roi_scale_y)))
        head_radius = nozzle_info.get("radius")
        if head_radius is not None:
            try:
                head_r = max(1.0, float(head_radius))
                min_side = int(
                    round(2.0 * head_r * max(1.4, self.tip_roi_radius_padding_scale))
                )
                roi_w = max(roi_w, min_side)
                roi_h = max(roi_h, min_side)
            except Exception:
                pass

        raw_radius = nozzle_info.get("radius")
        if raw_radius is not None:
            try:
                measured_radius = max(1.0, float(raw_radius))
            except Exception:
                measured_radius = float(min(bbox_w, bbox_h)) * 0.5
        else:
            measured_radius = float(min(bbox_w, bbox_h)) * 0.5
        measured_radius = self._apply_baseline_radius_anchor(
            zone_id=zone_id,
            measured_radius=measured_radius,
            nozzle_info=nozzle_info,
        )

        px, py, measured_radius = self._stabilize_analysis_geometry(
            zone_id=zone_id,
            measured_center=(float(position[0]), float(position[1])),
            measured_radius=measured_radius,
        )
        offset_x, offset_y = self._compute_tip_offset(nozzle_info, roi_h)
        x = int(round(px - (roi_w * 0.5) + offset_x))
        y = int(round(py - (roi_h * 0.5) + offset_y))

        height, width = frame.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(int(roi_w), width - x)
        h = min(int(roi_h), height - y)

        if w <= 0 or h <= 0:
            return None, None, None, None, None, None, None

        roi = frame[y : y + h, x : x + w].copy()
        # Use stabilized center for local-tip anchor to reduce annulus jitter.
        local_tip = (
            float(px - x),
            float(py - y),
        )
        local_tip = (
            float(np.clip(local_tip[0], 0.0, max(0.0, float(w - 1)))),
            float(np.clip(local_tip[1], 0.0, max(0.0, float(h - 1)))),
        )
        center = (int(round(local_tip[0])), int(round(local_tip[1])))
        head_radius = float(measured_radius) if measured_radius is not None else None
        notch_angle_deg = self._estimate_annulus_notch_angle_deg(
            zone_id=zone_id,
            center=center,
            orientation=nozzle_info.get("orientation"),
            spray_info=spray_info,
            roi_bbox=(x, y, w, h),
        )
        annular_mask, calib_info = self._build_annular_mask(
            (h, w),
            center,
            head_radius_px=head_radius,
            zone_id=zone_id,
            roi=roi,
            notch_angle_deg=notch_angle_deg,
        )
        spray_mask_raw = self._build_spray_exclusion_mask(
            (h, w),
            spray_info=spray_info,
            roi_bbox=(x, y, w, h),
            annular_mask=annular_mask,
            annulus_meta=calib_info,
            center=center,
            orientation=nozzle_info.get("orientation"),
        )
        # Force strict partition: annulus = spray_exclusion + analysis (no overlap, no outside).
        spray_mask = cv2.bitwise_and(spray_mask_raw, annular_mask)
        analysis_mask = cv2.bitwise_and(annular_mask, cv2.bitwise_not(spray_mask))

        if int(np.count_nonzero(analysis_mask)) < self.analysis_mask_min_pixels:
            return None, None, None, None, None, None, calib_info

        return (
            roi,
            analysis_mask,
            (x, y, w, h),
            analysis_mask,
            annular_mask,
            spray_mask,
            calib_info,
        )

    def _stabilize_analysis_geometry(
        self,
        zone_id: Optional[int],
        measured_center: Tuple[float, float],
        measured_radius: float,
    ) -> Tuple[float, float, float]:
        """Smooth the per-zone center/radius via EMA, rejecting large jumps until reacquire, and return the stabilized values."""
        cx, cy = float(measured_center[0]), float(measured_center[1])
        cr = max(1.0, float(measured_radius))
        if (not self.analysis_stabilization_enabled) or zone_id is None:
            return cx, cy, cr

        zone_key = int(zone_id)
        state = self.analysis_geometry_state.get(zone_key)
        if state is None:
            self.analysis_geometry_state[zone_key] = {
                "cx": cx,
                "cy": cy,
                "r": cr,
                "miss": 0.0,
            }
            return cx, cy, cr

        prev_cx = float(state.get("cx", cx))
        prev_cy = float(state.get("cy", cy))
        prev_r = max(1.0, float(state.get("r", cr)))
        jump = float(np.hypot(cx - prev_cx, cy - prev_cy))
        radius_delta_ratio = float(abs(cr - prev_r) / max(prev_r, 1e-6))

        reject_jump = jump > float(self.analysis_max_center_jump_px)
        reject_radius = radius_delta_ratio > float(self.analysis_max_radius_delta_ratio)
        if reject_jump or reject_radius:
            miss = float(state.get("miss", 0.0)) + 1.0
            if miss >= float(max(1, self.analysis_reacquire_frames)):
                self.analysis_geometry_state[zone_key] = {
                    "cx": cx,
                    "cy": cy,
                    "r": cr,
                    "miss": 0.0,
                }
                return cx, cy, cr
            state["miss"] = miss
            return prev_cx, prev_cy, prev_r

        alpha_c = float(np.clip(self.analysis_center_ema_alpha, 0.01, 1.0))
        alpha_r = float(np.clip(self.analysis_radius_ema_alpha, 0.01, 1.0))
        smoothed_cx = ((1.0 - alpha_c) * prev_cx) + (alpha_c * cx)
        smoothed_cy = ((1.0 - alpha_c) * prev_cy) + (alpha_c * cy)
        smoothed_r = ((1.0 - alpha_r) * prev_r) + (alpha_r * cr)
        self.analysis_geometry_state[zone_key] = {
            "cx": float(smoothed_cx),
            "cy": float(smoothed_cy),
            "r": float(max(1.0, smoothed_r)),
            "miss": 0.0,
        }
        return float(smoothed_cx), float(smoothed_cy), float(max(1.0, smoothed_r))

    def _compute_tip_offset(
        self, nozzle_info: Dict, roi_height: int
    ) -> Tuple[float, float]:
        """Return the (x, y) ROI shift along the nozzle orientation by tip_roi_offset_ratio*roi_height (zero if unset)."""
        orientation = nozzle_info.get("orientation")
        if orientation is None or abs(self.tip_roi_offset_ratio) < 1e-6:
            return 0.0, 0.0

        angle_rad = np.deg2rad(float(orientation))
        distance = float(roi_height) * self.tip_roi_offset_ratio
        return (
            float(np.cos(angle_rad) * distance),
            float(np.sin(angle_rad) * distance),
        )

    def _build_annular_mask(
        self,
        shape: Tuple[int, int],
        center: Tuple[int, int],
        head_radius_px: Optional[float] = None,
        zone_id: Optional[int] = None,
        roi: Optional[np.ndarray] = None,
        notch_angle_deg: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        """Build the ring (annulus) analysis mask around the nozzle head with a notch cut out; return it plus geometry/calibration meta."""
        h, w = shape
        min_dim = max(1, min(h, w))
        outer_radius = max(2, int(round(self.annulus_outer_ratio * min_dim)))
        inner_radius = max(
            0, min(outer_radius - 1, int(round(self.annulus_inner_ratio * min_dim)))
        )
        selected_offset = float(self.annulus_head_inner_offset_scale)
        selected_band = float(self.annulus_head_band_scale)
        selected_score = 0.0
        selected_samples = 0
        if (
            self.annulus_use_head_circle
            and head_radius_px is not None
            and head_radius_px > 1.0
        ):
            head_radius_px = float(head_radius_px)
            legacy_outer = int(
                round(head_radius_px * float(self.annulus_head_outer_scale))
            )
            legacy_inner = int(
                round(head_radius_px * float(self.annulus_head_inner_scale))
            )
            outer_radius = max(outer_radius, legacy_outer)
            inner_radius = max(inner_radius, legacy_inner)

            if self.annulus_head_outer_only:
                # Shift analysis ring to outside the nozzle head (where deposits typically accumulate).
                calib_mode = str(self.calibration_mode).strip().lower()
                if (
                    calib_mode == "auto"
                    and self.auto_calibration_enabled
                    and roi is not None
                    and roi.size > 0
                ):
                    selected_offset, selected_band, selected_score, selected_samples = (
                        self._resolve_auto_annulus_scales(
                            roi=roi,
                            center=center,
                            head_radius_px=head_radius_px,
                            zone_id=zone_id,
                        )
                    )
                else:
                    selected_offset = float(
                        np.clip(
                            float(self.annulus_head_inner_offset_scale),
                            float(self.annulus_head_inner_offset_min_scale),
                            float(self.annulus_head_inner_offset_max_scale),
                        )
                    )
                    selected_band = max(0.05, float(self.annulus_head_band_scale))

                outside_inner = int(round(head_radius_px * (1.0 + selected_offset)))
                outside_outer = int(
                    round(head_radius_px * (1.0 + selected_offset + selected_band))
                )
                outer_radius = max(outer_radius, outside_outer)
                if selected_offset < 0.0:
                    inner_radius = min(inner_radius, outside_inner)
                else:
                    inner_radius = max(inner_radius, outside_inner)

            inner_radius = min(inner_radius, max(0, outer_radius - 1))

        # Keep the annulus fully inside ROI so its shape stays circular (no square clipping at ROI borders).
        max_outer = max(2, int(round(0.46 * float(min_dim))))
        outer_radius = min(outer_radius, max_outer)
        inner_radius = min(inner_radius, max(0, outer_radius - 1))

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, center, outer_radius, 255, -1)
        if inner_radius > 0:
            cv2.circle(mask, center, inner_radius, 0, -1)
        mask = self._apply_annulus_notch(
            mask, center=center, notch_angle_deg=notch_angle_deg
        )
        return mask, {
            "mode": str(self.calibration_mode),
            "offset": float(selected_offset),
            "band": float(selected_band),
            "score": float(selected_score),
            "samples": int(selected_samples),
            "center_x": float(center[0]),
            "center_y": float(center[1]),
            "inner_radius_px": float(inner_radius),
            "outer_radius_px": float(outer_radius),
        }

    def _estimate_annulus_notch_angle_deg(
        self,
        zone_id: Optional[int],
        center: Tuple[int, int],
        orientation: Optional[float],
        spray_info: Optional[Dict],
        roi_bbox: Tuple[int, int, int, int],
    ) -> float:
        """Estimate the notch angle (deg) opposite the spray direction from spray polygon/lines, falling back to the last/stable angle."""
        estimate = None
        roi_x, roi_y, _, _ = roi_bbox
        cx, cy = float(center[0]), float(center[1])
        spray_vectors = []
        if spray_info is not None:
            polygon = spray_info.get("roi_polygon")
            if polygon is not None:
                poly = np.asarray(polygon, dtype=np.float32)
                if poly.ndim == 2 and poly.shape[0] >= 4 and poly.shape[1] == 2:
                    local_poly = poly.copy()
                    local_poly[:, 0] -= float(roi_x)
                    local_poly[:, 1] -= float(roi_y)
                    # Primary spray direction estimate: center -> ROI polygon centroid.
                    centroid_x = float(np.mean(local_poly[:, 0]))
                    centroid_y = float(np.mean(local_poly[:, 1]))
                    vx = centroid_x - cx
                    vy = centroid_y - cy
                    norm = float(np.hypot(vx, vy))
                    if norm > 1e-6:
                        # Heavier weight for polygon centroid to avoid line-noise flips.
                        for _ in range(3):
                            spray_vectors.append((vx / norm, vy / norm))

            if not spray_vectors:
                lines = spray_info.get("lines") or []
                for p1, p2 in lines:
                    x1 = float(p1[0]) - float(roi_x)
                    y1 = float(p1[1]) - float(roi_y)
                    x2 = float(p2[0]) - float(roi_x)
                    y2 = float(p2[1]) - float(roi_y)
                    d1 = float(np.hypot(x1 - cx, y1 - cy))
                    d2 = float(np.hypot(x2 - cx, y2 - cy))
                    if d1 >= d2:
                        vx = x1 - cx
                        vy = y1 - cy
                    else:
                        vx = x2 - cx
                        vy = y2 - cy
                    norm = float(np.hypot(vx, vy))
                    if norm > 1e-6:
                        spray_vectors.append((vx / norm, vy / norm))

        if spray_vectors:
            sx = float(sum(v[0] for v in spray_vectors))
            sy = float(sum(v[1] for v in spray_vectors))
            if abs(sx) > 1e-6 or abs(sy) > 1e-6:
                spray_deg = float(np.degrees(np.arctan2(sy, sx)))
                body_deg = spray_deg + 180.0
                estimate = float((body_deg + 360.0) % 360.0)

        # Do not trust nozzle orientation here; if spray is missing, keep previous/stable notch.
        if estimate is None and zone_id is not None:
            prev = self.annulus_notch_angle_state.get(int(zone_id))
            if prev is not None:
                estimate = float(prev)
        if estimate is None:
            estimate = float(
                (float(self.annulus_notch_fallback_angle_deg) + 360.0) % 360.0
            )
        if self.annulus_notch_reverse:
            estimate = float((float(estimate) + 180.0) % 360.0)
        return self._stabilize_notch_angle(zone_id, estimate)

    def _stabilize_notch_angle(self, zone_id: Optional[int], angle_deg: float) -> float:
        """Smooth the notch angle per zone with a turn-rate cap and circular EMA, returning the stabilized angle in degrees."""
        if zone_id is None:
            return float((float(angle_deg) + 360.0) % 360.0)
        zone_key = int(zone_id)
        prev = self.annulus_notch_angle_state.get(zone_key)
        current = float((float(angle_deg) + 360.0) % 360.0)
        if prev is None:
            self.annulus_notch_angle_state[zone_key] = current
            return current

        delta = ((current - float(prev) + 540.0) % 360.0) - 180.0
        max_turn = max(1.0, float(self.annulus_notch_max_turn_deg))
        delta = float(np.clip(delta, -max_turn, max_turn))
        limited = float((float(prev) + delta + 360.0) % 360.0)
        alpha = float(np.clip(self.annulus_notch_angle_ema_alpha, 0.01, 1.0))
        mixed = (
            (1.0 - alpha)
            * np.array(
                [np.cos(np.radians(prev)), np.sin(np.radians(prev))], dtype=np.float32
            )
        ) + (
            alpha
            * np.array(
                [np.cos(np.radians(limited)), np.sin(np.radians(limited))],
                dtype=np.float32,
            )
        )
        smoothed = float(
            (np.degrees(np.arctan2(float(mixed[1]), float(mixed[0]))) + 360.0) % 360.0
        )
        self.annulus_notch_angle_state[zone_key] = smoothed
        return smoothed

    def _apply_annulus_notch(
        self,
        annulus_mask: np.ndarray,
        center: Tuple[int, int],
        notch_angle_deg: Optional[float],
    ) -> np.ndarray:
        """Zero out an angular wedge (the notch) of the annulus mask centered on notch_angle_deg; return the modified mask."""
        if annulus_mask is None or annulus_mask.size == 0:
            return annulus_mask
        if not self.annulus_notch_enabled:
            return annulus_mask

        frac = float(np.clip(self.annulus_notch_fraction, 0.0, 0.5))
        if frac <= 1e-6:
            return annulus_mask
        notch_span = 360.0 * frac
        if notch_span <= 1e-3:
            return annulus_mask

        h, w = annulus_mask.shape[:2]
        cx, cy = float(center[0]), float(center[1])
        yy, xx = self._get_index_grid((h, w))
        ang = (np.degrees(np.arctan2(yy - cy, xx - cx)) + 360.0) % 360.0
        center_deg = (
            float((float(notch_angle_deg) + 360.0) % 360.0)
            if notch_angle_deg is not None
            else 0.0
        )
        delta = ((ang - center_deg + 540.0) % 360.0) - 180.0
        remove = np.abs(delta) <= (0.5 * notch_span)
        out = annulus_mask.copy()
        out[remove] = 0
        return out

    def _resolve_auto_annulus_scales(
        self,
        roi: np.ndarray,
        center: Tuple[int, int],
        head_radius_px: float,
        zone_id: Optional[int],
    ) -> Tuple[float, float, float, int]:
        """Auto-tune the annulus offset/band by scoring candidate rings on edge/contrast/temporal cues; return smoothed scales, score, samples."""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, self.edge_threshold_low, self.edge_threshold_high)
        grad_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        grad_norm = grad_mag / float(np.max(grad_mag) + 1e-6)

        h, w = edges.shape[:2]
        yy, xx = self._get_index_grid((h, w))
        cx, cy = float(center[0]), float(center[1])
        rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        head_r = max(1.0, float(head_radius_px))

        offset_min = float(np.clip(self.auto_offset_min, 0.0, 0.60))
        offset_max = float(np.clip(self.auto_offset_max, offset_min + 1e-6, 0.85))
        offset_step = max(0.01, float(self.auto_offset_step))
        band_min = float(np.clip(self.auto_band_min, 0.05, 1.6))
        band_max = float(np.clip(self.auto_band_max, band_min + 1e-6, 2.0))
        band_step = max(0.02, float(self.auto_band_step))

        offset_candidates = np.arange(
            offset_min, offset_max + (0.5 * offset_step), offset_step, dtype=np.float32
        )
        band_candidates = np.arange(
            band_min, band_max + (0.5 * band_step), band_step, dtype=np.float32
        )
        if offset_candidates.size == 0:
            offset_candidates = np.asarray(
                [float(self.annulus_head_inner_offset_scale)], dtype=np.float32
            )
        if band_candidates.size == 0:
            band_candidates = np.asarray(
                [float(self.annulus_head_band_scale)], dtype=np.float32
            )

        base_off = float(
            np.clip(self.annulus_head_inner_offset_scale, offset_min, offset_max)
        )
        base_band = float(np.clip(self.annulus_head_band_scale, band_min, band_max))
        zone_key = int(zone_id) if zone_id is not None else 0
        state = self.annulus_tune_state.get(zone_key)
        if state is None:
            state = {
                "offset": float(base_off),
                "band": float(base_band),
                "score": 0.0,
                "samples": 0.0,
                "edge_ratio": 0.0,
                "confidence": 0.0,
                "low_conf_streak": 0.0,
                "best_profile_offset": float(base_off),
                "best_profile_band": float(base_band),
                "best_profile_score": -1e9,
            }
            self.annulus_tune_state[zone_key] = state
        history = self.annulus_param_history.get(zone_key)
        if history is None:
            history = deque(maxlen=max(3, int(self.auto_tune_history_window)))
            self.annulus_param_history[zone_key] = history

        prev_off = float(state.get("offset", base_off))
        prev_band = float(state.get("band", base_band))
        prev_edge_ratio = float(state.get("edge_ratio", 0.0))
        prior_w = max(0.0, float(self.auto_tune_prior_weight))
        edge_floor = max(0.0, float(self.auto_tune_edge_floor))
        contrast_w = max(0.0, float(self.auto_tune_contrast_weight))
        temporal_w = max(0.0, float(self.auto_tune_temporal_weight))
        flicker_w = max(0.0, float(self.auto_tune_flicker_penalty))
        off_span = max(1e-6, float(offset_max - offset_min))
        band_span = max(1e-6, float(band_max - band_min))

        def eval_score(
            off_val: float, band_val: float
        ) -> Optional[Tuple[float, float]]:
            """Score one candidate (offset, band) ring by edge/contrast/temporal terms minus prior/flicker costs; return (score, edge_ratio) or None."""
            inner_r = head_r * (1.0 + float(off_val))
            outer_r = head_r * (1.0 + float(off_val) + float(band_val))
            if outer_r <= inner_r + 1.0:
                return None
            ring = (rr >= inner_r) & (rr <= outer_r)
            pixels = int(np.count_nonzero(ring))
            if pixels < 24:
                return None
            edge_ratio = float(np.count_nonzero(edges[ring])) / float(max(1, pixels))
            if edge_ratio < edge_floor:
                return None
            prior_cost = abs(float(off_val) - base_off) + (
                0.65 * abs(float(band_val) - base_band)
            )
            contrast_score = float(np.mean(grad_norm[ring])) if pixels > 0 else 0.0
            temporal_delta = (abs(float(off_val) - prev_off) / off_span) + (
                abs(float(band_val) - prev_band) / band_span
            )
            temporal_score = float(np.clip(1.0 - (0.5 * temporal_delta), 0.0, 1.0))
            flicker_penalty = abs(edge_ratio - prev_edge_ratio)
            score = (
                edge_ratio
                + (contrast_w * contrast_score)
                + (temporal_w * temporal_score)
                - (prior_w * prior_cost)
                - (flicker_w * flicker_penalty)
            )
            return float(score), float(edge_ratio)

        def search_grid(
            offset_vals: np.ndarray, band_vals: np.ndarray
        ) -> Tuple[float, float, float, float, float]:
            best_s = -1e9
            second_s = -1e9
            best_o = float(prev_off)
            best_b = float(prev_band)
            best_edge = 0.0
            for off in offset_vals:
                off_v = float(off)
                for band in band_vals:
                    band_v = float(band)
                    eval_out = eval_score(off_v, band_v)
                    if eval_out is None:
                        continue
                    cand_s, cand_edge = eval_out
                    if cand_s > best_s:
                        second_s = best_s
                        best_s = cand_s
                        best_o = off_v
                        best_b = band_v
                        best_edge = cand_edge
                    elif cand_s > second_s:
                        second_s = cand_s
            return best_s, second_s, best_o, best_b, best_edge

        use_coarse = (
            bool(self.auto_tune_coarse_refine_enabled)
            and (offset_candidates.size >= 3)
            and (band_candidates.size >= 3)
        )
        if use_coarse:
            off_stride = max(1, int(self.auto_tune_coarse_stride_offset))
            band_stride = max(1, int(self.auto_tune_coarse_stride_band))
            coarse_offsets = offset_candidates[::off_stride]
            coarse_bands = band_candidates[::band_stride]
            if coarse_offsets.size == 0:
                coarse_offsets = offset_candidates
            if coarse_bands.size == 0:
                coarse_bands = band_candidates
            coarse_best_s, _, coarse_best_o, coarse_best_b, _ = search_grid(
                coarse_offsets, coarse_bands
            )
            if coarse_best_s > -1e8:
                off_half = max(1, off_stride)
                band_half = max(1, band_stride)
                off_min_ref = coarse_best_o - (off_half * offset_step)
                off_max_ref = coarse_best_o + (off_half * offset_step)
                band_min_ref = coarse_best_b - (band_half * band_step)
                band_max_ref = coarse_best_b + (band_half * band_step)
                refine_offsets = offset_candidates[
                    (offset_candidates >= off_min_ref)
                    & (offset_candidates <= off_max_ref)
                ]
                refine_bands = band_candidates[
                    (band_candidates >= band_min_ref)
                    & (band_candidates <= band_max_ref)
                ]
                if refine_offsets.size == 0:
                    refine_offsets = offset_candidates
                if refine_bands.size == 0:
                    refine_bands = band_candidates
                best_score, second_score, best_off, best_band, best_edge_ratio = (
                    search_grid(refine_offsets, refine_bands)
                )
            else:
                best_score, second_score, best_off, best_band, best_edge_ratio = (
                    search_grid(offset_candidates, band_candidates)
                )
        else:
            best_score, second_score, best_off, best_band, best_edge_ratio = (
                search_grid(offset_candidates, band_candidates)
            )

        alpha = float(np.clip(self.auto_tune_ema_alpha, 0.02, 1.0))
        max_jump_off = max(0.005, float(self.auto_tune_max_jump_offset))
        max_jump_band = max(0.01, float(self.auto_tune_max_jump_band))
        confidence = 0.0
        if self.auto_tune_adaptive_enabled and best_score > -1e8:
            base_eval = eval_score(base_off, base_band)
            base_score = -1e9
            if base_eval is not None:
                base_score = float(base_eval[0])
            if not np.isfinite(second_score):
                second_score = -1e9
            margin = float(best_score - max(base_score, second_score))
            confidence = float(
                np.clip(
                    (margin - float(self.auto_tune_confidence_margin))
                    / float(self.auto_tune_confidence_scale),
                    0.0,
                    1.0,
                )
            )
            if confidence > 0.0:
                alpha_hi = max(alpha, float(self.auto_tune_alpha_max))
                alpha = ((1.0 - confidence) * alpha) + (confidence * alpha_hi)
                jump_boost = float(self.auto_tune_jump_boost)
                max_jump_off *= (1.0 - confidence) + (confidence * jump_boost)
                max_jump_band *= (1.0 - confidence) + (confidence * jump_boost)

        low_conf_streak = int(state.get("low_conf_streak", 0.0))
        if confidence < float(self.auto_tune_min_confidence):
            low_conf_streak += 1
        else:
            low_conf_streak = 0

        if low_conf_streak >= int(self.auto_tune_hold_low_conf_frames):
            # Fallback profile: use most reliable historical auto-fit instead of noisy current measurement.
            best_prof_score = float(state.get("best_profile_score", -1e9))
            if best_prof_score > -1e8:
                best_off = float(state.get("best_profile_offset", prev_off))
                best_band = float(state.get("best_profile_band", prev_band))
                best_score = float(best_prof_score)
                best_edge_ratio = float(state.get("edge_ratio", prev_edge_ratio))
            else:
                best_off = float(prev_off)
                best_band = float(prev_band)
                best_score = float(state.get("score", 0.0))
                best_edge_ratio = float(state.get("edge_ratio", prev_edge_ratio))

        history.append((float(best_off), float(best_band), float(best_score)))
        med_off = (
            float(np.median([v[0] for v in history])) if history else float(best_off)
        )
        med_band = (
            float(np.median([v[1] for v in history])) if history else float(best_band)
        )
        target_off = prev_off + float(
            np.clip(med_off - prev_off, -max_jump_off, max_jump_off)
        )
        target_band = prev_band + float(
            np.clip(med_band - prev_band, -max_jump_band, max_jump_band)
        )
        smooth_off = ((1.0 - alpha) * prev_off) + (alpha * target_off)
        smooth_band = ((1.0 - alpha) * prev_band) + (alpha * target_band)
        smooth_off = float(np.clip(smooth_off, offset_min, offset_max))
        smooth_band = float(np.clip(smooth_band, band_min, band_max))

        if confidence >= float(self.auto_tune_min_confidence) and best_score > float(
            state.get("best_profile_score", -1e9)
        ):
            state["best_profile_offset"] = float(best_off)
            state["best_profile_band"] = float(best_band)
            state["best_profile_score"] = float(best_score)

        state["offset"] = smooth_off
        state["band"] = smooth_band
        state["score"] = float(best_score if best_score > -1e8 else 0.0)
        state["edge_ratio"] = float(
            best_edge_ratio if best_score > -1e8 else prev_edge_ratio
        )
        state["confidence"] = float(confidence)
        state["low_conf_streak"] = float(max(0, low_conf_streak))
        state["samples"] = float(state.get("samples", 0.0) + 1.0)
        return smooth_off, smooth_band, float(state["score"]), int(state["samples"])

    def _build_spray_exclusion_mask(
        self,
        shape: Tuple[int, int],
        spray_info: Optional[Dict],
        roi_bbox: Tuple[int, int, int, int],
        annular_mask: Optional[np.ndarray] = None,
        annulus_meta: Optional[Dict] = None,
        center: Optional[Tuple[int, int]] = None,
        orientation: Optional[float] = None,
    ) -> np.ndarray:
        """Build the spray-exclusion mask from the spray ROI polygon, else an orientation ellipse, else a bottom strip; shaped to the annulus."""
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        roi_x, roi_y, _, _ = roi_bbox
        polygon = None if spray_info is None else spray_info.get("roi_polygon")

        if polygon is not None:
            polygon_array = np.asarray(polygon, dtype=np.int32)
            if (
                polygon_array.ndim == 2
                and polygon_array.shape[1] == 2
                and len(polygon_array) >= 3
            ):
                local_polygon = polygon_array.copy()
                local_polygon[:, 0] -= int(roi_x)
                local_polygon[:, 1] -= int(roi_y)
                cv2.fillPoly(mask, [local_polygon], 255)
                return self._shape_spray_exclusion(
                    mask, annular_mask, annulus_meta=annulus_meta
                )

        if center is not None and orientation is not None:
            ori_mask = np.zeros((h, w), dtype=np.uint8)
            cx, cy = int(center[0]), int(center[1])
            axis_len = max(8, int(round(0.42 * min(h, w))))
            axis_w = max(4, int(round(0.14 * min(h, w))))
            cv2.ellipse(
                ori_mask,
                (cx, cy),
                (axis_len, axis_w),
                float(orientation),
                0,
                360,
                255,
                -1,
            )
            return self._shape_spray_exclusion(
                ori_mask, annular_mask, annulus_meta=annulus_meta
            )

        ignore_rows = int(round(h * self.tip_roi_bottom_ignore_ratio))
        if ignore_rows > 0:
            start_row = max(0, h - ignore_rows)
            mask[start_row:, :] = 255

        return self._shape_spray_exclusion(
            mask, annular_mask, annulus_meta=annulus_meta
        )

    def _shape_spray_exclusion(
        self,
        raw_mask: np.ndarray,
        annular_mask: Optional[np.ndarray],
        annulus_meta: Optional[Dict] = None,
    ) -> np.ndarray:
        """Constrain the raw spray mask to an inner radial band of the annulus (or its boundary), returning the final exclusion mask."""
        if raw_mask is None:
            if annular_mask is not None and annular_mask.size > 0:
                return np.zeros_like(annular_mask, dtype=np.uint8)
            return np.zeros((1, 1), dtype=np.uint8)
        if raw_mask.size == 0:
            return np.zeros_like(raw_mask, dtype=np.uint8)
        if annular_mask is None or annular_mask.size == 0:
            return raw_mask

        def compute_band_px(meta: Dict) -> float:
            """Return the spray-exclusion band width in px as a clamped fraction of the annulus thickness."""
            inner_r = float(meta.get("inner_radius_px", 0.0))
            outer_r = float(meta.get("outer_radius_px", inner_r))
            annulus_thickness = max(1.0, float(outer_r - inner_r))
            band = annulus_thickness * float(self.spray_exclusion_band_ratio_of_annulus)
            band = max(float(self.spray_exclusion_min_band_px), float(band))
            band = min(float(self.spray_exclusion_max_band_px), float(band))
            return max(1.0, band)

        if self.spray_exclusion_from_inner_circle and isinstance(annulus_meta, dict):
            cx = float(annulus_meta.get("center_x", 0.0))
            cy = float(annulus_meta.get("center_y", 0.0))
            inner_r = float(annulus_meta.get("inner_radius_px", 0.0))
            outer_r = float(annulus_meta.get("outer_radius_px", inner_r))
            h, w = annular_mask.shape[:2]
            yy, xx = self._get_index_grid((h, w))
            rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            grow_px = compute_band_px(annulus_meta)
            stop_r = min(float(outer_r), float(inner_r + grow_px))
            radial_band = np.zeros_like(annular_mask, dtype=np.uint8)
            radial_band[(rr >= inner_r) & (rr <= stop_r)] = 255
            return cv2.bitwise_and(radial_band, annular_mask)

        mask_work = raw_mask.copy()
        blur_k = max(1, int(self.spray_exclusion_blur_ksize))
        if blur_k % 2 == 0:
            blur_k += 1
        if blur_k >= 3:
            smooth = cv2.GaussianBlur(mask_work, (blur_k, blur_k), 0)
            _, mask_work = cv2.threshold(smooth, 1, 255, cv2.THRESH_BINARY)

        morph_iter = max(1, int(self.spray_exclusion_morph_iterations))
        morph_kernel = self._get_ellipse_kernel(3)
        mask_work = cv2.morphologyEx(
            mask_work, cv2.MORPH_CLOSE, morph_kernel, iterations=morph_iter
        )

        outer_kernel = self._get_ellipse_kernel(int(self.analysis_ring_dilate_px))
        ring_outer = cv2.dilate(annular_mask, outer_kernel, iterations=1)

        inner_kernel = self._get_ellipse_kernel(
            int(self.spray_exclusion_inner_erode_px)
        )
        ring_inner = cv2.erode(annular_mask, inner_kernel, iterations=1)

        boundary_band = cv2.bitwise_and(ring_outer, cv2.bitwise_not(ring_inner))
        if isinstance(annulus_meta, dict):
            cx = float(annulus_meta.get("center_x", 0.0))
            cy = float(annulus_meta.get("center_y", 0.0))
            inner_r = float(annulus_meta.get("inner_radius_px", 0.0))
            outer_r = float(annulus_meta.get("outer_radius_px", inner_r))
            if outer_r > inner_r + 1e-6:
                grow_px = compute_band_px(annulus_meta)
                stop_r = min(float(outer_r), float(inner_r + grow_px))
                h, w = annular_mask.shape[:2]
                yy, xx = self._get_index_grid((h, w))
                rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
                boundary_band = np.zeros_like(annular_mask, dtype=np.uint8)
                boundary_band[(rr >= inner_r) & (rr <= stop_r)] = 255
                boundary_band = cv2.bitwise_and(boundary_band, annular_mask)
        if self.spray_exclusion_cover_analysis:
            boundary_band = cv2.bitwise_or(boundary_band, annular_mask)

        return cv2.bitwise_and(mask_work, boundary_band)

    def _detect_edges(self, roi: np.ndarray) -> np.ndarray:
        """Return a Canny edge mask of the ROI after grayscale conversion and Gaussian blur."""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, self.edge_threshold_low, self.edge_threshold_high)
        return edges

    def _analyze_texture(self, roi: np.ndarray, mask: np.ndarray) -> float:
        """Return the Laplacian variance (texture/sharpness score) of the ROI within the masked region."""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
        laplacian = cv2.Laplacian(masked_gray, cv2.CV_64F)
        valid_pixels = laplacian[mask > 0]
        variance = valid_pixels.var() if valid_pixels.size > 0 else 0.0
        return float(variance)

    def _build_inner_anchor_mask(
        self,
        roi_mask: np.ndarray,
        annulus_meta: Optional[Dict] = None,
    ) -> Optional[np.ndarray]:
        """Build a mask of the inner radial band of the annulus used to anchor buildup components; None if unavailable/empty."""
        if roi_mask is None or roi_mask.size == 0:
            return None
        if not isinstance(annulus_meta, dict):
            return None
        inner_r = float(annulus_meta.get("inner_radius_px", 0.0))
        outer_r = float(annulus_meta.get("outer_radius_px", inner_r))
        cx = float(annulus_meta.get("center_x", 0.0))
        cy = float(annulus_meta.get("center_y", 0.0))
        if outer_r <= (inner_r + 1e-6):
            return None

        annulus_thickness = max(1.0, float(outer_r - inner_r))
        band_px = max(
            1.0,
            annulus_thickness * float(self.buildup_component_inner_anchor_band_ratio),
        )
        stop_r = min(float(outer_r), float(inner_r + band_px))
        h, w = roi_mask.shape[:2]
        yy, xx = self._get_index_grid((h, w))
        rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        anchor = np.zeros_like(roi_mask, dtype=np.uint8)
        anchor[(rr >= inner_r) & (rr <= stop_r)] = 255
        anchor = cv2.bitwise_and(anchor, roi_mask.astype(np.uint8))
        if int(np.count_nonzero(anchor)) <= 0:
            return None
        return anchor

    def _detect_buildup_regions(
        self,
        edge_mask: np.ndarray,
        roi_mask: np.ndarray,
        annulus_meta: Optional[Dict] = None,
    ) -> np.ndarray:
        """Morph-close/dilate edges within the ROI and keep only connected components passing area and inner-anchor filters."""
        closed = cv2.morphologyEx(
            edge_mask,
            cv2.MORPH_CLOSE,
            self.kernel,
            iterations=self.morph_iterations,
        )
        dilated = cv2.dilate(closed, self.kernel, iterations=1)
        buildup_mask = cv2.bitwise_and(dilated, dilated, mask=roi_mask)
        if not self.buildup_component_filter_enabled:
            return buildup_mask
        if buildup_mask is None or buildup_mask.size == 0:
            return buildup_mask

        source_bin = (buildup_mask > 0).astype(np.uint8)
        if int(np.count_nonzero(source_bin)) <= 0:
            return buildup_mask

        anchor_mask = None
        if self.buildup_component_inner_anchor_required:
            anchor_mask = self._build_inner_anchor_mask(
                roi_mask=roi_mask, annulus_meta=annulus_meta
            )
        total_roi_pixels = int(np.count_nonzero(roi_mask > 0))
        max_comp_area_px = int(
            round(
                float(total_roi_pixels) * float(self.buildup_component_max_area_ratio)
            )
        )
        max_comp_area_px = max(max_comp_area_px, self.buildup_component_min_area_px)

        labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            source_bin, connectivity=8
        )
        filtered = np.zeros_like(buildup_mask, dtype=np.uint8)
        for label_id in range(1, int(labels_count)):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < int(self.buildup_component_min_area_px):
                continue
            if area > max_comp_area_px:
                continue
            comp = labels == label_id
            if anchor_mask is not None:
                overlap = int(np.count_nonzero(comp & (anchor_mask > 0)))
                overlap_ratio = float(overlap) / float(max(1, area))
                if overlap_ratio < float(
                    self.buildup_component_inner_anchor_min_overlap_ratio
                ):
                    continue
            filtered[comp] = 255
        return filtered

    def _calculate_buildup_percentage(
        self, buildup_mask: np.ndarray, roi_mask: np.ndarray
    ) -> float:
        """Return the fraction (0-1) of analysis-mask pixels covered by the buildup mask."""
        if roi_mask is None or np.sum(roi_mask) == 0:
            return 0.0

        buildup_pixels = np.sum(buildup_mask > 0)
        total_pixels = np.sum(roi_mask > 0)

        if total_pixels == 0:
            return 0.0

        percentage = buildup_pixels / total_pixels
        return float(percentage)

    def _determine_severity(self, buildup_percentage: float) -> str:
        """Map a buildup fraction to a severity label: none / low / medium / high."""
        if buildup_percentage < 0.05:
            return "none"
        if buildup_percentage < 0.15:
            return "low"
        if buildup_percentage < 0.30:
            return "medium"
        return "high"

    def _update_temporal_ema(
        self,
        zone_id: Optional[int],
        current_value: float,
        store: Dict[int, float],
    ) -> float:
        """Update and return the zone's EMA of a value in the given store, also appending it to the zone history."""
        if zone_id is None:
            return float(current_value)

        zone_key = int(zone_id)
        previous = store.get(zone_key)
        if previous is None:
            store[zone_key] = float(current_value)
        else:
            store[zone_key] = (self.ema_alpha * float(current_value)) + (
                (1.0 - self.ema_alpha) * float(previous)
            )

        history = self.history.setdefault(zone_key, deque(maxlen=self.max_history))
        if store is self.ema_percentage:
            history.append({"buildup_percentage": float(current_value)})
        elif store is self.ema_diff:
            if history:
                history[-1]["diff_mean"] = float(current_value)
            else:
                history.append(
                    {"buildup_percentage": 0.0, "diff_mean": float(current_value)}
                )
        return float(store[zone_key])

    def _empty_result(self) -> Dict:
        """Return the default no-detection result dict (no buildup, null masks, neutral severity/threshold fields)."""
        return {
            "has_buildup": False,
            "buildup_percentage": 0.0,
            "buildup_mask": None,
            "texture_score": 0.0,
            "severity": "none",
            "roi": None,
            "roi_mask": None,
            "roi_bbox": None,
            "analysis_mask": None,
            "annular_mask": None,
            "spray_exclusion_mask": None,
            "calibration_mode": str(self.calibration_mode),
            "annulus_inner_offset_scale": float(self.annulus_head_inner_offset_scale),
            "annulus_band_scale": float(self.annulus_head_band_scale),
            "annulus_auto_score": 0.0,
            "annulus_auto_samples": 0,
            "temporal_enabled": self.temporal_enabled,
            "has_baseline": False,
            "diff_mean": None,
            "ssim": None,
            "ema_buildup_percentage": None,
            "ema_diff_mean": None,
            "temporal_increase": False,
            "temporal_score": None,
            "trend": None,
            "glcm_enabled": self.glcm_enabled,
            "glcm_contrast": None,
            "glcm_buildup": False,
            "baseline_locked": False,
            "baseline_lock_enabled": bool(self.baseline_lock_user_enabled),
            "diff_percentage": 0.0,
            "diff_trigger": False,
            "diff_reliable": False,
            "diff_edge_overlap_ratio": 0.0,
            "effective_area_threshold": float(self.buildup_area_threshold),
        }

    def visualize_buildup(
        self, frame: np.ndarray, nozzle_info: Dict, buildup_info: Dict
    ) -> np.ndarray:
        """Overlay the detected buildup mask as a translucent color region on a copy of the frame; return it unchanged if no buildup."""
        output = frame.copy()
        bbox = buildup_info.get("roi_bbox", nozzle_info.get("bounding_box"))
        buildup_mask = buildup_info.get("buildup_mask")
        has_buildup = bool(buildup_info.get("has_buildup", False))
        if not has_buildup:
            return output

        if bbox is None:
            return output

        x, y, w, h = bbox
        x, y, w, h = int(x), int(y), int(w), int(h)

        height, width = frame.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, width - x)
        h = min(h, height - y)

        if w <= 0 or h <= 0:
            return output

        overlay = np.zeros_like(output)

        if has_buildup and buildup_mask is not None:
            if buildup_mask.shape[0] != h or buildup_mask.shape[1] != w:
                buildup_mask = cv2.resize(buildup_mask, (w, h))
            overlay[y : y + h, x : x + w][buildup_mask > 0] = [0, 120, 255]
        alpha = 0.30
        output = cv2.addWeighted(output, 1, overlay, alpha, 0)

        return output
