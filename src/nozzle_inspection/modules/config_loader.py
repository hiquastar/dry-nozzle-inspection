"""
Configuration loader for the nozzle inspection system.
"""

import os
from typing import Any, Dict, Optional

import yaml


class ConfigLoader:
    """
    Loads and manages configuration for the nozzle inspection system.
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize object."""
        if config_path is None:
            # Default to config/config.yaml relative to project root
            project_root = os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )
            )
            config_path = os.path.join(project_root, "config", "config.yaml")

        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load the YAML config (with BOM fallback) merged over defaults, returning defaults on error."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as file:
                config = yaml.safe_load(file)
                return self._deep_merge(
                    self._get_default_config(), config if config else {}
                )
        except UnicodeDecodeError:
            # Fallback for files saved with BOM or legacy editors.
            with open(self.config_path, "r", encoding="utf-8-sig") as file:
                config = yaml.safe_load(file)
                return self._deep_merge(
                    self._get_default_config(), config if config else {}
                )
        except FileNotFoundError:
            print(f"Warning: Config file not found at {self.config_path}")
            return self._get_default_config()
        except yaml.YAMLError as e:
            print(f"Error parsing config file: {e}")
            return self._get_default_config()

    def _deep_merge(
        self, base: Dict[str, Any], overrides: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recursively merge overrides into base, returning a new dict where nested dicts are combined."""
        merged = dict(base)
        for key, value in (overrides or {}).items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = self._deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    def _get_default_config(self) -> Dict[str, Any]:
        """Return the built-in default configuration dict used as the merge base and fallback."""
        return {
            "camera": {
                "source": 0,
                "width": 640,
                "height": 480,
                "fps": 30,
                "file_playback_fps_override": 0.0,
                "file_playback_fps_min": 1.0,
                "file_playback_fps_max": 120.0,
                "video_auto_loop": True,
            },
            "runtime_modes": {
                "enable_dyn": True,
                "enable_fix": True,
                "enable_manual": True,
                "startup_mode": "FIX",
                "start_in_video_only": True,
            },
            "nozzle_detection": {
                "min_contour_area": 500,
                "max_contour_area": 120000,
                "use_circle_detection": True,
                "grid_rows": 2,
                "grid_cols": 3,
                "dynamic_zoning_enabled": True,
                "dynamic_tracking_mode": "lock_then_track",
                "fullframe_center_min_contour_area": 220.0,
                "fullframe_center_max_contour_area": 120000.0,
                "dynamic_zone_margin_px": 28.0,
                "dynamic_zone_ema_alpha": 0.3,
                "dynamic_zone_min_size_px": 58,
                "dynamic_zone_max_size_ratio": 0.55,
                "dynamic_zone_box_scale": 0.62,
                "dynamic_zone_kalman_enabled": True,
                "dynamic_zone_kalman_process_noise": 0.02,
                "dynamic_zone_kalman_measurement_noise": 8.0,
                "dynamic_zone_max_center_jump_px": 36.0,
                "dynamic_zone_anti_overlap_enabled": True,
                "dynamic_zone_min_center_spacing_ratio": 0.85,
                "dynamic_zone_lost_hold_frames": 8,
                "dynamic_zone_reacquire_max_dist_px": 180.0,
                "dynamic_zone_bootstrap_search_scale": 0.80,
                "dynamic_zone_center_deadband_px": 12.0,
                "dynamic_zone_center_smooth_alpha": 0.18,
                "dynamic_zone_update_interval_frames": 2,
                "dynamic_zone_use_raw_for_reacquire_only": True,
                "dynamic_lock_min_votes": 6,
                "dynamic_lock_hold_frames": 18,
                "dynamic_lock_update_alpha": 0.08,
                "dynamic_lock_roi_scale": 0.72,
                "zone_duplicate_claim_distance_px": 95.0,
                "fullframe_center_topk": 24,
                "fullframe_center_diverse_min_dist_px": 180.0,
                "fullframe_center_var_threshold": 5000.0,
                "fullframe_center_canny_low_default": 50,
                "fullframe_center_canny_low_noisy": 30,
                "fullframe_center_canny_high": 150,
                "fullframe_center_dual_path_enabled": True,
                "fullframe_center_min_aspect": 0.10,
                "fullframe_center_max_aspect": 0.50,
                "fullframe_center_relaxed_min_aspect": 0.06,
                "fullframe_center_relaxed_max_aspect": 0.85,
                "fullframe_center_relax_center_ratio": 0.05,
                "fullframe_center_relax_max_aspect": 0.50,
                "dynamic_bootstrap_gate_enabled": False,
                "dynamic_bootstrap_min_bbox_area_ratio": 0.0012,
                "dynamic_bootstrap_border_margin_ratio": 0.05,
                "fullframe_rescue_enabled": True,
                "fullframe_rescue_min_fail_streak": 3,
                "fullframe_rescue_max_assign_distance_px": 260.0,
                "fullframe_rescue_claim_radius_px": 70.0,
                "ownership_mode": "hybrid",
                "ownership_center_max_dist_px": 220.0,
                "hide_undetected_zone_boxes": True,
                "show_dynamic_zone_debug_overlay": True,
                "use_blob_detection": True,
                "blob_min_threshold": 10.0,
                "blob_max_threshold": 220.0,
                "blob_threshold_step": 10.0,
                "blob_filter_by_area": True,
                "blob_min_area": 80.0,
                "blob_max_area": 50000.0,
                "blob_filter_by_circularity": True,
                "blob_min_circularity": 0.70,
                "blob_filter_by_convexity": False,
                "blob_min_convexity": 0.80,
                "blob_filter_by_inertia": False,
                "blob_min_inertia_ratio": 0.20,
                "dp": 1.2,
                "min_dist": 50,
                "param1": 50,
                "param2": 24,
                "min_radius": 10,
                "max_radius": 160,
                "head_stability_enabled": True,
                "head_stability_center_alpha": 0.24,
                "head_stability_radius_alpha": 0.20,
                "head_stability_max_center_jump_px": 16.0,
                "head_stability_max_radius_delta_ratio": 0.10,
                "head_stability_center_deadband_px": 1.5,
                "head_stability_radius_deadband_px": 1.0,
                "head_radius_consistency_weight": 2.4,
                "head_stability_strength": 0.60,
                "head_radius_history_size": 9,
                "head_radius_outlier_mad_scale": 3.0,
                "head_reliability_circle": 1.0,
                "head_reliability_circle_relaxed": 0.18,
                "head_reliability_circle_blob": 0.08,
                "head_reliability_contour": 0.70,
                "head_reliability_kalman_predict": 0.0,
                "head_radius_anchor_enabled": True,
                "head_radius_anchor_update_reliability_min": 0.82,
                "head_radius_anchor_alpha": 0.14,
                "head_radius_anchor_min_ratio": 0.95,
                "head_radius_anchor_max_ratio": 1.08,
                "head_radius_freeze_on_weak": True,
                "head_radius_weak_shrink_limit_ratio": 0.0,
                "head_radius_freeze_expand_on_weak": True,
                "head_radius_weak_expand_limit_ratio": 0.03,
                "head_geometry_user_lock_enabled": False,
                "head_geometry_user_lock_capture_reliability_min": 0.85,
                "confirmation_window": 8,
                "enter_confirmations": 4,
                "exit_rejections": 4,
                "max_position_jump_px": 45.0,
                "confirm_enter_trusted_only": True,
                "confirm_enter_require_kalman": True,
                "confirm_enter_allowed_base_methods": ["circle", "contour"],
                "wall_filter_enabled": True,
                "wall_filter_bottom_only": True,
                "wall_filter_center_x_ratio": 0.5,
                "wall_filter_center_y_ratio": 0.5,
                "wall_filter_inner_ratio": 0.70,
                "wall_filter_outer_ratio": 1.08,
                "wall_filter_reacquire_px": 85.0,
                "show_wall_filter_overlay": True,
                "adaptive_wall_filter_enabled": True,
                "wall_model_update_interval_frames": 4,
                "wall_model_ema_alpha": 0.18,
                "wall_model_band_base_px": 14.0,
                "wall_model_band_sigma_gain": 2.2,
                "wall_model_min_band_px": 10.0,
                "wall_model_max_band_ratio": 0.16,
                "wall_model_min_radius_ratio": 0.30,
                "wall_model_max_radius_ratio": 0.62,
                "wall_model_method": "contour_ring",
                "wall_model_canny_low": 40,
                "wall_model_canny_high": 120,
                "wall_model_contour_min_area_ratio": 0.02,
                "wall_model_contour_circularity_weight": 1.1,
                "wall_model_max_center_shift_ratio": 0.12,
                "wall_model_max_radius_delta_ratio": 0.10,
                "wall_model_hough_param1": 120.0,
                "wall_model_hough_param2": 32.0,
                "anchor_gate_enabled": True,
                "anchor_gate_debug_overlay": True,
                "anchor_gate_bootstrap_frames": 50,
                "anchor_gate_base_px": 48.0,
                "anchor_gate_radius_gain": 1.4,
                "anchor_gate_ema_alpha": 0.15,
                "anchor_gate_min_radius_tol": 8.0,
                "anchor_gate_radius_tol_ratio": 0.40,
                "anchor_gate_center_zone_relax": 1.9,
                "anchor_gate_reacquire_after_rejects": 12,
                "anchor_gate_reacquire_frames": 18,
                "quality_gate_enabled": True,
                "quality_gate_border_margin_ratio": 0.04,
                "quality_gate_contour_max_area_ratio": 0.18,
                "quality_gate_contour_min_area_ratio": 0.0005,
                "quality_gate_min_aspect_ratio": 0.45,
                "quality_gate_max_aspect_ratio": 2.2,
                "quality_gate_max_radius_ratio": 0.24,
                "center_object_enabled": True,
                "center_object_promote_score": 0.24,
                "center_object_min_area_ratio": 0.00035,
                "center_object_max_area_ratio": 0.30,
                "center_object_min_aspect_ratio": 0.55,
                "center_object_max_aspect_ratio": 3.8,
                "center_object_max_radius_ratio": 0.38,
                "center_object_hough_dp": 1.2,
                "center_object_hough_param1": 110.0,
                "center_object_hough_param2": 18.0,
                "single_primary_zone_mode": True,
                "single_primary_zone_min_score": 8.0,
                "origin_big_nozzle_enabled": True,
                "origin_x_ratio": 0.50,
                "origin_y_ratio": 0.42,
                "origin_gate_x_ratio": 0.24,
                "origin_roi_half_width_ratio": 0.30,
                "origin_roi_top_ratio": 0.00,
                "origin_roi_bottom_ratio": 0.72,
                "origin_min_height_ratio": 0.12,
                "origin_min_area_ratio": 0.0025,
                "origin_max_area_ratio": 0.26,
                "origin_min_aspect_ratio": 0.16,
                "origin_max_aspect_ratio": 1.15,
                "origin_min_score": 0.18,
                "origin_tip_min_y_ratio": 0.12,
                "origin_tip_max_y_ratio": 0.48,
                "origin_max_bbox_width_ratio": 0.28,
                "origin_max_bbox_height_ratio": 0.58,
                "origin_shape_ema_alpha": 0.28,
                "origin_disable_deviation_warnings": True,
                "origin_tip_step_px": 26.0,
                "origin_bbox_xy_step_px": 24.0,
                "origin_bbox_growth_limit_ratio": 0.14,
                "spray_cadence_weak_frames": 2,
                "spray_cadence_confirming_frames": 2,
            },
            "buildup_detection": {
                "edge_threshold_low": 50,
                "edge_threshold_high": 150,
                "kernel_size": 5,
                "morph_iterations": 2,
                "buildup_area_threshold": 0.07,
                "buildup_area_hard_floor": True,
                "buildup_require_strong_detection": False,
                "buildup_weak_override_ratio": 0.10,
                "buildup_weak_threshold_scale": 1.25,
                "glcm_min_buildup_ratio": 0.006,
                "temporal_min_buildup_ratio": 0.010,
                "buildup_yes_hold_frames": 6,
                "annulus_head_outer_only": True,
                "annulus_head_inner_offset_min_scale": -0.35,
                "annulus_head_inner_offset_max_scale": 0.50,
                "annulus_head_inner_offset_scale": 0.01,
                "annulus_head_band_scale": 1.02,
                "annulus_notch_enabled": True,
                "annulus_notch_fraction": 0.25,
                "annulus_notch_fallback_angle_deg": 90.0,
                "annulus_notch_reverse": False,
                "annulus_notch_angle_ema_alpha": 0.30,
                "annulus_notch_max_turn_deg": 20.0,
                "calibration_mode": "manual",
                "auto_calibration_enabled": True,
                "auto_offset_min": 0.01,
                "auto_offset_max": 0.12,
                "auto_offset_step": 0.02,
                "auto_band_min": 0.40,
                "auto_band_max": 0.80,
                "auto_band_step": 0.05,
                "auto_tune_ema_alpha": 0.22,
                "auto_tune_prior_weight": 0.18,
                "auto_tune_edge_floor": 0.002,
                "auto_tune_max_jump_offset": 0.06,
                "auto_tune_max_jump_band": 0.12,
                "auto_tune_adaptive_enabled": True,
                "auto_tune_confidence_margin": 0.0015,
                "auto_tune_confidence_scale": 0.010,
                "auto_tune_alpha_max": 0.65,
                "auto_tune_jump_boost": 2.2,
                "auto_tune_min_confidence": 0.20,
                "auto_tune_hold_low_conf_frames": 4,
                "auto_tune_history_window": 11,
                "auto_tune_contrast_weight": 0.35,
                "auto_tune_temporal_weight": 0.30,
                "auto_tune_flicker_penalty": 0.10,
                "auto_tune_coarse_refine_enabled": True,
                "auto_tune_coarse_stride_offset": 2,
                "auto_tune_coarse_stride_band": 2,
                "calib_manual_offset_step": 0.01,
                "calib_manual_band_step": 0.02,
                "spray_line_thickness_px": 10,
                "analysis_ring_dilate_px": 6,
                "spray_exclusion_band_ratio_of_annulus": 0.27,
                "spray_exclusion_min_band_px": 1,
                "spray_exclusion_max_band_px": 24,
                "spray_exclusion_from_inner_circle": True,
                "spray_exclusion_cover_analysis": True,
                "spray_exclusion_inner_erode_px": 3,
                "spray_exclusion_blur_ksize": 5,
                "spray_exclusion_morph_iterations": 2,
                "calib_spray_line_step_px": 1,
                "calib_spray_ring_step_px": 1,
                "calib_spray_band_ratio_step": 0.01,
                "baseline_lock_enabled": False,
                "baseline_lock_user_enabled": False,
                "baseline_lock_max_buildup_ratio": 0.003,
                "baseline_lock_required_stable_frames": 35,
                "baseline_lock_max_center_jitter_px": 2.0,
                "baseline_lock_max_radius_jitter_ratio": 0.06,
                "baseline_lock_allow_refresh": False,
                "baseline_lock_require_real_detection": True,
                "baseline_lock_require_kalman_smoothed": True,
                "baseline_lock_require_strong_detection": True,
                "baseline_lock_allow_relaxed_methods": False,
                "baseline_lock_allow_blob_method": False,
                "baseline_lock_allow_rescue_methods": False,
                "baseline_lock_radius_anchor_enabled": True,
                "baseline_lock_radius_anchor_min_ratio": 0.95,
                "baseline_lock_radius_anchor_max_ratio": 1.30,
                "baseline_lock_radius_anchor_on_weak_only": True,
                "buildup_diff_enabled": True,
                "buildup_diff_threshold": 14,
                "buildup_diff_blur_kernel": 3,
                "buildup_diff_min_ratio": 0.006,
                "buildup_diff_use_for_mask": True,
                "buildup_diff_requires_strong_detection": False,
                "buildup_diff_allow_circle_relaxed": True,
                "buildup_diff_allow_circle_blob": True,
                "buildup_diff_allow_kalman_predict": True,
                "buildup_diff_edge_overlap_required": True,
                "buildup_diff_edge_overlap_kernel": 3,
                "buildup_diff_edge_overlap_min_ratio": 0.25,
                "buildup_component_filter_enabled": True,
                "buildup_component_min_area_px": 12,
                "buildup_component_max_area_ratio": 1.0,
                "buildup_component_inner_anchor_required": True,
                "buildup_component_inner_anchor_band_ratio": 0.30,
                "buildup_component_inner_anchor_min_overlap_ratio": 0.08,
                "buildup_cadence_weak_frames": 2,
                "buildup_cadence_confirming_frames": 2,
            },
            "warning": {
                "position_deviation_threshold": 20,
                "orientation_deviation_threshold": 15,
                "buildup_warning_level": 0.07,
                "buildup_critical_level": 0.2,
            },
            "display": {
                "show_video": True,
                "show_detection_overlay": True,
                "show_metrics": True,
                "strict_input_output_size": True,
                "lock_window_size": True,
                "window_scale_percent": 100,
                "window_name": "Nozzle Inspection",
                "normal_color": [0, 255, 0],
                "warning_color": [0, 165, 255],
                "critical_color": [0, 0, 255],
            },
            "logging": {
                "enabled": True,
                "log_file": "logs/inspection.log",
                "log_level": "INFO",
            },
            "recording": {
                "auto_record": True,
                "codec_preference": ["avc1", "mp4v", "XVID", "MJPG"],
                "output_fps_override": 0.0,
            },
            "dl": {
                "spray_override_enabled": True,
                "spray_override_max_cv_diff_deg": 10.0,
            },
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Look up a dotted key path in the config, returning default if any segment is missing."""
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_section(self, section: str) -> Dict[str, Any]:
        """Return the top-level config section as a dict, or an empty dict if absent."""
        return self.config.get(section, {})
