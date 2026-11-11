"""
Nozzle detection module for identifying nozzle position and orientation.
"""

from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class NozzleDetector:
    """
    Detects nozzles in camera frames and determines their position and orientation.
    """

    def __init__(self, config: Dict):
        """Load detection parameters from config and initialize Kalman filter and cached blob detector."""
        self.min_contour_area = config.get("min_contour_area", 500)
        self.max_contour_area = config.get("max_contour_area", 50000)
        self.use_circle_detection = config.get("use_circle_detection", True)
        self.use_blob_detection = bool(config.get("use_blob_detection", True))
        self.dp = config.get("dp", 1.2)
        self.min_dist = config.get("min_dist", 50)
        self.param1 = config.get("param1", 50)
        self.param2 = config.get("param2", 30)
        self.min_radius = config.get("min_radius", 10)
        self.max_radius = config.get("max_radius", 100)
        self.relaxed_fallback_enabled = bool(
            config.get("relaxed_fallback_enabled", True)
        )
        self.relaxed_param1_scale = float(config.get("relaxed_param1_scale", 0.85))
        self.relaxed_param2_scale = float(config.get("relaxed_param2_scale", 0.75))
        self.relaxed_min_radius_scale = float(
            config.get("relaxed_min_radius_scale", 0.80)
        )
        self.relaxed_contour_area_scale = float(
            config.get("relaxed_contour_area_scale", 0.70)
        )
        self.blob_min_threshold = float(config.get("blob_min_threshold", 10.0))
        self.blob_max_threshold = float(config.get("blob_max_threshold", 220.0))
        self.blob_threshold_step = float(config.get("blob_threshold_step", 10.0))
        self.blob_filter_by_area = bool(config.get("blob_filter_by_area", True))
        self.blob_min_area = float(config.get("blob_min_area", 80.0))
        self.blob_max_area = float(config.get("blob_max_area", 50000.0))
        self.blob_filter_by_circularity = bool(
            config.get("blob_filter_by_circularity", True)
        )
        self.blob_min_circularity = float(config.get("blob_min_circularity", 0.70))
        self.blob_filter_by_convexity = bool(
            config.get("blob_filter_by_convexity", False)
        )
        self.blob_min_convexity = float(config.get("blob_min_convexity", 0.80))
        self.blob_filter_by_inertia = bool(config.get("blob_filter_by_inertia", False))
        self.blob_min_inertia_ratio = float(config.get("blob_min_inertia_ratio", 0.20))
        # Build blob detector once to avoid per-frame allocation
        if self.use_blob_detection:
            _bp = cv2.SimpleBlobDetector.Params()
            _bp.minThreshold = float(self.blob_min_threshold)
            _bp.maxThreshold = float(self.blob_max_threshold)
            _bp.thresholdStep = float(self.blob_threshold_step)
            _bp.filterByArea = bool(self.blob_filter_by_area)
            _bp.minArea = float(self.blob_min_area)
            _bp.maxArea = float(self.blob_max_area)
            _bp.filterByCircularity = bool(self.blob_filter_by_circularity)
            _bp.minCircularity = float(self.blob_min_circularity)
            _bp.filterByConvexity = bool(self.blob_filter_by_convexity)
            _bp.minConvexity = float(self.blob_min_convexity)
            _bp.filterByInertia = bool(self.blob_filter_by_inertia)
            _bp.minInertiaRatio = float(self.blob_min_inertia_ratio)
            self._blob_detector = cv2.SimpleBlobDetector.create(_bp)
        else:
            self._blob_detector = None
        self.use_color_filter = config.get("use_color_filter", False)
        self.hsv_lower = np.array(config.get("hsv_lower", [0, 0, 50]))
        self.hsv_upper = np.array(config.get("hsv_upper", [180, 50, 255]))

        # Head geometry stabilization (center/radius/bbox)
        self.head_stability_enabled = bool(config.get("head_stability_enabled", True))
        self.head_stability_center_alpha = float(
            config.get("head_stability_center_alpha", 0.24)
        )
        self.head_stability_radius_alpha = float(
            config.get("head_stability_radius_alpha", 0.20)
        )
        self.head_stability_max_center_jump_px = float(
            config.get("head_stability_max_center_jump_px", 16.0)
        )
        self.head_stability_max_radius_delta_ratio = float(
            config.get("head_stability_max_radius_delta_ratio", 0.20)
        )
        self.head_stability_center_deadband_px = float(
            config.get("head_stability_center_deadband_px", 1.5)
        )
        self.head_stability_radius_deadband_px = float(
            config.get("head_stability_radius_deadband_px", 1.0)
        )
        self.head_radius_consistency_weight = float(
            config.get("head_radius_consistency_weight", 2.4)
        )
        self.head_stability_strength = float(
            np.clip(config.get("head_stability_strength", 0.0), 0.0, 1.0)
        )
        self.head_radius_history_size = int(
            max(3, config.get("head_radius_history_size", 9))
        )
        self.head_radius_outlier_mad_scale = float(
            max(1.0, config.get("head_radius_outlier_mad_scale", 3.0))
        )
        self.head_reliability_circle = float(
            np.clip(config.get("head_reliability_circle", 1.0), 0.0, 1.0)
        )
        self.head_reliability_circle_relaxed = float(
            np.clip(config.get("head_reliability_circle_relaxed", 0.55), 0.0, 1.0)
        )
        self.head_reliability_circle_blob = float(
            np.clip(config.get("head_reliability_circle_blob", 0.45), 0.0, 1.0)
        )
        self.head_reliability_contour = float(
            np.clip(config.get("head_reliability_contour", 0.70), 0.0, 1.0)
        )
        self.head_reliability_kalman_predict = float(
            np.clip(config.get("head_reliability_kalman_predict", 0.0), 0.0, 1.0)
        )
        self.head_radius_anchor_enabled = bool(
            config.get("head_radius_anchor_enabled", True)
        )
        self.head_radius_anchor_update_reliability_min = float(
            np.clip(
                config.get("head_radius_anchor_update_reliability_min", 0.82), 0.0, 1.0
            )
        )
        self.head_radius_anchor_alpha = float(
            np.clip(config.get("head_radius_anchor_alpha", 0.14), 0.01, 1.0)
        )
        self.head_radius_anchor_min_ratio = float(
            np.clip(config.get("head_radius_anchor_min_ratio", 0.95), 0.10, 1.20)
        )
        self.head_radius_anchor_max_ratio = float(
            np.clip(config.get("head_radius_anchor_max_ratio", 1.08), 0.80, 2.00)
        )
        if self.head_radius_anchor_max_ratio < self.head_radius_anchor_min_ratio:
            self.head_radius_anchor_max_ratio = self.head_radius_anchor_min_ratio
        self.head_radius_freeze_on_weak = bool(
            config.get("head_radius_freeze_on_weak", True)
        )
        self.head_radius_weak_shrink_limit_ratio = float(
            np.clip(config.get("head_radius_weak_shrink_limit_ratio", 0.0), 0.0, 0.50)
        )
        self.head_radius_freeze_expand_on_weak = bool(
            config.get("head_radius_freeze_expand_on_weak", True)
        )
        self.head_radius_weak_expand_limit_ratio = float(
            np.clip(config.get("head_radius_weak_expand_limit_ratio", 0.03), 0.0, 0.50)
        )
        self.head_state = None
        self.head_radius_history = deque(maxlen=self.head_radius_history_size)
        self.head_radius_anchor = None

        # Optional Kalman smoothing / short-gap prediction
        self.kalman_enabled = bool(config.get("kalman_enabled", False))
        self.kalman_predict_on_miss = bool(config.get("kalman_predict_on_miss", True))
        self.kalman_max_predict_frames = int(config.get("kalman_max_predict_frames", 5))
        self.kalman_process_noise = float(config.get("kalman_process_noise", 0.03))
        self.kalman_measurement_noise = float(
            config.get("kalman_measurement_noise", 10.0)
        )
        self.kalman_error_cov = float(config.get("kalman_error_cov", 1.0))
        self.kalman = None
        self.kalman_initialized = False
        self.kalman_miss_count = 0
        self.last_orientation = None
        self.user_geometry_lock_enabled = bool(
            config.get("head_geometry_user_lock_enabled", False)
        )
        self.user_geometry_lock_capture_reliability_min = float(
            np.clip(
                config.get("head_geometry_user_lock_capture_reliability_min", 0.85),
                0.0,
                1.0,
            )
        )
        self.user_geometry_lock_state = None
        if self.kalman_enabled:
            self.kalman = cv2.KalmanFilter(4, 2)
            self.kalman.measurementMatrix = np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32
            )
            self.kalman.transitionMatrix = np.array(
                [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
            )
            self.kalman.processNoiseCov = (
                np.eye(4, dtype=np.float32) * self.kalman_process_noise
            )
            self.kalman.measurementNoiseCov = (
                np.eye(2, dtype=np.float32) * self.kalman_measurement_noise
            )
            self.kalman.errorCovPost = (
                np.eye(4, dtype=np.float32) * self.kalman_error_cov
            )

        # Store reference position for tracking
        self.reference_position = None
        self.reference_orientation = None
        self.last_circle_source = None

    def detect(self, frame: np.ndarray) -> Dict:
        """Run full detection pipeline on frame and return a dict with position, radius, orientation, and confidence info."""
        if frame is None or frame.size == 0:
            out = self._empty_result()
            out["reject_reason"] = "empty_frame"
            return out

        # Preprocess frame
        processed = self._preprocess(frame)

        # Detect circles using Hough Transform
        circles = None
        if self.use_circle_detection:
            circles = self._detect_circles(processed)

        # Detect contours
        contours = self._detect_contours(processed)

        # Find best nozzle candidate
        nozzle_info = self._find_best_nozzle(frame, contours, circles)
        if not nozzle_info.get("detected", False):
            nozzle_info["reject_reason"] = "no_circle_no_contour"
            nozzle_info["debug_details"] = {
                "num_contours": int(len(contours)),
                "num_circles": int(0 if circles is None else len(circles)),
            }

        nozzle_info = self._apply_kalman(nozzle_info)
        nozzle_info = self._stabilize_head_geometry(nozzle_info)
        nozzle_info = self._apply_user_geometry_lock(nozzle_info)

        # Update reference if this is first detection
        if nozzle_info.get("detected") and self.reference_position is None:
            self.reference_position = nozzle_info.get("position")
            self.reference_orientation = nozzle_info.get("orientation")

        return nozzle_info

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Convert frame to grayscale, apply Gaussian blur, and optionally mask by HSV color range."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply color filtering if enabled
        if self.use_color_filter:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
            blurred = cv2.bitwise_and(blurred, blurred, mask=mask)

        return blurred

    def _detect_circles(self, processed: np.ndarray) -> Optional[np.ndarray]:
        """Try Hough circle detection, fall back to relaxed parameters, then blob detection; return (N,3) int array [x,y,r] or None."""
        max_radius = max(0, int(self.max_radius))

        circles = cv2.HoughCircles(
            processed,
            cv2.HOUGH_GRADIENT,
            dp=self.dp,
            minDist=self.min_dist,
            param1=self.param1,
            param2=self.param2,
            minRadius=self.min_radius,
            maxRadius=max_radius,
        )

        if circles is not None:
            self.last_circle_source = "hough"
            circles = np.round(circles[0, :]).astype("int")
            return circles

        if self.relaxed_fallback_enabled:
            relaxed_param1 = int(
                max(30, round(float(self.param1) * float(self.relaxed_param1_scale)))
            )
            relaxed_param2 = int(
                max(10, round(float(self.param2) * float(self.relaxed_param2_scale)))
            )
            relaxed_min_radius = int(
                max(
                    1,
                    round(
                        float(self.min_radius) * float(self.relaxed_min_radius_scale)
                    ),
                )
            )
            circles = cv2.HoughCircles(
                processed,
                cv2.HOUGH_GRADIENT,
                dp=self.dp,
                minDist=self.min_dist,
                param1=relaxed_param1,
                param2=relaxed_param2,
                minRadius=relaxed_min_radius,
                maxRadius=max_radius,
            )
            if circles is not None:
                self.last_circle_source = "hough_relaxed"
                circles = np.round(circles[0, :]).astype("int")
                return circles

        if not self.use_blob_detection or self._blob_detector is None:
            self.last_circle_source = None
            return None

        keypoints = self._blob_detector.detect(processed)
        if not keypoints:
            self.last_circle_source = None
            return None

        self.last_circle_source = "blob"
        circles = np.array(
            [[kp.pt[0], kp.pt[1], max(1.0, 0.5 * kp.size)] for kp in keypoints],
            dtype=np.float32,
        )
        return np.round(circles).astype("int")

    def _detect_contours(self, processed: np.ndarray) -> List[np.ndarray]:
        """Run Canny edge detection and return contours filtered by area; retries with relaxed thresholds when nothing is found."""
        edges = cv2.Canny(processed, 50, 150)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if self.max_contour_area is None or self.max_contour_area <= 0:
            filtered_contours = [
                c for c in contours if cv2.contourArea(c) >= self.min_contour_area
            ]
        else:
            filtered_contours = [
                c
                for c in contours
                if self.min_contour_area <= cv2.contourArea(c) <= self.max_contour_area
            ]
        if filtered_contours or not self.relaxed_fallback_enabled:
            return filtered_contours

        edges_relaxed = cv2.Canny(processed, 35, 120)
        contours_relaxed, _ = cv2.findContours(
            edges_relaxed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        relaxed_min_area = max(
            20.0, float(self.min_contour_area) * float(self.relaxed_contour_area_scale)
        )
        if self.max_contour_area is None or self.max_contour_area <= 0:
            return [
                c for c in contours_relaxed if cv2.contourArea(c) >= relaxed_min_area
            ]
        return [
            c
            for c in contours_relaxed
            if relaxed_min_area <= cv2.contourArea(c) <= self.max_contour_area
        ]

    def _find_best_nozzle(
        self,
        frame: np.ndarray,
        contours: List[np.ndarray],
        circles: Optional[np.ndarray],
    ) -> Dict:
        """Pick the best nozzle candidate from circles and contours using a tracking-anchor cost function."""
        # Prioritize circle detection if available
        if circles is not None and len(circles) > 0:
            frame_h, frame_w = frame.shape[:2]
            max_reasonable_radius = int(min(frame_w, frame_h) * 0.48)

            candidate_circles = [c for c in circles if c[2] <= max_reasonable_radius]
            if not candidate_circles:
                candidate_circles = list(circles)

            contour_anchor = None
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x0, y0, w0, h0 = cv2.boundingRect(largest_contour)
                contour_anchor = (x0 + w0 // 2, y0 + h0 // 2)

            tracking_anchor = None
            prev_radius = None
            if isinstance(self.head_state, dict):
                sx = self.head_state.get("x")
                sy = self.head_state.get("y")
                sr = self.head_state.get("r")
                if sx is not None and sy is not None:
                    tracking_anchor = (float(sx), float(sy))
                if sr is not None:
                    prev_radius = float(sr)
            if len(self.head_radius_history) >= 3:
                try:
                    prev_radius = float(
                        np.median(
                            np.asarray(self.head_radius_history, dtype=np.float32)
                        )
                    )
                except Exception:
                    pass
            if tracking_anchor is None and self.reference_position is not None:
                tracking_anchor = (
                    float(self.reference_position[0]),
                    float(self.reference_position[1]),
                )

            if (
                tracking_anchor is not None
                or contour_anchor is not None
                or prev_radius is not None
            ):
                ref_x, ref_y = (
                    tracking_anchor if tracking_anchor is not None else (None, None)
                )
                anchor_x, anchor_y = (
                    contour_anchor if contour_anchor is not None else (None, None)
                )

                def circle_cost(c):
                    """Score a circle candidate by distance to tracking anchor, contour anchor, and previous radius."""
                    cx, cy, cr = float(c[0]), float(c[1]), float(c[2])
                    cost = 0.0
                    if ref_x is not None and ref_y is not None:
                        dx = cx - ref_x
                        dy = cy - ref_y
                        cost += (dx * dx) + (dy * dy)
                    if anchor_x is not None and anchor_y is not None:
                        dx = cx - float(anchor_x)
                        dy = cy - float(anchor_y)
                        cost += 0.70 * ((dx * dx) + (dy * dy))
                    if prev_radius is not None:
                        radius_delta = abs(cr - prev_radius)
                        radius_cost = radius_delta * max(
                            0.0, float(self.head_radius_consistency_weight)
                        )
                        cost += radius_cost * radius_cost
                    return cost

                x, y, r = min(candidate_circles, key=circle_cost)
            else:
                # HoughCircles usually returns strongest candidates first
                x, y, r = candidate_circles[0]

            # Calculate orientation (default to 0 for circles)
            orientation = 0.0

            # Create bounding box
            bbox = (x - r, y - r, 2 * r, 2 * r)

            return {
                "detected": True,
                "position": (int(x), int(y)),
                "orientation": orientation,
                "radius": int(r),
                "contour": None,
                "bounding_box": bbox,
                "detection_method": (
                    "circle_blob"
                    if self.last_circle_source == "blob"
                    else (
                        "circle_relaxed"
                        if self.last_circle_source == "hough_relaxed"
                        else "circle"
                    )
                ),
            }

        # Fall back to contour-based detection
        if contours:
            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Calculate center
            center_x = x + w // 2
            center_y = y + h // 2

            # Calculate orientation using fitted ellipse or moments
            orientation = self._calculate_orientation(largest_contour)

            # Estimate radius
            radius = max(w, h) // 2

            return {
                "detected": True,
                "position": (center_x, center_y),
                "orientation": orientation,
                "radius": radius,
                "contour": largest_contour,
                "bounding_box": (x, y, w, h),
                "detection_method": "contour",
            }

        return self._empty_result()

    def _calculate_orientation(self, contour: np.ndarray) -> float:
        """Estimate nozzle orientation in degrees via ellipse fitting, falling back to image moments."""
        try:
            # Fit an ellipse if possible
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                angle = ellipse[2]
                return angle
        except cv2.error:
            pass

        # Fall back to moments-based calculation
        moments = cv2.moments(contour)
        if moments["mu20"] != moments["mu02"]:
            angle = 0.5 * np.arctan2(
                2 * moments["mu11"], moments["mu20"] - moments["mu02"]
            )
            angle = np.degrees(angle)
            return angle

        return 0.0

    def _empty_result(self) -> Dict:
        """Return a dict representing a non-detection with all fields set to None/False."""
        return {
            "detected": False,
            "position": None,
            "orientation": None,
            "radius": None,
            "contour": None,
            "bounding_box": None,
            "detection_method": None,
            "is_prediction": False,
            "reject_reason": None,
            "debug_details": {},
        }

    @staticmethod
    def _clip_jump(
        prev_pt: Tuple[float, float], new_pt: Tuple[float, float], max_jump: float
    ) -> Tuple[float, float]:
        """Clip displacement from prev_pt to new_pt to at most max_jump pixels."""
        if prev_pt is None or max_jump <= 0.0:
            return float(new_pt[0]), float(new_pt[1])
        dx = float(new_pt[0]) - float(prev_pt[0])
        dy = float(new_pt[1]) - float(prev_pt[1])
        dist = float(np.hypot(dx, dy))
        if dist <= max_jump or dist <= 1e-6:
            return float(new_pt[0]), float(new_pt[1])
        scale = float(max_jump / dist)
        return float(prev_pt[0]) + (dx * scale), float(prev_pt[1]) + (dy * scale)

    @staticmethod
    def _radius_from_bbox(bbox) -> Optional[float]:
        """Estimate radius as half the longer side of a bounding box."""
        if bbox is None:
            return None
        try:
            _, _, bw, bh = bbox
            bw = max(1.0, float(bw))
            bh = max(1.0, float(bh))
            return 0.5 * max(bw, bh)
        except Exception:
            return None

    def _stabilize_bbox(
        self,
        bbox,
        detection_method: str,
        center: Tuple[float, float],
        stable_radius: float,
        measured_radius: float,
    ):
        """Recompute bounding box from a stabilized center and radius."""
        cx, cy = float(center[0]), float(center[1])
        sr = max(1.0, float(stable_radius))
        method = str(detection_method or "").strip().lower()
        if bbox is None:
            r = int(max(1, round(sr)))
            return int(round(cx - r)), int(round(cy - r)), int(2 * r), int(2 * r)

        try:
            _, _, bw, bh = [float(v) for v in bbox]
        except Exception:
            r = int(max(1, round(sr)))
            return int(round(cx - r)), int(round(cy - r)), int(2 * r), int(2 * r)

        bw = max(2.0, bw)
        bh = max(2.0, bh)
        aspect = min(bw, bh) / max(bw, bh)
        measured_r = max(1.0, float(measured_radius))
        scale = float(np.clip(sr / measured_r, 0.60, 1.70))
        if method.startswith("circle") or aspect >= 0.75:
            half_w = sr
            half_h = sr
        else:
            half_w = max(1.0, 0.5 * bw * scale)
            half_h = max(1.0, 0.5 * bh * scale)

        x = int(round(cx - half_w))
        y = int(round(cy - half_h))
        w = int(max(2, round(2.0 * half_w)))
        h = int(max(2, round(2.0 * half_h)))
        return x, y, w, h

    @staticmethod
    def _base_detection_method(detection_method: str) -> str:
        """Extract the base method name, stripping any '+kalman' suffix."""
        method = str(detection_method or "").strip().lower()
        if not method:
            return ""
        return method.split("+")[0].strip()

    def _measurement_reliability(self, nozzle_info: Dict) -> float:
        """Return a [0,1] reliability score for a detection based on its method."""
        if bool(nozzle_info.get("is_prediction", False)):
            return float(self.head_reliability_kalman_predict)
        method = self._base_detection_method(nozzle_info.get("detection_method", ""))
        if method == "circle":
            return float(self.head_reliability_circle)
        if method == "circle_relaxed":
            return float(self.head_reliability_circle_relaxed)
        if method == "circle_blob":
            return float(self.head_reliability_circle_blob)
        if method == "contour":
            return float(self.head_reliability_contour)
        if method == "kalman_predict":
            return float(self.head_reliability_kalman_predict)
        return float(self.head_reliability_contour)

    def _radius_outlier_filter(
        self, measured_radius: float, fallback_radius: float
    ) -> float:
        """Reject measured radius if it deviates beyond a MAD-based threshold; return fallback instead."""
        r = max(1.0, float(measured_radius))
        hist = list(self.head_radius_history)
        if len(hist) < 4:
            return r
        try:
            hist_np = np.asarray(hist, dtype=np.float32)
            med = float(np.median(hist_np))
            mad = float(np.median(np.abs(hist_np - med)))
        except Exception:
            return r
        robust_sigma = max(1.0, 1.4826 * mad)
        max_delta = float(self.head_radius_outlier_mad_scale) * robust_sigma
        if abs(r - med) > max_delta:
            return float(max(1.0, fallback_radius))
        return r

    def _effective_stability_params(
        self,
    ) -> Tuple[float, float, float, float, float, float]:
        # Single "strength" knob: 0.0 keeps base settings, 1.0 aggressively stabilizes.
        """Scale EMA alphas and jump limits by the head_stability_strength knob and return all six params."""
        s = float(np.clip(self.head_stability_strength, 0.0, 1.0))
        alpha_c = float(
            np.clip(self.head_stability_center_alpha * (1.0 - (0.75 * s)), 0.01, 1.0)
        )
        alpha_r = float(
            np.clip(self.head_stability_radius_alpha * (1.0 - (0.85 * s)), 0.01, 1.0)
        )
        max_jump = max(
            1.0, float(self.head_stability_max_center_jump_px) * (1.0 - (0.75 * s))
        )
        max_r_ratio = max(
            0.01, float(self.head_stability_max_radius_delta_ratio) * (1.0 - (0.85 * s))
        )
        center_deadband = max(
            0.0, float(self.head_stability_center_deadband_px) + (1.5 * s)
        )
        radius_deadband = max(
            0.0, float(self.head_stability_radius_deadband_px) + (1.0 * s)
        )
        return alpha_c, alpha_r, max_jump, max_r_ratio, center_deadband, radius_deadband

    def _update_radius_anchor(
        self, base_method: str, reliability: float, radius: float, is_prediction: bool
    ):
        """EMA-update the radius anchor from a strong, reliable measurement."""
        if (not self.head_radius_anchor_enabled) or is_prediction:
            return
        if base_method not in {"circle", "contour"}:
            return
        if float(reliability) < float(self.head_radius_anchor_update_reliability_min):
            return
        r = max(1.0, float(radius))
        if self.head_radius_anchor is None:
            self.head_radius_anchor = r
            return
        a = float(self.head_radius_anchor_alpha)
        self.head_radius_anchor = ((1.0 - a) * float(self.head_radius_anchor)) + (a * r)

    def _stabilize_head_geometry(self, nozzle_info: Dict) -> Dict:
        """Apply EMA smoothing, jump clipping, and anchor constraints to position and radius."""
        if not nozzle_info.get("detected", False):
            return nozzle_info

        pos = nozzle_info.get("position")
        if pos is None:
            return nozzle_info

        mx = float(pos[0])
        my = float(pos[1])
        measured_radius = nozzle_info.get("radius")
        if measured_radius is None:
            measured_radius = self._radius_from_bbox(nozzle_info.get("bounding_box"))
        if measured_radius is not None:
            try:
                measured_radius = max(1.0, float(measured_radius))
            except Exception:
                measured_radius = None

        if not isinstance(self.head_state, dict):
            init_r = float(measured_radius) if measured_radius is not None else 1.0
            self.head_state = {"x": mx, "y": my, "r": init_r}
            self.head_radius_history.append(float(max(1.0, init_r)))
            if self.head_radius_anchor_enabled:
                self.head_radius_anchor = float(max(1.0, init_r))
            nozzle_info["position"] = (int(round(mx)), int(round(my)))
            if measured_radius is None:
                nozzle_info["radius"] = int(round(init_r))
            else:
                nozzle_info["radius"] = int(round(float(measured_radius)))
            nozzle_info["bounding_box"] = self._stabilize_bbox(
                nozzle_info.get("bounding_box"),
                nozzle_info.get("detection_method", ""),
                (mx, my),
                init_r,
                init_r,
            )
            return nozzle_info

        prev_x = float(self.head_state.get("x", mx))
        prev_y = float(self.head_state.get("y", my))
        prev_r = max(
            1.0,
            float(
                self.head_state.get(
                    "r", measured_radius if measured_radius is not None else 1.0
                )
            ),
        )

        if measured_radius is None:
            measured_radius = prev_r
        else:
            measured_radius = max(1.0, float(measured_radius))
        measured_radius = self._radius_outlier_filter(measured_radius, prev_r)

        base_method = self._base_detection_method(
            nozzle_info.get("detection_method", "")
        )
        is_prediction = bool(nozzle_info.get("is_prediction", False))
        reliability = float(
            np.clip(self._measurement_reliability(nozzle_info), 0.0, 1.0)
        )
        strong_measurement = (
            (not is_prediction)
            and (base_method in {"circle", "contour"})
            and (reliability >= float(self.head_radius_anchor_update_reliability_min))
        )
        weak_measurement = not strong_measurement
        if reliability <= 1e-6:
            mx, my = prev_x, prev_y
            measured_radius = prev_r
        elif reliability < 0.999:
            mx = prev_x + (reliability * (mx - prev_x))
            my = prev_y + (reliability * (my - prev_y))
            measured_radius = prev_r + (reliability * (measured_radius - prev_r))

        geometry_lock_active = bool(
            self.user_geometry_lock_enabled
            and (self.user_geometry_lock_state is not None)
        )
        if weak_measurement:
            # Keep weak-frame hard clamps only when user geometry lock is active.
            # When lock is OFF, rely on soft smoothing (EMA) so radius can adapt.
            if (
                geometry_lock_active
                and self.head_radius_anchor_enabled
                and self.head_radius_anchor is not None
            ):
                min_anchor_radius = float(self.head_radius_anchor) * float(
                    self.head_radius_anchor_min_ratio
                )
                measured_radius = max(measured_radius, min_anchor_radius)
                max_anchor_radius = float(self.head_radius_anchor) * float(
                    self.head_radius_anchor_max_ratio
                )
                measured_radius = min(measured_radius, max_anchor_radius)
            if geometry_lock_active and self.head_radius_freeze_on_weak:
                shrink_limit = float(self.head_radius_weak_shrink_limit_ratio)
                if shrink_limit <= 1e-9:
                    measured_radius = max(measured_radius, prev_r)
                else:
                    measured_radius = max(
                        measured_radius, float(prev_r) * (1.0 - shrink_limit)
                    )
            if geometry_lock_active and self.head_radius_freeze_expand_on_weak:
                expand_limit = float(self.head_radius_weak_expand_limit_ratio)
                if expand_limit <= 1e-9:
                    measured_radius = min(measured_radius, prev_r)
                else:
                    measured_radius = min(
                        measured_radius, float(prev_r) * (1.0 + expand_limit)
                    )

        if not self.head_stability_enabled:
            sm_x, sm_y, sm_r = mx, my, measured_radius
        else:
            (
                alpha_c,
                alpha_r,
                max_jump,
                max_r_ratio,
                center_deadband,
                radius_deadband,
            ) = self._effective_stability_params()
            if float(np.hypot(mx - prev_x, my - prev_y)) <= center_deadband:
                mx, my = prev_x, prev_y
            mx, my = self._clip_jump(
                (prev_x, prev_y),
                (mx, my),
                max_jump,
            )

            if abs(measured_radius - prev_r) <= radius_deadband:
                measured_radius = prev_r
            max_r_delta = max(1.0, prev_r * max_r_ratio)
            measured_radius = float(
                np.clip(measured_radius, prev_r - max_r_delta, prev_r + max_r_delta)
            )

            # Decrease adaptation speed when the measurement is weak.
            alpha_c = float(np.clip(alpha_c * (0.35 + (0.65 * reliability)), 0.01, 1.0))
            alpha_r = float(np.clip(alpha_r * (0.30 + (0.70 * reliability)), 0.01, 1.0))
            sm_x = ((1.0 - alpha_c) * prev_x) + (alpha_c * mx)
            sm_y = ((1.0 - alpha_c) * prev_y) + (alpha_c * my)
            sm_r = ((1.0 - alpha_r) * prev_r) + (alpha_r * measured_radius)

        self._update_radius_anchor(base_method, reliability, sm_r, is_prediction)
        self.head_radius_history.append(float(max(1.0, sm_r)))
        self.head_state = {
            "x": float(sm_x),
            "y": float(sm_y),
            "r": float(max(1.0, sm_r)),
        }
        nozzle_info["position"] = (int(round(sm_x)), int(round(sm_y)))
        nozzle_info["radius"] = int(round(max(1.0, sm_r)))
        nozzle_info["bounding_box"] = self._stabilize_bbox(
            nozzle_info.get("bounding_box"),
            nozzle_info.get("detection_method", ""),
            (sm_x, sm_y),
            max(1.0, sm_r),
            max(1.0, measured_radius),
        )
        return nozzle_info

    def _apply_kalman(self, nozzle_info: Dict) -> Dict:
        """Correct or predict position with Kalman filter; set is_prediction=True on a miss frame."""
        if not self.kalman_enabled or self.kalman is None:
            nozzle_info["is_prediction"] = False
            return nozzle_info

        if nozzle_info.get("detected") and nozzle_info.get("position") is not None:
            x, y = nozzle_info["position"]
            if not self.kalman_initialized:
                self.kalman.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
                self.kalman.statePre = self.kalman.statePost.copy()
                self.kalman_initialized = True
                corrected = self.kalman.statePost
            else:
                self.kalman.predict()
                measurement = np.array(
                    [[np.float32(x)], [np.float32(y)]], dtype=np.float32
                )
                corrected = self.kalman.correct(measurement)

            nozzle_info["position"] = (int(corrected[0][0]), int(corrected[1][0]))
            nozzle_info["detection_method"] = (
                nozzle_info.get("detection_method") or "unknown"
            ) + "+kalman"
            nozzle_info["is_prediction"] = False
            self.kalman_miss_count = 0
            if nozzle_info.get("orientation") is not None:
                self.last_orientation = nozzle_info["orientation"]
            return nozzle_info

        if (
            self.kalman_initialized
            and self.kalman_predict_on_miss
            and self.kalman_miss_count < self.kalman_max_predict_frames
        ):
            predicted = self.kalman.predict()
            self.kalman_miss_count += 1
            nozzle_info["detected"] = True
            nozzle_info["position"] = (int(predicted[0][0]), int(predicted[1][0]))
            nozzle_info["orientation"] = (
                self.last_orientation
                if self.last_orientation is not None
                else self.reference_orientation
            )
            nozzle_info["detection_method"] = "kalman_predict"
            nozzle_info["is_prediction"] = True
            return nozzle_info

        nozzle_info["is_prediction"] = False
        return nozzle_info

    def set_user_geometry_lock(
        self, enabled: bool, clear_on_disable: bool = True
    ) -> Dict[str, bool]:
        """Enable or disable the user geometry lock; clears stored lock state on disable."""
        self.user_geometry_lock_enabled = bool(enabled)
        if (not self.user_geometry_lock_enabled) and clear_on_disable:
            self.user_geometry_lock_state = None
        return {
            "user_geometry_lock_enabled": bool(self.user_geometry_lock_enabled),
            "has_user_geometry_lock": bool(self.user_geometry_lock_state is not None),
        }

    def request_user_geometry_lock_capture(
        self, nozzle_info: Dict, force: bool = False
    ) -> bool:
        """Attempt to capture the current detection as the user geometry lock anchor if reliable enough."""
        if not self.user_geometry_lock_enabled:
            return False
        info = nozzle_info or {}
        if not bool(info.get("detected", False)):
            return False
        pos = info.get("position")
        radius = info.get("radius")
        if pos is None or radius is None:
            return False
        if (not force) and (not self._can_capture_user_geometry_lock(info)):
            return False
        try:
            self.user_geometry_lock_state = {
                "x": float(pos[0]),
                "y": float(pos[1]),
                "r": max(1.0, float(radius)),
            }
            return True
        except Exception:
            return False

    def _can_capture_user_geometry_lock(self, nozzle_info: Dict) -> bool:
        """Return True if nozzle_info is reliable enough to use as a geometry lock anchor."""
        if not nozzle_info.get("detected", False):
            return False
        if bool(nozzle_info.get("is_prediction", False)):
            return False
        method_txt = str(nozzle_info.get("detection_method") or "").strip().lower()
        if "+kalman" not in method_txt:
            return False
        base_method = self._base_detection_method(method_txt)
        if base_method not in {"circle", "contour"}:
            return False
        reliability = float(
            np.clip(self._measurement_reliability(nozzle_info), 0.0, 1.0)
        )
        return reliability >= float(self.user_geometry_lock_capture_reliability_min)

    def _apply_user_geometry_lock(self, nozzle_info: Dict) -> Dict:
        """Override detected position and radius with the locked values when the lock is active."""
        if not self.user_geometry_lock_enabled:
            return nozzle_info

        pos = nozzle_info.get("position")
        radius = nozzle_info.get("radius")
        if self.user_geometry_lock_state is None:
            self.request_user_geometry_lock_capture(nozzle_info, force=False)
            return nozzle_info

        lock = self.user_geometry_lock_state
        if (not nozzle_info.get("detected", False)) or lock is None:
            return nozzle_info

        lx = float(lock.get("x", 0.0))
        ly = float(lock.get("y", 0.0))
        lr = max(1.0, float(lock.get("r", 1.0)))
        nozzle_info["position"] = (int(round(lx)), int(round(ly)))
        nozzle_info["radius"] = int(round(lr))
        nozzle_info["bounding_box"] = (
            int(round(lx - lr)),
            int(round(ly - lr)),
            int(round(2.0 * lr)),
            int(round(2.0 * lr)),
        )
        nozzle_info["geometry_locked"] = True
        return nozzle_info

    def get_position_deviation(self, current_position: Tuple[int, int]) -> float:
        """Return pixel distance from current_position to the stored reference position."""
        if self.reference_position is None or current_position is None:
            return 0.0

        dx = current_position[0] - self.reference_position[0]
        dy = current_position[1] - self.reference_position[1]
        return np.sqrt(dx * dx + dy * dy)

    def get_orientation_deviation(self, current_orientation: float) -> float:
        """Return shortest angular difference in degrees from the stored reference orientation."""
        if self.reference_orientation is None or current_orientation is None:
            return 0.0

        # Calculate shortest angle difference
        diff = abs(current_orientation - self.reference_orientation)
        if diff > 180:
            diff = 360 - diff

        return diff

    def reset_reference(self):
        """Clear all tracking state, Kalman filter, and geometry lock to restart from scratch."""
        self.reference_position = None
        self.reference_orientation = None
        self.last_orientation = None
        self.head_state = None
        self.head_radius_history.clear()
        self.head_radius_anchor = None
        self.user_geometry_lock_state = None
        self.kalman_initialized = False
        self.kalman_miss_count = 0
        if self.kalman_enabled and self.kalman is not None:
            self.kalman.statePost = np.zeros((4, 1), dtype=np.float32)
            self.kalman.statePre = np.zeros((4, 1), dtype=np.float32)
