"""
Camera capture module for acquiring video frames.
"""

from typing import Optional, Tuple

import cv2
import numpy as np


class CameraCapture:
    """
    Handles camera/video capture for the nozzle inspection system.
    """

    def __init__(
        self, source: int = 0, width: int = 640, height: int = 480, fps: int = 30
    ):
        """Initialize object."""
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.is_opened = False

    def open(self) -> bool:
        """Open the capture source, set frame size for live cameras, and return True on success."""
        try:
            self.cap = cv2.VideoCapture(self.source)

            if not self.cap.isOpened():
                print(f"Error: Could not open camera/video source {self.source}")
                return False

            # Set camera properties only for live camera devices.
            if not isinstance(self.source, str):
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

            self.is_opened = True
            print(
                f"Camera opened successfully: {self.width}x{self.height} @ {self.fps} FPS"
            )
            return True

        except Exception as e:
            print(f"Error opening camera: {e}")
            return False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Grab the next frame, returning (success, frame) or (False, None) if unavailable."""
        if not self.is_opened or self.cap is None:
            return False, None

        ret, frame = self.cap.read()

        if not ret:
            return False, None

        return True, frame

    def release(self):
        """Release the underlying capture device and mark the camera as closed."""
        if self.cap is not None:
            self.cap.release()
            self.is_opened = False
            print("Camera released")

    def get_fps(self) -> float:
        """Return the measured capture FPS if within 1-240, else the configured FPS, else 0.0."""
        if self.cap is not None and self.is_opened:
            measured = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if 1.0 <= measured <= 240.0:
                return measured
            configured = float(self.fps or 0.0)
            if 1.0 <= configured <= 240.0:
                return configured
        return 0.0

    def get_frame_size(self) -> Tuple[int, int]:
        """Return the capture's actual (width, height) in pixels, or (0, 0) if not open."""
        if self.cap is not None and self.is_opened:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return width, height
        return 0, 0

    def __del__(self):
        """Release the capture on garbage collection if it is still open."""
        if hasattr(self, "cap") and self.cap is not None:
            self.release()
