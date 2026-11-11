"""
Dry Nozzle Inspection System

A real-time computer vision system for inspecting dry nozzles,
detecting their position, orientation, and potential material buildup.

Package role:
    This is the public package entrypoint (`import nozzle_inspection`).
    Keep only stable/high-level exports here for external usage.
"""

__version__ = "1.0.0"
__author__ = "Quang Hieu Tran"

from .modules.buildup_detector import BuildupDetector
from .modules.camera import CameraCapture
from .modules.config_loader import ConfigLoader
from .modules.grid_zone_orchestrator import GridZoneOrchestrator
from .modules.nozzle_detector import NozzleDetector
from .modules.spray_angle_detector import SprayAngleDetector
from .modules.warning_system import WarningSystem

__all__ = [
    "CameraCapture",
    "NozzleDetector",
    "BuildupDetector",
    "SprayAngleDetector",
    "GridZoneOrchestrator",
    "WarningSystem",
    "ConfigLoader",
]
