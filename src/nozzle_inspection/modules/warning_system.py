"""
Warning system module for alerting on detected abnormalities.
"""

from enum import IntEnum
from typing import Dict


class AlertLevel(IntEnum):
    """
    Alert severity levels.
    """

    NORMAL = 0
    WARNING = 1
    CRITICAL = 2


class WarningSystem:
    """
    Manages warnings and alerts for the nozzle inspection system.
    """

    def __init__(self, config: Dict):
        """Initialize object."""
        self.position_threshold = config.get("position_deviation_threshold", 20)
        self.orientation_threshold = config.get("orientation_deviation_threshold", 15)
        self.buildup_warning_level = config.get("buildup_warning_level", 0.1)
        self.buildup_critical_level = config.get("buildup_critical_level", 0.2)

    def evaluate(
        self,
        nozzle_info: Dict,
        buildup_info: Dict,
        position_deviation: float,
        orientation_deviation: float,
    ) -> Dict:
        """Assess detection/deviation/buildup against thresholds and return alert_level and warnings."""
        warnings = []
        alert_level = AlertLevel.NORMAL

        # Check if nozzle is detected
        if not nozzle_info.get("detected", False):
            warnings.append("Nozzle not detected")
            alert_level = AlertLevel.CRITICAL
        else:
            # Check position deviation
            if position_deviation > self.position_threshold:
                warning_msg = f"Position deviation: {position_deviation:.1f} px (threshold: {self.position_threshold} px)"
                warnings.append(warning_msg)
                alert_level = max(alert_level, AlertLevel.WARNING)

            # Check orientation deviation
            if orientation_deviation > self.orientation_threshold:
                warning_msg = f"Orientation deviation: {orientation_deviation:.1f}° (threshold: {self.orientation_threshold}°)"
                warnings.append(warning_msg)
                alert_level = max(alert_level, AlertLevel.WARNING)

            # Check buildup
            buildup_percentage = buildup_info.get("buildup_percentage", 0.0)

            if buildup_percentage >= self.buildup_critical_level:
                warning_msg = (
                    f"Critical buildup detected: {buildup_percentage*100:.1f}%"
                )
                warnings.append(warning_msg)
                alert_level = AlertLevel.CRITICAL
            elif buildup_percentage >= self.buildup_warning_level:
                warning_msg = f"Buildup detected: {buildup_percentage*100:.1f}%"
                warnings.append(warning_msg)
                alert_level = max(alert_level, AlertLevel.WARNING)

        return {"alert_level": alert_level, "warnings": warnings}

    def get_color(self, alert_level: AlertLevel, config: Dict) -> tuple:
        """Return the configured BGR color tuple for the given alert level (critical/warning/normal)."""
        if alert_level == AlertLevel.CRITICAL:
            return tuple(config.get("critical_color", [0, 0, 255]))
        elif alert_level == AlertLevel.WARNING:
            return tuple(config.get("warning_color", [0, 165, 255]))
        else:
            return tuple(config.get("normal_color", [0, 255, 0]))
