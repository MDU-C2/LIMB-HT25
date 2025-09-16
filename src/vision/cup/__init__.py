"""Cup detection package (markerless)."""

from .cup_detector import CupDetector, CupDetectionResult
from .yolo11_detector import UltralyticsYOLOCupDetector

__all__ = [
    "CupDetector",
    "CupDetectionResult",
    "UltralyticsYOLOCupDetector",
]


