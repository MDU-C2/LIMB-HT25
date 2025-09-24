"""Tag detection package (ArUco/AprilTag)."""

from .tag_detector import TagDetector, TagDetectionResult
from .tag_alignment import TagAlignmentDetector, TagAlignment, RobotArmTagConfig

__all__ = [
    "TagDetector",
    "TagDetectionResult",
    "TagAlignmentDetector",
    "TagAlignment",
    "RobotArmTagConfig",
]


