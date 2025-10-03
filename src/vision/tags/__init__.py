"""Tag detection package (ArUco/AprilTag)."""

from .tag_detector import TagDetector, TagDetectionResult
from .tag_alignment import TagAlignmentDetector, TagAlignment, RobotArmTagConfig
from .tag_utils import (
    get_tag_transform_by_id,
    get_tag_reprojection_error_by_id,
    get_tag_timestamp_by_id,
    filter_tags_by_reprojection_error,
    get_tag_statistics,
    compare_tag_detections,
    extract_transform_matrices
)

__all__ = [
    "TagDetector",
    "TagDetectionResult",
    "TagAlignmentDetector",
    "TagAlignment",
    "RobotArmTagConfig",
    # Utility functions
    "get_tag_transform_by_id",
    "get_tag_reprojection_error_by_id", 
    "get_tag_timestamp_by_id",
    "filter_tags_by_reprojection_error",
    "get_tag_statistics",
    "compare_tag_detections",
    "extract_transform_matrices",
]


