from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tags import TagDetectionResult
from cup import CupDetectionResult


class OutputFormatter:
    """Formats results into structured Python dicts suitable for JSON serialization."""

    @staticmethod
    def build_output(
        tag_result: TagDetectionResult,
        cup_result: CupDetectionResult,
    ) -> Dict[str, Any]:
        # Format tag result with all available fields
        tag_output = {
            "tag_ids": tag_result.tag_ids.tolist() if tag_result.tag_ids is not None else [],
            "rvecs": tag_result.rvecs.tolist() if tag_result.rvecs is not None else None,
            "tvecs": tag_result.tvecs.tolist() if tag_result.tvecs is not None else None,
            "corners": [corner.tolist() for corner in tag_result.corners] if tag_result.corners is not None else None,
            "transforms": [transform.tolist() for transform in tag_result.transforms] if tag_result.transforms is not None else None,
            "reproj_errors": tag_result.reproj_errors.tolist() if tag_result.reproj_errors is not None else None,
            "timestamps": tag_result.timestamps
        }
        
        return {
           "tag_result": tag_output,
           "cup_result": cup_result,
        }