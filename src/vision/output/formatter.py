from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class OutputFormatter:
    """Formats results into structured Python dicts suitable for JSON serialization."""

    @staticmethod
    def pose_to_dict(T: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
        if T is None:
            return None
        return {
            "matrix": T.tolist(),
        }

    @staticmethod
    def cup_position_to_dict(
        cup_detected: bool,
        relative_position_m: Optional[Tuple[float, float, float]],
    ) -> Dict[str, Any]:
        return {
            "cup_detected": bool(cup_detected),
            "cup_relative_position": None
            if relative_position_m is None
            else {
                "x": float(relative_position_m[0]),
                "y": float(relative_position_m[1]),
                "z": float(relative_position_m[2]),
            },
        }

    @staticmethod
    def tag_detection_to_dict(
        tag_detected: bool,
        tag_ids: Optional[List[int]] = None,
        tag_positions: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        return {
            "tag_detected": bool(tag_detected),
            "tag_ids": tag_ids if tag_ids is not None else [],
            "tag_positions": tag_positions if tag_positions is not None else [],
        }

    @staticmethod
    def build_output(
        cup_detected: bool,
        cup_relative_position_m: Optional[Tuple[float, float, float]],
        arm_pose_corrected_cam_from_arm: Optional[np.ndarray],
        tag_detected: bool = False,
        tag_ids: Optional[List[int]] = None,
        tag_positions: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        result = OutputFormatter.cup_position_to_dict(
            cup_detected, cup_relative_position_m
        )
        result["arm_pose_corrected"] = OutputFormatter.pose_to_dict(
            arm_pose_corrected_cam_from_arm
        )
        result["tag_detection"] = OutputFormatter.tag_detection_to_dict(
            tag_detected, tag_ids, tag_positions
        )
        return result


