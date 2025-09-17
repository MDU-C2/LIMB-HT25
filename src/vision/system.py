from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from tags import TagDetector
from cup import CupDetector, UltralyticsYOLOCupDetector
from imu import IMUFusion
from output import OutputFormatter
from utils.geometry import pixel_and_depth_to_cam_point, relative_position_of_point_in_arm_frame


class VisionSystem:
    """
    Orchestrates tag detection, cup detection, and IMU fusion, returning structured output.
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        marker_length_m: float,
        assumed_cup_diameter_m: float = 0.08,
        detector_kind: str = "contour",
        yolo_model_path: Optional[str] = None,
        yolo_device: Optional[str] = None,
    ) -> None:
        self.tag_detector = TagDetector(camera_matrix, dist_coeffs, marker_length_m)
        if detector_kind == "yolo11" and yolo_model_path is not None:
            self.cup_detector = UltralyticsYOLOCupDetector(
                camera_matrix=camera_matrix,
                weights_path=yolo_model_path,
                assumed_cup_diameter_m=assumed_cup_diameter_m,
                device=yolo_device,
            )
        else:
            self.cup_detector = CupDetector(camera_matrix, assumed_cup_diameter_m)
        self.imu_fusion = IMUFusion()
        self.camera_matrix = camera_matrix

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        imu_delta_T_cam_from_arm: Optional[np.ndarray] = None,
        mode: str = "cup",
    ) -> Dict[str, Any]:
        # 1) IMU predict
        if imu_delta_T_cam_from_arm is not None:
            self.imu_fusion.predict_with_imu(imu_delta_T_cam_from_arm)

        # 2) Tag detection (optional by mode) → vision pose for arm/hand in camera frame
        vision_T_cam_from_arm = None
        if mode == "tag":
            tag_result = self.tag_detector.detect_and_estimate(frame_bgr)
            if tag_result.tvecs is not None and tag_result.rvecs is not None and len(tag_result.tag_ids) > 0:
                # Use first detected tag as reference
                rvec = tag_result.rvecs[0]
                tvec = tag_result.tvecs[0]
                vision_T_cam_from_arm = self.tag_detector.rvec_tvec_to_matrix(rvec, tvec)

        # 3) Correct IMU with vision if available
        if vision_T_cam_from_arm is not None:
            self.imu_fusion.correct_with_vision(vision_T_cam_from_arm)

        arm_pose_corrected = self.imu_fusion.pose.transform_cam_from_arm

        # 4) Cup detection (only in cup mode)
        if mode == "cup":
            cup_det = self.cup_detector.detect(frame_bgr)
        else:
            from cup import CupDetectionResult  # lightweight dataclass import
            cup_det = CupDetectionResult(False, None, None, None)

        # 5) Compute relative position of cup in arm frame
        cup_relative_position_m: Optional[Tuple[float, float, float]] = None
        if cup_det.detected and cup_det.pixel_center is not None and cup_det.distance_m is not None:
            u, v = cup_det.pixel_center
            cup_cam_xyz = pixel_and_depth_to_cam_point(u, v, cup_det.distance_m, self.camera_matrix)
            cup_arm_xyz = relative_position_of_point_in_arm_frame(
                arm_pose_corrected, cup_cam_xyz
            )
            cup_relative_position_m = (float(cup_arm_xyz[0]), float(cup_arm_xyz[1]), float(cup_arm_xyz[2]))

        # 6) Build output
        result = OutputFormatter.build_output(
            cup_detected=cup_det.detected,
            cup_relative_position_m=cup_relative_position_m,
            arm_pose_corrected_cam_from_arm=arm_pose_corrected,
        )
        return result


