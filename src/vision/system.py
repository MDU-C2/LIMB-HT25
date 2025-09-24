from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from tags import TagDetector
from cup import CupDetector
from imu import IMUFusion
from visualization import calculate_camera_relative_coordinates


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
        yolo_model_path: str = "yolo11s.pt",
        yolo_device: Optional[str] = None,
        yolo_conf: float = 0.35,
        yolo_iou: float = 0.45,
    ) -> None:
        self.tag_detector = TagDetector(camera_matrix, dist_coeffs, marker_length_m)
        
        # Always use YOLO for cup detection
        self.cup_detector = CupDetector(
            camera_matrix=camera_matrix,
            weights_path=yolo_model_path,
            assumed_cup_diameter_m=assumed_cup_diameter_m,
            device=yolo_device,
            conf=yolo_conf,
            iou=yolo_iou,
        )
        
        self.imu_fusion = IMUFusion()
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        imu_delta_T_cam_from_arm: Optional[np.ndarray] = None,
        mode: str = "cup",
    ) -> Dict[str, Any]:
        

        # 1) Tag detection (optional by mode) → vision pose for arm/hand in camera frame
        tag_result = None
        tag_coordinates = None
        if mode == "tag":
            tag_result = self.tag_detector.detect_and_estimate(frame_bgr)
            # Calculate camera-relative coordinates for all detected tags
            if tag_result is not None:
                tag_coordinates = calculate_camera_relative_coordinates(tag_result)
           
        # 2) Cup detection (only in cup mode)
        cup_result = None
        if mode == "cup":
            cup_result = self.cup_detector.detect(frame_bgr)
        
        return {
           "tag_result": tag_result,
           "tag_coordinates": tag_coordinates,
           "cup_result": cup_result,
        }
    


