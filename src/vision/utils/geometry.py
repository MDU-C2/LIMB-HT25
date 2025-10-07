from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np


def pixel_and_depth_to_cam_point(
    u: int, v: int, depth_m: float, camera_matrix: np.ndarray
) -> np.ndarray:
    """Back-project a pixel with scalar depth into a 3D camera-frame point."""
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])

    x = (u - cx) * depth_m / fx
    y = (v - cy) * depth_m / fy
    z = depth_m
    return np.array([x, y, z], dtype=np.float64)


def transform_point(T_to_from: np.ndarray, p_from: np.ndarray) -> np.ndarray:
    p_h = np.ones(4, dtype=np.float64)
    p_h[:3] = p_from
    q = T_to_from @ p_h
    return q[:3]


def relative_position_of_point_in_arm_frame(
    cam_from_arm: np.ndarray,
    cup_cam_xyz: np.ndarray,
) -> np.ndarray:
    """Compute cup position in arm frame given arm pose in camera frame."""
    # arm_from_cam is inverse of cam_from_arm
    R = cam_from_arm[:3, :3]
    t = cam_from_arm[:3, 3]
    arm_from_cam = np.eye(4)
    arm_from_cam[:3, :3] = R.T
    arm_from_cam[:3, 3] = -R.T @ t
    cup_arm = transform_point(arm_from_cam, cup_cam_xyz)
    return cup_arm


def calculate_camera_relative_coordinates(tag_detection_result) -> Dict[int, Dict[str, Any]]:
    """
    Calculate camera-relative coordinates for all detected tags.
    
    This is a core utility function that converts tag detection results into
    structured coordinate data with position, orientation, and distance.
    
    Args:
        tag_detection_result: TagDetectionResult from TagDetector
    
    Returns:
        Dictionary mapping tag_id to coordinate information:
        {
            tag_id: {
                'position': {'x': float, 'y': float, 'z': float},  # meters
                'orientation': {'roll': float, 'pitch': float, 'yaw': float},  # degrees
                'distance': float,  # meters from camera
                'coordinate_system': 'camera_relative'
            }
        }
    """
    coordinates = {}
    
    if (tag_detection_result.tag_ids is None or 
        tag_detection_result.tvecs is None or 
        tag_detection_result.rvecs is None):
        return coordinates
    
    for i, tag_id in enumerate(tag_detection_result.tag_ids):
        if (i < len(tag_detection_result.tvecs) and 
            i < len(tag_detection_result.rvecs)):
            
            # Get position and rotation
            tvec = tag_detection_result.tvecs[i].flatten()
            rvec = tag_detection_result.rvecs[i].flatten()
            
            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            
            # Calculate Euler angles (roll, pitch, yaw)
            sy = np.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])
            singular = sy < 1e-6
            
            if not singular:
                roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                pitch = np.arctan2(-rotation_matrix[2, 0], sy)
                yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            else:
                roll = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
                pitch = np.arctan2(-rotation_matrix[2, 0], sy)
                yaw = 0
            
            # Convert to degrees
            roll_deg = np.degrees(roll)
            pitch_deg = np.degrees(pitch)
            yaw_deg = np.degrees(yaw)
            
            # Calculate distance from camera
            distance = np.linalg.norm(tvec)
            
            coordinates[int(tag_id)] = {
                'position': {
                    'x': float(tvec[0]),  # meters
                    'y': float(tvec[1]),  # meters
                    'z': float(tvec[2])   # meters
                },
                'orientation': {
                    'roll': float(roll_deg),   # degrees
                    'pitch': float(pitch_deg), # degrees
                    'yaw': float(yaw_deg)      # degrees
                },
                'distance': float(distance),  # meters from camera
                'coordinate_system': 'camera_relative'
            }
    
    return coordinates


