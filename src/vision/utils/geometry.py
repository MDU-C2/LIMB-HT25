from typing import Optional, Tuple

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


