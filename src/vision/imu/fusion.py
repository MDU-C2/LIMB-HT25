from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ArmPose:
    """Pose of arm/hand in camera frame as 4x4 transform."""
    transform_cam_from_arm: np.ndarray  # 4x4


class IMUFusion:
    """
    Placeholder for IMU + vision fusion.

    Provide methods to update with IMU integration and correct with vision pose using either
    a complementary or Kalman-style update.
    """

    def __init__(self, alpha: float = 0.02) -> None:
        # alpha for complementary filter: small alpha trusts IMU more; large alpha trusts vision more
        self.alpha = float(alpha)
        self._pose_cam_from_arm_estimate = np.eye(4, dtype=np.float64)

    @property
    def pose(self) -> ArmPose:
        return ArmPose(self._pose_cam_from_arm_estimate.copy())

    def predict_with_imu(self, delta_transform_cam_from_arm: np.ndarray) -> None:
        """Integrate IMU motion model: left-multiply by delta in camera frame.

        For a real system, delta would come from integrating angular rates and linear acceleration.
        """
        self._pose_cam_from_arm_estimate = (
            delta_transform_cam_from_arm @ self._pose_cam_from_arm_estimate
        )

    def correct_with_vision(self, vision_transform_cam_from_arm: np.ndarray) -> None:
        """Complementary-style blend between IMU estimate and vision absolute pose."""
        # Simple SE(3) blend via matrix interpolation (not strictly correct). For production, use Lie algebra.
        imu_T = self._pose_cam_from_arm_estimate
        vis_T = vision_transform_cam_from_arm

        blended = (1.0 - self.alpha) * imu_T + self.alpha * vis_T

        # Re-orthonormalize rotation via SVD
        U, _, Vt = np.linalg.svd(blended[:3, :3])
        blended[:3, :3] = U @ Vt

        self._pose_cam_from_arm_estimate = blended


