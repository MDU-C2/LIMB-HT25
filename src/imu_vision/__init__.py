"""IMU-Vision fusion package for fiducial + depth approach.

This package implements a simple fiducial + depth system without VIO:
- Hand pose estimation from ArUco markers
- Cup 3D position from YOLO + depth
- Relative pose calculation for control
- Optional IMU smoothing when ArUco is temporarily lost
"""

from .frames import FrameType, TransformManager
from .calibration import CalibrationManager
from .hand_pose import HandPoseEstimator
from .cup_3d import Cup3DEstimator
from .relative_pose import RelativePoseCalculator
from .smoothing import IMUSmoother
from .fusion_system import FiducialDepthSystem

__all__ = [
    "FrameType",
    "TransformManager", 
    "CalibrationManager",
    "HandPoseEstimator",
    "Cup3DEstimator",
    "RelativePoseCalculator",
    "IMUSmoother",
    "FiducialDepthSystem"
]
