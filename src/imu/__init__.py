"""
IMU system for the robotic arm project.

This package provides functionality for:
- Reading data from 4 IMUs (2 robotic arm, 2 EMG band)
- Calibrating IMU sensors
- Processing and filtering IMU data
- Integrating IMU data with vision system
- Pose estimation from IMU data
"""

from .data_structures import (
    IMUReading,
    IMULocation,
    IMUCalibration,
    ProcessedIMUReading,
    IMUDataCollection
)

from .imu_interface import (
    IMUManager,
    IMUProtocol,
    SerialIMUProtocol,
    SimulatedIMUProtocol
)

from .calibration import (
    IMUCalibrator,
    IMUDataProcessor,
    IMUCalibrationCollector
)

from .vision_integration import (
    IMUPoseEstimator,
    MultiIMUPoseEstimator,
    VisionIMUIntegrator,
    create_imu_delta_transform
)

__all__ = [
    # Data structures
    'IMUReading',
    'IMULocation',
    'IMUCalibration',
    'ProcessedIMUReading',
    'IMUDataCollection',
    
    # Interface and communication
    'IMUManager',
    'IMUProtocol',
    'SerialIMUProtocol',
    'SimulatedIMUProtocol',
    
    # Calibration and processing
    'IMUCalibrator',
    'IMUDataProcessor',
    'IMUCalibrationCollector',
    
    # Vision integration
    'IMUPoseEstimator',
    'MultiIMUPoseEstimator',
    'VisionIMUIntegrator',
    'create_imu_delta_transform',
]
