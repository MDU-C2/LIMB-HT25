"""
Integration module for combining IMU data with the vision system.

This module provides functionality to:
- Convert IMU data to pose transformations
- Integrate IMU data with vision-based pose estimation
- Provide enhanced pose estimation for the robotic arm system
"""

import numpy as np
from typing import Optional, Dict, List
import logging
from scipy.spatial.transform import Rotation
from scipy.integrate import cumtrapz

from .data_structures import IMUDataCollection, ProcessedIMUReading, IMULocation
from .imu_interface import IMUManager


class IMUPoseEstimator:
    """Estimates pose from IMU data using sensor fusion techniques."""
    
    def __init__(self, sample_rate: float = 100.0):
        """
        Initialize IMU pose estimator.
        
        Args:
            sample_rate: Sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
        
        # Complementary filter parameters
        self.alpha = 0.02  # Trust factor for accelerometer/magnetometer
        
        # Initialize pose state
        self.current_pose = np.eye(4)  # 4x4 transformation matrix
        self.velocity = np.zeros(3)    # Linear velocity
        self.angular_velocity = np.zeros(3)  # Angular velocity
        
        # Previous readings for integration
        self.prev_reading: Optional[ProcessedIMUReading] = None
        self.prev_timestamp: Optional[float] = None
        
        self.logger = logging.getLogger(__name__)
    
    def update_pose(self, reading: ProcessedIMUReading) -> np.ndarray:
        """
        Update pose estimate from IMU reading.
        
        Args:
            reading: Processed IMU reading
            
        Returns:
            4x4 transformation matrix representing pose
        """
        if self.prev_reading is None:
            self.prev_reading = reading
            self.prev_timestamp = reading.timestamp
            return self.current_pose
        
        # Calculate time delta
        dt = reading.timestamp - self.prev_timestamp
        if dt <= 0:
            return self.current_pose
        
        # Update angular velocity
        self.angular_velocity = reading.gyro
        
        # Integrate angular velocity to get rotation
        rotation_delta = self._integrate_rotation(self.angular_velocity, dt)
        
        # Update linear velocity (simplified - assumes no gravity compensation)
        acceleration = reading.accel
        self.velocity += acceleration * dt
        
        # Update position
        position_delta = self.velocity * dt
        
        # Create transformation matrix
        delta_transform = np.eye(4)
        delta_transform[:3, :3] = rotation_delta
        delta_transform[:3, 3] = position_delta
        
        # Update current pose
        self.current_pose = self.current_pose @ delta_transform
        
        # Store current reading
        self.prev_reading = reading
        self.prev_timestamp = reading.timestamp
        
        return self.current_pose
    
    def _integrate_rotation(self, angular_velocity: np.ndarray, dt: float) -> np.ndarray:
        """Integrate angular velocity to get rotation matrix."""
        # Convert angular velocity to rotation vector
        rotation_vector = angular_velocity * dt
        
        # Convert to rotation matrix
        rotation = Rotation.from_rotvec(rotation_vector)
        return rotation.as_matrix()
    
    def reset_pose(self, initial_pose: Optional[np.ndarray] = None) -> None:
        """Reset pose estimate to initial value."""
        self.current_pose = initial_pose if initial_pose is not None else np.eye(4)
        self.velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        self.prev_reading = None
        self.prev_timestamp = None


class MultiIMUPoseEstimator:
    """Estimates pose from multiple IMUs using sensor fusion."""
    
    def __init__(self, sample_rate: float = 100.0):
        """
        Initialize multi-IMU pose estimator.
        
        Args:
            sample_rate: Sample rate in Hz
        """
        self.sample_rate = sample_rate
        
        # Individual pose estimators for each IMU
        self.pose_estimators: Dict[IMULocation, IMUPoseEstimator] = {}
        for location in IMULocation:
            self.pose_estimators[location] = IMUPoseEstimator(sample_rate)
        
        # Fused pose estimate
        self.fused_pose = np.eye(4)
        
        # Weights for different IMUs (can be adjusted based on reliability)
        self.imu_weights = {
            IMULocation.ROBOT_FOREARM: 0.3,
            IMULocation.ROBOT_UPPERARM: 0.3,
            IMULocation.EMG_FOREARM: 0.2,
            IMULocation.EMG_UPPERARM: 0.2,
        }
        
        self.logger = logging.getLogger(__name__)
    
    def update_poses(self, data_collection: IMUDataCollection) -> Dict[IMULocation, np.ndarray]:
        """
        Update pose estimates from IMU data collection.
        
        Args:
            data_collection: Collection of IMU readings
            
        Returns:
            Dictionary of pose estimates for each IMU
        """
        poses = {}
        
        for location in IMULocation:
            reading = data_collection.get_reading(location)
            if reading:
                pose = self.pose_estimators[location].update_pose(reading)
                poses[location] = pose
        
        # Fuse poses if multiple IMUs are available
        if len(poses) > 1:
            self.fused_pose = self._fuse_poses(poses)
        
        return poses
    
    def _fuse_poses(self, poses: Dict[IMULocation, np.ndarray]) -> np.ndarray:
        """
        Fuse multiple pose estimates into a single pose.
        
        Args:
            poses: Dictionary of pose estimates
            
        Returns:
            Fused pose estimate
        """
        if not poses:
            return self.fused_pose
        
        # Simple weighted average of poses
        # This is a simplified approach - in practice, more sophisticated fusion would be used
        weighted_rotation = np.zeros((3, 3))
        weighted_translation = np.zeros(3)
        total_weight = 0.0
        
        for location, pose in poses.items():
            weight = self.imu_weights.get(location, 0.0)
            if weight > 0:
                weighted_rotation += weight * pose[:3, :3]
                weighted_translation += weight * pose[:3, 3]
                total_weight += weight
        
        if total_weight > 0:
            # Normalize
            weighted_rotation /= total_weight
            weighted_translation /= total_weight
            
            # Re-orthonormalize rotation matrix
            U, _, Vt = np.linalg.svd(weighted_rotation)
            weighted_rotation = U @ Vt
            
            # Create fused pose
            fused_pose = np.eye(4)
            fused_pose[:3, :3] = weighted_rotation
            fused_pose[:3, 3] = weighted_translation
            
            return fused_pose
        
        return self.fused_pose
    
    def get_fused_pose(self) -> np.ndarray:
        """Get the current fused pose estimate."""
        return self.fused_pose.copy()
    
    def reset_poses(self, initial_pose: Optional[np.ndarray] = None) -> None:
        """Reset all pose estimates."""
        for estimator in self.pose_estimators.values():
            estimator.reset_pose(initial_pose)
        self.fused_pose = initial_pose if initial_pose is not None else np.eye(4)


class VisionIMUIntegrator:
    """Integrates IMU data with vision system for enhanced pose estimation."""
    
    def __init__(self, imu_manager: IMUManager, sample_rate: float = 100.0):
        """
        Initialize vision-IMU integrator.
        
        Args:
            imu_manager: IMU manager instance
            sample_rate: Sample rate in Hz
        """
        self.imu_manager = imu_manager
        self.multi_pose_estimator = MultiIMUPoseEstimator(sample_rate)
        
        # Integration parameters
        self.vision_weight = 0.1  # Weight for vision updates
        self.imu_weight = 0.9     # Weight for IMU predictions
        
        # State
        self.last_vision_pose: Optional[np.ndarray] = None
        self.last_imu_pose: Optional[np.ndarray] = None
        self.integrated_pose = np.eye(4)
        
        self.logger = logging.getLogger(__name__)
    
    def update_with_vision(self, vision_pose: np.ndarray, timestamp: float) -> np.ndarray:
        """
        Update pose estimate with vision data.
        
        Args:
            vision_pose: 4x4 transformation matrix from vision system
            timestamp: Timestamp of vision update
            
        Returns:
            Updated integrated pose
        """
        self.last_vision_pose = vision_pose
        
        # Get current IMU pose
        current_imu_pose = self.multi_pose_estimator.get_fused_pose()
        
        # Integrate vision and IMU poses
        self.integrated_pose = self._integrate_poses(vision_pose, current_imu_pose)
        
        return self.integrated_pose
    
    def update_with_imu(self, data_collection: IMUDataCollection) -> np.ndarray:
        """
        Update pose estimate with IMU data.
        
        Args:
            data_collection: Collection of IMU readings
            
        Returns:
            Updated integrated pose
        """
        # Update individual IMU poses
        imu_poses = self.multi_pose_estimator.update_poses(data_collection)
        
        # Get fused IMU pose
        current_imu_pose = self.multi_pose_estimator.get_fused_pose()
        self.last_imu_pose = current_imu_pose
        
        # If we have vision data, integrate it
        if self.last_vision_pose is not None:
            self.integrated_pose = self._integrate_poses(self.last_vision_pose, current_imu_pose)
        else:
            # Use IMU pose directly if no vision data
            self.integrated_pose = current_imu_pose
        
        return self.integrated_pose
    
    def _integrate_poses(self, vision_pose: np.ndarray, imu_pose: np.ndarray) -> np.ndarray:
        """
        Integrate vision and IMU poses using weighted combination.
        
        Args:
            vision_pose: Pose from vision system
            imu_pose: Pose from IMU system
            
        Returns:
            Integrated pose
        """
        # Simple weighted combination
        # In practice, more sophisticated fusion algorithms would be used
        
        # Combine rotations
        vision_rotation = vision_pose[:3, :3]
        imu_rotation = imu_pose[:3, :3]
        
        # Weighted combination of rotation matrices
        combined_rotation = (self.vision_weight * vision_rotation + 
                           self.imu_weight * imu_rotation)
        
        # Re-orthonormalize
        U, _, Vt = np.linalg.svd(combined_rotation)
        combined_rotation = U @ Vt
        
        # Combine translations
        combined_translation = (self.vision_weight * vision_pose[:3, 3] + 
                              self.imu_weight * imu_pose[:3, 3])
        
        # Create integrated pose
        integrated_pose = np.eye(4)
        integrated_pose[:3, :3] = combined_rotation
        integrated_pose[:3, 3] = combined_translation
        
        return integrated_pose
    
    def get_integrated_pose(self) -> np.ndarray:
        """Get the current integrated pose estimate."""
        return self.integrated_pose.copy()
    
    def reset_integration(self, initial_pose: Optional[np.ndarray] = None) -> None:
        """Reset the integration state."""
        self.multi_pose_estimator.reset_poses(initial_pose)
        self.last_vision_pose = None
        self.last_imu_pose = None
        self.integrated_pose = initial_pose if initial_pose is not None else np.eye(4)


def create_imu_delta_transform(data_collection: IMUDataCollection, 
                              prev_data_collection: Optional[IMUDataCollection] = None) -> np.ndarray:
    """
    Create IMU delta transform for use with existing vision system.
    
    Args:
        data_collection: Current IMU data collection
        prev_data_collection: Previous IMU data collection
        
    Returns:
        4x4 transformation matrix representing motion delta
    """
    if prev_data_collection is None:
        return np.eye(4)
    
    # Calculate average angular velocity across all IMUs
    total_angular_velocity = np.zeros(3)
    count = 0
    
    for location in IMULocation:
        current_reading = data_collection.get_reading(location)
        prev_reading = prev_data_collection.get_reading(location)
        
        if current_reading and prev_reading:
            # Calculate angular velocity delta
            angular_velocity = current_reading.gyro
            total_angular_velocity += angular_velocity
            count += 1
    
    if count == 0:
        return np.eye(4)
    
    # Average angular velocity
    avg_angular_velocity = total_angular_velocity / count
    
    # Calculate time delta
    dt = data_collection.timestamp - prev_data_collection.timestamp
    
    # Create rotation from angular velocity
    rotation_vector = avg_angular_velocity * dt
    rotation = Rotation.from_rotvec(rotation_vector)
    
    # Create delta transform
    delta_transform = np.eye(4)
    delta_transform[:3, :3] = rotation.as_matrix()
    
    return delta_transform
