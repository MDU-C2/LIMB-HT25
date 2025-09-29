"""
IMU Preintegration for tightly coupled VIO.

This module implements IMU preintegration to accumulate IMU measurements
between keyframes, providing motion constraints for the VIO system.
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass
import logging

from ..imu.simple_imu import IMUData


@dataclass
class PreintegratedIMU:
    """Preintegrated IMU measurements between two keyframes."""
    
    # Preintegrated rotation (delta rotation)
    delta_R: np.ndarray  # 3x3 rotation matrix
    
    # Preintegrated velocity
    delta_v: np.ndarray  # 3x1 velocity vector
    
    # Preintegrated position
    delta_p: np.ndarray  # 3x1 position vector
    
    # Time interval
    dt: float
    
    # Covariance matrix of preintegrated measurements
    covariance: np.ndarray  # 9x9 covariance matrix
    
    # Jacobians with respect to bias
    dR_dbg: np.ndarray  # 3x3 jacobian of rotation w.r.t. gyro bias
    dV_dbg: np.ndarray  # 3x3 jacobian of velocity w.r.t. gyro bias
    dV_dba: np.ndarray  # 3x3 jacobian of velocity w.r.t. accel bias
    dP_dbg: np.ndarray  # 3x3 jacobian of position w.r.t. gyro bias
    dP_dba: np.ndarray  # 3x3 jacobian of position w.r.t. accel bias


class IMUPreintegration:
    """IMU preintegration for tightly coupled VIO."""
    
    def __init__(self, gravity: np.ndarray = np.array([0, 0, -9.81])):
        """
        Initialize IMU preintegration.
        
        Args:
            gravity: Gravity vector in world frame (m/s²)
        """
        self.gravity = gravity
        self.logger = logging.getLogger(__name__)
        
        # IMU noise parameters (these should be calibrated)
        self.gyro_noise_std = 0.01  # rad/s
        self.accel_noise_std = 0.1  # m/s²
        self.gyro_bias_std = 0.001  # rad/s
        self.accel_bias_std = 0.01  # m/s²
        
        # Initialize preintegration state
        self.reset()
    
    def reset(self):
        """Reset preintegration state."""
        # Preintegrated measurements
        self.delta_R = np.eye(3)
        self.delta_v = np.zeros(3)
        self.delta_p = np.zeros(3)
        self.dt = 0.0
        
        # Covariance matrix (9x9 for [delta_R, delta_v, delta_p])
        self.covariance = np.eye(9) * 1e-6
        
        # Jacobians w.r.t. bias
        self.dR_dbg = np.zeros((3, 3))
        self.dV_dbg = np.zeros((3, 3))
        self.dV_dba = np.zeros((3, 3))
        self.dP_dbg = np.zeros((3, 3))
        self.dP_dba = np.zeros((3, 3))
        
        # Accumulated measurements
        self.measurements: List[IMUData] = []
    
    def integrate_measurement(self, imu_data: IMUData, prev_imu_data: IMUData):
        """
        Integrate a single IMU measurement.
        
        Args:
            imu_data: Current IMU measurement
            prev_imu_data: Previous IMU measurement
        """
        dt = imu_data.timestamp - prev_imu_data.timestamp
        if dt <= 0:
            return
        
        # Extract measurements
        accel = imu_data.accel
        gyro = imu_data.gyro
        
        # Update preintegrated rotation
        delta_angle = gyro * dt
        delta_R_measurement = self._angle_to_rotation_matrix(delta_angle)
        self.delta_R = self.delta_R @ delta_R_measurement
        
        # Update preintegrated velocity
        self.delta_v += self.delta_R @ accel * dt
        
        # Update preintegrated position
        self.delta_p += self.delta_v * dt + 0.5 * self.delta_R @ accel * dt * dt
        
        # Update time
        self.dt += dt
        
        # Update covariance (simplified)
        self._update_covariance(dt, accel, gyro)
        
        # Update Jacobians (simplified)
        self._update_jacobians(dt, accel, gyro)
        
        # Store measurement
        self.measurements.append(imu_data)
    
    def _angle_to_rotation_matrix(self, angle: np.ndarray) -> np.ndarray:
        """Convert angle vector to rotation matrix using Rodrigues' formula."""
        angle_norm = np.linalg.norm(angle)
        if angle_norm < 1e-8:
            return np.eye(3)
        
        axis = angle / angle_norm
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        return np.eye(3) + np.sin(angle_norm) * K + (1 - np.cos(angle_norm)) * K @ K
    
    def _update_covariance(self, dt: float, accel: np.ndarray, gyro: np.ndarray):
        """Update covariance matrix (simplified implementation)."""
        # This is a simplified covariance update
        # In practice, you'd use the full covariance propagation equations
        
        # Process noise
        Q = np.diag([
            self.gyro_noise_std**2 * dt**2,  # Rotation noise
            self.gyro_noise_std**2 * dt**2,
            self.gyro_noise_std**2 * dt**2,
            self.accel_noise_std**2 * dt**2,  # Velocity noise
            self.accel_noise_std**2 * dt**2,
            self.accel_noise_std**2 * dt**2,
            self.accel_noise_std**2 * dt**4,  # Position noise
            self.accel_noise_std**2 * dt**4,
            self.accel_noise_std**2 * dt**4
        ])
        
        # Simple covariance update (not the full propagation)
        self.covariance += Q
    
    def _update_jacobians(self, dt: float, accel: np.ndarray, gyro: np.ndarray):
        """Update Jacobians w.r.t. bias (simplified implementation)."""
        # This is a simplified Jacobian update
        # In practice, you'd use the full Jacobian propagation equations
        
        # Update rotation Jacobian w.r.t. gyro bias
        self.dR_dbg += -self.delta_R * dt
        
        # Update velocity Jacobian w.r.t. gyro bias
        self.dV_dbg += -self.delta_R @ self._skew_symmetric(accel) * dt
        
        # Update velocity Jacobian w.r.t. accel bias
        self.dV_dba += -self.delta_R * dt
        
        # Update position Jacobian w.r.t. gyro bias
        self.dP_dbg += -0.5 * self.delta_R @ self._skew_symmetric(accel) * dt * dt
        
        # Update position Jacobian w.r.t. accel bias
        self.dP_dba += -0.5 * self.delta_R * dt * dt
    
    def _skew_symmetric(self, v: np.ndarray) -> np.ndarray:
        """Create skew-symmetric matrix from vector."""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
    
    def get_preintegrated_measurement(self) -> PreintegratedIMU:
        """Get the current preintegrated measurement."""
        return PreintegratedIMU(
            delta_R=self.delta_R.copy(),
            delta_v=self.delta_v.copy(),
            delta_p=self.delta_p.copy(),
            dt=self.dt,
            covariance=self.covariance.copy(),
            dR_dbg=self.dR_dbg.copy(),
            dV_dbg=self.dV_dbg.copy(),
            dV_dba=self.dV_dba.copy(),
            dP_dbg=self.dP_dbg.copy(),
            dP_dba=self.dP_dba.copy()
        )
    
    def predict_next_state(self, prev_state: 'VIOState', bias_gyro: np.ndarray, bias_accel: np.ndarray) -> 'VIOState':
        """
        Predict next state using preintegrated measurements.
        
        Args:
            prev_state: Previous VIO state
            bias_gyro: Gyroscope bias
            bias_accel: Accelerometer bias
            
        Returns:
            Predicted next state
        """
        # Apply bias correction
        delta_R_corrected = self.delta_R @ self._angle_to_rotation_matrix(-self.dR_dbg @ bias_gyro)
        delta_v_corrected = self.delta_v - self.dV_dbg @ bias_gyro - self.dV_dba @ bias_accel
        delta_p_corrected = self.delta_p - self.dP_dbg @ bias_gyro - self.dP_dba @ bias_accel
        
        # Predict next state
        next_rotation = prev_state.rotation @ delta_R_corrected
        next_velocity = prev_state.velocity + prev_state.rotation @ delta_v_corrected + self.gravity * self.dt
        next_position = prev_state.position + prev_state.velocity * self.dt + 0.5 * self.gravity * self.dt**2 + prev_state.rotation @ delta_p_corrected
        
        return VIOState(
            position=next_position,
            velocity=next_velocity,
            rotation=next_rotation,
            timestamp=prev_state.timestamp + self.dt
        )


@dataclass
class VIOState:
    """VIO state representation."""
    
    position: np.ndarray  # 3x1 position vector
    velocity: np.ndarray  # 3x1 velocity vector
    rotation: np.ndarray  # 3x3 rotation matrix
    timestamp: float
    
    def __post_init__(self):
        """Ensure arrays are numpy arrays."""
        self.position = np.asarray(self.position, dtype=np.float64)
        self.velocity = np.asarray(self.velocity, dtype=np.float64)
        self.rotation = np.asarray(self.rotation, dtype=np.float64)
