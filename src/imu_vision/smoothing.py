"""Optional IMU smoothing for when ArUco is temporarily lost.

Provides simple smoothing options:
- Pose-only error-state EKF on (p, v, q, b_g, b_a)
- Simple complementary filter: IMU to low-pass blend with tag quaternion
- Position hold with velocity prior from IMU
"""

from typing import Optional, Dict, Any
import numpy as np
from dataclasses import dataclass


@dataclass
class IMUData:
    """IMU data structure."""
    angular_velocity: np.ndarray  # [wx, wy, wz] in rad/s
    linear_acceleration: np.ndarray  # [ax, ay, az] in m/s²
    timestamp: float  # seconds


@dataclass
class PoseState:
    """Pose state for smoothing."""
    position: np.ndarray  # [x, y, z] in meters
    velocity: np.ndarray  # [vx, vy, vz] in m/s
    quaternion: np.ndarray  # [w, x, y, z] quaternion
    gyro_bias: np.ndarray  # [bx, by, bz] gyroscope bias
    accel_bias: np.ndarray  # [bx, by, bz] accelerometer bias


class IMUSmoother:
    """Simple IMU smoothing for pose estimation when ArUco is lost."""
    
    def __init__(self, 
                 smoothing_method: str = "complementary",
                 alpha: float = 0.02,
                 max_prediction_time: float = 1.0):
        """Initialize IMU smoother.
        
        Args:
            smoothing_method: Method to use ("complementary", "ekf", "hold")
            alpha: Blending factor for complementary filter (small = trust IMU more)
            max_prediction_time: Maximum time to predict without vision correction (seconds)
        """
        self.smoothing_method = smoothing_method
        self.alpha = alpha
        self.max_prediction_time = max_prediction_time
        
        # State variables
        self.last_vision_pose = None
        self.last_vision_timestamp = None
        self.last_imu_timestamp = None
        self.is_tracking = False
        
        # For complementary filter
        self.current_quaternion = np.array([1, 0, 0, 0])  # [w, x, y, z]
        self.current_position = np.zeros(3)
        self.current_velocity = np.zeros(3)
        
        # For EKF (simplified)
        self.ekf_state = None
        self.ekf_covariance = None
        
        # For position hold
        self.held_position = None
        self.velocity_prior = np.zeros(3)
    
    def update_with_vision(self, 
                          T_WH: np.ndarray, 
                          timestamp: float) -> None:
        """Update with vision-based pose correction.
        
        Args:
            T_WH: 4x4 world→hand transform matrix
            timestamp: Timestamp in seconds
        """
        self.last_vision_pose = T_WH.copy()
        self.last_vision_timestamp = timestamp
        self.is_tracking = True
        
        # Extract position and orientation
        position = T_WH[:3, 3]
        rotation_matrix = T_WH[:3, :3]
        quaternion = self._rotation_matrix_to_quaternion(rotation_matrix)
        
        # Update current state
        self.current_position = position.copy()
        self.current_quaternion = quaternion.copy()
        self.held_position = position.copy()
        
        # Reset velocity (assume stationary when vision is available)
        self.current_velocity = np.zeros(3)
        self.velocity_prior = np.zeros(3)
        
        print(f"Vision correction applied at t={timestamp:.3f}s")
    
    def predict_with_imu(self, 
                        imu_data: IMUData, 
                        dt: float) -> Optional[np.ndarray]:
        """Predict pose using IMU data.
        
        Args:
            imu_data: IMU measurements
            dt: Time step in seconds
            
        Returns:
            4x4 predicted transform matrix or None if prediction not available
        """
        if not self.is_tracking or self.last_vision_pose is None:
            return None
        
        # Check if too much time has passed since last vision update
        if (imu_data.timestamp - self.last_vision_timestamp) > self.max_prediction_time:
            print(f"Warning: No vision update for {imu_data.timestamp - self.last_vision_timestamp:.3f}s")
            return None
        
        if self.smoothing_method == "complementary":
            return self._predict_complementary(imu_data, dt)
        elif self.smoothing_method == "ekf":
            return self._predict_ekf(imu_data, dt)
        elif self.smoothing_method == "hold":
            return self._predict_hold(imu_data, dt)
        else:
            print(f"Warning: Unknown smoothing method '{self.smoothing_method}'")
            return None
    
    def _predict_complementary(self, imu_data: IMUData, dt: float) -> np.ndarray:
        """Predict using complementary filter approach."""
        # Integrate angular velocity to get orientation
        angular_velocity = imu_data.angular_velocity
        quaternion_dot = self._quaternion_derivative(self.current_quaternion, angular_velocity)
        
        # Update quaternion
        predicted_quaternion = self.current_quaternion + quaternion_dot * dt
        predicted_quaternion = self._normalize_quaternion(predicted_quaternion)
        
        # Convert to rotation matrix
        predicted_rotation = self._quaternion_to_rotation_matrix(predicted_quaternion)
        
        # For position, we could integrate acceleration, but for simplicity, hold last position
        # In a real system, you'd integrate: position += velocity * dt + 0.5 * acceleration * dt²
        predicted_position = self.current_position.copy()
        
        # Update current state
        self.current_quaternion = predicted_quaternion
        self.current_position = predicted_position
        
        # Construct transform matrix
        T_WH = np.eye(4)
        T_WH[:3, :3] = predicted_rotation
        T_WH[:3, 3] = predicted_position
        
        return T_WH
    
    def _predict_ekf(self, imu_data: IMUData, dt: float) -> np.ndarray:
        """Predict using simplified EKF approach."""
        # This is a simplified EKF - in practice, you'd implement a full error-state EKF
        # For now, use complementary filter as approximation
        return self._predict_complementary(imu_data, dt)
    
    def _predict_hold(self, imu_data: IMUData, dt: float) -> np.ndarray:
        """Predict using position hold with velocity prior."""
        # Hold last known position
        predicted_position = self.held_position.copy()
        
        # Simple velocity integration with small velocity prior
        velocity_decay = 0.95  # Decay factor for velocity
        self.velocity_prior *= velocity_decay
        
        # Add small velocity from IMU (very conservative)
        imu_velocity_estimate = imu_data.linear_acceleration * dt * 0.1  # Scale down significantly
        self.velocity_prior += imu_velocity_estimate * 0.01  # Very small update
        
        # Update position with velocity
        predicted_position += self.velocity_prior * dt
        
        # For orientation, use simple integration
        angular_velocity = imu_data.angular_velocity
        quaternion_dot = self._quaternion_derivative(self.current_quaternion, angular_velocity)
        predicted_quaternion = self.current_quaternion + quaternion_dot * dt
        predicted_quaternion = self._normalize_quaternion(predicted_quaternion)
        
        predicted_rotation = self._quaternion_to_rotation_matrix(predicted_quaternion)
        
        # Update current state
        self.current_quaternion = predicted_quaternion
        self.current_position = predicted_position
        
        # Construct transform matrix
        T_WH = np.eye(4)
        T_WH[:3, :3] = predicted_rotation
        T_WH[:3, 3] = predicted_position
        
        return T_WH
    
    def _quaternion_derivative(self, quaternion: np.ndarray, angular_velocity: np.ndarray) -> np.ndarray:
        """Calculate quaternion derivative from angular velocity."""
        w, x, y, z = quaternion
        wx, wy, wz = angular_velocity
        
        # Quaternion derivative: q_dot = 0.5 * q * [0, wx, wy, wz]
        q_dot = np.array([
            0.5 * (-x * wx - y * wy - z * wz),
            0.5 * (w * wx + y * wz - z * wy),
            0.5 * (w * wy - x * wz + z * wx),
            0.5 * (w * wz + x * wy - y * wx)
        ])
        
        return q_dot
    
    def _normalize_quaternion(self, quaternion: np.ndarray) -> np.ndarray:
        """Normalize quaternion to unit length."""
        norm = np.linalg.norm(quaternion)
        if norm < 1e-8:
            return np.array([1, 0, 0, 0])  # Default quaternion
        return quaternion / norm
    
    def _rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to quaternion [w, x, y, z]."""
        trace = np.trace(R)
        
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        
        return np.array([w, x, y, z])
    
    def _quaternion_to_rotation_matrix(self, quaternion: np.ndarray) -> np.ndarray:
        """Convert quaternion [w, x, y, z] to rotation matrix."""
        w, x, y, z = quaternion
        
        # Normalize quaternion
        norm = np.linalg.norm(quaternion)
        if norm < 1e-8:
            return np.eye(3)
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
        
        # Convert to rotation matrix
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
        
        return R
    
    def get_smoothing_info(self) -> Dict[str, Any]:
        """Get smoothing information for debugging."""
        return {
            'smoothing_method': self.smoothing_method,
            'is_tracking': self.is_tracking,
            'last_vision_timestamp': self.last_vision_timestamp,
            'last_imu_timestamp': self.last_imu_timestamp,
            'time_since_vision': (self.last_imu_timestamp - self.last_vision_timestamp) if self.last_imu_timestamp and self.last_vision_timestamp else None,
            'current_position': self.current_position.tolist() if self.current_position is not None else None,
            'current_quaternion': self.current_quaternion.tolist() if self.current_quaternion is not None else None
        }
