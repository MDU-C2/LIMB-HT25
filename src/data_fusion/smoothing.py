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


class IMUValidator:
    """IMU-based validation for vision system results."""
    
    def __init__(self, 
                 validation_thresholds: Optional[Dict[str, float]] = None,
                 enable_motion_validation: bool = True,
                 enable_orientation_validation: bool = True,
                 enable_consistency_checking: bool = True):
        """Initialize IMU validator.
        
        Args:
            validation_thresholds: Thresholds for validation
            enable_motion_validation: Check if motion is physically reasonable
            enable_orientation_validation: Check orientation consistency
            enable_consistency_checking: Check temporal consistency
        """
        self.enable_motion_validation = enable_motion_validation
        self.enable_orientation_validation = enable_orientation_validation
        self.enable_consistency_checking = enable_consistency_checking
        
        # Default validation thresholds
        self.thresholds = {
            'max_acceleration': 50.0,  # m/s² (reasonable human motion limit)
            'max_angular_velocity': 10.0,  # rad/s (reasonable rotation speed)
            'max_position_jump': 0.5,  # meters (max position change between frames)
            'max_orientation_jump': 0.5,  # radians (max orientation change)
            'min_gravity_magnitude': 8.0,  # m/s² (gravity should be ~9.8)
            'max_gravity_deviation': 2.0,  # m/s² (acceptable gravity deviation)
        }
        
        if validation_thresholds:
            self.thresholds.update(validation_thresholds)
        
        # History for temporal consistency
        self.last_pose = None
        self.last_imu_data = None
        self.last_timestamp = None
        self.pose_history = []
        self.max_history = 10
        
    def validate_pose(self, 
                     vision_pose: np.ndarray,
                     imu_data: IMUData,
                     timestamp: float) -> Dict[str, Any]:
        """Validate a vision-based pose using IMU data.
        
        Args:
            vision_pose: 4x4 transformation matrix from vision
            imu_data: Current IMU measurements
            timestamp: Current timestamp
            
        Returns:
            Validation results with confidence score and issues
        """
        validation_result = {
            'is_valid': True,
            'confidence': 1.0,
            'issues': [],
            'warnings': [],
            'validation_details': {}
        }
        
        if vision_pose is None or imu_data is None:
            validation_result['is_valid'] = False
            validation_result['confidence'] = 0.0
            validation_result['issues'].append('Missing vision pose or IMU data')
            return validation_result
        
        # Extract position and orientation from vision pose
        vision_position = vision_pose[:3, 3]
        vision_rotation = vision_pose[:3, :3]
        
        # 1. Motion validation (check if motion is physically reasonable)
        if self.enable_motion_validation:
            motion_result = self._validate_motion(vision_pose, imu_data, timestamp)
            validation_result['validation_details']['motion'] = motion_result
            if not motion_result['is_valid']:
                validation_result['is_valid'] = False
                validation_result['issues'].extend(motion_result['issues'])
            validation_result['confidence'] *= motion_result['confidence']
        
        # 2. Orientation validation (check orientation consistency with IMU)
        if self.enable_orientation_validation:
            orientation_result = self._validate_orientation(vision_rotation, imu_data)
            validation_result['validation_details']['orientation'] = orientation_result
            if not orientation_result['is_valid']:
                validation_result['is_valid'] = False
                validation_result['issues'].extend(orientation_result['issues'])
            validation_result['confidence'] *= orientation_result['confidence']
        
        # 3. Temporal consistency checking
        if self.enable_consistency_checking:
            consistency_result = self._validate_temporal_consistency(vision_pose, timestamp)
            validation_result['validation_details']['consistency'] = consistency_result
            if not consistency_result['is_valid']:
                validation_result['is_valid'] = False
                validation_result['issues'].extend(consistency_result['issues'])
            validation_result['confidence'] *= consistency_result['confidence']
        
        # Update history
        self._update_history(vision_pose, imu_data, timestamp)
        
        return validation_result
    
    def _validate_motion(self, pose: np.ndarray, imu_data: IMUData, timestamp: float) -> Dict[str, Any]:
        """Validate motion is physically reasonable."""
        result = {'is_valid': True, 'confidence': 1.0, 'issues': []}
        
        # Check IMU acceleration magnitude
        accel_magnitude = np.linalg.norm(imu_data.linear_acceleration)
        if accel_magnitude > self.thresholds['max_acceleration']:
            result['is_valid'] = False
            result['issues'].append(f'Excessive acceleration: {accel_magnitude:.2f} m/s²')
            result['confidence'] *= 0.3
        
        # Check angular velocity magnitude
        angular_magnitude = np.linalg.norm(imu_data.angular_velocity)
        if angular_magnitude > self.thresholds['max_angular_velocity']:
            result['is_valid'] = False
            result['issues'].append(f'Excessive angular velocity: {angular_magnitude:.2f} rad/s')
            result['confidence'] *= 0.5
        
        # Check if gravity is reasonable (should be ~9.8 m/s² in stationary conditions)
        if accel_magnitude < self.thresholds['min_gravity_magnitude']:
            result['warnings'] = result.get('warnings', [])
            result['warnings'].append('Low acceleration magnitude - possible sensor issue')
            result['confidence'] *= 0.8
        elif accel_magnitude > 9.8 + self.thresholds['max_gravity_deviation']:
            result['warnings'] = result.get('warnings', [])
            result['warnings'].append('High acceleration magnitude - possible motion or sensor issue')
            result['confidence'] *= 0.9
        
        return result
    
    def _validate_orientation(self, vision_rotation: np.ndarray, imu_data: IMUData) -> Dict[str, Any]:
        """Validate orientation consistency with IMU."""
        result = {'is_valid': True, 'confidence': 1.0, 'issues': []}
        
        # Check if rotation matrix is valid (orthogonal)
        if not np.allclose(vision_rotation @ vision_rotation.T, np.eye(3), atol=1e-6):
            result['is_valid'] = False
            result['issues'].append('Invalid rotation matrix')
            result['confidence'] *= 0.1
        
        # Check determinant (should be 1 for rotation matrix)
        det = np.linalg.det(vision_rotation)
        if abs(det - 1.0) > 1e-6:
            result['is_valid'] = False
            result['issues'].append(f'Invalid rotation matrix determinant: {det:.6f}')
            result['confidence'] *= 0.1
        
        return result
    
    def _validate_temporal_consistency(self, pose: np.ndarray, timestamp: float) -> Dict[str, Any]:
        """Validate temporal consistency with previous poses."""
        result = {'is_valid': True, 'confidence': 1.0, 'issues': []}
        
        if self.last_pose is None or self.last_timestamp is None:
            return result
        
        dt = timestamp - self.last_timestamp
        if dt <= 0:
            result['is_valid'] = False
            result['issues'].append('Invalid timestamp')
            return result
        
        # Check position jump
        position_change = np.linalg.norm(pose[:3, 3] - self.last_pose[:3, 3])
        if position_change > self.thresholds['max_position_jump']:
            result['is_valid'] = False
            result['issues'].append(f'Large position jump: {position_change:.3f}m')
            result['confidence'] *= 0.2
        
        # Check orientation jump
        rotation_change = np.linalg.norm(pose[:3, :3] - self.last_pose[:3, :3])
        if rotation_change > self.thresholds['max_orientation_jump']:
            result['is_valid'] = False
            result['issues'].append(f'Large orientation jump: {rotation_change:.3f}')
            result['confidence'] *= 0.3
        
        return result
    
    def _update_history(self, pose: np.ndarray, imu_data: IMUData, timestamp: float):
        """Update pose and IMU history."""
        self.last_pose = pose.copy()
        self.last_imu_data = imu_data
        self.last_timestamp = timestamp
        
        # Keep limited history
        self.pose_history.append({
            'pose': pose.copy(),
            'timestamp': timestamp,
            'imu_data': imu_data
        })
        
        if len(self.pose_history) > self.max_history:
            self.pose_history.pop(0)
    
    def get_validation_info(self) -> Dict[str, Any]:
        """Get validation system information."""
        return {
            'enabled_features': {
                'motion_validation': self.enable_motion_validation,
                'orientation_validation': self.enable_orientation_validation,
                'consistency_checking': self.enable_consistency_checking
            },
            'thresholds': self.thresholds,
            'history_length': len(self.pose_history),
            'has_recent_data': self.last_timestamp is not None
        }
