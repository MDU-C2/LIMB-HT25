"""
Complementary Filter for IMU-Vision Fusion

This module implements a simple complementary filter that fuses:
- IMU data (gyroscope, accelerometer) for high-frequency updates
- Vision data (position, orientation from AprilTags) for drift correction

The filter uses weighted averaging:
- High-pass filter on IMU: Tracks fast motions but drifts over time
- Low-pass filter on Vision: Stable but noisy and lower rate

Key parameter: alpha (trust factor)
- alpha = 0.98 means 98% trust in IMU prediction, 2% trust in vision correction
- Higher alpha = more responsive but more drift
- Lower alpha = more stable but more lag
"""

import numpy as np
from typing import Optional, Tuple
import time


class ComplementaryFilter:
    """
    Simple complementary filter for sensor fusion.
    
    State:
        - position: [x, y, z] in mm
        - velocity: [vx, vy, vz] in mm/s
        - orientation: quaternion [w, x, y, z]
        - angular_velocity: [wx, wy, wz] in rad/s
    """
    
    def __init__(self, alpha: float = 0.98, alpha_position: float = 0.95):
        """
        Initialize complementary filter.
        
        Args:
            alpha: Trust factor for orientation (0-1). Higher = trust IMU more.
            alpha_position: Trust factor for position (0-1). Higher = trust velocity integration more.
        """
        self.alpha = alpha
        self.alpha_position = alpha_position
        
        # State variables
        self.position = np.array([0.0, 0.0, 0.0])  # mm
        self.velocity = np.array([0.0, 0.0, 0.0])  # mm/s
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0])  # quaternion [w, x, y, z]
        self.angular_velocity = np.array([0.0, 0.0, 0.0])  # rad/s
        
        # Timing
        self.last_update_time = None
        self.initialized = False
        
        # Gravity vector (for accelerometer-based orientation)
        self.gravity = np.array([0.0, 0.0, 9.81])  # m/s² (Z-up convention)
        
        #print(f"ComplementaryFilter initialized (alpha={alpha}, alpha_pos={alpha_position})")
    
    def initialize(self, position: np.ndarray, orientation: Optional[np.ndarray] = None):
        """
        Initialize filter state with first measurement.
        
        Args:
            position: Initial position [x, y, z] in mm
            orientation: Initial orientation quaternion [w, x, y, z] (optional)
        """
        self.position = np.array(position, dtype=np.float64)
        
        if orientation is not None:
            self.orientation = np.array(orientation, dtype=np.float64)
            self.orientation = self._normalize_quaternion(self.orientation)
        
        self.velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        self.last_update_time = time.time()
        self.initialized = True
        
        #print(f"Filter initialized: pos={self.position}, orient={self.orientation}")
    
    def predict_with_imu(self, gyro: np.ndarray, accel: np.ndarray, dt: Optional[float] = None) -> dict:
        """
        Prediction step using IMU data (high-pass filter).
        
        Args:
            gyro: Angular velocity [wx, wy, wz] in rad/s
            accel: Linear acceleration [ax, ay, az] in m/s²
            dt: Time step in seconds (auto-computed if None)
        
        Returns:
            dict: Current state estimate
        """
        if not self.initialized:
            print("Warning: Filter not initialized. Call initialize() first.")
            return self.get_state()
        
        # Compute time step
        current_time = time.time()
        if dt is None:
            if self.last_update_time is not None:
                dt = current_time - self.last_update_time
            else:
                dt = 0.01  # Default 10ms
        
        self.last_update_time = current_time
        
        # Limit dt to avoid instability
        dt = min(dt, 0.1)  # Max 100ms
        
        # 1. Update orientation using gyroscope (integrate angular velocity)
        self.angular_velocity = np.array(gyro, dtype=np.float64)
        #print(f"angular_velocity: {self.angular_velocity}")
        self.orientation = self._integrate_quaternion(self.orientation, self.angular_velocity, dt)
        #print(f"orientation: {self.orientation}")
        self.orientation = self._normalize_quaternion(self.orientation)
        
        # 2. Update velocity and position using accelerometer
        # Remove gravity from accelerometer reading (gravity compensation)
        accel_world = self._rotate_vector_by_quaternion(accel, self.orientation)
        #print(f"accel_world: {accel_world}")
        accel_world = accel_world - self.gravity  # Remove gravity
        
        # Convert acceleration to mm/s²
        accel_world_mm = accel_world * 1000.0
        
        # Integrate acceleration to velocity
        self.velocity += accel_world_mm * dt
        
        # Integrate velocity to position
        self.position += self.velocity * dt
        
        return self.get_state()
    
    def update_with_vision(self, vision_position: np.ndarray, 
                          vision_orientation: Optional[np.ndarray] = None) -> dict:
        """
        Update step using vision data (low-pass filter).
        
        Args:
            vision_position: Measured position [x, y, z] in mm
            vision_orientation: Measured orientation quaternion [w, x, y, z] (optional)
        
        Returns:
            dict: Corrected state estimate
        """
        if not self.initialized:
            # First measurement - initialize the filter
            self.initialize(vision_position, vision_orientation)
            return self.get_state()
        
        # 1. Correct position (complementary filter)
        vision_pos = np.array(vision_position, dtype=np.float64)
        self.position = self.alpha_position * self.position + (1 - self.alpha_position) * vision_pos
        
        # 2. Correct velocity (estimate from position change)
        # This helps dampen drift
        position_error = vision_pos - self.position
        velocity_correction = position_error * 0.1  # Simple proportional correction
        self.velocity = self.alpha_position * self.velocity + (1 - self.alpha_position) * velocity_correction
        
        # 3. Correct orientation if available
        if vision_orientation is not None:
            vision_orient = np.array(vision_orientation, dtype=np.float64)
            vision_orient = self._normalize_quaternion(vision_orient)
            
            # Spherical linear interpolation (SLERP) for smooth quaternion blending
            self.orientation = self._slerp(self.orientation, vision_orient, 1 - self.alpha)
            self.orientation = self._normalize_quaternion(self.orientation)
        
        return self.get_state()
    
    def get_state(self) -> dict:
        """
        Get current filter state.
        
        Returns:
            dict: Current state with position, velocity, orientation, angular_velocity
        """
        return {
            'position': self.position.copy(),  # mm
            'velocity': self.velocity.copy(),  # mm/s
            'orientation': self.orientation.copy(),  # quaternion [w, x, y, z]
            'angular_velocity': self.angular_velocity.copy(),  # rad/s
            'initialized': self.initialized
        }
    
    def reset(self):
        """Reset filter to uninitialized state."""
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0])
        self.angular_velocity = np.zeros(3)
        self.last_update_time = None
        self.initialized = False
        #print("Filter reset")
    
    # ========== Utility Functions ==========
    
    @staticmethod
    def _normalize_quaternion(q: np.ndarray) -> np.ndarray:
        """Normalize quaternion to unit length."""
        norm = np.linalg.norm(q)
        if norm > 1e-8:
            return q / norm
        return np.array([1.0, 0.0, 0.0, 0.0])
    
    @staticmethod
    def _integrate_quaternion(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
        """
        Integrate quaternion using angular velocity.
        
        Args:
            q: Current quaternion [w, x, y, z]
            omega: Angular velocity [wx, wy, wz] in rad/s
            dt: Time step in seconds
        
        Returns:
            Updated quaternion
        """
        # Quaternion derivative: q_dot = 0.5 * q ⊗ [0, omega]
        omega_quat = np.array([0.0, omega[0], omega[1], omega[2]])
        q_dot = 0.5 * ComplementaryFilter._quaternion_multiply(q, omega_quat)
        
        # Integrate: q_new = q + q_dot * dt
        q_new = q + q_dot * dt
        return ComplementaryFilter._normalize_quaternion(q_new)
    
    @staticmethod
    def _quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """
        Multiply two quaternions: q1 ⊗ q2
        
        Args:
            q1, q2: Quaternions [w, x, y, z]
        
        Returns:
            Product quaternion
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    @staticmethod
    def _rotate_vector_by_quaternion(v: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        Rotate a vector by a quaternion: v' = q ⊗ [0, v] ⊗ q*
        
        Args:
            v: Vector [x, y, z]
            q: Quaternion [w, x, y, z]
        
        Returns:
            Rotated vector
        """
        # Convert vector to quaternion
        v_quat = np.array([0.0, v[0], v[1], v[2]])
        
        # Conjugate of q
        q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
        
        # Rotation: q ⊗ v ⊗ q*
        temp = ComplementaryFilter._quaternion_multiply(q, v_quat)
        result = ComplementaryFilter._quaternion_multiply(temp, q_conj)
        
        return result[1:]  # Return vector part [x, y, z]
    
    @staticmethod
    def _slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """
        Spherical linear interpolation between two quaternions.
        
        Args:
            q1, q2: Quaternions [w, x, y, z]
            t: Interpolation parameter (0 = q1, 1 = q2)
        
        Returns:
            Interpolated quaternion
        """
        # Ensure unit quaternions
        q1 = ComplementaryFilter._normalize_quaternion(q1)
        q2 = ComplementaryFilter._normalize_quaternion(q2)
        
        # Compute dot product
        dot = np.dot(q1, q2)
        
        # If dot < 0, negate q2 to take shorter path
        if dot < 0.0:
            q2 = -q2
            dot = -dot
        
        # Clamp dot to avoid numerical issues
        dot = np.clip(dot, -1.0, 1.0)
        
        # If quaternions are very close, use linear interpolation
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            return ComplementaryFilter._normalize_quaternion(result)
        
        # Compute angle between quaternions
        theta = np.arccos(dot)
        sin_theta = np.sin(theta)
        
        # Spherical interpolation
        w1 = np.sin((1 - t) * theta) / sin_theta
        w2 = np.sin(t * theta) / sin_theta
        
        return w1 * q1 + w2 * q2
    
    @staticmethod
    def quaternion_to_euler(q: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert quaternion to Euler angles (roll, pitch, yaw).
        
        Args:
            q: Quaternion [w, x, y, z]
        
        Returns:
            (roll, pitch, yaw) in radians
        """
        w, x, y, z = q
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw

