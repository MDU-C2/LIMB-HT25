"""
Extended Kalman Filter for IMU-Vision Fusion

This module implements a hierarchical EKF-based sensor fusion:
- Stage 1: Pre-filter orientation from gyro + accel (complementary filter)
- Stage 2: EKF fuses pre-filtered IMU orientation with vision position + orientation

The EKF tracks:
- Position [x, y, z] in mm
- Velocity [vx, vy, vz] in mm/s
- Orientation quaternion [qw, qx, qy, qz]
- Angular velocity [wx, wy, wz] in rad/s

Key advantages over complementary filter:
- Optimal fusion (minimum variance estimate)
- Tracks uncertainty (covariance matrix)
- Adaptive weighting based on measurement quality
- Better handling of nonlinear dynamics
"""

import numpy as np
from typing import Optional, Dict, Tuple
import time
from dataclasses import dataclass


@dataclass
class EKFState:
    """EKF state structure."""
    position: np.ndarray          # [x, y, z] in mm
    velocity: np.ndarray          # [vx, vy, vz] in mm/s
    orientation: np.ndarray       # [qw, qx, qy, qz] quaternion
    angular_velocity: np.ndarray  # [wx, wy, wz] in rad/s
    timestamp: float              # seconds
    covariance: np.ndarray        # 13x13 state covariance matrix


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for IMU-Vision sensor fusion.
    
    State vector (13D):
        [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
    
    Process model:
        - Position: p' = p + v*dt
        - Velocity: v' = v + a*dt - g (gravity compensation)
        - Orientation: q' = q + 0.5*q⊗[0,w]*dt (quaternion integration)
        - Angular velocity: w' = w (constant model)
    
    Measurements:
        - IMU: Pre-filtered orientation (from complementary filter on ESP32)
        - Vision: Position + orientation from AprilTag pose estimation
    """
    
    def __init__(
        self,
        process_noise_pos: float = 1.0,      # Position process noise (mm²)
        process_noise_vel: float = 10.0,     # Velocity process noise (mm²/s²)
        process_noise_orient: float = 0.01,  # Orientation process noise (rad²)
        process_noise_angvel: float = 0.1,   # Angular velocity process noise (rad²/s²)
        measurement_noise_vision_pos: float = 25.0,    # Vision position noise (mm²)
        measurement_noise_vision_orient: float = 0.01, # Vision orientation noise (rad²)
        measurement_noise_imu_orient: float = 0.005,   # IMU orientation noise (rad²)
    ):
        """
        Initialize EKF.
        
        Args:
            process_noise_*: Process noise parameters (tuning parameters)
            measurement_noise_*: Measurement noise parameters (from sensor specs)
        """
        # State dimension
        self.state_dim = 13
        
        # State vector: [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
        self.x = np.zeros(self.state_dim)
        self.x[6] = 1.0  # Initialize quaternion to identity
        
        # State covariance matrix (13x13)
        self.P = np.eye(self.state_dim) * 100.0  # Initial uncertainty
        
        # Process noise covariance matrix Q (13x13)
        self.Q = np.diag([
            process_noise_pos, process_noise_pos, process_noise_pos,        # position
            process_noise_vel, process_noise_vel, process_noise_vel,        # velocity
            process_noise_orient, process_noise_orient, 
            process_noise_orient, process_noise_orient,                     # orientation (quaternion)
            process_noise_angvel, process_noise_angvel, process_noise_angvel # angular velocity
        ])
        
        # Measurement noise covariances
        self.R_vision_pos = np.eye(3) * measurement_noise_vision_pos
        self.R_vision_orient = np.eye(4) * measurement_noise_vision_orient
        self.R_vision_full = np.eye(7)
        self.R_vision_full[:3, :3] = self.R_vision_pos
        self.R_vision_full[3:, 3:] = self.R_vision_orient
        
        self.R_imu_orient = np.eye(4) * measurement_noise_imu_orient
        
        # Gravity vector (world frame, Z-up)
        self.gravity = np.array([0.0, 0.0, 9.81])  # m/s²
        
        # Timing
        self.last_update_time = None
        self.initialized = False
        
        print(f"EKF initialized (state_dim={self.state_dim})")
    
    def initialize(
        self,
        position: np.ndarray,
        orientation: Optional[np.ndarray] = None,
        velocity: Optional[np.ndarray] = None
    ):
        """
        Initialize filter state.
        
        Args:
            position: Initial position [x, y, z] in mm
            orientation: Initial orientation quaternion [qw, qx, qy, qz] (optional)
            velocity: Initial velocity [vx, vy, vz] in mm/s (optional)
        """
        self.x[0:3] = position
        
        if velocity is not None:
            self.x[3:6] = velocity
        else:
            self.x[3:6] = 0.0
        
        if orientation is not None:
            self.x[6:10] = self._normalize_quaternion(orientation)
        else:
            self.x[6:10] = np.array([1.0, 0.0, 0.0, 0.0])
        
        self.x[10:13] = 0.0  # Angular velocity
        
        # Reset covariance
        self.P = np.eye(self.state_dim) * 10.0
        self.P[0:3, 0:3] *= 10.0   # Higher initial position uncertainty
        self.P[3:6, 3:6] *= 50.0   # Higher initial velocity uncertainty
        
        self.last_update_time = time.time()
        self.initialized = True
        
        print(f"EKF initialized at position: {position}")
    
    def predict(
        self,
        angular_velocity: np.ndarray,
        linear_acceleration: np.ndarray,
        dt: Optional[float] = None
    ) -> EKFState:
        """
        Prediction step using IMU measurements.
        
        Args:
            angular_velocity: [wx, wy, wz] in rad/s
            linear_acceleration: [ax, ay, az] in m/s²
            dt: Time step in seconds (auto-computed if None)
        
        Returns:
            Current state estimate
        """
        if not self.initialized:
            print("Warning: EKF not initialized")
            return self.get_state()
        
        # Compute time step
        current_time = time.time()
        if dt is None:
            if self.last_update_time is not None:
                dt = current_time - self.last_update_time
            else:
                dt = 0.01
        
        self.last_update_time = current_time
        dt = min(dt, 0.1)  # Limit dt to avoid instability
        
        # Extract current state
        p = self.x[0:3]    # position
        v = self.x[3:6]    # velocity
        q = self.x[6:10]   # orientation
        w = angular_velocity  # angular velocity (from measurement)
        
        # --- Prediction (state propagation) ---
        
        # 1. Update orientation using angular velocity (quaternion integration)
        q_new = self._integrate_quaternion(q, w, dt)
        q_new = self._normalize_quaternion(q_new)
        
        # 2. Rotate acceleration to world frame and remove gravity
        accel_world = self._rotate_vector_by_quaternion(linear_acceleration, q_new)
        accel_world = accel_world - self.gravity  # Remove gravity
        accel_world_mm = accel_world * 1000.0  # Convert to mm/s²
        
        # 3. Update velocity and position
        v_new = v + accel_world_mm * dt
        p_new = p + v_new * dt
        
        # 4. Update angular velocity (constant model)
        w_new = w
        
        # Update state vector
        self.x[0:3] = p_new
        self.x[3:6] = v_new
        self.x[6:10] = q_new
        self.x[10:13] = w_new
        
        # --- Covariance prediction: P = F*P*F' + Q ---
        
        # Compute Jacobian F (linearized state transition)
        F = self._compute_state_jacobian(dt, q, w, accel_world_mm)
        
        # Propagate covariance
        self.P = F @ self.P @ F.T + self.Q * dt
        
        # Ensure symmetry
        self.P = (self.P + self.P.T) / 2.0
        
        return self.get_state()
    
    def update_with_vision(
        self,
        position: np.ndarray,
        orientation: Optional[np.ndarray] = None
    ) -> EKFState:
        """
        Update step using vision measurements.
        
        Args:
            position: Measured position [x, y, z] in mm
            orientation: Measured orientation quaternion [qw, qx, qy, qz] (optional)
        
        Returns:
            Corrected state estimate
        """
        if not self.initialized:
            self.initialize(position, orientation)
            return self.get_state()
        
        # Prepare measurement vector and matrices
        if orientation is not None:
            # Full measurement: position + orientation
            z = np.hstack([position, self._normalize_quaternion(orientation)])
            H = self._measurement_jacobian_full()
            R = self.R_vision_full
        else:
            # Position-only measurement
            z = position
            H = self._measurement_jacobian_position()
            R = self.R_vision_pos
        
        # Innovation (measurement residual)
        h = self._measurement_function(self.x, orientation is not None)
        y = z - h
        
        # Handle quaternion sign ambiguity (q and -q represent same rotation)
        if orientation is not None and len(y) == 7:
            if np.dot(orientation, self.x[6:10]) < 0:
                y[3:7] = z[3:7] - (-self.x[6:10])
        
        # Innovation covariance
        S = H @ self.P @ H.T + R
        
        # Kalman gain
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            print("Warning: Singular innovation covariance, skipping update")
            return self.get_state()
        
        # State update
        self.x = self.x + K @ y
        
        # Normalize quaternion after update
        self.x[6:10] = self._normalize_quaternion(self.x[6:10])
        
        # Covariance update (Joseph form for numerical stability)
        I = np.eye(self.state_dim)
        I_KH = I - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        
        # Ensure symmetry
        self.P = (self.P + self.P.T) / 2.0
        
        return self.get_state()
    
    def update_with_imu_orientation(self, orientation: np.ndarray) -> EKFState:
        """
        Update step using pre-filtered IMU orientation (from complementary filter).
        
        Args:
            orientation: Pre-filtered orientation quaternion [qw, qx, qy, qz]
        
        Returns:
            Corrected state estimate
        """
        if not self.initialized:
            return self.get_state()
        
        # Measurement vector (orientation only)
        z = self._normalize_quaternion(orientation)
        
        # Measurement Jacobian (observes orientation only)
        H = np.zeros((4, self.state_dim))
        H[0:4, 6:10] = np.eye(4)  # Observe quaternion states
        
        # Measurement function
        h = self.x[6:10]
        
        # Innovation
        y = z - h
        
        # Handle quaternion sign ambiguity
        if np.dot(z, h) < 0:
            y = z - (-h)
        
        # Innovation covariance
        S = H @ self.P @ H.T + self.R_imu_orient
        
        # Kalman gain
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return self.get_state()
        
        # State update
        self.x = self.x + K @ y
        
        # Normalize quaternion
        self.x[6:10] = self._normalize_quaternion(self.x[6:10])
        
        # Covariance update
        I = np.eye(self.state_dim)
        I_KH = I - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R_imu_orient @ K.T
        
        # Ensure symmetry
        self.P = (self.P + self.P.T) / 2.0
        
        return self.get_state()
    
    def get_state(self) -> EKFState:
        """Get current state estimate with covariance."""
        return EKFState(
            position=self.x[0:3].copy(),
            velocity=self.x[3:6].copy(),
            orientation=self.x[6:10].copy(),
            angular_velocity=self.x[10:13].copy(),
            timestamp=time.time(),
            covariance=self.P.copy()
        )
    
    def get_position_uncertainty(self) -> float:
        """Get position uncertainty (3D standard deviation in mm)."""
        pos_cov = self.P[0:3, 0:3]
        return np.sqrt(np.trace(pos_cov))
    
    def get_orientation_uncertainty(self) -> float:
        """Get orientation uncertainty (quaternion norm)."""
        orient_cov = self.P[6:10, 6:10]
        return np.sqrt(np.trace(orient_cov))
    
    def reset(self):
        """Reset filter to uninitialized state."""
        self.x = np.zeros(self.state_dim)
        self.x[6] = 1.0
        self.P = np.eye(self.state_dim) * 100.0
        self.last_update_time = None
        self.initialized = False
        print("EKF reset")
    
    # ========== Helper Functions ==========
    
    def _compute_state_jacobian(
        self,
        dt: float,
        q: np.ndarray,
        w: np.ndarray,
        accel: np.ndarray
    ) -> np.ndarray:
        """
        Compute Jacobian of state transition function.
        
        Linearizes: x_{k+1} = f(x_k, u_k)
        Returns: F = ∂f/∂x
        """
        F = np.eye(self.state_dim)
        
        # Position depends on velocity
        F[0:3, 3:6] = np.eye(3) * dt
        
        # Velocity depends on orientation (through rotated acceleration)
        # ∂(R(q)*a)/∂q is complex, use small angle approximation
        F[3:6, 6:10] = self._accel_rotation_jacobian(q, accel) * dt
        
        # Orientation depends on angular velocity
        # ∂(q + 0.5*q⊗[0,w]*dt)/∂q
        F[6:10, 6:10] = np.eye(4) + 0.5 * self._quaternion_omega_matrix(w) * dt
        F[6:10, 10:13] = 0.5 * self._quaternion_derivative_matrix(q) * dt
        
        return F
    
    def _measurement_jacobian_position(self) -> np.ndarray:
        """Jacobian for position-only measurement."""
        H = np.zeros((3, self.state_dim))
        H[0:3, 0:3] = np.eye(3)  # Observe position states
        return H
    
    def _measurement_jacobian_full(self) -> np.ndarray:
        """Jacobian for position + orientation measurement."""
        H = np.zeros((7, self.state_dim))
        H[0:3, 0:3] = np.eye(3)  # Observe position
        H[3:7, 6:10] = np.eye(4)  # Observe orientation
        return H
    
    def _measurement_function(self, x: np.ndarray, include_orientation: bool) -> np.ndarray:
        """
        Measurement function h(x).
        
        Args:
            x: State vector
            include_orientation: Whether to include orientation in measurement
        
        Returns:
            Expected measurement
        """
        if include_orientation:
            return np.hstack([x[0:3], x[6:10]])  # position + orientation
        else:
            return x[0:3]  # position only
    
    def _accel_rotation_jacobian(self, q: np.ndarray, accel: np.ndarray) -> np.ndarray:
        """
        Jacobian of rotated acceleration w.r.t. quaternion.
        Simplified approximation for numerical stability.
        """
        # Use finite differences for numerical Jacobian
        epsilon = 1e-6
        J = np.zeros((3, 4))
        
        for i in range(4):
            q_plus = q.copy()
            q_plus[i] += epsilon
            q_plus = self._normalize_quaternion(q_plus)
            
            q_minus = q.copy()
            q_minus[i] -= epsilon
            q_minus = self._normalize_quaternion(q_minus)
            
            accel_plus = self._rotate_vector_by_quaternion(accel / 1000.0, q_plus) * 1000.0
            accel_minus = self._rotate_vector_by_quaternion(accel / 1000.0, q_minus) * 1000.0
            
            J[:, i] = (accel_plus - accel_minus) / (2 * epsilon)
        
        return J
    
    def _quaternion_omega_matrix(self, w: np.ndarray) -> np.ndarray:
        """
        Omega matrix for quaternion derivative: q_dot = 0.5 * Omega(w) * q
        """
        wx, wy, wz = w
        return np.array([
            [0, -wx, -wy, -wz],
            [wx, 0, wz, -wy],
            [wy, -wz, 0, wx],
            [wz, wy, -wx, 0]
        ])
    
    def _quaternion_derivative_matrix(self, q: np.ndarray) -> np.ndarray:
        """
        Matrix for quaternion derivative w.r.t. angular velocity.
        """
        qw, qx, qy, qz = q
        return np.array([
            [-qx, -qy, -qz],
            [qw, -qz, qy],
            [qz, qw, -qx],
            [-qy, qx, qw]
        ])
    
    @staticmethod
    def _normalize_quaternion(q: np.ndarray) -> np.ndarray:
        """Normalize quaternion to unit length."""
        norm = np.linalg.norm(q)
        if norm > 1e-8:
            return q / norm
        return np.array([1.0, 0.0, 0.0, 0.0])
    
    @staticmethod
    def _integrate_quaternion(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
        """Integrate quaternion using angular velocity."""
        omega_quat = np.array([0.0, omega[0], omega[1], omega[2]])
        q_dot = 0.5 * ExtendedKalmanFilter._quaternion_multiply(q, omega_quat)
        q_new = q + q_dot * dt
        return ExtendedKalmanFilter._normalize_quaternion(q_new)
    
    @staticmethod
    def _quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions: q1 ⊗ q2"""
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
        """Rotate a vector by a quaternion: v' = q ⊗ [0, v] ⊗ q*"""
        v_quat = np.array([0.0, v[0], v[1], v[2]])
        q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
        temp = ExtendedKalmanFilter._quaternion_multiply(q, v_quat)
        result = ExtendedKalmanFilter._quaternion_multiply(temp, q_conj)
        return result[1:]
    
    @staticmethod
    def quaternion_to_euler(q: np.ndarray) -> Tuple[float, float, float]:
        """Convert quaternion to Euler angles (roll, pitch, yaw) in radians."""
        w, x, y, z = q
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw

