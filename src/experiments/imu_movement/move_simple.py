#!/usr/bin/env python3
"""Simple script to read IMU data and detect high-level movements."""

import sys
import time
import serial
import json
import numpy as np
from collections import deque
import threading
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from imu_visualizer import IMUVisualizer
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
from data_fusion.complementary_filter import ComplementaryFilter


class MovementStateMachine:

    def __init__(self, thresholds: dict, neutral_timeout: float = 5.0, consecutive_detections: int = 5):
        self.x_state = "neutral" # "neutral", "right", "left"
        self.y_state = "neutral" # "neutral", "forward", "backward"
        self.z_state = "neutral" # "neutral", "up", "down"

        self.x_threshold = thresholds["x"]
        self.y_threshold = thresholds["y"]
        self.z_threshold = thresholds["z"]

        self.neutral_timeout = neutral_timeout # timeout in seconds
        self.state_change_time = None

        self.last_detection_time_x = None
        self.last_detection_time_y = None
        self.last_detection_time_z = None
        self.consecutive_detections = consecutive_detections # Number of consecutive detections to consider as up movement

        # Counter for X-axis
        self.consecutive_right_count = 0
        self.consecutive_left_count = 0

        # Counter for Y-axis
        self.consecutive_forward_count = 0
        self.consecutive_backward_count = 0

        # Counter for Z-axis
        self.consecutive_up_count = 0
        self.consecutive_down_count = 0

    def update(self, accel_world, current_time):
        """Update the state machine on current acceleration"""
        if self.state_change_time is None:
            self.stae_change_time = current_time

        # Check if any axis is currently active (not neutral)
        x_active = self.x_state != "neutral"
        y_active = self.y_state != "neutral"
        z_active = self.z_state != "neutral"

        # Update X-axis state
        if not y_active and not z_active:
            x_direction = self._update_x_axis(accel_world[0], current_time)
        else:
            # Another axis is active, keep current state
            x_direction = "neutral"
            self.consecutive_right_count = 0
            self.consecutive_left_count = 0

        # Update Y-axis state
        if not x_active and not z_active:
            y_direction = self._update_y_axis(accel_world[1], current_time)
        else:
            # Another axis is active, keep current state
            y_direction = "neutral"
            self.consecutive_forward_count = 0
            self.consecutive_backward_count = 0

        # Update Z-axis state
        if not x_active and not y_active:
            z_direction = self._update_z_axis(accel_world[2], current_time)
        else:
            # Another axis is active, keep current state
            z_direction = "neutral"
            self.consecutive_up_count = 0
            self.consecutive_down_count = 0

        return {"x_direction": x_direction, "y_direction": y_direction, "z_direction": z_direction}

    def _update_x_axis(self, accel_x, current_time):
        """Update the x-axis state machine"""
        #print(f"Acc X: {accel_x}")
        if self.x_state == "neutral":
            if self.last_detection_time_x is None or (current_time - self.last_detection_time_x) >= self.neutral_timeout:
                if accel_x > self.x_threshold:
                    self.consecutive_right_count += 1
                    self.consecutive_left_count = 0

                    if self.consecutive_right_count >= self.consecutive_detections:
                        self.x_state = "right"
                        self.last_detection_time_x = current_time
                        return "right"
                elif accel_x < -self.x_threshold:
                    self.consecutive_left_count += 1
                    self.consecutive_right_count = 0

                    if self.consecutive_left_count >= self.consecutive_detections:
                        self.x_state = "left"
                        self.last_detection_time_x = current_time
                        return "left"
                else:
                    self.consecutive_right_count = 0
                    self.consecutive_left_count = 0
                
                return "neutral"
            else:
                self.consecutive_right_count = 0
                self.consecutive_left_count = 0
                return "neutral"
        elif self.x_state == "right" or self.x_state == "left":
            self.x_state = "neutral"
            self.consecutive_right_count = 0
            self.consecutive_left_count = 0
            return "neutral"

        return self.x_state


    def _update_y_axis(self, accel_y, current_time):
        """Update the y-axis state machine"""
        #print(f"Acc Y: {accel_y}")
        if self.y_state == "neutral":
            if self.last_detection_time_y is None or (current_time - self.last_detection_time_y) >= self.neutral_timeout:
                if accel_y > self.y_threshold:
                    self.consecutive_forward_count += 1
                    self.consecutive_backward_count = 0

                    if self.consecutive_forward_count >= self.consecutive_detections:
                        self.y_state = "forward"
                        self.last_detection_time_y = current_time
                        return "forward"
                elif accel_y < -self.y_threshold:
                    self.consecutive_backward_count += 1
                    self.consecutive_forward_count = 0

                    if self.consecutive_backward_count >= self.consecutive_detections:
                        self.y_state = "backward"
                        self.last_detection_time_y = current_time
                        return "backward"
                else:
                    self.consecutive_forward_count = 0
                    self.consecutive_backward_count = 0
                return "neutral"
            else:
                self.consecutive_forward_count = 0
                self.consecutive_backward_count = 0
                return "neutral"

        elif self.y_state == "forward" or self.y_state == "backward":
            self.y_state = "neutral"
            self.consecutive_forward_count = 0
            self.consecutive_backward_count = 0
            return "neutral"

        return self.y_state

    def _update_z_axis(self, accel_z, current_time):
        """Update the z-axis state machine"""
        #print(f"Acc Z: {accel_z}")
        if self.z_state == "neutral":
            if self.last_detection_time_z is None or (current_time - self.last_detection_time_z) >= self.neutral_timeout:
                if accel_z > self.z_threshold:
                    self.consecutive_up_count += 1
                    self.consecutive_down_count = 0

                    if self.consecutive_up_count >= self.consecutive_detections:
                        self.z_state = "up"
                        self.last_detection_time_z = current_time
                        return "up"
                    
                elif accel_z < -self.z_threshold:
                    self.consecutive_down_count += 1
                    self.consecutive_up_count = 0

                    if self.consecutive_down_count >= self.consecutive_detections:
                        self.z_state = "down"
                        self.last_detection_time_z = current_time
                        return "down"
                else:
                    self.consecutive_up_count = 0
                    self.consecutive_down_count = 0
                    return "neutral"
            else:
                self.consecutive_up_count = 0
                self.consecutive_down_count = 0
                return "neutral"

        elif self.z_state == "up" or self.z_state == "down":
            self.z_state = "neutral"
            self.consecutive_up_count = 0
            self.consecutive_down_count = 0
            return "neutral"
        return self.z_state

movement_fsm = MovementStateMachine(thresholds={"x": 0.5, "y": 0.5, "z": 0.8}, neutral_timeout=2.0, consecutive_detections=5)



# Movement detection parameters
WINDOW_TIME_MS = 250  # Rolling window size in milliseconds
THRESHOLD = 1.0  # Minimum magnitude threshold (m/s²) - tune between 0.6-1.5
MIN_SAMPLES_FOR_DETECTION = 15  # Minimum samples needed before detecting
GRAVITY_MAGNITUDE = 9.81  # Gravity magnitude in m/s²
CALIBRATION_SAMPLES = 50  # Number of samples to use for initial orientation calibration
CALIBRATION_TIME_MS = 5000  # Time to wait for calibration (ms)

def quaternion_multiply(q1, q2):
    """Multiply two quaternions: q1 ⊗ q2"""
    return np.array([
        q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3],
        q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2],
        q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1],
        q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]
    ])

def quaternion_to_euler(q):
    """
    Convert quaternion [w, x, y, z] to Euler angles (roll, pitch, yaw) in radians.
    
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

def rotate_vector_by_quaternion(v, q):
    """
    Rotate a body-frame vector to world frame: v_world = q* ⊗ [0, v_body] ⊗ q
    
    Args:
        v: Vector [x, y, z] in body frame
        q: Quaternion [w, x, y, z] representing body orientation relative to world
    
    Returns:
        Rotated vector [x, y, z] in world frame
    """
    v_quat = np.array([0.0, v[0], v[1], v[2]])
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])  # Conjugate (inverse rotation)
    # Rotate body → world: q* ⊗ v ⊗ q
    temp = quaternion_multiply(q_conj, v_quat)
    result = quaternion_multiply(temp, q)
    return result[1:]  # Return vector part

def visualize_imu_orientation(quaternion, accel_body=None, accel_world=None, ax=None, show_plot=True):
    """
    Visualize IMU orientation and acceleration as Euler angles and 3D coordinate frames.
    
    Args:
        quaternion: Quaternion [w, x, y, z] representing IMU orientation
        accel_body: Optional acceleration vector [ax, ay, az] in body frame (m/s²)
        accel_world: Optional acceleration vector [ax, ay, az] in world frame (m/s²)
        ax: Optional matplotlib 3D axis (if None, creates new figure)
        show_plot: Whether to display the plot immediately
    
    Returns:
        fig, ax: Figure and axis objects (if ax was None, returns new ones)
    """
    # Convert quaternion to Euler angles
    roll, pitch, yaw = quaternion_to_euler(quaternion)
    roll_deg = np.rad2deg(roll)
    pitch_deg = np.rad2deg(pitch)
    yaw_deg = np.rad2deg(yaw)
    
    # Create figure if needed
    if ax is None:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure
        ax.clear()
    
    # Build title with orientation and acceleration info
    title = f'IMU Orientation & Acceleration\nRoll: {roll_deg:.1f}° | Pitch: {pitch_deg:.1f}° | Yaw: {yaw_deg:.1f}°'
    if accel_body is not None:
        accel_body_mag = np.linalg.norm(accel_body)
        title += f'\nBody Accel: [{accel_body[0]:.2f}, {accel_body[1]:.2f}, {accel_body[2]:.2f}] m/s² (mag: {accel_body_mag:.2f})'
    if accel_world is not None:
        accel_world_mag = np.linalg.norm(accel_world)
        title += f'\nWorld Accel: [{accel_world[0]:.2f}, {accel_world[1]:.2f}, {accel_world[2]:.2f}] m/s² (mag: {accel_world_mag:.2f})'
    
    # Set up 3D plot
    ax.set_title(title, fontsize=11, pad=25)
    ax.set_xlabel('X (Forward)', fontsize=10)
    ax.set_ylabel('Y (Right)', fontsize=10)
    ax.set_zlabel('Z (Up)', fontsize=10)
    
    # Determine appropriate axis limits based on acceleration magnitude
    max_accel = 0
    if accel_body is not None:
        max_accel = max(max_accel, np.linalg.norm(accel_body))
    if accel_world is not None:
        max_accel = max(max_accel, np.linalg.norm(accel_world))
    
    # Scale limits: show at least 1.5 for orientation, but extend for acceleration
    # Normalize acceleration to reasonable scale (divide by gravity for visualization)
    scale_limit = max(1.5, (max_accel / GRAVITY_MAGNITUDE) * 1.5)
    ax.set_xlim([-scale_limit, scale_limit])
    ax.set_ylim([-scale_limit, scale_limit])
    ax.set_zlim([-scale_limit, scale_limit])
    
    # Define unit vectors in world frame
    x_world = np.array([1.0, 0.0, 0.0])
    y_world = np.array([0.0, 1.0, 0.0])
    z_world = np.array([0.0, 0.0, 1.0])
    
    # Rotate world frame vectors to body frame (to show sensor orientation)
    # Note: We rotate world vectors by the quaternion to see where they point in body frame
    # Actually, we want to show body frame axes in world frame, so we rotate body frame axes
    # Body frame axes are [1,0,0], [0,1,0], [0,0,1] in body frame
    # To show them in world frame, we rotate them using q* ⊗ v ⊗ q
    x_body = np.array([1.0, 0.0, 0.0])
    y_body = np.array([0.0, 1.0, 0.0])
    z_body = np.array([0.0, 0.0, 1.0])
    
    x_rotated = rotate_vector_by_quaternion(x_body, quaternion)
    y_rotated = rotate_vector_by_quaternion(y_body, quaternion)
    z_rotated = rotate_vector_by_quaternion(z_body, quaternion)
    
    # Draw origin
    origin = np.array([0.0, 0.0, 0.0])
    
    # Draw body frame axes (sensor orientation)
    scale = 1.0
    ax.plot([origin[0], x_rotated[0] * scale], 
            [origin[1], x_rotated[1] * scale],
            [origin[2], x_rotated[2] * scale], 
            'r-', linewidth=3, label='X (Forward)')
    ax.plot([origin[0], y_rotated[0] * scale],
            [origin[1], y_rotated[1] * scale],
            [origin[2], y_rotated[2] * scale], 
            'g-', linewidth=3, label='Y (Right)')
    ax.plot([origin[0], z_rotated[0] * scale],
            [origin[1], z_rotated[1] * scale],
            [origin[2], z_rotated[2] * scale], 
            'b-', linewidth=3, label='Z (Up)')
    
    # Draw world frame axes (reference, thinner lines)
    ax.plot([origin[0], x_world[0] * scale * 0.5], 
            [origin[1], x_world[1] * scale * 0.5],
            [origin[2], x_world[2] * scale * 0.5], 
            'r--', linewidth=1, alpha=0.3, label='World X')
    ax.plot([origin[0], y_world[0] * scale * 0.5],
            [origin[1], y_world[1] * scale * 0.5],
            [origin[2], y_world[2] * scale * 0.5], 
            'g--', linewidth=1, alpha=0.3, label='World Y')
    ax.plot([origin[0], z_world[0] * scale * 0.5],
            [origin[1], z_world[1] * scale * 0.5],
            [origin[2], z_world[2] * scale * 0.5], 
            'b--', linewidth=1, alpha=0.3, label='World Z')
    
    # Add text labels at arrow tips
    ax.text(x_rotated[0] * scale * 1.1, x_rotated[1] * scale * 1.1, x_rotated[2] * scale * 1.1, 
            'X', color='red', fontsize=12, weight='bold')
    ax.text(y_rotated[0] * scale * 1.1, y_rotated[1] * scale * 1.1, y_rotated[2] * scale * 1.1, 
            'Y', color='green', fontsize=12, weight='bold')
    ax.text(z_rotated[0] * scale * 1.1, z_rotated[1] * scale * 1.1, z_rotated[2] * scale * 1.1, 
            'Z', color='blue', fontsize=12, weight='bold')
    
    # Draw acceleration vectors
    accel_scale = 0.1  # Scale factor to make acceleration visible (m/s² -> plot units)
    
    if accel_body is not None:
        # Rotate body-frame acceleration to world frame for visualization
        accel_body_world_frame = rotate_vector_by_quaternion(accel_body, quaternion)
        accel_vec = accel_body_world_frame * accel_scale
        # Draw line
        ax.plot([origin[0], accel_vec[0]], 
                [origin[1], accel_vec[1]],
                [origin[2], accel_vec[2]], 
                'm-', linewidth=3, alpha=0.9, label='Body Accel')
        # Add marker at tip to show direction
        ax.scatter([accel_vec[0]], [accel_vec[1]], [accel_vec[2]], 
                  c='magenta', s=100, marker='o', alpha=0.9)
    
    if accel_world is not None:
        # World-frame acceleration (already in world frame)
        accel_vec_world = accel_world * accel_scale
        # Draw line
        ax.plot([origin[0], accel_vec_world[0]], 
                [origin[1], accel_vec_world[1]],
                [origin[2], accel_vec_world[2]], 
                color='orange', linewidth=3, alpha=0.9, linestyle='--', label='World Accel (linear)')
        # Add marker at tip to show direction
        ax.scatter([accel_vec_world[0]], [accel_vec_world[1]], [accel_vec_world[2]], 
                  c='orange', s=100, marker='^', alpha=0.9)
    
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    if show_plot:
        plt.draw()
        plt.pause(0.001)  # Small pause to allow plot to update
    
    return fig, ax

def get_direction(norm_accel_world, current_time):
    """Get the up direction of the acceleration in the world frame"""
    # Compute the mean acceleration over the window
    #print(f"Norm acceleration: {norm_accel_world}")
    mean_accel = np.mean(norm_accel_world, axis=0)
    #print(f"Mean acceleration: {mean_accel}")
    direction = movement_fsm.update(mean_accel, current_time)
    
    # direction is a dictionary with the keys "x_direction", "y_direction", "z_direction"
    if direction["x_direction"] != "neutral":
        pass
        #print(f"X-axis direction: {direction['x_direction']}")
    if direction["y_direction"] != "neutral":
        pass
        #print(f"Y-axis direction: {direction['y_direction']}")
    if direction["z_direction"] != "neutral":
        pass
        #print(f"Z-axis direction: {direction['z_direction']}")

    return direction

def init_orientation_from_accel(accel_body):
    """Initialize the orientation from the acceleration in the body frame
    
    The accelerometer reading IS gravity in the body frame. We need to rotate it to the world frame.
    """
    # Normalize gravity vector in body frame
    gravity_body = accel_body / np.linalg.norm(accel_body)
    #print(f"Gravity body: {gravity_body}")

    # Gravirt in world frame is [0, 0, 1] (normalized, pointing up)
    gravity_world = np.array([0.0, 0.0, 1.0])

    # Find rotation that aligns gravity_body to gravity_world
    v = np.cross(gravity_body, gravity_world)
    s = np.linalg.norm(v)
    c = np.dot(gravity_body, gravity_world)

    if s < 1e-6:
        if c > 0:
            return np.array([1.0, 0.0, 0.0, 0.0]) # Already aligned
        else:
            # 180 degree rotation, need to pick axis
            if abs(gravity_body[0]) < 0.9:
                axis = np.array([1.0, 0.0, 0.0])
            else:
                axis = np.array([0.0, 1.0, 0.0])
            axis = axis / np.dot(axis, gravity_body) * gravity_body
            axis = axis / np.linalg.norm(axis)
            return np.array([0.0, axis[0], axis[1], axis[2]]) # 180 deg rotation

    # Rodrigues' rotation formula to get rotation matrix
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

    R = np.eye(3) + vx + vx @ vx * (1 - c) / (s**2) # Rotation matrix

    # Convert rotation matrix to quaternion
    trace = np.trace(R)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    q = np.array([w, x, y, z])
    norm = np.linalg.norm(q)
    if norm > 1e-8:
        return q/norm
    return np.array([1.0, 0.0, 0.0, 0.0])

def calibrate_imu(ser, calibration_time=2000):#, sample_rate_hz=100):
    """Calibrate the IMU by collecting samples while stationary"""
    calibration_time = calibration_time / 1000 # Convert to seconds
    print(f"\n{'='*60}")
    print("Calibrating IMU...")
    print(f"="*60)
    print(f"Keep sensor still for {calibration_time} seconds...")
    print("Collecting samples...\n")

    calib_samples = []
    start_time = time.time()
    buffer = ""
    samples_collected = 0
    
    #target_samples = int(calibration_time * sample_rate_hz)

    while time.time() - start_time < calibration_time:
        if ser.in_waiting > 0:
            data = ser.read(ser.in_waiting).decode("utf-8", errors="ignore")
            buffer += data

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()

                if not line or not line.startswith("{"):
                    continue

                try:
                    data = json.loads(line)
                    accel = data["accel"]
                    gyro = data["gyro"]
                    sample = {
                        "accel": np.array([accel["x"], accel["y"], accel["z"]]),
                        "gyro": np.array([gyro["x"], gyro["y"], gyro["z"]])
                    }
                    calib_samples.append(sample)
                    samples_collected += 1

                    if samples_collected & 50 == 0:
                        elapsed = time.time() - start_time
                        progress = min(100, (elapsed/calibration_time)*100)
                        print(f"   Progress: {progress:.0f}% | {samples_collected} samples.", end="\r")

                except (json.JSONDecodeError, KeyError):
                    continue
        time.sleep(0.001)
    print(f"Collected {samples_collected} samples in {calibration_time:.2f} seconds.")

    # Compute ygro bias (mean of gyro readings when stationary)
    gyro_readings = np.array([sample["gyro"] for sample in calib_samples])
    gyro_bias = np.mean(gyro_readings, axis=0)

    # Compute initial orientation from accelerometer
    accel_readings = np.array([sample["accel"] for sample in calib_samples])
    mean_accel = np.mean(accel_readings, axis=0)
    initial_quaternion = init_orientation_from_accel(mean_accel)

    return {
        "gyro_bias": gyro_bias,
        "initial_quaternion": initial_quaternion,
        "initial_accel": mean_accel
    }

def detect_movement(imu_samples_with_timestamps, current_time, initial_quaternion=None, gyro_bias=None):
    """
    Logic is in here.
    """

    # Filter samples within the last WINDOW_TIME_MS
    window_samples_with_timestamps = [
        (timestamp, sample) for timestamp, sample in imu_samples_with_timestamps
        if (current_time - timestamp) * 1000 < WINDOW_TIME_MS
    ]
    timestamps = np.array([ts for ts, _ in window_samples_with_timestamps]) # Shape (N,), max (WINDOW_TIME_MS,) sometimes (WINDOW_TIME_MS+1,)
    samples = np.array([sample for _, sample in window_samples_with_timestamps]) # Shape (N, 6), max (WINDOW_TIME_MS, 6) (sometimes (WINDOW_TIME_MS+1, 6))

    # Step 1 - Compute quaternion derivatie from gyro data
    gyro_data = samples[:, 3:] # Shape (N, 3), max (WINDOW_TIME_MS, 3)
    #print(f"Gyro data: {gyro_data}", gyro_data.shape)

    if gyro_bias is not None:
        #print(f"Gyro bias: {gyro_bias}")
        gyro_data = gyro_data - gyro_bias
        #print(f"Gyro data: {gyro_data}")
    
    if initial_quaternion is not None:
        quaternions = np.tile(initial_quaternion, (len(gyro_data), 1))
    else:
        quaternions = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (len(gyro_data), 1))
    """   
    quaternions = np.zeros((len(gyro_data), 4)) # (N, 4), max (WINDOW_TIME_MS, 4)

    if initial_quaternion is not None:
        quaternions[0] = initial_quaternion.copy()
    else:
        
        quaternions[0] = np.array([1.0, 0.0, 0.0, 0.0]) # Initial: no rotation
    

    for i in range(1, len(gyro_data)):
        dt = timestamps[i] - timestamps[i-1]
        if dt <= 0 or dt > 0.1:
            dt = 0.01
        
        omega = gyro_data[i] # One sample of gyro data
        #print(f"Omega: {omega}")
        omega_quat = np.array([0.0, omega[0], omega[1], omega[2]]) 
        #print(f"Omega quaternion: {omega_quat}")
        q_prev = quaternions[i-1]
        #print(f"Q prev: {q_prev}") # Right now zeros apart from first element
        q_dot = 0.5 * quaternion_multiply(q_prev, omega_quat)
        #print(f"Q dot: {q_dot}")
        # Integrate
        q_new = q_prev + q_dot * dt
        norm = np.linalg.norm(q_new)
        if norm > 1e-8: 
            q_new = q_new / norm
        else:
            q_new = np.array([1.0, 0.0, 0.0, 0.0])

        quaternions[i] = q_new
    """
    # Rotate acceleration into world frame and subtract gravity
    accel_data = samples[:, :3] # Shape (N, 3), max (WINDOW_TIME_MS, 3)
    gravity_world = np.array([0.0, 0.0, GRAVITY_MAGNITUDE])
    linear_accel_world = np.zeros_like(accel_data)

    for i in range(len(accel_data)):
        v_q = np.array([0.0, accel_data[i][0], accel_data[i][1], accel_data[i][2]])
        q_conj = np.array([quaternions[i][0], -quaternions[i][1], -quaternions[i][2], -quaternions[i][3]])
        #temp = quaternion_multiply(quaternions[i], v_q)
        #accel_world = quaternion_multiply(temp, q_conj)[1:]

        temp = quaternion_multiply(q_conj, v_q)
        accel_world = quaternion_multiply(temp, quaternions[i])[1:]
        
        linear_accel_world[i] = accel_world - gravity_world
        #print(f"Linear acceleration world: {linear_accel_world[i]}")

    if len(linear_accel_world) > 0:
        mean_accel_world = np.mean(linear_accel_world, axis=0)
        mean_accel_body = np.mean(accel_data, axis=0)
        
        # Debug: Check what the quaternion is doing
        print(f"\nDebug info:")
        print(f"  Mean body accel: {mean_accel_body}")
        print(f"  Mean world accel (after rotation): {mean_accel_world}")
        print(f"  Quaternion used: {quaternions[0]}")
        
        # Check: Rotate body accel with quaternion and see result
        test_body = mean_accel_body
        test_quat = quaternions[0]
        q_conj = np.array([test_quat[0], -test_quat[1], -test_quat[2], -test_quat[3]])
        v_q = np.array([0.0, test_body[0], test_body[1], test_body[2]])
        temp = quaternion_multiply(q_conj, v_q)
        rotated = quaternion_multiply(temp, test_quat)[1:]
        print(f"  Rotated body accel: {rotated}")
        print(f"  Expected gravity in world: [0, 0, 9.81]")
        print(f"  Difference: {rotated - np.array([0, 0, 9.81])}")

    # Return the latest quaternion for visualization
    latest_quaternion = quaternions[-1] if len(quaternions) > 0 else np.array([1.0, 0.0, 0.0, 0.0])


    # linear_accel_world[0] is one sample [ax, ay, az], linear_accel_world is matrix shape (N, 3), max (WINDOW_TIME_MS+1, 3)
    #print(f"Mean in X-axis: {np.mean(linear_accel_world[:, 0])}") # Mean of linear acceleration in x-axis
    #print(f"Mean in Y-axis: {np.mean(linear_accel_world[:, 1])}") # Mean of linear acceleration in y-axis
    #print(f"Mean in Z-axis: {np.mean(linear_accel_world[:, 2])}") # Mean of linear acceleration in z-axis
    #print(f"Linear acceleration world: {linear_accel_world}")
    # Print linear acceleration if one axis has a value greater than 2
    #print(f"Linear acceleration world greater than 3: {np.any(np.abs(linear_accel_world) > 3)}")

    direction = get_direction(linear_accel_world, current_time)
    
    # Return the latest acceleration for visualization
    latest_accel_body = accel_data[-1] if len(accel_data) > 0 else np.array([0.0, 0.0, 0.0])
    latest_accel_world = linear_accel_world[-1] if len(linear_accel_world) > 0 else np.array([0.0, 0.0, 0.0])
    
    return {
        "direction": "up", 
        "quaternion": latest_quaternion,
        "accel_body": latest_accel_body,
        "accel_world": latest_accel_world
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python move_simple.py <serial_port>")
        print("Example: python move_simple.py /dev/ttyUSB0")
        sys.exit(1)
    
    port = sys.argv[1]
    visualize = sys.argv[2] if len(sys.argv) > 2 else "False"
    
    print(f"Reading IMU data from {port}...")
    print(f"Using {WINDOW_TIME_MS}ms rolling window for movement detection")
    print(f"Threshold: {THRESHOLD} m/s² | Window: {WINDOW_TIME_MS}ms")
    print("Visualization window opened. Press Ctrl+C to stop.\n")
    
    try:
        ser = serial.Serial(port, 115200, timeout=1)
        ser.reset_input_buffer()
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        sys.exit(1)

    # Calibrate IMU
    calibration = calibrate_imu(ser, calibration_time=CALIBRATION_TIME_MS)
    current_quaternion = calibration["initial_quaternion"].copy()
    gyro_bias = calibration["gyro_bias"]
    last_gyro_time = None
    
    ser.reset_input_buffer()
    time.sleep(0.1)

    # Shared data structures
    imu_samples = deque(maxlen=100)
    buffer = ""
    line_count = 0
    last_detection_time = 0
    last_visualization_time = 0
    
    # Initialize visualization
    plt.ion()  # Turn on interactive mode
    fig = None
    ax = None
    
    # Start serial reader in background thread
    while True:
        if ser.in_waiting > 0:
            data = ser.read(ser.in_waiting).decode("utf-8", errors="ignore")
            #print(f"DATA: {data}")
            buffer += data

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()

                if not line:
                    continue

                if line_count < 5:
                    line_count += 1
            
                if line.startswith("{"):
                    data = json.loads(line)
                    accel = data["accel"]
                    gyro = data["gyro"]
                    #print(f"ACCEL: {accel} | GYRO: {gyro}")

                    current_time = time.time()

                    if last_gyro_time is not None:
                        dt = current_time - last_gyro_time
                        if dt > 0 and dt < 0.1:
                            # Apply gyro bias correction
                            omega = np.array([gyro["x"], gyro["y"], gyro["z"]]) - calibration["gyro_bias"]
                            omega_quat = np.array([0.0, omega[0], omega[1], omega[2]])
                            q_dot = 0.5 * quaternion_multiply(current_quaternion, omega_quat)
                            current_quaternion = current_quaternion + q_dot * dt
                            # Normalize
                            norm = np.linalg.norm(current_quaternion)
                            if norm > 1e-8:
                                current_quaternion = current_quaternion / norm
                            else:
                                current_quaternion = calibration["initial_quaternion"].copy()

                    last_gyro_time = current_time

                    # Add to buffer:
                    sample = [
                        accel["x"], accel["y"], accel["z"],
                        gyro["x"], gyro["y"], gyro["z"]
                    ]
                    #print(sample)
                    imu_samples.append((current_time, sample))
                    # Detect movement using rolling window (check every 50 ms to avoid spam)
                    if current_time - last_detection_time >= 0.05:
                        movement = detect_movement(imu_samples, current_time, current_quaternion, gyro_bias)
                        # Logic here
                        if movement: # and movemnt["direction"] != "none":
                            #len(timestamps)
                            #print(f"Direction: {movement['direction']}")
                            pass
                        last_detection_time = current_time
                        
                        # Update orientation visualization (every 100ms to avoid too frequent updates)
                        if visualize == "True" and current_time - last_visualization_time >= 0.1:
                            if movement and 'quaternion' in movement:
                                accel_body = movement.get('accel_body', None)
                                accel_world = movement.get('accel_world', None)
                                fig, ax = visualize_imu_orientation(
                                    movement['quaternion'], 
                                    accel_body=accel_body,
                                    accel_world=accel_world,
                                    ax=ax, 
                                    show_plot=True
                                )
                                last_visualization_time = current_time
                           

if __name__ == "__main__":
    main()

