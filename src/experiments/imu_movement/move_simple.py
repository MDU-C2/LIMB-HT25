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
from imu_visualizer import IMUVisualizer
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
from data_fusion.complementary_filter import ComplementaryFilter

# Quaternion helper functions
def quaternion_multiply(q1, q2):
    """Multiply two quaternions: q1 ⊗ q2"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def normalize_quaternion(q):
    """Normalize quaternion to unit length."""
    norm = np.linalg.norm(q)
    if norm > 1e-8:
        return q / norm
    return np.array([1.0, 0.0, 0.0, 0.0])

def rotate_vector_by_quaternion(v, q):
    """
    Rotate a vector by a quaternion: v' = q ⊗ [0, v] ⊗ q*
    
    Args:
        v: Vector [x, y, z]
        q: Quaternion [w, x, y, z]
    
    Returns:
        Rotated vector [x, y, z]
    """
    v_quat = np.array([0.0, v[0], v[1], v[2]])
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
    temp = quaternion_multiply(q, v_quat)
    result = quaternion_multiply(temp, q_conj)
    return result[1:]  # Return vector part

def quaternion_from_accelerometer(accel):
    """
    Estimate quaternion from accelerometer reading (assuming it's gravity).
    This gives us the orientation of the sensor relative to gravity.
    
    Args:
        accel: Accelerometer reading [ax, ay, az] in m/s²
    
    Returns:
        quaternion: [w, x, y, z] representing rotation from world to sensor frame
    """
    # Normalize accelerometer reading (should be gravity when stationary)
    accel_norm = np.linalg.norm(accel)
    if accel_norm < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0])  # No rotation if no acceleration
    
    accel_normalized = accel / accel_norm
    
    # Gravity in world frame points down (negative Z)
    gravity_world = np.array([0.0, 0.0, -1.0])
    
    # Find rotation that aligns gravity_world with accel_normalized
    # Using method from: https://math.stackexchange.com/questions/180418
    v = np.cross(gravity_world, accel_normalized)
    s = np.linalg.norm(v)  # Sine of angle
    c = np.dot(gravity_world, accel_normalized)  # Cosine of angle
    
    if s < 1e-6:
        # Vectors are parallel (or anti-parallel)
        if c > 0:
            return np.array([1.0, 0.0, 0.0, 0.0])  # No rotation
        else:
            # 180 degree rotation around any perpendicular axis
            # Use X axis as rotation axis
            return np.array([0.0, 1.0, 0.0, 0.0])
    
    # Normalize rotation axis
    v = v / s
    
    # Build quaternion: q = [cos(θ/2), sin(θ/2) * axis]
    # For small angles: cos(θ/2) ≈ 1, sin(θ/2) ≈ θ/2 ≈ s/2
    # More accurate: use half-angle formulas
    half_angle = np.arccos(np.clip(c, -1.0, 1.0)) / 2.0
    q_w = np.cos(half_angle)
    q_xyz = np.sin(half_angle) * v
    
    quaternion = np.array([q_w, q_xyz[0], q_xyz[1], q_xyz[2]])
    return normalize_quaternion(quaternion)

def integrate_quaternion_from_gyro(gyro_data, timestamps, initial_quaternion=None):
    """
    Integrate gyroscope data to get orientation quaternion.
    
    Args:
        gyro_data: (N, 3) array of angular velocities [wx, wy, wz] in rad/s
        timestamps: (N,) array of timestamps
        initial_quaternion: Initial quaternion [w, x, y, z] (optional)
    
    Returns:
        quaternions: (N, 4) array of quaternions [w, x, y, z]
    """
    quaternions = np.zeros((len(gyro_data), 4))
    if initial_quaternion is not None:
        quaternions[0] = initial_quaternion.copy()
    else:
        quaternions[0] = np.array([1.0, 0.0, 0.0, 0.0])  # Default: no rotation
    
    for i in range(1, len(gyro_data)):
        dt = timestamps[i] - timestamps[i-1]
        if dt <= 0 or dt > 0.1:
            dt = 0.01
        
        # Quaternion derivative: q_dot = 0.5 * q ⊗ [0, wx, wy, wz]
        omega = gyro_data[i]
        omega_quat = np.array([0.0, omega[0], omega[1], omega[2]])
        
        # Multiply quaternion with angular velocity quaternion
        q_prev = quaternions[i-1]
        q_dot = 0.5 * quaternion_multiply(q_prev, omega_quat)
        
        # Integrate
        q_new = q_prev + q_dot * dt
        quaternions[i] = normalize_quaternion(q_new)
    
    return quaternions

# Movement detection parameters
WINDOW_TIME_MS = 250  # Rolling window size in milliseconds
THRESHOLD = 1.0  # Minimum magnitude threshold (m/s²) - tune between 0.6-1.5
MIN_SAMPLES_FOR_DETECTION = 15  # Minimum samples needed before detecting
GRAVITY_MAGNITUDE = 9.81  # Gravity magnitude in m/s²
CALIBRATION_SAMPLES = 50  # Number of samples to use for initial orientation calibration
CALIBRATION_TIME_MS = 1000  # Time to wait for calibration (ms)

def detect_movement(imu_samples_with_timestamps, current_time, complementary_filter):
    """
    Minimal movement detection algorithm:
    1. Rotate accel into world frame using orientation estimate (from complementary filter)
    2. Remove gravity
    3. Low-pass/average over window
    4. Threshold + dominant axis detection
    """
    # Filter samples within the last WINDOW_TIME_MS
    window_samples_with_timestamps = [
        (timestamp, sample) for timestamp, sample in imu_samples_with_timestamps
        if (current_time - timestamp) * 1000 < WINDOW_TIME_MS
    ]
    
    if len(window_samples_with_timestamps) < MIN_SAMPLES_FOR_DETECTION:
        return None
    
    # Extract timestamps and samples
    timestamps = np.array([ts for ts, _ in window_samples_with_timestamps])
    samples = np.array([sample for _, sample in window_samples_with_timestamps])
    
    # Extract accel and gyro
    accel_data = samples[:, :3]  # (N, 3) - [ax, ay, az] in m/s² (sensor frame)
    gyro_data = samples[:, 3:]   # (N, 3) - [gx, gy, gz] in rad/s
    
    # Step 1: Get current orientation from complementary filter
    # The filter maintains orientation state, so we use the latest orientation
    filter_state = complementary_filter.get_state()
    current_orientation = filter_state['orientation']  # Quaternion [w, x, y, z]
    
    # Step 2: Rotate acceleration into world frame
    # Gravity in world frame points DOWN (negative Z): [0, 0, -9.81]
    gravity_world = np.array([0.0, 0.0, -GRAVITY_MAGNITUDE])  # Gravity vector in world frame (points down)
    accel_world = np.zeros_like(accel_data)
    
    for i in range(len(accel_data)):
        # Rotate sensor-frame acceleration to world frame using complementary filter's rotation method
        accel_world[i] = ComplementaryFilter._rotate_vector_by_quaternion(accel_data[i], current_orientation)
    
    # Step 3: Remove gravity
    a_lin = accel_world - gravity_world
    
    # Step 4: Low-pass / average over window (simple mean)
    m = np.mean(a_lin, axis=0)  # Mean linear acceleration vector
    
    # Step 5: Threshold + dominant axis
    magnitude = np.linalg.norm(m)
    
    if magnitude < THRESHOLD:
        return {
            "direction": "none",
            "confidence": 0.0,
            "magnitude": magnitude,
            "vector": m.tolist()
        }
    
    # Pick axis with largest absolute value
    abs_m = np.abs(m)
    max_axis = np.argmax(abs_m)
    max_value = abs_m[max_axis]
    
    # Determine direction based on sign and axis
    if max_axis == 0:  # X axis
        direction = "forward" if m[0] > 0 else "backward"
    elif max_axis == 1:  # Y axis
        direction = "right" if m[1] > 0 else "left"
    else:  # Z axis
        direction = "up" if m[2] > 0 else "down"
    
    # Simple confidence: how much above threshold (normalized to 0-1)
    confidence = min((magnitude - THRESHOLD) / THRESHOLD + 0.5, 1.0)
    confidence = max(0.0, confidence)
    
    return {
        "direction": direction,
        "confidence": confidence * 100.0,  # Convert to percentage
        "magnitude": magnitude,
        "vector": m.tolist()
    }

def calibrate_initial_orientation(ser, num_samples=CALIBRATION_SAMPLES):
    """
    Calibrate initial orientation from accelerometer when sensor is stationary.
    
    Args:
        ser: Serial port object
        num_samples: Number of samples to use for calibration
    
    Returns:
        initial_quaternion: [w, x, y, z] representing initial orientation
    """
    print(f"Calibrating initial orientation... Keep sensor STILL for {CALIBRATION_TIME_MS/1000:.1f} seconds")
    
    accel_samples = []
    buffer = ""
    samples_collected = 0
    start_time = time.time()
    
    while samples_collected < num_samples and (time.time() - start_time) < (CALIBRATION_TIME_MS / 1000.0):
        if ser.in_waiting > 0:
            data = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
            buffer += data
            
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                line = line.strip()
                
                if line.startswith('{'):
                    try:
                        data = json.loads(line)
                        accel = data['accel']
                        accel_vec = np.array([accel['x'], accel['y'], accel['z']])
                        accel_samples.append(accel_vec)
                        samples_collected += 1
                    except json.JSONDecodeError:
                        pass
        
        time.sleep(0.01)
        if samples_collected % 10 == 0:
            print(f"  Collected {samples_collected}/{num_samples} samples...")
    
    if len(accel_samples) < 10:
        print("Warning: Not enough samples for calibration. Using default orientation.")
        return np.array([1.0, 0.0, 0.0, 0.0])
    
    # Average accelerometer readings (should be gravity when stationary)
    mean_accel = np.mean(accel_samples, axis=0)
    
    # Estimate quaternion from gravity direction
    initial_quaternion = quaternion_from_accelerometer(mean_accel)
    
    print(f"Calibration complete. Initial orientation estimated from gravity.")
    print(f"  Mean accel: [{mean_accel[0]:.3f}, {mean_accel[1]:.3f}, {mean_accel[2]:.3f}] m/s²")
    
    return initial_quaternion

def serial_reader(port, visualizer, imu_samples, stop_event):
    """Read serial data in background thread."""
    try:
        ser = serial.Serial(port, 115200, timeout=1)
        ser.reset_input_buffer()
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        return
    
    # Calibrate initial orientation
    initial_quaternion = calibrate_initial_orientation(ser)
    
    # Initialize complementary filter with initial orientation
    # Note: complementary filter uses Z-up convention (gravity = [0, 0, 9.81])
    # We need to adjust for Z-down convention
    complementary_filter = ComplementaryFilter(alpha=0.98)
    complementary_filter.initialize(
        position=np.array([0.0, 0.0, 0.0]),  # Initial position (not used for movement detection)
        orientation=initial_quaternion
    )
    # Override gravity to use Z-down convention
    complementary_filter.gravity = np.array([0.0, 0.0, -GRAVITY_MAGNITUDE])
    
    line_count = 0
    buffer = ""
    last_detection_time = 0
    last_imu_update = 0
    
    try:
        while not stop_event.is_set():
            try:
                if ser.in_waiting > 0:
                    data = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
                    buffer += data
                    
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        
                        if not line:
                            continue
                        
                        if line.startswith('{'):
                            try:
                                data = json.loads(line)
                                accel = data['accel']
                                gyro = data['gyro']
                                
                                current_time = time.time()
                                
                                sample = [
                                    accel['x'], accel['y'], accel['z'],
                                    gyro['x'], gyro['y'], gyro['z']
                                ]
                                imu_samples.append((current_time, sample))
                                
                                # Update complementary filter with IMU data
                                if current_time - last_imu_update >= 0.01:  # Update at ~100Hz
                                    dt = current_time - last_imu_update
                                    if dt > 0 and dt < 0.1:
                                        accel_vec = np.array([accel['x'], accel['y'], accel['z']])
                                        gyro_vec = np.array([gyro['x'], gyro['y'], gyro['z']])
                                        filter_state = complementary_filter.predict_with_imu(gyro_vec, accel_vec, dt)
                                        
                                        # Update visualization with current orientation
                                        visualizer.update_orientation(filter_state['orientation'])
                                    last_imu_update = current_time
                                
                                # Detect movement
                                if current_time - last_detection_time >= 0.05:
                                    movement = detect_movement(imu_samples, current_time, complementary_filter)
                                    visualizer.update_movement(movement)
                                    
                                    if movement and movement['direction'] != 'none':
                                        window_samples_count = len([
                                            ts for ts, _ in imu_samples
                                            if (current_time - ts) * 1000 < WINDOW_TIME_MS
                                        ])
                                        print(f"Movement: {movement['direction']:8s} | "
                                              f"Magnitude: {movement['magnitude']:.3f} m/s² | "
                                              f"CONFIDENCE: {movement['confidence']:.2f}% | "
                                              f"Window: {window_samples_count} samples")
                                    last_detection_time = current_time
                            
                            except json.JSONDecodeError as e:
                                if line_count < 10:
                                    print(f"JSON decode error: {e}")
                                pass
                else:
                    time.sleep(0.01)
                    
            except serial.SerialException as e:
                print(f"Serial error: {e}")
                time.sleep(1)
    
    except Exception as e:
        print(f"Serial reader error: {e}")
    finally:
        if 'ser' in locals():
            ser.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python move_simple.py <serial_port>")
        print("Example: python move_simple.py /dev/ttyUSB0")
        sys.exit(1)
    
    port = sys.argv[1]
    
    # Initialize visualizer (must be in main thread)
    visualizer = IMUVisualizer(threshold=THRESHOLD)
    visualizer.start(rotate_vector_by_quaternion)
    
    print(f"Reading IMU data from {port}...")
    print(f"Using {WINDOW_TIME_MS}ms rolling window for movement detection")
    print(f"Threshold: {THRESHOLD} m/s² | Window: {WINDOW_TIME_MS}ms")
    print("Visualization window opened. Press Ctrl+C to stop.\n")
    
    # Shared data structures
    imu_samples = deque(maxlen=100)
    stop_event = threading.Event()
    
    # Start serial reader in background thread
    serial_thread = threading.Thread(
        target=serial_reader, 
        args=(port, visualizer, imu_samples, stop_event),
        daemon=True
    )
    serial_thread.start()
    
    try:
        # Main thread runs matplotlib event loop
        while True:
            visualizer.update_plot()
            time.sleep(0.05)  # Update visualization at ~20 Hz
            if not plt.get_fignums():  # Check if window is closed
                break
    
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        stop_event.set()
        plt.close('all')

if __name__ == "__main__":
    main()

