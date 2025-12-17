#!/usr/bin/env python3
"""Simple script to read IMU data and detect high-level movements."""

import sys
import time
import serial
import json
import numpy as np
from collections import deque

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

def integrate_quaternion_from_gyro(gyro_data, timestamps):
    """
    Integrate gyroscope data to get orientation quaternion.
    
    Args:
        gyro_data: (N, 3) array of angular velocities [wx, wy, wz] in rad/s
        timestamps: (N,) array of timestamps
    
    Returns:
        quaternions: (N, 4) array of quaternions [w, x, y, z]
    """
    quaternions = np.zeros((len(gyro_data), 4))
    quaternions[0] = np.array([1.0, 0.0, 0.0, 0.0])  # Initial: no rotation
    
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
ENERGY_MIN = 0.5  # Minimum energy per sample to detect movement
GYRO_ACTIVITY_MAX = 2.0  # Maximum gyro activity (rad²/s²) - above this = rotating, not moving
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence (0-1) to register movement (50%)
WINDOW_TIME_MS = 300  # Rolling window size in milliseconds
MIN_SAMPLES_FOR_DETECTION = 15  # Minimum samples needed before detecting
GRAVITY_MAGNITUDE = 9.82  # Gravity magnitude in m/s²

def detect_movement(imu_samples_with_timestamps, current_time):
    """
    Detect movement intention from IMU samples using:
    1. Orientation-based gravity removal (rotate accel to world frame)
    2. Energy + stationarity gating
    3. 6-unit-vector projection for direction detection
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
    accel_data = samples[:, :3]  # (N, 3) - [ax, ay, az] in m/s² (body frame)
    gyro_data = samples[:, 3:]   # (N, 3) - [gx, gy, gz] in rad/s
    
    # Step 1: Integrate gyro to get orientation quaternions
    quaternions = integrate_quaternion_from_gyro(gyro_data, timestamps)
    
    # Step 2: Rotate acceleration into world frame and subtract gravity
    gravity_world = np.array([0.0, 0.0, GRAVITY_MAGNITUDE])  # Gravity vector in world frame (m/s²)
    linear_accel_world = np.zeros_like(accel_data)
    
    for i in range(len(accel_data)):
        # Rotate body-frame acceleration to world frame
        accel_world = rotate_vector_by_quaternion(accel_data[i], quaternions[i])
        # Subtract gravity
        linear_accel_world[i] = accel_world - gravity_world
    
    # Step 3: Energy + Stationarity Gating
    # Compute window energy (sum of squared accelerations)
    energy = np.sum(linear_accel_world**2)
    energy_per_sample = energy / len(linear_accel_world)
    
    # Compute stationarity (variance of accel and gyro rates)
    accel_variance = np.var(linear_accel_world, axis=0)
    accel_stationarity = np.mean(accel_variance)  # Lower = more stationary
    
    gyro_variance = np.var(gyro_data, axis=0)
    gyro_activity = np.mean(gyro_variance)  # Higher = more rotation
    
    # Gating: if energy too low OR gyro activity too high -> "none"
    if energy_per_sample < ENERGY_MIN or gyro_activity > GYRO_ACTIVITY_MAX:
        return {
            "direction": "none",
            "confidence": 0.0,
            "magnitude": 0.0,
            "vector": [0.0, 0.0, 0.0]
        }
    
    # Step 4: Project onto 6 unit vectors and score
    # Mean acceleration in world frame
    mean_accel_world = np.mean(linear_accel_world, axis=0)
    
    # 6 unit vectors: forward, backward, right, left, up, down
    unit_vectors = np.array([
        [1.0, 0.0, 0.0],   # forward (+X)
        [-1.0, 0.0, 0.0],  # backward (-X)
        [0.0, 1.0, 0.0],   # right (+Y)
        [0.0, -1.0, 0.0],  # left (-Y)
        [0.0, 0.0, 1.0],   # up (+Z)
        [0.0, 0.0, -1.0]   # down (-Z)
    ])
    
    direction_names = ["forward", "backward", "right", "left", "up", "down"]
    
    # Score each direction by projection (dot product)
    scores = np.dot(unit_vectors, mean_accel_world)
    
    # Find best direction
    best_idx = np.argmax(scores)
    best_score = scores[best_idx]
    
    # Only register if score is positive (movement in that direction)
    if best_score <= 0:
        return {
            "direction": "none",
            "confidence": 0.0,
            "magnitude": np.linalg.norm(mean_accel_world),
            "vector": mean_accel_world.tolist()
        }
    
    direction = direction_names[best_idx]
    
    # Calculate confidence based on score magnitude and energy
    magnitude = np.linalg.norm(mean_accel_world)
    
    # Normalize score: best_score / magnitude gives cosine of angle (0-1 range)
    # The dot product of a unit vector with any vector is ≤ magnitude (Cauchy-Schwarz)
    # Clamp best_score to magnitude to prevent numerical errors
    if magnitude > 1e-6:  # Avoid division by zero
        # Clamp best_score to magnitude (should never exceed, but protect against errors)
        best_score_clamped = min(abs(best_score), magnitude)
        score_normalized = best_score_clamped / magnitude
        # Ensure score_normalized is in [0, 1]
        score_normalized = max(0.0, min(score_normalized, 1.0))
    else:
        score_normalized = 0.0
    
    # Energy factor: how much above minimum energy (normalized to 0.5-1.0 range)
    # Cap energy_factor to prevent it from inflating confidence
    energy_factor = min(energy_per_sample / ENERGY_MIN, 2.0) / 2.0
    energy_factor = max(0.5, min(energy_factor, 1.0))  # Clamp to [0.5, 1.0]
    
    # Confidence combines alignment (score_normalized) and energy
    # Both are in [0, 1] range, so result should be in [0, 1]
    confidence = score_normalized * energy_factor
    
    # Final clamp to [0, 1] range - CRITICAL: ensure it's never > 1.0
    confidence = max(0.0, min(confidence, 1.0))
    
    # Convert to percentage (0-100 range)
    confidence_percent = confidence * 100.0
    
    # Final sanity check: ensure percentage is in valid range
    confidence_percent = max(0.0, min(confidence_percent, 100.0))
    #print(f"confidence_percent: {confidence_percent}")
    # Apply confidence threshold (compare against original confidence, not percentage)
    if confidence < CONFIDENCE_THRESHOLD:
        direction = "none"
        confidence_percent = 0.0
    
    return {
        "direction": direction,
        "confidence": confidence_percent,
        "magnitude": magnitude,
        "vector": mean_accel_world.tolist()
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python process_movement.py <serial_port>")
        print("Example: python process_movement.py /dev/ttyUSB0")
        sys.exit(1)
    
    port = sys.argv[1]
    
    # Open serial port
    try:
        ser = serial.Serial(port, 115200, timeout=1)
        # Clear any buffered data
        ser.reset_input_buffer()
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        sys.exit(1)
    
    print(f"Reading IMU data from {port}...")
    print(f"Using {WINDOW_TIME_MS}ms rolling window for movement detection")
    print(f"Energy threshold: {ENERGY_MIN} | Gyro activity max: {GYRO_ACTIVITY_MAX}")
    print("Waiting for data...")
    print("Press Ctrl+C to stop.\n")
    
    # Buffer for samples with timestamps: (timestamp, [ax, ay, az, gx, gy, gz])
    # Use a deque with reasonable maxlen to prevent memory issues
    # At 100Hz, 200ms = 20 samples, so 100 samples gives us ~1 second of history
    imu_samples = deque(maxlen=100)
    line_count = 0
    buffer = ""
    last_detection_time = 0
    
    try:
        while True:
            try:
                # Read available bytes
                if ser.in_waiting > 0:
                    data = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
                    buffer += data
                    
                    # Process complete lines
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        
                        if not line:
                            continue
                        
                        # Debug: print first few lines to see what we're receiving
                        if line_count < 5:
                            #print(f"DEBUG: Received line: {line[:80]}")
                            line_count += 1
                        
                        if line.startswith('{'):
                            try:
                                data = json.loads(line)
                                accel = data['accel']
                                gyro = data['gyro']
                                
                                # Get current timestamp
                                current_time = time.time()
                                
                                # Add to buffer: (timestamp, [ax, ay, az, gx, gy, gz])
                                sample = [
                                    accel['x'], accel['y'], accel['z'],
                                    gyro['x'], gyro['y'], gyro['z']
                                ]
                                imu_samples.append((current_time, sample))
                                
                                # Detect movement using rolling window (check every ~50ms to avoid spam)
                                if current_time - last_detection_time >= 0.05:
                                    movement = detect_movement(imu_samples, current_time)
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
                                if line_count < 10:  # Only print first few errors
                                    print(f"JSON decode error: {e}")
                                    print(f"Line: {line[:80]}")
                                pass
                else:
                    # No data available, small sleep to avoid busy waiting
                    time.sleep(0.01)
                    
            except serial.SerialException as e:
                print(f"Serial error: {e}")
                print("Attempting to reconnect...")
                try:
                    ser.close()
                    time.sleep(1)
                    ser = serial.Serial(port, 115200, timeout=1)
                    ser.reset_input_buffer()
                    buffer = ""
                    print("Reconnected.")
                except:
                    print("Failed to reconnect. Exiting.")
                    break
    
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        ser.close()

if __name__ == "__main__":
    main()

