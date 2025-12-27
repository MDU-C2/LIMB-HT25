#!/usr/bin/env python3
"""
Simple script to read IMU data from serial and print it.
Supports three gravity removal methods:
  1. highpass - High-pass filter (removes DC bias)
  2. estimate - Estimate gravity vector when still
  3. madgwick - Full orientation estimation using Madgwick filter
"""

import serial
import serial.tools.list_ports
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ============================================================================
# Quaternion utilities for Madgwick filter (Option 3)
# ============================================================================

def normalize_quaternion(q):
    """Normalize quaternion to unit length."""
    norm = np.linalg.norm(q)
    if norm > 1e-8:
        return q / norm
    return np.array([1.0, 0.0, 0.0, 0.0])


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


def quaternion_conjugate(q):
    """Return quaternion conjugate (inverse rotation)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def rotate_vector_by_quaternion(v, q):
    """
    Rotate vector from sensor frame to world frame using quaternion.
    v_world = q* ⊗ [0, v_sensor] ⊗ q
    """
    v_quat = np.array([0.0, v[0], v[1], v[2]])
    q_conj = quaternion_conjugate(q)
    temp = quaternion_multiply(q_conj, v_quat)
    result = quaternion_multiply(temp, q)
    return result[1:]


def madgwick_update(q, accel, gyro, dt, beta=0.1):
    """
    Madgwick filter update for IMU orientation estimation.
    
    Args:
        q: Initial quaternion [w, x, y, z]
        accel: Acceleration [ax, ay, az] in m/s²
        gyro: Angular velocity [wx, wy, wz] in rad/s
        dt: Time step in seconds
        beta: Filter gain
    
    Returns:
        Updated quaternion [w, x, y, z]
    """
    accel_norm = np.linalg.norm(accel)
    
    if accel_norm < 1e-8:
        # No acceleration, pure gyro integration
        omega_quat = np.array([0.0, gyro[0], gyro[1], gyro[2]])
        q_dot = 0.5 * quaternion_multiply(q, omega_quat)
        q_new = q + q_dot * dt
        return normalize_quaternion(q_new)
    
    accel_normalized = accel / accel_norm
    
    # Compute objective function and Jacobian
    w, x, y, z = q
    
    f_g = np.array([
        2*(x*z - w*y) - accel_normalized[0],
        2*(w*x + y*z) - accel_normalized[1],
        2*(0.5 - x*x - y*y) - accel_normalized[2]
    ])
    
    J_g = np.array([
        [-2*y, 2*z, -2*w, 2*x],
        [2*x, 2*w, 2*z, 2*y],
        [0, -4*x, -4*y, 0]
    ])
    
    # Gradient descent step
    step = J_g.T @ f_g
    step_norm = np.linalg.norm(step)
    if step_norm > 1e-8:
        step = step / step_norm
    
    # Update quaternion
    omega_quat = np.array([0.0, gyro[0], gyro[1], gyro[2]])
    q_dot_gyro = 0.5 * quaternion_multiply(q, omega_quat)
    q_dot_correction = -beta * step
    q_dot = q_dot_gyro + q_dot_correction
    q_new = q + q_dot * dt
    
    return normalize_quaternion(q_new)


# ============================================================================
# Main function
# ============================================================================

def read_and_print_imu(port=None, baudrate=115200, method='estimate', visualize=True):
    """
    Read IMU data from serial port and print it.
    
    Args:
        port: Serial port path (e.g., '/dev/ttyUSB0' or 'COM3'). If None, auto-detects.
        baudrate: Serial baudrate (default 115200)
        method: Gravity removal method - 'highpass', 'estimate', or 'madgwick'
        visualize: Whether to show real-time velocity plot (default: True)
    """
    # Auto-detect port if not specified
    if port is None:
        ports = serial.tools.list_ports.comports()
        for p in ports:
            # Look for common ESP32 identifiers
            if any(identifier in p.description.lower() for identifier in 
                   ['cp210', 'ch340', 'ftdi', 'usb serial', 'esp32', 'usbmodem']):
                port = p.device
                print(f"Auto-detected ESP32 device: {port} ({p.description})")
                break
        
        if port is None:
            print("Error: Could not auto-detect ESP32 device")
            print("Available ports:")
            for p in ports:
                print(f"  {p.device}: {p.description}")
            return
    
    # Open serial connection
    try:
        ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=1.0,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS
        )
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        return
    
    print(f"Connected to {port} at {baudrate} baud")
    print(f"Gravity removal method: {method}")
    print("Reading IMU data... (Press Ctrl+C to stop)\n")
    
    # Set up visualization if enabled
    if visualize:
        plt.ion()  # Turn on interactive mode
        fig = plt.figure(figsize=(12, 6))
        
        # Create main plot (left side)
        ax = fig.add_subplot(121)  # 1 row, 2 cols, position 1
        ax.set_xlabel('Axis', fontsize=12)
        ax.set_ylabel('Velocity (m/s)', fontsize=12)
        ax.set_title('Real-time Velocity by Axis', fontsize=14)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['X', 'Y', 'Z'])
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-3, 3)  # Initial y-axis range, will auto-adjust
        bars = ax.bar([0, 1, 2], [0, 0, 0], color=['red', 'green', 'blue'], alpha=0.7)
        
        # Add direction text at the bottom of the plot
        direction_text = ax.text(0.5, -0.15, 'Direction: None', 
                                transform=ax.transAxes, 
                                fontsize=16, fontweight='bold',
                                ha='center', va='top')
        
        # Add countdown timer text
        countdown_text = ax.text(0.5, -0.22, 'Next detection: Ready', 
                                 transform=ax.transAxes, 
                                 fontsize=12,
                                 ha='center', va='top',
                                 color='gray')
        
        # Create history panel (right side)
        ax_history = fig.add_subplot(122)  # 1 row, 2 cols, position 2
        ax_history.axis('off')  # Hide axes
        ax_history.set_title('Direction History', fontsize=14, fontweight='bold', pad=20)
        
        # History list storage
        direction_history = []
        MAX_HISTORY = 20  # Maximum number of directions to store
        last_direction = None
        last_direction_time = None
        displayed_direction = None  # The direction currently being displayed
        DIRECTION_TIMEOUT = 4.0  # seconds - minimum time between direction detections
        
        # Initial history text
        history_text = ax_history.text(0.1, 0.95, 'No directions yet', 
                                       transform=ax_history.transAxes,
                                       fontsize=11, va='top', ha='left',
                                       family='monospace')
        
        plt.tight_layout()
        plt.show(block=False)
    
    buffer = ""
    sample_count = 0
    
    # Velocity tracking
    velocity = [0.0, 0.0, 0.0]  # [vx, vy, vz] in m/s
    last_timestamp = None
    
    # Stillness detection parameters
    GRAVITY = 9.81  # m/s²
    ACCEL_STILL_THRESH = 0.5  # m/s² - how close to gravity to consider still
    GYRO_STILL_THRESH = 0.1  # rad/s - max gyro magnitude to consider still
    
    # Velocity integration parameters
    DEADBAND_THRESH = 0.15  # m/s² - ignore accelerations below this
    VELOCITY_DECAY = 0.95  # Decay factor per sample when acceleration is small
    GRAVITY_EMA_ALPHA = 0.05  # EMA smoothing factor for gravity vector estimation
    
    # Gyro bias tracking (for Madgwick)
    gyro_bias = np.array([0.0, 0.0, 0.0])
    still_time_for_bias = 0.0
    BIAS_UPDATE_DURATION = 0.5  # seconds of stillness before updating bias
    
    # Initialize gravity removal based on method
    if method == 'highpass':
        # Option 1: High-pass filter
        ALPHA = 0.98  # High-pass filter coefficient
        ax_filtered = 0.0
        ay_filtered = 0.0
        az_filtered = 0.0
        print("Using high-pass filter for gravity removal")
    elif method == 'estimate':
        # Option 2: Estimate gravity vector when still
        gravity_vector = np.array([0.0, 0.0, GRAVITY])  # Default: gravity in Z
        print("Using gravity vector estimation (updates when still)")
    elif method == 'madgwick':
        # Option 3: Full orientation estimation
        q_ws = None  # Will be initialized when still
        q_ws_initialized = False
        beta = 0.05  # Madgwick filter gain (lower = more stable)
        gyro_bias = np.array([0.0, 0.0, 0.0])  # Gyro bias tracking
        still_time_for_bias = 0.0
        BIAS_UPDATE_DURATION = 0.5  # seconds of stillness before updating bias
        print("Using Madgwick filter for full orientation estimation")
        print("Waiting for stillness to initialize orientation...")
    else:
        print(f"Unknown method: {method}. Using 'estimate'")
        method = 'estimate'
        gravity_vector = np.array([0.0, 0.0, GRAVITY])
    
    try:
        while True:
            # Read available data
            if ser.in_waiting > 0:
                data = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
                buffer += data
                
                # Process complete lines
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    
                    if not line:
                        continue
                    
                    # Skip ESP-IDF log messages
                    if line.startswith(('I (', 'W (', 'E (', 'D (')):
                        continue
                    
                    # Parse JSON
                    if line.startswith('{'):
                        try:
                            data = json.loads(line)
                            
                            # Extract accel and gyro
                            accel_data = data.get('accel', {})
                            gyro_data = data.get('gyro', {})
                            temp = data.get('temp', None)
                            
                            if accel_data and gyro_data:
                                sample_count += 1
                                
                                # Get current timestamp
                                current_timestamp = time.time()
                                
                                # Compute time step
                                if last_timestamp is not None:
                                    dt = current_timestamp - last_timestamp
                                    # Limit dt to avoid instability
                                    dt = min(max(dt, 0.0), 0.1)
                                else:
                                    dt = 0.01  # Default 10ms for first sample
                                
                                last_timestamp = current_timestamp
                                
                                # Get acceleration and gyro values
                                ax_raw = accel_data.get('x', 0.0)
                                ay_raw = accel_data.get('y', 0.0)
                                az_raw = accel_data.get('z', 0.0)
                                gx = gyro_data.get('x', 0.0)
                                gy = gyro_data.get('y', 0.0)
                                gz = gyro_data.get('z', 0.0)
                                
                                accel_raw = np.array([ax_raw, ay_raw, az_raw])
                                gyro_raw = np.array([gx, gy, gz])
                                
                                # Update gyro bias for Madgwick method (before computing linear accel)
                                if method == 'madgwick':
                                    accel_mag_check = np.linalg.norm(accel_raw)
                                    gyro_mag_check = np.linalg.norm(gyro_raw)
                                    accel_diff_check = abs(accel_mag_check - GRAVITY)
                                    
                                    is_still_for_bias = (accel_diff_check < ACCEL_STILL_THRESH and 
                                                         gyro_mag_check < GYRO_STILL_THRESH)
                                    
                                    if is_still_for_bias:
                                        still_time_for_bias += dt
                                        if still_time_for_bias >= BIAS_UPDATE_DURATION:
                                            # Update bias using EMA
                                            BIAS_EMA_ALPHA = 0.02
                                            gyro_bias = (1 - BIAS_EMA_ALPHA) * gyro_bias + BIAS_EMA_ALPHA * gyro_raw
                                    else:
                                        still_time_for_bias = 0.0
                                    
                                    # Apply gyro bias correction
                                    gyro_corrected = gyro_raw - gyro_bias
                                else:
                                    gyro_corrected = gyro_raw
                                
                                # Compute linear acceleration based on selected method
                                if method == 'highpass':
                                    # Option 1: High-pass filter
                                    ax_filtered = ALPHA * ax_filtered + (1 - ALPHA) * ax_raw
                                    ay_filtered = ALPHA * ay_filtered + (1 - ALPHA) * ay_raw
                                    az_filtered = ALPHA * az_filtered + (1 - ALPHA) * az_raw
                                    
                                    ax_linear = ax_raw - ax_filtered
                                    ay_linear = ay_raw - ay_filtered
                                    az_linear = az_raw - az_filtered
                                    
                                elif method == 'estimate':
                                    # Option 2: Estimate gravity vector when still (with EMA smoothing)
                                    accel_mag = np.linalg.norm(accel_raw)
                                    gyro_mag = np.linalg.norm(gyro_raw)
                                    accel_diff_from_gravity = abs(accel_mag - GRAVITY)
                                    
                                    is_still_check = (accel_diff_from_gravity < ACCEL_STILL_THRESH and 
                                                     gyro_mag < GYRO_STILL_THRESH)
                                    
                                    if is_still_check and accel_mag > 0.1:
                                        # Update gravity vector estimate with EMA (smoother)
                                        new_gravity = accel_raw * (GRAVITY / accel_mag)
                                        gravity_vector = (1 - GRAVITY_EMA_ALPHA) * gravity_vector + GRAVITY_EMA_ALPHA * new_gravity
                                    
                                    # Remove gravity from all axes
                                    ax_linear = ax_raw - gravity_vector[0]
                                    ay_linear = ay_raw - gravity_vector[1]
                                    az_linear = az_raw - gravity_vector[2]
                                    
                                elif method == 'madgwick':
                                    # Option 3: Full orientation estimation
                                    # Initialize orientation when still (not on first sample)
                                    if not q_ws_initialized:
                                        accel_mag_init = np.linalg.norm(accel_raw)
                                        gyro_mag_init = np.linalg.norm(gyro_raw)
                                        accel_diff_init = abs(accel_mag_init - GRAVITY)
                                        
                                        # Wait for stillness before initializing
                                        is_still_init = (accel_diff_init < ACCEL_STILL_THRESH and 
                                                        gyro_mag_init < GYRO_STILL_THRESH)
                                        
                                        if is_still_init and accel_mag_init > 0.1:
                                            # Gravity in sensor frame (normalized)
                                            g_s_normalized = accel_raw / accel_mag_init
                                            # Gravity in world frame (Z up)
                                            g_w_normalized = np.array([0.0, 0.0, 1.0])
                                            
                                            # Find rotation that aligns g_s to g_w
                                            v = np.cross(g_s_normalized, g_w_normalized)
                                            s = np.linalg.norm(v)
                                            c = np.dot(g_s_normalized, g_w_normalized)
                                            
                                            if s < 1e-6:
                                                if c > 0:
                                                    q_ws = np.array([1.0, 0.0, 0.0, 0.0])
                                                else:
                                                    # 180 degree rotation
                                                    q_ws = np.array([0.0, 1.0, 0.0, 0.0])
                                            else:
                                                # Rodrigues' rotation formula
                                                vx = np.array([
                                                    [0, -v[2], v[1]],
                                                    [v[2], 0, -v[0]],
                                                    [-v[1], v[0], 0]
                                                ])
                                                R = np.eye(3) + vx + vx @ vx * (1 - c) / (s**2)
                                                
                                                # Convert rotation matrix to quaternion
                                                trace = np.trace(R)
                                                if trace > 0:
                                                    s_q = np.sqrt(trace + 1.0) * 2
                                                    w = 0.25 * s_q
                                                    x = (R[2, 1] - R[1, 2]) / s_q
                                                    y = (R[0, 2] - R[2, 0]) / s_q
                                                    z = (R[1, 0] - R[0, 1]) / s_q
                                                    q_ws = normalize_quaternion(np.array([w, x, y, z]))
                                                else:
                                                    # Fallback
                                                    q_ws = np.array([1.0, 0.0, 0.0, 0.0])
                                            q_ws_initialized = True
                                            print(f"Madgwick initialized! Sample #{sample_count}")
                                    
                                    # Safety check: ensure q_ws is initialized
                                    if q_ws is None or not q_ws_initialized:
                                        # Not initialized yet - use raw acceleration (no gravity removal)
                                        ax_linear = ax_raw
                                        ay_linear = ay_raw
                                        az_linear = az_raw - GRAVITY  # Simple Z-axis gravity removal
                                    else:
                                        # Update Madgwick filter with bias-corrected gyro
                                        q_ws = madgwick_update(q_ws, accel_raw, gyro_corrected, dt, beta)
                                        
                                        # Compute gravity in sensor frame using rotation matrix (more reliable)
                                        # Convert quaternion to rotation matrix
                                        w, x, y, z = q_ws
                                        R = np.array([
                                            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                                            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                                            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
                                        ])
                                        
                                        # Gravity in world frame (Z up)
                                        g_world = np.array([0.0, 0.0, GRAVITY])
                                        # Rotate to sensor frame: g_sensor = R^T * g_world
                                        g_sensor = R.T @ g_world
                                        
                                        # Remove gravity from acceleration
                                        ax_linear = ax_raw - g_sensor[0]
                                        ay_linear = ay_raw - g_sensor[1]
                                        az_linear = az_raw - g_sensor[2]
                                
                                # Check if stationary (using LINEAR acceleration for better detection)
                                a_lin_mag = np.sqrt(ax_linear**2 + ay_linear**2 + az_linear**2)
                                gyro_mag = np.linalg.norm(gyro_raw)
                                
                                is_still = (a_lin_mag < ACCEL_STILL_THRESH and 
                                           gyro_mag < GYRO_STILL_THRESH)
                                
                                if is_still:
                                    # Reset velocity to zero when stationary
                                    velocity = [0.0, 0.0, 0.0]
                                else:
                                    # Apply deadband to filter out small accelerations
                                    if a_lin_mag < DEADBAND_THRESH:
                                        # Decay velocity when acceleration is small (reduces drift)
                                        velocity[0] *= VELOCITY_DECAY
                                        velocity[1] *= VELOCITY_DECAY
                                        velocity[2] *= VELOCITY_DECAY
                                    else:
                                        # Integrate LINEAR acceleration (gravity removed) to get velocity: v = v0 + a * dt
                                        velocity[0] += ax_linear * dt
                                        velocity[1] += ay_linear * dt
                                        velocity[2] += az_linear * dt
                                
                                # Update visualization and print every 25 samples
                                if sample_count % 25 == 0:
                                    print(f"Sample #{sample_count}:")
                                    print(f"  Accel (raw): x={ax_raw:7.3f}  y={ay_raw:7.3f}  z={az_raw:7.3f} m/s²")
                                    print(f"  Accel (linear): x={ax_linear:7.3f}  y={ay_linear:7.3f}  z={az_linear:7.3f} m/s²")
                                    if method == 'estimate':
                                        print(f"  Gravity vector: x={gravity_vector[0]:7.3f}  y={gravity_vector[1]:7.3f}  z={gravity_vector[2]:7.3f} m/s²")
                                    print(f"  Gyro:  x={gyro_data.get('x', 0.0):7.3f}  "
                                          f"y={gyro_data.get('y', 0.0):7.3f}  "
                                          f"z={gyro_data.get('z', 0.0):7.3f} rad/s")
                                    print(f"  Vel:   x={velocity[0]:7.3f}  y={velocity[1]:7.3f}  z={velocity[2]:7.3f} m/s")
                                    if temp is not None:
                                        print(f"  Temp:  {temp:.1f} °C")
                                    print()
                                    
                                    # Update visualization
                                    if visualize:
                                        # Update bar heights
                                        bars[0].set_height(velocity[0])
                                        bars[1].set_height(velocity[1])
                                        bars[2].set_height(velocity[2])
                                        
                                        # Auto-adjust y-axis range
                                        max_vel = max(abs(v) for v in velocity)
                                        if max_vel > 0:
                                            y_range = max(5, max_vel * 1.2)
                                            ax.set_ylim(-y_range, y_range)
                                        
                                        # Update colors based on sign
                                        for i, bar in enumerate(bars):
                                            if velocity[i] >= 0:
                                                bar.set_color(['red', 'green', 'blue'][i])
                                            else:
                                                bar.set_color(['darkred', 'darkgreen', 'darkblue'][i])
                                        
                                        # Check if enough time has passed to allow new direction detection
                                        time_since_last = current_timestamp - last_direction_time if last_direction_time is not None else float('inf')
                                        can_detect_new = time_since_last >= DIRECTION_TIMEOUT
                                        
                                        # Update countdown timer (always update, even if direction hasn't changed)
                                        if last_direction_time is not None:
                                            time_remaining = DIRECTION_TIMEOUT - time_since_last
                                            if time_remaining > 0:
                                                countdown_str = f'Next detection: {time_remaining:.1f}s'
                                                countdown_text.set_color('orange')
                                            else:
                                                countdown_str = 'Next detection: Ready'
                                                countdown_text.set_color('green')
                                        else:
                                            countdown_str = 'Next detection: Ready'
                                            countdown_text.set_color('green')
                                        countdown_text.set_text(countdown_str)
                                        
                                        # Determine dominant direction based on velocity magnitude
                                        VEL_THRESH = 0.2  # Minimum velocity to show direction
                                        
                                        # Find the axis with the largest absolute velocity
                                        abs_velocities = [abs(velocity[0]), abs(velocity[1]), abs(velocity[2])]
                                        max_idx = np.argmax(abs_velocities)
                                        max_vel_mag = abs_velocities[max_idx]
                                        
                                        # Only detect/update direction if timeout has passed
                                        if can_detect_new:
                                            if max_vel_mag > VEL_THRESH:
                                                # Determine direction based on dominant axis and sign
                                                if max_idx == 0:  # X-axis
                                                    if velocity[0] < 0:
                                                        current_direction = "Forward"
                                                    else:
                                                        current_direction = "Back"
                                                elif max_idx == 1:  # Y-axis
                                                    if velocity[1] < 0:
                                                        current_direction = "Left"
                                                    else:
                                                        current_direction = "Right"
                                                else:  # Z-axis
                                                    if velocity[2] > 0:
                                                        current_direction = "Up"
                                                    else:
                                                        current_direction = "Down"
                                                
                                                # Only update if direction changed
                                                if current_direction != displayed_direction:
                                                    displayed_direction = current_direction
                                                    last_direction = current_direction
                                                    last_direction_time = current_timestamp
                                                    
                                                    # Add to history
                                                    direction_history.append(current_direction)
                                                    if len(direction_history) > MAX_HISTORY:
                                                        direction_history.pop(0)  # Remove oldest
                                                    
                                                    # Update history text
                                                    history_lines = direction_history[-MAX_HISTORY:]  # Show last N
                                                    history_str = '\n'.join([f"{i+1}. {d}" for i, d in enumerate(history_lines)])
                                                    if not history_str:
                                                        history_str = "No directions yet"
                                                    history_text.set_text(history_str)
                                                    
                                                    # Reset countdown display
                                                    countdown_str = f'Next detection: {DIRECTION_TIMEOUT:.1f}s'
                                                    countdown_text.set_color('orange')
                                                    countdown_text.set_text(countdown_str)
                                            else:
                                                # Below threshold - show None only if we can detect
                                                displayed_direction = None
                                                current_direction = None
                                        # else: keep showing the last displayed direction during countdown
                                        
                                        # Display the current direction (either new or frozen during countdown)
                                        if displayed_direction is not None:
                                            direction_str = f"Direction: {displayed_direction}"
                                        else:
                                            direction_str = "Direction: None"
                                        
                                        direction_text.set_text(direction_str)
                                        
                                        fig.canvas.draw()
                                        fig.canvas.flush_events()
                                
                        except json.JSONDecodeError:
                            # Skip invalid JSON lines
                            pass
                        except Exception as e:
                            print(f"Error parsing line: {e}")
                            print(f"Line: {line[:100]}\n")
            
            # Small sleep to prevent busy waiting
            time.sleep(0.001)
            
    except KeyboardInterrupt:
        print(f"\n\nStopped. Read {sample_count} samples total.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        ser.close()
        print("Serial connection closed")
        if visualize:
            plt.ioff()  # Turn off interactive mode
            print("Close the plot window to exit.")
            plt.show(block=True)  # Keep plot open until closed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Read and print IMU data from serial port',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Gravity removal methods:
  highpass  - High-pass filter removes DC bias (simple, fast)
  estimate  - Estimate gravity vector when still (good balance)
  madgwick  - Full orientation estimation (most accurate, slower)
        """
    )
    parser.add_argument('--port', type=str, default=None,
                       help='Serial port (e.g., /dev/ttyUSB0 or COM3). Auto-detects if not specified.')
    parser.add_argument('--baudrate', type=int, default=115200,
                       help='Serial baudrate (default: 115200)')
    parser.add_argument('--method', type=str, default='estimate',
                       choices=['highpass', 'estimate', 'madgwick'],
                       help='Gravity removal method (default: estimate)')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable real-time velocity visualization')
    
    args = parser.parse_args()
    
    read_and_print_imu(port=args.port, baudrate=args.baudrate, 
                       method=args.method, visualize=not args.no_viz)
