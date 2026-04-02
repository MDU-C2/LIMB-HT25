#!/usr/bin/env python3
"""
Simple script to read IMU data from BLE and print it.
Supports three gravity removal methods:
  1. highpass - High-pass filter (removes DC bias)
  2. estimate - Estimate gravity vector when still
  3. madgwick - Full orientation estimation using Madgwick filter
"""

import asyncio
import struct
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from bleak import BleakClient, BleakScanner
from bleak.backends.characteristic import BleakGATTCharacteristic
import sys
import os

# Add path for sensor packet serialization
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../ble_central'))
from sensor_packet_serialization import decode_packet, deserialize_packet_data

# BLE UUIDs
SERVICE_UUID = "23011525-1212-efde-1523-785feabcd122"
IMU_CHARACTERISTIC_UUID = "25011525-1212-efde-1523-785feabcd122"

# IMU data format constants (from sensors_service.h)
IMU_BYTES_PER_VALUE = 2  # int16_t
IMU_VALUES_PER_SAMPLE = 6  # gyro x,y,z + accel x,y,z
IMU_SENSOR_COUNT = 2  # We only have 1 sensor, but protocol expects 2
IMU_FREQUENCY = 100  # Hz

# Conversion factors for LSM6DSO32
# Default config: accel range = 4g, gyro range = 250 dps
ACCEL_RANGE_G = 4.0
GRAVITY_MS2 = 9.80665
ACCEL_SCALE_FACTOR = ACCEL_RANGE_G * GRAVITY_MS2 / 32768.0  # m/s² per LSB

GYRO_RANGE_DPS = 250.0
DEG_TO_RAD = 0.0174532925
GYRO_SCALE_FACTOR = GYRO_RANGE_DPS * DEG_TO_RAD / 32768.0  # rad/s per LSB


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

class IMUDataProcessor:
    """Processes IMU data from BLE notifications"""
    
    def __init__(self, method='estimate', visualize=True):
        self.method = method
        self.visualize = visualize
        self.sample_count = 0
        self.velocity = [0.0, 0.0, 0.0]
        self.last_timestamp = None
        
        # Stillness detection parameters
        self.GRAVITY = 9.81
        self.ACCEL_STILL_THRESH = 0.5
        self.GYRO_STILL_THRESH = 0.1
        
        # Velocity integration parameters
        self.DEADBAND_THRESH = 0.15
        self.VELOCITY_DECAY = 0.95
        self.GRAVITY_EMA_ALPHA = 0.05
        
        # Gyro bias tracking
        self.gyro_bias = np.array([0.0, 0.0, 0.0])
        self.still_time_for_bias = 0.0
        self.BIAS_UPDATE_DURATION = 0.5
        
        # Initialize gravity removal based on method
        if method == 'highpass':
            self.ALPHA = 0.98
            self.ax_filtered = 0.0
            self.ay_filtered = 0.0
            self.az_filtered = 0.0
            print("Using high-pass filter for gravity removal")
        elif method == 'estimate':
            self.gravity_vector = np.array([0.0, 0.0, self.GRAVITY])
            print("Using gravity vector estimation (updates when still)")
        elif method == 'madgwick':
            self.q_ws = None
            self.q_ws_initialized = False
            self.beta = 0.05
            self.gyro_bias = np.array([0.0, 0.0, 0.0])
            self.still_time_for_bias = 0.0
            print("Using Madgwick filter for full orientation estimation")
            print("Waiting for stillness to initialize orientation...")
        else:
            print(f"Unknown method: {method}. Using 'estimate'")
            self.method = 'estimate'
            self.gravity_vector = np.array([0.0, 0.0, self.GRAVITY])
        
        # Visualization setup
        if visualize:
            self._setup_visualization()
            # Throttle visualization updates for performance
            # Update every N samples (1 = every sample, 2 = every other sample, etc.)
            self.VIZ_UPDATE_INTERVAL = 10  # Update every 2 samples for smooth 50Hz visualization
            self.last_viz_update_sample = 2
        else:
            self.VIZ_UPDATE_INTERVAL = 999999
            self.last_viz_update_sample = 0
    
    def _setup_visualization(self):
        """Set up matplotlib visualization"""
        plt.ion()
        self.fig = plt.figure(figsize=(12, 6))
        
        # Main plot
        self.ax = self.fig.add_subplot(121)
        self.ax.set_xlabel('Axis', fontsize=12)
        self.ax.set_ylabel('Velocity (m/s)', fontsize=12)
        self.ax.set_title('Real-time Velocity by Axis', fontsize=14)
        self.ax.set_xticks([0, 1, 2])
        self.ax.set_xticklabels(['X', 'Y', 'Z'])
        self.ax.grid(True, alpha=0.3)
        self.ax.set_ylim(-3, 3)
        self.bars = self.ax.bar([0, 1, 2], [0, 0, 0], color=['red', 'green', 'blue'], alpha=0.7)
        
        self.direction_text = self.ax.text(0.5, -0.15, 'Direction: None', 
                                          transform=self.ax.transAxes, 
                                          fontsize=16, fontweight='bold',
                                          ha='center', va='top')
        
        self.countdown_text = self.ax.text(0.5, -0.22, 'Next detection: Ready', 
                                           transform=self.ax.transAxes, 
                                           fontsize=12,
                                           ha='center', va='top',
                                           color='gray')
        
        # History panel
        self.ax_history = self.fig.add_subplot(122)
        self.ax_history.axis('off')
        self.ax_history.set_title('Direction History', fontsize=14, fontweight='bold', pad=20)
        
        self.direction_history = []
        self.MAX_HISTORY = 20
        self.last_direction = None
        self.last_direction_time = None
        self.displayed_direction = None
        self.DIRECTION_TIMEOUT = 4.0
        
        self.history_text = self.ax_history.text(0.1, 0.95, 'No directions yet', 
                                                 transform=self.ax_history.transAxes,
                                                 fontsize=11, va='top', ha='left',
                                                 family='monospace')
        
        plt.tight_layout()
        plt.show(block=False)
    
    def process_imu_data(self, accel_raw, gyro_raw):
        """Process a single IMU sample"""
        self.sample_count += 1
        current_timestamp = time.time()
        
        # Compute time step
        if self.last_timestamp is not None:
            dt = current_timestamp - self.last_timestamp
            dt = min(max(dt, 0.0), 0.1)
        else:
            dt = 0.01
        
        self.last_timestamp = current_timestamp
        
        # Update gyro bias for Madgwick method
        if self.method == 'madgwick':
            accel_mag_check = np.linalg.norm(accel_raw)
            gyro_mag_check = np.linalg.norm(gyro_raw)
            accel_diff_check = abs(accel_mag_check - self.GRAVITY)
            
            is_still_for_bias = (accel_diff_check < self.ACCEL_STILL_THRESH and 
                                 gyro_mag_check < self.GYRO_STILL_THRESH)
            
            if is_still_for_bias:
                self.still_time_for_bias += dt
                if self.still_time_for_bias >= self.BIAS_UPDATE_DURATION:
                    BIAS_EMA_ALPHA = 0.02
                    self.gyro_bias = (1 - BIAS_EMA_ALPHA) * self.gyro_bias + BIAS_EMA_ALPHA * gyro_raw
            else:
                self.still_time_for_bias = 0.0
            
            gyro_corrected = gyro_raw - self.gyro_bias
        else:
            gyro_corrected = gyro_raw
        
        # Compute linear acceleration based on selected method
        if self.method == 'highpass':
            self.ax_filtered = self.ALPHA * self.ax_filtered + (1 - self.ALPHA) * accel_raw[0]
            self.ay_filtered = self.ALPHA * self.ay_filtered + (1 - self.ALPHA) * accel_raw[1]
            self.az_filtered = self.ALPHA * self.az_filtered + (1 - self.ALPHA) * accel_raw[2]
            
            ax_linear = accel_raw[0] - self.ax_filtered
            ay_linear = accel_raw[1] - self.ay_filtered
            az_linear = accel_raw[2] - self.az_filtered
        
        elif self.method == 'estimate':
            accel_mag = np.linalg.norm(accel_raw)
            gyro_mag = np.linalg.norm(gyro_raw)
            accel_diff_from_gravity = abs(accel_mag - self.GRAVITY)
            
            is_still_check = (accel_diff_from_gravity < self.ACCEL_STILL_THRESH and 
                             gyro_mag < self.GYRO_STILL_THRESH)
            
            if is_still_check and accel_mag > 0.1:
                new_gravity = accel_raw * (self.GRAVITY / accel_mag)
                self.gravity_vector = (1 - self.GRAVITY_EMA_ALPHA) * self.gravity_vector + self.GRAVITY_EMA_ALPHA * new_gravity
            
            ax_linear = accel_raw[0] - self.gravity_vector[0]
            ay_linear = accel_raw[1] - self.gravity_vector[1]
            az_linear = accel_raw[2] - self.gravity_vector[2]
        
        elif self.method == 'madgwick':
            if not self.q_ws_initialized:
                accel_mag_init = np.linalg.norm(accel_raw)
                gyro_mag_init = np.linalg.norm(gyro_raw)
                accel_diff_init = abs(accel_mag_init - self.GRAVITY)
                
                is_still_init = (accel_diff_init < self.ACCEL_STILL_THRESH and 
                                gyro_mag_init < self.GYRO_STILL_THRESH)
                
                if is_still_init and accel_mag_init > 0.1:
                    g_s_normalized = accel_raw / accel_mag_init
                    g_w_normalized = np.array([0.0, 0.0, 1.0])
                    
                    v = np.cross(g_s_normalized, g_w_normalized)
                    s = np.linalg.norm(v)
                    c = np.dot(g_s_normalized, g_w_normalized)
                    
                    if s < 1e-6:
                        if c > 0:
                            self.q_ws = np.array([1.0, 0.0, 0.0, 0.0])
                        else:
                            self.q_ws = np.array([0.0, 1.0, 0.0, 0.0])
                    else:
                        vx = np.array([
                            [0, -v[2], v[1]],
                            [v[2], 0, -v[0]],
                            [-v[1], v[0], 0]
                        ])
                        R = np.eye(3) + vx + vx @ vx * (1 - c) / (s**2)
                        
                        trace = np.trace(R)
                        if trace > 0:
                            s_q = np.sqrt(trace + 1.0) * 2
                            w = 0.25 * s_q
                            x = (R[2, 1] - R[1, 2]) / s_q
                            y = (R[0, 2] - R[2, 0]) / s_q
                            z = (R[1, 0] - R[0, 1]) / s_q
                            self.q_ws = normalize_quaternion(np.array([w, x, y, z]))
                        else:
                            self.q_ws = np.array([1.0, 0.0, 0.0, 0.0])
                    self.q_ws_initialized = True
                    print(f"Madgwick initialized! Sample #{self.sample_count}")
            
            if self.q_ws is None or not self.q_ws_initialized:
                ax_linear = accel_raw[0]
                ay_linear = accel_raw[1]
                az_linear = accel_raw[2] - self.GRAVITY
            else:
                self.q_ws = madgwick_update(self.q_ws, accel_raw, gyro_corrected, dt, self.beta)
                
                w, x, y, z = self.q_ws
                R = np.array([
                    [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                    [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                    [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
                ])
                
                g_world = np.array([0.0, 0.0, self.GRAVITY])
                g_sensor = R.T @ g_world
                
                ax_linear = accel_raw[0] - g_sensor[0]
                ay_linear = accel_raw[1] - g_sensor[1]
                az_linear = accel_raw[2] - g_sensor[2]
        
        # Check if stationary
        a_lin_mag = np.sqrt(ax_linear**2 + ay_linear**2 + az_linear**2)
        gyro_mag = np.linalg.norm(gyro_raw)
        
        is_still = (a_lin_mag < self.ACCEL_STILL_THRESH and 
                   gyro_mag < self.GYRO_STILL_THRESH)
        
        if is_still:
            print(f"I still right now...")
            self.velocity = [0.0, 0.0, 0.0]
        else:
            if a_lin_mag < self.DEADBAND_THRESH:
                self.velocity[0] *= self.VELOCITY_DECAY
                self.velocity[1] *= self.VELOCITY_DECAY
                self.velocity[2] *= self.VELOCITY_DECAY
            else:
                self.velocity[0] += ax_linear * dt
                self.velocity[1] += ay_linear * dt
                self.velocity[2] += az_linear * dt
        
        # Print to console every 25 samples (to avoid spam)
        if 1 < 0:
            print(f"Sample #{self.sample_count}:")
            print(f"  Accel (raw): x={accel_raw[0]:7.3f}  y={accel_raw[1]:7.3f}  z={accel_raw[2]:7.3f} m/s²")
            print(f"  Accel (linear): x={ax_linear:7.3f}  y={ay_linear:7.3f}  z={az_linear:7.3f} m/s²")
            if self.method == 'estimate':
                print(f"  Gravity vector: x={self.gravity_vector[0]:7.3f}  y={self.gravity_vector[1]:7.3f}  z={self.gravity_vector[2]:7.3f} m/s²")
            print(f"  Gyro:  x={gyro_raw[0]:7.3f}  y={gyro_raw[1]:7.3f}  z={gyro_raw[2]:7.3f} rad/s")
            print(f"  Vel:   x={self.velocity[0]:7.3f}  y={self.velocity[1]:7.3f}  z={self.velocity[2]:7.3f} m/s")
            print()
        
        # Update visualization live (throttled for smooth performance)
        if self.visualize and (self.sample_count - self.last_viz_update_sample) >= self.VIZ_UPDATE_INTERVAL:
            self.last_viz_update_sample = self.sample_count
            self._update_visualization(current_timestamp)
    
    def _update_visualization(self, current_timestamp):
        """Update the visualization"""
        # Update bar heights
        self.bars[0].set_height(self.velocity[0])
        self.bars[1].set_height(self.velocity[1])
        self.bars[2].set_height(self.velocity[2])
        
        # Auto-adjust y-axis range
        max_vel = max(abs(v) for v in self.velocity)
        if max_vel > 0:
            y_range = max(5, max_vel * 1.2)
            self.ax.set_ylim(-y_range, y_range)
        
        # Update colors
        for i, bar in enumerate(self.bars):
            if self.velocity[i] >= 0:
                bar.set_color(['red', 'green', 'blue'][i])
            else:
                bar.set_color(['darkred', 'darkgreen', 'darkblue'][i])
        
        # Direction detection
        time_since_last = current_timestamp - self.last_direction_time if self.last_direction_time is not None else float('inf')
        can_detect_new = time_since_last >= self.DIRECTION_TIMEOUT
        
        if self.last_direction_time is not None:
            time_remaining = self.DIRECTION_TIMEOUT - time_since_last
            if time_remaining > 0:
                countdown_str = f'Next detection: {time_remaining:.1f}s'
                self.countdown_text.set_color('orange')
            else:
                countdown_str = 'Next detection: Ready'
                self.countdown_text.set_color('green')
        else:
            countdown_str = 'Next detection: Ready'
            self.countdown_text.set_color('green')
        self.countdown_text.set_text(countdown_str)
        
        VEL_THRESH = 0.2
        abs_velocities = [abs(self.velocity[0]), abs(self.velocity[1]), abs(self.velocity[2])]
        max_idx = np.argmax(abs_velocities)
        max_vel_mag = abs_velocities[max_idx]
        
        if can_detect_new:
            if max_vel_mag > VEL_THRESH:
                if max_idx == 0:
                    current_direction = "Forward" if self.velocity[0] < 0 else "Back"
                elif max_idx == 1:
                    current_direction = "Left" if self.velocity[1] < 0 else "Right"
                else:
                    current_direction = "Up" if self.velocity[2] > 0 else "Down"
                
                if current_direction != self.displayed_direction:
                    self.displayed_direction = current_direction
                    self.last_direction = current_direction
                    self.last_direction_time = current_timestamp
                    
                    self.direction_history.append(current_direction)
                    if len(self.direction_history) > self.MAX_HISTORY:
                        self.direction_history.pop(0)
                    
                    history_lines = self.direction_history[-self.MAX_HISTORY:]
                    history_str = '\n'.join([f"{i+1}. {d}" for i, d in enumerate(history_lines)])
                    if not history_str:
                        history_str = "No directions yet"
                    self.history_text.set_text(history_str)
            else:
                self.displayed_direction = None
        
        direction_str = f"Direction: {self.displayed_direction}" if self.displayed_direction else "Direction: None"
        self.direction_text.set_text(direction_str)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


async def read_and_print_imu_ble(device_name="LIMBServer", method='estimate', visualize=True):
    """
    Read IMU data from BLE device and print it.
    
    Args:
        device_name: BLE device name to connect to (default: LIMBServer)
        method: Gravity removal method - 'highpass', 'estimate', or 'madgwick'
        visualize: Whether to show real-time velocity plot (default: True)
    """
    print(f"Scanning for BLE device: {device_name}")
    device = await BleakScanner.find_device_by_name(device_name, timeout=10.0)
    
    if device is None:
        print(f"Error: Could not find BLE device '{device_name}'")
        print("Make sure the device is powered on and advertising.")
        return
    
    print(f"Found device: {device.name} ({device.address})")
    
    processor = IMUDataProcessor(method=method, visualize=visualize)
    
    def notification_handler(characteristic: BleakGATTCharacteristic, data: bytearray):
        """Handle IMU BLE notifications"""
        try:
            # Decode packet: [4 bytes sequence][sensor data]
            sequence_number, sensor_data = decode_packet(memoryview(data))
            
            # Deserialize sensor data
            # Format: For each sample, for each sensor: gyro.x, gyro.y, gyro.z, accel.x, accel.y, accel.z (int16_t)
            sensors = deserialize_packet_data(
                sensor_data,
                bytes_per_value=IMU_BYTES_PER_VALUE,
                values_per_sample=IMU_VALUES_PER_SAMPLE,
                sensor_count=IMU_SENSOR_COUNT,
                signed=True,
            )
            
            # Process first sensor's first sample (we only have 1 sensor)
            if len(sensors) > 0 and len(sensors[0]) > 0:
                sample = sensors[0][0]  # First sensor, first sample
                
                # Extract values: [gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z]
                gyro_raw = np.array([
                    sample[0] * GYRO_SCALE_FACTOR,  # Convert to rad/s
                    sample[1] * GYRO_SCALE_FACTOR,
                    sample[2] * GYRO_SCALE_FACTOR
                ])
                
                accel_raw = np.array([
                    sample[3] * ACCEL_SCALE_FACTOR,  # Convert to m/s²
                    sample[4] * ACCEL_SCALE_FACTOR,
                    sample[5] * ACCEL_SCALE_FACTOR
                ])
                
                processor.process_imu_data(accel_raw, gyro_raw)
        
        except Exception as e:
            print(f"Error processing BLE data: {e}")
            import traceback
            traceback.print_exc()
    
    async with BleakClient(device) as client:
        print(f"Connected to {device_name}")
        print(f"Gravity removal method: {method}")
        print("Reading IMU data... (Press Ctrl+C to stop)\n")
        
        # Subscribe to IMU notifications
        await client.start_notify(IMU_CHARACTERISTIC_UUID, notification_handler)
        
        try:
            # Keep running until interrupted
            while True:
                await asyncio.sleep(1.0)
        except KeyboardInterrupt:
            print(f"\n\nStopped. Read {processor.sample_count} samples total.")
        finally:
            await client.stop_notify(IMU_CHARACTERISTIC_UUID)
            print("BLE connection closed")
            if visualize:
                plt.ioff()
                print("Close the plot window to exit.")
                plt.show(block=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Read and print IMU data from BLE device',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Gravity removal methods:
  highpass  - High-pass filter removes DC bias (simple, fast)
  estimate  - Estimate gravity vector when still (good balance)
  madgwick  - Full orientation estimation (most accurate, slower)
        """
    )
    parser.add_argument('--device', type=str, default='LIMBServer',
                       help='BLE device name (default: LIMBServer)')
    parser.add_argument('--method', type=str, default='madgwick',
                       choices=['highpass', 'estimate', 'madgwick'],
                       help='Gravity removal method (default: estimate)')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable real-time velocity visualization')
    
    args = parser.parse_args()
    
    asyncio.run(read_and_print_imu_ble(device_name=args.device, 
                                       method=args.method, 
                                       visualize=not args.no_viz))
