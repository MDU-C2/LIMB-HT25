#!/usr/bin/env python3
"""
Main script for IMU-Vision sensor fusion using complementary filter.

This script:
1. Reads IMU data from ESP32
2. Reads vision data from OAK-D
3. Fuses them using complementary filter
4. Displays real-time pose estimation

Usage:
    python fusion_main.py
"""

import sys
import os
import time
import numpy as np

# Add parent directory to path to import sensors and vision modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sensors.imu_reader import IMUReader
from data_fusion.complementary_filter import ComplementaryFilter
from data_fusion.ekf_filter import ExtendedKalmanFilter
from vision.system import VisionSystem


def print_state(state: dict, frame_count: int):
    """Pretty print the current state."""
    pos = state['position']
    vel = state['velocity']
    orient = state['orientation']
    ang_vel = state['angular_velocity']
    
    # Convert orientation to Euler angles for readability
    roll, pitch, yaw = ComplementaryFilter.quaternion_to_euler(orient)
    roll_deg = np.rad2deg(roll)
    pitch_deg = np.rad2deg(pitch)
    yaw_deg = np.rad2deg(yaw)
    
    print(f"\n{'='*60}")
    print(f"Frame {frame_count} - Fused Pose Estimate")
    print(f"{'='*60}")
    print(f"Position (mm):      X={pos[0]:8.1f}  Y={pos[1]:8.1f}  Z={pos[2]:8.1f}")
    print(f"Velocity (mm/s):    X={vel[0]:8.1f}  Y={vel[1]:8.1f}  Z={vel[2]:8.1f}")
    print(f"Orientation (deg):  R={roll_deg:7.1f}  P={pitch_deg:7.1f}  Y={yaw_deg:7.1f}")
    print(f"Angular Vel (°/s):  X={np.rad2deg(ang_vel[0]):7.1f}  Y={np.rad2deg(ang_vel[1]):7.1f}  Z={np.rad2deg(ang_vel[2]):7.1f}")
    print(f"{'='*60}")


def run_imu_only_demo():
    """
    Run IMU-only tracking (no vision).
    This demonstrates the filter with just IMU data.
    """
    print("\n" + "="*60)
    print("IMU-Only Tracking Demo")
    print("="*60)
    print("\nInitializing IMU...")
    
    # Create IMU reader
    imu = IMUReader()
    
    # Activate IMU
    if not imu.activate():
        print("Failed to activate IMU!")
        return
    
    print("IMU activated. Waiting for first data...")
    
    # Wait for first IMU data
    first_data = None
    for _ in range(50):
        first_data = imu.get_latest_data()
        if first_data is not None:
            break
        time.sleep(0.1)
    
    if first_data is None:
        print("No IMU data received!")
        imu.deactivate()
        return
    
    print("First IMU data received!")
    
    # Create complementary filter
    fusion_filter = ComplementaryFilter(alpha=0.98, alpha_position=0.95)
    
    # Initialize filter at origin (no vision available)
    fusion_filter.initialize(position=np.array([0.0, 0.0, 0.0]))
    
    print("\n" + "="*60)
    print("Starting IMU-only tracking...")
    print("Move the IMU to see pose estimates")
    print("Press Ctrl+C to stop")
    print("="*60)
    
    frame_count = 0
    last_print_time = time.time()
    
    try:
        while True:
            # Get IMU data
            imu_data = imu.get_latest_data()
            
            if imu_data is not None:
                # Predict with IMU
                state = fusion_filter.predict_with_imu(
                    gyro=imu_data.angular_velocity,
                    accel=imu_data.linear_acceleration
                )
                
                # Print state every 0.5 seconds
                current_time = time.time()
                if current_time - last_print_time >= 0.5:
                    frame_count += 1
                    print_state(state, frame_count)
                    last_print_time = current_time
            
            time.sleep(0.01)  # 100 Hz loop
    
    except KeyboardInterrupt:
        print("\n\nStopping...")
    
    finally:
        imu.deactivate()
        print("IMU deactivated")


def run_imu_vision_demo():
    """
    Run full IMU-Vision fusion.
    This requires both IMU and OAK-D camera with AprilTag.
    """
    print("\n" + "="*60)
    print("IMU-Vision Fusion Demo")
    print("="*60)
    print("\nInitializing systems...")
    
    # Initialize IMU and Vision systems
    imu = IMUReader()
    vision = VisionSystem(
        tag_size=0.05,  # 10 cm AprilTags
        enable_visualization=True
    )

    # Activate IMU
    print("Activating IMU...")
    if not imu.activate():
        print("Failed to activate IMU!")
        return
    
    print("IMU activated!")

    # Start Vision pipeline
    print("Starting vision pipeline...")
    if not vision.start_pipeline():
        print("Failed to start vision pipeline!")
        imu.deactivate()
        return
    
    print("Vision pipeline started!")
    
    # Wait for first vision measurement to initialize filter
    print("\nWaiting for AprilTag detection to initialize filter...")
    print("Please ensure AprilTag ID 0 is visible to the camera...")
    
    vision_pose = None
    for i in range(100):  # Try for 10 seconds
        vision.update()
        vision_pose = vision.get_latest_pose(tag_id=0)
        if vision_pose is not None:
            print(f"\nAprilTag detected! Position: {vision_pose['position']}")
            break
        time.sleep(0.1)
    
    if vision_pose is None:
        print("\nNo AprilTag detected after 10 seconds!")
        print("Please check:")
        print("  - AprilTag ID 0 is in camera view")
        print("  - Tag is well-lit and not blurry")
        print("  - Tag size is set correctly (10 cm)")
        vision.shutdown()
        imu.deactivate()
        return
    
    # Initialize complementary filter
    print("\nInitializing complementary filter...")
    fusion_filter = ComplementaryFilter(alpha=0.98, alpha_position=0.95)
    fusion_filter.initialize(
        position=vision_pose['position'],
        orientation=vision_pose['orientation']
    )
    print("Filter initialized!")
    
    # Main fusion loop
    print("\n" + "="*60)
    print("Running IMU-Vision Fusion")
    print("Move the arm to see fused pose estimates")
    print("Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    frame_count = 0
    last_print_time = time.time()
    
    try:
        while vision.is_pipeline_running():
            # 1. Update vision system (polls queues)
            vision.update()
            
            # 2. Predict with IMU (high rate)
            imu_data = imu.get_latest_data()
            if imu_data is not None:
                state = fusion_filter.predict_with_imu(
                    gyro=imu_data.angular_velocity,
                    accel=imu_data.linear_acceleration
                )
            else:
                state = fusion_filter.get_state()
            
            # 3. Update with vision when available
            vision_pose = vision.get_latest_pose(tag_id=0)
            if vision_pose is not None:
                state = fusion_filter.update_with_vision(
                    vision_position=vision_pose['position'],
                    vision_orientation=vision_pose['orientation']
                )
            
            # 4. Display fused state periodically
            current_time = time.time()
            if current_time - last_print_time >= 0.5:
                frame_count += 1
                print_state(state, frame_count)
                last_print_time = current_time
            
            time.sleep(0.01)  # 100 Hz loop
    
    except KeyboardInterrupt:
        print("\n\nStopping...")
    
    finally:
        vision.shutdown()
        imu.deactivate()
        print("Systems shut down")


def run_ekf_fusion_demo():
    """
    Run EKF-based IMU-Vision fusion.
    This uses Extended Kalman Filter for optimal sensor fusion.
    """
    print("\n" + "="*60)
    print("IMU-Vision Fusion with EKF")
    print("="*60)
    print("\nInitializing systems...")
    
    # Initialize IMU and Vision systems
    imu = IMUReader()
    vision = VisionSystem(
        tag_size=0.1,  # 10 cm AprilTags
        enable_visualization=True
    )

    # Activate IMU
    print("Activating IMU...")
    if not imu.activate():
        print("Failed to activate IMU!")
        return
    
    print("IMU activated!")

    # Start Vision pipeline
    print("Starting vision pipeline...")
    if not vision.start_pipeline():
        print("Failed to start vision pipeline!")
        imu.deactivate()
        return
    
    print("Vision pipeline started!")
    
    # Wait for first vision measurement to initialize EKF
    print("\nWaiting for AprilTag detection to initialize EKF...")
    print("Please ensure AprilTag ID 0 is visible to the camera...")
    
    vision_pose = None
    for i in range(100):  # Try for 10 seconds
        vision.update()
        vision_pose = vision.get_latest_pose(tag_id=0)
        if vision_pose is not None:
            print(f"\nAprilTag detected! Position: {vision_pose['position']}")
            break
        time.sleep(0.1)
    
    if vision_pose is None:
        print("\nNo AprilTag detected after 10 seconds!")
        print("Please check:")
        print("  - AprilTag ID 0 is in camera view")
        print("  - Tag is well-lit and not blurry")
        print("  - Tag size is set correctly (10 cm)")
        vision.shutdown()
        imu.deactivate()
        return
    
    # Initialize Extended Kalman Filter
    print("\nInitializing Extended Kalman Filter...")
    ekf = ExtendedKalmanFilter(
        process_noise_pos=1.0,
        process_noise_vel=10.0,
        process_noise_orient=0.01,
        process_noise_angvel=0.1,
        measurement_noise_vision_pos=25.0,
        measurement_noise_vision_orient=0.01
    )
    
    ekf.initialize(
        position=vision_pose['position'],
        orientation=vision_pose['orientation']
    )
    print("Filter initialized!")
    print(f"Initial position uncertainty: {ekf.get_position_uncertainty():.1f} mm")
    print(f"Initial orientation uncertainty: {ekf.get_orientation_uncertainty():.3f} rad")
    
    # Main fusion loop
    print("\n" + "="*60)
    print("Running EKF-based IMU-Vision Fusion")
    print("Move the arm to see fused pose estimates")
    print("Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    frame_count = 0
    last_print_time = time.time()
    vision_updates = 0
    imu_updates = 0
    
    try:
        while vision.is_pipeline_running():
            # 1. Update vision system (polls queues)
            vision.update()
            
            # 2. Predict with IMU (high rate)
            imu_data = imu.get_latest_data()
            if imu_data is not None:
                ekf_state = ekf.predict(
                    angular_velocity=imu_data.angular_velocity,
                    linear_acceleration=imu_data.linear_acceleration
                )
                imu_updates += 1
            else:
                ekf_state = ekf.get_state()
            
            # 3. Update with vision when available
            vision_pose = vision.get_latest_pose(tag_id=0)
            if vision_pose is not None:
                ekf_state = ekf.update_with_vision(
                    position=vision_pose['position'],
                    orientation=vision_pose['orientation']
                )
                vision_updates += 1
            
            # 4. Display fused state periodically
            current_time = time.time()
            if current_time - last_print_time >= 0.5:
                frame_count += 1
                
                # Convert EKFState to dict for print_state
                state_dict = {
                    'position': ekf_state.position,
                    'velocity': ekf_state.velocity,
                    'orientation': ekf_state.orientation,
                    'angular_velocity': ekf_state.angular_velocity
                }
                
                print_state(state_dict, frame_count)
                
                # Print EKF-specific info
                print(f"EKF Uncertainty: pos={ekf.get_position_uncertainty():.1f} mm, "
                      f"orient={ekf.get_orientation_uncertainty():.3f} rad")
                print(f"Updates: IMU={imu_updates}, Vision={vision_updates}")
                print("="*60)
                
                last_print_time = current_time
                imu_updates = 0
                vision_updates = 0
            
            time.sleep(0.01)  # 100 Hz loop
    
    except KeyboardInterrupt:
        print("\n\nStopping...")
    
    finally:
        vision.shutdown()
        imu.deactivate()
        print("Systems shut down")


def main():
    """Main entry point."""
    print("\n" + "="*60)
    print("IMU-Vision Sensor Fusion")
    print("="*60)
    
    print("\nSelect fusion algorithm:")
    print("1. IMU-only tracking (Complementary Filter)")
    print("2. IMU-Vision fusion (Complementary Filter)")
    print("3. IMU-Vision fusion (Extended Kalman Filter) ⭐")
    print("4. Quit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        run_imu_only_demo()
    elif choice == "2":
        run_imu_vision_demo()
    elif choice == "3":
        run_ekf_fusion_demo()
    elif choice == "4":
        print("Exiting...")
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()

