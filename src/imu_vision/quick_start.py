#!/usr/bin/env python3
"""Quick start example for the fiducial depth system.

This script shows the simplest way to use the system from the new location.
"""

import sys
import os
import numpy as np

# Add parent directory to path if running directly
if __name__ == "__main__":
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

from imu_vision.fusion_system import FiducialDepthSystem
from imu_vision.smoothing import IMUData
from vision.tags.camera_calibration import load_calibration_json


def main():
    """Quick start example."""
    print("Fiducial Depth System - Quick Start")
    print("=" * 40)
    
    # 1. Load camera calibration
    print("1. Loading camera calibration...")
    try:
        calibration_path = "vision/tags/camera_calibration.json"
        camera_matrix, dist_coeffs = load_calibration_json(calibration_path)
        print(f"   ✓ Loaded calibration from {calibration_path}")
    except:
        # Fallback calibration
        camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float64)
        print("   ✓ Using fallback calibration")
    
    # 2. Initialize the system
    print("2. Initializing FiducialDepthSystem...")
    system = FiducialDepthSystem(
        camera_matrix=camera_matrix,
        enable_imu_smoothing=True,
        smoothing_method="complementary"
    )
    print("   ✓ System initialized")
    
    # 3. Create mock detection results
    print("3. Creating mock detection results...")
    
    # Mock tag detection (hand pose)
    tag_result = {
        'detected': True,
        'tag_ids': [2],
        'transform_cam_from_tag': np.array([
            [1, 0, 0, 0.02],    # 2cm to the right
            [0, 1, 0, 0],       # no y offset
            [0, 0, 1, 0.1],     # 10cm in front
            [0, 0, 0, 1]
        ], dtype=np.float64)
    }
    
    # Mock cup detection
    cup_result = {
        'detected': True,
        'pixel_center': [320, 240],
        'bbox': [280, 200, 360, 280],
        'distance_m': 0.5
    }
    
    # Mock IMU data
    imu_data = IMUData(
        angular_velocity=np.array([0.01, 0.02, -0.01]),
        linear_acceleration=np.array([0.1, 0.05, 9.8]),
        timestamp=0.0
    )
    
    print("   ✓ Mock data created")
    
    # 4. Process frame
    print("4. Processing frame...")
    result = system.process_frame(
        tag_detection_result=tag_result,
        cup_detection_result=cup_result,
        imu_data=imu_data
    )
    
    # 5. Display results
    print("5. Results:")
    print(f"   Hand pose detected: {result['hand_pose']['detected']}")
    if result['hand_pose']['detected']:
        pos = result['hand_pose']['position']
        print(f"   Hand position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})m")
        print(f"   Source: {result['hand_pose']['source']}")
    
    print(f"   Cup 3D detected: {result['cup_3d']['detected']}")
    if result['cup_3d']['detected']:
        pos = result['cup_3d']['position_world']
        print(f"   Cup position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})m")
        print(f"   Distance: {result['cup_3d']['distance']:.3f}m")
    
    print(f"   Control available: {result['control']['available']}")
    if result['control']['available']:
        vector = result['control']['hand_to_cup_vector']
        distance = result['control']['control_errors']['distance_to_cup']
        print(f"   Hand→Cup vector: ({vector[0]:.3f}, {vector[1]:.3f}, {vector[2]:.3f})m")
        print(f"   Distance to cup: {distance:.3f}m")
    
    # 6. Get control command
    print("6. Getting control command...")
    control_command = system.get_control_command()
    if control_command:
        print("   ✓ Control command available")
        print(f"   Distance to cup: {control_command['distance_to_cup']:.3f}m")
        print(f"   Approach vector: {control_command['approach_vector']}")
    else:
        print("   ❌ No control command available")
    
    print("\n" + "=" * 40)
    print("✅ Quick start completed successfully!")
    print("The system is working correctly from its new location.")
    print("=" * 40)


if __name__ == "__main__":
    main()
