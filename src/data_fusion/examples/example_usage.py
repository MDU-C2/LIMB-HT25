"""Example usage of the fiducial + depth system.

This script demonstrates how to use the FiducialDepthSystem for:
- Hand pose estimation from ArUco markers
- Cup 3D position from YOLO + depth
- Relative pose calculation for control
- Optional IMU smoothing
"""

import numpy as np
import cv2
import time
import json
import os
import sys
from typing import Optional

# Add parent directory to path if running directly
if __name__ == "__main__":
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

# Import the existing vision system components
from vision.system import VisionSystem
from vision.tags.utils.camera_calibration import load_calibration_json

# Import the new fiducial depth system
from imu_vision.fusion_system import FiducialDepthSystem
from imu_vision.smoothing import IMUData


def load_camera_calibration(calibration_path: Optional[str] = None) -> tuple[np.ndarray, np.ndarray]:
    """Load camera calibration."""
    if calibration_path is None:
        default_path = "vision/tags/camera_calibration.json"
    else:
        default_path = calibration_path
    
    try:
        if os.path.isfile(default_path):
            return load_calibration_json(default_path)
    except Exception:
        pass
    
    # Fallback calibration
    fx, fy = 800.0, 800.0
    cx, cy = 640.0, 360.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((5, 1), dtype=np.float64)
    return K, dist


def create_mock_tag_detection() -> dict:
    """Create mock tag detection result for testing."""
    # Mock transform: tag 10cm in front of camera, slightly rotated
    T_CH = np.eye(4, dtype=np.float64)
    T_CH[2, 3] = 0.1  # 10cm in front
    T_CH[0, 3] = 0.02  # 2cm to the right
    
    # Add small rotation
    angle = 0.1  # ~6 degrees
    T_CH[:3, :3] = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    
    return {
        'detected': True,
        'tag_ids': [2],
        'transform_cam_from_tag': T_CH,
        'corners': None
    }


def create_mock_cup_detection() -> dict:
    """Create mock cup detection result for testing."""
    return {
        'detected': True,
        'pixel_center': [320, 240],  # Center of 640x480 image
        'bbox': [280, 200, 360, 280],  # [x1, y1, x2, y2]
        'distance_m': 0.5,  # 50cm away
        'confidence': 0.85
    }


def create_mock_imu_data() -> IMUData:
    """Create mock IMU data for testing."""
    return IMUData(
        angular_velocity=np.array([0.01, 0.02, -0.01]),  # Small rotation rates
        linear_acceleration=np.array([0.1, 0.05, 9.8]),  # Gravity + small acceleration
        timestamp=time.time()
    )


def demonstrate_basic_usage():
    """Demonstrate basic usage of the fiducial depth system."""
    print("=== Basic Usage Demo ===")
    
    # Load camera calibration
    camera_matrix, dist_coeffs = load_camera_calibration()
    
    # Initialize the fiducial depth system
    system = FiducialDepthSystem(
        camera_matrix=camera_matrix,
        enable_imu_smoothing=True,
        smoothing_method="complementary"
    )
    
    # Create mock detection results
    tag_result = create_mock_tag_detection()
    cup_result = create_mock_cup_detection()
    imu_data = create_mock_imu_data()
    
    # Process frame
    result = system.process_frame(
        tag_detection_result=tag_result,
        cup_detection_result=cup_result,
        imu_data=imu_data
    )
    
    # Print results
    print("Processing Results:")
    print(f"  Hand pose detected: {result['hand_pose']['detected']}")
    if result['hand_pose']['detected']:
        print(f"    Position: {result['hand_pose']['position']}")
        print(f"    Source: {result['hand_pose']['source']}")
    
    print(f"  Cup 3D detected: {result['cup_3d']['detected']}")
    if result['cup_3d']['detected']:
        print(f"    World position: {result['cup_3d']['position_world']}")
        print(f"    Distance: {result['cup_3d']['distance']:.3f}m")
    
    print(f"  Control info available: {result['control']['available']}")
    if result['control']['available']:
        print(f"    Hand to cup vector: {result['control']['hand_to_cup_vector']}")
        print(f"    Distance to cup: {result['control']['control_errors']['distance_to_cup']:.3f}m")
    
    return system, result


def demonstrate_imu_smoothing():
    """Demonstrate IMU smoothing when ArUco is temporarily lost."""
    print("\n=== IMU Smoothing Demo ===")
    
    # Load camera calibration
    camera_matrix, dist_coeffs = load_camera_calibration()
    
    # Initialize system with IMU smoothing
    system = FiducialDepthSystem(
        camera_matrix=camera_matrix,
        enable_imu_smoothing=True,
        smoothing_method="complementary"
    )
    
    # First frame: ArUco detected
    print("Frame 1: ArUco detected")
    tag_result = create_mock_tag_detection()
    cup_result = create_mock_cup_detection()
    imu_data = create_mock_imu_data()
    
    result1 = system.process_frame(tag_detection_result=tag_result, cup_detection_result=cup_result, depth_image=None, imu_data=imu_data)
    print(f"  Hand pose source: {result1['hand_pose']['source']}")
    
    # Second frame: ArUco lost, but IMU available
    print("Frame 2: ArUco lost, using IMU prediction")
    time.sleep(0.1)  # Simulate time passing
    
    # Create new IMU data with different values
    imu_data2 = IMUData(
        angular_velocity=np.array([0.02, 0.01, -0.02]),
        linear_acceleration=np.array([0.05, 0.1, 9.8]),
        timestamp=time.time()
    )
    
    result2 = system.process_frame(
        tag_detection_result=None,  # No ArUco detection
        cup_detection_result=cup_result,
        imu_data=imu_data2
    )
    
    print(f"  Hand pose source: {result2['hand_pose']['source']}")
    if result2['hand_pose']['detected']:
        print(f"  IMU smoothing info: {result2['system_info']['imu_smoothing']}")


def demonstrate_control_commands():
    """Demonstrate control command generation."""
    print("\n=== Control Commands Demo ===")
    
    # Load camera calibration
    camera_matrix, dist_coeffs = load_camera_calibration()
    
    # Initialize system
    system = FiducialDepthSystem(camera_matrix=camera_matrix)
    
    # Process frame with detections
    tag_result = create_mock_tag_detection()
    cup_result = create_mock_cup_detection()
    
    result = system.process_frame(tag_result, cup_result)
    
    # Get control command
    control_command = system.get_control_command()
    
    if control_command is not None:
        print("Control Command:")
        print(f"  Hand position: {control_command['hand_pose']['position']}")
        print(f"  Cup position (world): {control_command['cup_position']['world_frame']}")
        print(f"  Cup position (hand): {control_command['cup_position']['hand_frame']}")
        print(f"  Distance to cup: {control_command['distance_to_cup']:.3f}m")
        print(f"  Approach vector: {control_command['approach_vector']}")
        print(f"  Target position: {control_command['target_position']}")
        
        # Position errors
        errors = control_command['position_error']
        print(f"  Position errors: x={errors['x_error']:.3f}, y={errors['y_error']:.3f}, z={errors['z_error']:.3f}")
    else:
        print("No control command available (missing detections)")


def demonstrate_calibration():
    """Demonstrate calibration management."""
    print("\n=== Calibration Demo ===")
    
    # Load camera calibration
    camera_matrix, dist_coeffs = load_camera_calibration()
    
    # Initialize system
    system = FiducialDepthSystem(camera_matrix=camera_matrix)
    
    # Check current calibration
    calib_info = system.calibration_manager.get_calibration_info()
    print("Current calibration:")
    print(f"  Calibrated: {calib_info['is_calibrated']}")
    print(f"  Has T_WC: {calib_info['has_T_WC']}")
    print(f"  Has T_CH_fixed: {calib_info['has_T_CH_fixed']}")
    
    # Set custom calibration
    print("\nSetting custom calibration...")
    T_WC = np.eye(4, dtype=np.float64)
    T_WC[2, 3] = 0.05  # Camera 5cm above world origin
    
    T_CH_fixed = np.eye(4, dtype=np.float64)
    T_CH_fixed[2, 3] = 0.15  # Hand tag 15cm in front of camera
    
    system.set_calibration(T_WC, T_CH_fixed)
    
    # Check updated calibration
    calib_info = system.calibration_manager.get_calibration_info()
    print("Updated calibration:")
    print(f"  Calibrated: {calib_info['is_calibrated']}")
    if calib_info['has_T_WC']:
        print(f"  T_WC: {calib_info['T_WC']}")


def main():
    """Run all demonstrations."""
    print("Fiducial + Depth System Demo")
    print("=" * 50)
    
    try:
        # Run demonstrations
        demonstrate_basic_usage()
        demonstrate_imu_smoothing()
        demonstrate_control_commands()
        demonstrate_calibration()
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
