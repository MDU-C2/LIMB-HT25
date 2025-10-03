#!/usr/bin/env python3
"""
Example usage of IMU validation in the fusion system.

This demonstrates how to use IMU validation independently of IMU smoothing,
allowing you to validate vision results with IMU data without using IMU for prediction.
"""

import numpy as np
import time
from typing import Dict, Any, Optional

from ..fusion_system import FiducialDepthSystem
from ..smoothing import IMUData


def create_sample_imu_data(angular_vel: np.ndarray = None, 
                          linear_accel: np.ndarray = None,
                          timestamp: float = None) -> IMUData:
    """Create sample IMU data for testing."""
    if angular_vel is None:
        angular_vel = np.array([0.1, 0.05, -0.02])  # Small rotation
    if linear_accel is None:
        linear_accel = np.array([0.2, -0.1, 9.8])   # Gravity + small motion
    if timestamp is None:
        timestamp = time.time()
    
    return IMUData(
        angular_velocity=angular_vel,
        linear_acceleration=linear_accel,
        timestamp=timestamp
    )


def create_sample_tag_detection() -> Dict[str, Any]:
    """Create sample tag detection result."""
    return {
        'detected_tags': [
            {
                'id': 0,
                'corners': np.array([
                    [100, 150],
                    [200, 150], 
                    [200, 250],
                    [100, 250]
                ]),
                'pose': np.array([
                    [1.0, 0.0, 0.0, 0.1],
                    [0.0, 1.0, 0.0, 0.2],
                    [0.0, 0.0, 1.0, 0.5],
                    [0.0, 0.0, 0.0, 1.0]
                ])
            }
        ]
    }


def example_validation_only():
    """Example: Use IMU validation without smoothing."""
    print("=== IMU Validation Only (No Smoothing) ===")
    
    # Create system with validation enabled but smoothing disabled
    camera_matrix = np.array([
        [800, 0, 320],
        [0, 800, 240],
        [0, 0, 1]
    ])
    
    system = FiducialDepthSystem(
        camera_matrix=camera_matrix,
        calibration_file=None,
        enable_imu_smoothing=False,  # No smoothing
        enable_imu_validation=True   # But validate with IMU
    )
    
    # Create sample data
    tag_result = create_sample_tag_detection()
    imu_data = create_sample_imu_data()
    
    print(f"System initialized:")
    print(f"  IMU Smoothing: {system.enable_imu_smoothing}")
    print(f"  IMU Validation: {system.enable_imu_validation}")
    print()
    
    # Process frame
    result = system.process_frame(
        tag_detection_result=tag_result,
        imu_data=imu_data
    )
    
    # Check results
    hand_pose = result['hand_pose']
    if hand_pose['detected']:
        print(f"✅ Hand pose detected from: {hand_pose['source']}")
        print(f"   Position: {hand_pose['position']}")
        
        # Check validation results
        if hand_pose['validation'] is not None:
            validation = hand_pose['validation']
            print(f"   Validation:")
            print(f"     Valid: {validation['is_valid']}")
            print(f"     Confidence: {validation['confidence']:.3f}")
            
            if validation['issues']:
                print(f"     Issues: {validation['issues']}")
            if validation['warnings']:
                print(f"     Warnings: {validation['warnings']}")
        else:
            print(f"   No validation performed")
    else:
        print(f"❌ No hand pose detected")
    
    # Print system info
    system_info = result['system_info']
    print(f"\nSystem Info:")
    print(f"  IMU Validation Enabled: {system_info['imu_validation_enabled']}")
    if 'imu_validation' in system_info:
        validation_info = system_info['imu_validation']
        print(f"  Validation Features: {validation_info['enabled_features']}")
        print(f"  History Length: {validation_info['history_length']}")


def example_with_bad_imu_data():
    """Example: Test validation with problematic IMU data."""
    print("\n=== Testing with Bad IMU Data ===")
    
    camera_matrix = np.array([
        [800, 0, 320],
        [0, 800, 240],
        [0, 0, 1]
    ])
    
    system = FiducialDepthSystem(
        camera_matrix=camera_matrix,
        calibration_file=None,
        enable_imu_smoothing=False,
        enable_imu_validation=True
    )
    
    tag_result = create_sample_tag_detection()
    
    # Test with excessive acceleration
    print("Testing with excessive acceleration...")
    bad_imu_data = create_sample_imu_data(
        angular_vel=np.array([0.1, 0.05, -0.02]),
        linear_accel=np.array([100.0, 50.0, 200.0])  # Excessive acceleration
    )
    
    result = system.process_frame(
        tag_detection_result=tag_result,
        imu_data=bad_imu_data
    )
    
    hand_pose = result['hand_pose']
    if hand_pose['validation'] is not None:
        validation = hand_pose['validation']
        print(f"Validation result: Valid={validation['is_valid']}, Confidence={validation['confidence']:.3f}")
        if validation['issues']:
            print(f"Issues found: {validation['issues']}")
    
    # Test with excessive angular velocity
    print("\nTesting with excessive angular velocity...")
    bad_imu_data2 = create_sample_imu_data(
        angular_vel=np.array([20.0, 15.0, 10.0]),  # Excessive rotation
        linear_accel=np.array([0.2, -0.1, 9.8])
    )
    
    result2 = system.process_frame(
        tag_detection_result=tag_result,
        imu_data=bad_imu_data2
    )
    
    hand_pose2 = result2['hand_pose']
    if hand_pose2['validation'] is not None:
        validation2 = hand_pose2['validation']
        print(f"Validation result: Valid={validation2['is_valid']}, Confidence={validation2['confidence']:.3f}")
        if validation2['issues']:
            print(f"Issues found: {validation2['issues']}")


def example_custom_thresholds():
    """Example: Use custom validation thresholds."""
    print("\n=== Custom Validation Thresholds ===")
    
    camera_matrix = np.array([
        [800, 0, 320],
        [0, 800, 240],
        [0, 0, 1]
    ])
    
    # Custom thresholds - more lenient
    custom_thresholds = {
        'max_acceleration': 100.0,  # Allow higher acceleration
        'max_angular_velocity': 20.0,  # Allow faster rotation
        'max_position_jump': 1.0,  # Allow larger position jumps
    }
    
    system = FiducialDepthSystem(
        camera_matrix=camera_matrix,
        calibration_file=None,
        enable_imu_smoothing=False,
        enable_imu_validation=True,
        validation_thresholds=custom_thresholds
    )
    
    tag_result = create_sample_tag_detection()
    
    # Test with data that would fail with default thresholds
    moderate_imu_data = create_sample_imu_data(
        angular_vel=np.array([15.0, 12.0, 8.0]),  # Would fail with default (10.0 max)
        linear_accel=np.array([60.0, 40.0, 80.0])  # Would fail with default (50.0 max)
    )
    
    result = system.process_frame(
        tag_detection_result=tag_result,
        imu_data=moderate_imu_data
    )
    
    hand_pose = result['hand_pose']
    if hand_pose['validation'] is not None:
        validation = hand_pose['validation']
        print(f"Validation with custom thresholds:")
        print(f"  Valid: {validation['is_valid']}")
        print(f"  Confidence: {validation['confidence']:.3f}")
        if validation['issues']:
            print(f"  Issues: {validation['issues']}")


def example_validation_and_smoothing():
    """Example: Use both validation and smoothing."""
    print("\n=== IMU Validation + Smoothing ===")
    
    camera_matrix = np.array([
        [800, 0, 320],
        [0, 800, 240],
        [0, 0, 1]
    ])
    
    system = FiducialDepthSystem(
        camera_matrix=camera_matrix,
        calibration_file=None,
        enable_imu_smoothing=True,   # Enable smoothing
        enable_imu_validation=True,  # And validation
        smoothing_method="complementary"
    )
    
    print(f"System initialized:")
    print(f"  IMU Smoothing: {system.enable_imu_smoothing}")
    print(f"  IMU Validation: {system.enable_imu_validation}")
    
    # First frame with vision
    tag_result = create_sample_tag_detection()
    imu_data = create_sample_imu_data()
    
    result1 = system.process_frame(
        tag_detection_result=tag_result,
        imu_data=imu_data
    )
    
    hand_pose1 = result1['hand_pose']
    print(f"\nFrame 1 - With vision:")
    print(f"  Detected: {hand_pose1['detected']}")
    print(f"  Source: {hand_pose1['source']}")
    if hand_pose1['validation']:
        print(f"  Validation: Valid={hand_pose1['validation']['is_valid']}")
    
    # Second frame without vision (should use IMU prediction)
    time.sleep(0.1)  # Small delay
    imu_data2 = create_sample_imu_data(
        angular_vel=np.array([0.2, 0.1, -0.05]),
        linear_accel=np.array([0.3, -0.2, 9.7])
    )
    
    result2 = system.process_frame(
        tag_detection_result=None,  # No vision
        imu_data=imu_data2
    )
    
    hand_pose2 = result2['hand_pose']
    print(f"\nFrame 2 - No vision:")
    print(f"  Detected: {hand_pose2['detected']}")
    print(f"  Source: {hand_pose2['source']}")
    if hand_pose2['validation']:
        print(f"  Validation: Valid={hand_pose2['validation']['is_valid']}")


if __name__ == "__main__":
    print("IMU Validation Examples")
    print("======================")
    
    try:
        example_validation_only()
        example_with_bad_imu_data()
        example_custom_thresholds()
        example_validation_and_smoothing()
        
        print("\n✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
