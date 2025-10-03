#!/usr/bin/env python3
"""
Quick test script to verify IMU validation integration works.
"""

import numpy as np
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from fusion_system import FiducialDepthSystem
    from smoothing import IMUData, IMUValidator
    print("✅ Imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_imu_validator():
    """Test the IMUValidator class directly."""
    print("\n=== Testing IMUValidator ===")
    
    validator = IMUValidator()
    
    # Create test data
    vision_pose = np.array([
        [1.0, 0.0, 0.0, 0.1],
        [0.0, 1.0, 0.0, 0.2],
        [0.0, 0.0, 1.0, 0.5],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    imu_data = IMUData(
        angular_velocity=np.array([0.1, 0.05, -0.02]),
        linear_acceleration=np.array([0.2, -0.1, 9.8]),
        timestamp=1234567890.0
    )
    
    # Test validation
    result = validator.validate_pose(vision_pose, imu_data, 1234567890.0)
    
    print(f"Validation result:")
    print(f"  Valid: {result['is_valid']}")
    print(f"  Confidence: {result['confidence']:.3f}")
    print(f"  Issues: {result['issues']}")
    
    return result['is_valid']


def test_fusion_system():
    """Test the FiducialDepthSystem with IMU validation."""
    print("\n=== Testing FiducialDepthSystem ===")
    
    # Create system
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
    
    print(f"System created:")
    print(f"  IMU Smoothing: {system.enable_imu_smoothing}")
    print(f"  IMU Validation: {system.enable_imu_validation}")
    print(f"  IMU Validator: {system.imu_validator is not None}")
    
    # Create test data
    tag_result = {
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
    
    imu_data = IMUData(
        angular_velocity=np.array([0.1, 0.05, -0.02]),
        linear_acceleration=np.array([0.2, -0.1, 9.8]),
        timestamp=1234567890.0
    )
    
    # Process frame
    result = system.process_frame(
        tag_detection_result=tag_result,
        imu_data=imu_data
    )
    
    print(f"\nProcessing result:")
    print(f"  Hand pose detected: {result['hand_pose']['detected']}")
    print(f"  Source: {result['hand_pose']['source']}")
    
    if result['hand_pose']['validation'] is not None:
        validation = result['hand_pose']['validation']
        print(f"  Validation:")
        print(f"    Valid: {validation['is_valid']}")
        print(f"    Confidence: {validation['confidence']:.3f}")
        print(f"    Issues: {validation['issues']}")
    
    # Check system info
    system_info = result['system_info']
    print(f"\nSystem info:")
    print(f"  IMU validation enabled: {system_info['imu_validation_enabled']}")
    if 'imu_validation' in system_info:
        validation_info = system_info['imu_validation']
        print(f"  Validation features: {validation_info['enabled_features']}")
    
    return result['hand_pose']['detected']


def main():
    """Run all tests."""
    print("IMU Validation Integration Test")
    print("==============================")
    
    try:
        # Test IMUValidator
        validator_ok = test_imu_validator()
        
        # Test FusionSystem
        fusion_ok = test_fusion_system()
        
        print(f"\n=== Test Results ===")
        print(f"IMUValidator: {'✅ PASS' if validator_ok else '❌ FAIL'}")
        print(f"FusionSystem: {'✅ PASS' if fusion_ok else '❌ FAIL'}")
        
        if validator_ok and fusion_ok:
            print(f"\n🎉 All tests passed! IMU validation is working correctly.")
            return True
        else:
            print(f"\n⚠️  Some tests failed.")
            return False
            
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
