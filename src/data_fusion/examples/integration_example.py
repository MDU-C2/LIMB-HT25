"""Integration example showing how to use the new fiducial depth system with existing vision components.

This script demonstrates how to integrate the new FiducialDepthSystem with the existing
VisionSystem for a complete pipeline.
"""

import numpy as np
import cv2
import time
import argparse
import os
import sys

# Add parent directory to path if running directly
if __name__ == "__main__":
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

from vision.system import VisionSystem
from vision.tags.camera_calibration import load_calibration_json
from imu_vision.fusion_system import FiducialDepthSystem
from imu_vision.smoothing import IMUData


def load_camera_calibration(calibration_path: str = None) -> tuple[np.ndarray, np.ndarray]:
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


def create_mock_imu_data() -> IMUData:
    """Create mock IMU data."""
    return IMUData(
        angular_velocity=np.array([0.01, 0.02, -0.01]),
        linear_acceleration=np.array([0.1, 0.05, 9.8]),
        timestamp=time.time()
    )


def print_results(result: dict, frame_count: int):
    """Print processing results in a clean format."""
    print(f"\n--- Frame {frame_count} ---")
    
    # Hand pose information
    hand_pose = result['hand_pose']
    if hand_pose['detected']:
        pos = hand_pose['position']
        print(f"Hand Pose ({hand_pose['source']}): Pos({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})m")
    else:
        print("Hand Pose: Not detected")
    
    # Cup 3D information
    cup_3d = result['cup_3d']
    if cup_3d['detected']:
        pos = cup_3d['position_world']
        print(f"Cup 3D: Pos({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})m, Dist: {cup_3d['distance']:.3f}m")
    else:
        print("Cup 3D: Not detected")
    
    # Control information
    control = result['control']
    if control['available']:
        vector = control['hand_to_cup_vector']
        distance = control['control_errors']['distance_to_cup']
        print(f"Control: Hand→Cup({vector[0]:.3f}, {vector[1]:.3f}, {vector[2]:.3f})m, Dist: {distance:.3f}m")
    else:
        print("Control: Not available")


def main():
    """Main integration example."""
    parser = argparse.ArgumentParser(description="Fiducial Depth System Integration Example")
    parser.add_argument("--show", action="store_true", help="Display camera frames")
    parser.add_argument("--calib", type=str, default=None, help="Path to calibration file")
    parser.add_argument("--enable-imu", action="store_true", help="Enable IMU smoothing")
    parser.add_argument("--smoothing-method", choices=["complementary", "ekf", "hold"], 
                       default="complementary", help="IMU smoothing method")
    args = parser.parse_args()
    
    # Load camera calibration
    camera_matrix, dist_coeffs = load_camera_calibration(args.calib)
    
    # Initialize existing vision system
    vision_system = VisionSystem(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        marker_length_m=0.03,
        yolo_model_path="yolo11s.pt"
    )
    
    # Initialize new fiducial depth system
    fiducial_system = FiducialDepthSystem(
        camera_matrix=camera_matrix,
        enable_imu_smoothing=args.enable_imu,
        smoothing_method=args.smoothing_method
    )
    
    print("Fiducial Depth System Integration Example")
    print("=" * 50)
    print(f"IMU smoothing: {'Enabled' if args.enable_imu else 'Disabled'}")
    if args.enable_imu:
        print(f"Smoothing method: {args.smoothing_method}")
    print("Press 'q' to quit, 'c' to show control command")
    print("=" * 50)
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process with existing vision system
            vision_result = vision_system.process_frame(frame, mode="combined")
            
            # Create mock IMU data (in real system, this would come from IMU)
            imu_data = create_mock_imu_data() if args.enable_imu else None
            
            # Process with new fiducial depth system
            fiducial_result = fiducial_system.process_frame(
                tag_detection_result=vision_result.get("tag_result"),
                cup_detection_result=vision_result.get("cup_result"),
                imu_data=imu_data
            )
            
            # Print results every 30 frames to avoid spam
            if frame_count % 30 == 0:
                print_results(fiducial_result, frame_count)
            
            # Show visualization if requested
            if args.show:
                # Draw detections on frame
                if vision_result.get("tag_result") is not None:
                    # Draw tag detection (simplified)
                    cv2.putText(frame, "Tag Detected", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if vision_result.get("cup_result") is not None:
                    # Draw cup detection (simplified)
                    cv2.putText(frame, "Cup Detected", (10, 70), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Show control information
                if fiducial_result['control']['available']:
                    distance = fiducial_result['control']['control_errors']['distance_to_cup']
                    cv2.putText(frame, f"Dist: {distance:.3f}m", (10, 110), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                
                cv2.imshow("Fiducial Depth System", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    # Show control command
                    control_command = fiducial_system.get_control_command()
                    if control_command is not None:
                        print("\n--- Control Command ---")
                        print(f"Hand position: {control_command['hand_pose']['position']}")
                        print(f"Cup position (hand frame): {control_command['cup_position']['hand_frame']}")
                        print(f"Distance to cup: {control_command['distance_to_cup']:.3f}m")
                        print(f"Approach vector: {control_command['approach_vector']}")
                    else:
                        print("No control command available")
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released")


if __name__ == "__main__":
    main()
