import json
import time
import argparse
import os
from typing import Optional

import cv2
import numpy as np

from system import VisionSystem
from tags.camera_calibration import load_calibration_json
from visualization import visualize_tag_detection, visualize_cup_detection, visualize_combined_detection, calculate_tag_cup_distance


def load_camera_calibration(calibration_path: Optional[str] = None) -> tuple[np.ndarray, np.ndarray]:
    # Try to load from provided path or default utils/camera_calibration.json; fallback to placeholder
    if calibration_path is None:
        default_path = os.path.join(os.path.dirname(__file__), "tags", "camera_calibration.json")
    else:
        default_path = calibration_path
    try:
        if os.path.isfile(default_path):
            return load_calibration_json(default_path)
    except Exception:
        pass
    fx, fy = 800.0, 800.0
    cx, cy = 640.0, 360.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((5, 1), dtype=np.float64)
    return K, dist


def normalize_yolo_model_path(model_name: str) -> str:
    """
    Ensure the YOLO model path always points to the models folder in src/vision/models.
    
    Args:
        model_name: Model name (e.g., "yolo11s.pt", "yolo11n.pt", "yolo11m.pt")
    
    Returns:
        Full path to the model in src/vision/models/
        
    Raises:
        FileNotFoundError: If the model file doesn't exist in the models directory
    """
    # Get the directory of this file (src/vision/)
    vision_dir = os.path.dirname(__file__)
    models_dir = os.path.join(vision_dir, "models")
    
    # Extract just the filename if a full path was provided
    model_filename = os.path.basename(model_name)
    
    # Ensure it has .pt extension
    if not model_filename.endswith('.pt'):
        model_filename += '.pt'
    
    # Construct the full path
    full_path = os.path.join(models_dir, model_filename)
    
    # Check if the model file exists
    if not os.path.isfile(full_path):
        # List available models for helpful error message
        available_models = []
        if os.path.isdir(models_dir):
            available_models = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
        
        error_msg = f"YOLO model '{model_filename}' not found in {models_dir}"
        if available_models:
            error_msg += f"\nAvailable models: {', '.join(available_models)}"
        else:
            error_msg += f"\nNo .pt model files found in {models_dir}"
        
        raise FileNotFoundError(error_msg)
    
    return full_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true", help="Display camera frames in a window")
    parser.add_argument("--yolo-model", type=str, default="yolo11s.pt", help="YOLO model name (e.g., yolo11s.pt, yolo11n.pt). Will be loaded from src/vision/models/. Available: yolo11n.pt, yolo11s.pt")
    parser.add_argument("--yolo-device", type=str, default=None, help="Device for YOLO (e.g., cuda:0 on Jetson)")
    parser.add_argument("--yolo-conf", type=float, default=0.35, help="YOLO confidence threshold")
    parser.add_argument("--yolo-iou", type=float, default=0.45, help="YOLO IoU threshold")
    parser.add_argument("--mode", choices=["cup", "tag", "combined"], default="cup", help="Select detection pipeline: cup, tag, or combined")
    parser.add_argument("--calib", type=str, default=None, help="Path to calibration JSON (defaults to utils/camera_calibration.json)")
    args = parser.parse_args()
    camera_matrix, dist_coeffs = load_camera_calibration(args.calib)
    
    try:
        # Normalize the YOLO model path to always point to src/vision/models/
        normalized_model_path = normalize_yolo_model_path(args.yolo_model)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    system = VisionSystem(
        camera_matrix,
        dist_coeffs,
        marker_length_m=0.03,
        yolo_model_path=normalized_model_path,
        yolo_device=args.yolo_device,
        yolo_conf=args.yolo_conf,
        yolo_iou=args.yolo_iou,
    )

    cap = cv2.VideoCapture(0)
   

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # Optional: Provide IMU delta transform per frame. Using identity here as placeholder.
        imu_delta = np.eye(4, dtype=np.float64)
      
      
        result = system.process_frame(frame, imu_delta, mode=args.mode)
        #print(json.dumps(result), end="\r")
        '''
        result = OutputFormatter.build_output(
            cup_detected=cup_det.detected,
            cup_relative_position_m=cup_relative_position_m,
            arm_pose_corrected_cam_from_arm=arm_pose_corrected,
            tag_detected=tag_detected,
            tag_ids=tag_result.tag_ids,
            tag_positions=tag_positions,
        )
        '''


        if args.show:
            
            if args.mode == "cup":
                
                print("Cup mode: frame shape {frame.shape}, showing window...", end="\r", flush=True)
                frame = visualize_cup_detection(frame, result["cup_result"])

            elif args.mode == "tag":
                
                print("Tag mode: frame shape {frame.shape}, showing window...", end="\r", flush=True)
                #print(f"Tag result: {result['tag_result']}")
                frame = visualize_tag_detection(frame, result["tag_result"], system.camera_matrix, system.dist_coeffs)
                
                # Print coordinate information to terminal
                if result["tag_coordinates"] is not None and len(result["tag_coordinates"]) > 0:
                    print(f"\nDetected {len(result['tag_coordinates'])} tags with coordinates:")
                    for tag_id, coord_info in result["tag_coordinates"].items():
                        pos = coord_info['position']
                        orient = coord_info['orientation']
                        distance = coord_info['distance']
                        print(f"  Tag {tag_id}: Pos({pos['x']:.3f}, {pos['y']:.3f}, {pos['z']:.3f})m, "
                              f"Dist: {distance:.3f}m, RPY({orient['roll']:.1f}, {orient['pitch']:.1f}, {orient['yaw']:.1f})°")
                else:
                    print("No tags detected", end="\r")
            
            elif args.mode == "combined":
                
                print("Combined mode: frame shape {frame.shape}, showing window...", end="\r", flush=True)
                frame = visualize_combined_detection(frame, result["tag_result"], result["cup_result"], system.camera_matrix, system.dist_coeffs)
                
                # Print combined information to terminal
                output_lines = []
                
                # Cup information
                if result["cup_result"] is not None and result["cup_result"].detected:
                    cup_info = result["cup_result"]
                    output_lines.append(f"Cup detected: center=({cup_info.pixel_center[0]}, {cup_info.pixel_center[1]}), distance={cup_info.distance_m:.3f}m")
                else:
                    output_lines.append("Cup: Not detected")
                
                # Tag information
                if result["tag_coordinates"] is not None and len(result["tag_coordinates"]) > 0:
                    output_lines.append(f"Tags detected: {len(result['tag_coordinates'])}")
                    for tag_id, coord_info in result["tag_coordinates"].items():
                        pos = coord_info['position']
                        orient = coord_info['orientation']
                        distance = coord_info['distance']
                        output_lines.append(f"  Tag {tag_id}: Pos({pos['x']:.3f}, {pos['y']:.3f}, {pos['z']:.3f})m, Dist: {distance:.3f}m")
                else:
                    output_lines.append("Tags: None detected")
                
                # Tag-Cup distance information
                distance_info = calculate_tag_cup_distance(result["tag_result"], result["cup_result"], system.camera_matrix, target_tag_id=2)
                if distance_info is not None:
                    distance_mag = distance_info['distance_magnitude']
                    distance_components = distance_info['distance_components']
                    output_lines.append(f"Tag 2 -> Cup Distance: {distance_mag:.3f}m")
                    output_lines.append(f"  deltaX: {distance_components['x']:.3f}m, deltaY: {distance_components['y']:.3f}m, deltaZ: {distance_components['z']:.3f}m")
                else:
                    output_lines.append("Tag 2 -> Cup Distance: Not available (Tag 2 or cup not detected)")
                
                # Print all information
                #print(f"\nCombined Detection:")
                #for line in output_lines:
                #    print(f"  {line}")
             
                
            
            # Always show the camera window
           
            cv2.imshow("Vision", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break


        # Throttle loop slightly
        #time.sleep(0.005 if args.show else 0.01)
    cap.release()
    cv2.destroyAllWindows()
    


if __name__ == "__main__":
    main()


