import json
import time
import argparse
import os
from typing import Optional

import cv2
import numpy as np

from system import VisionSystem
from tags.camera_calibration import load_calibration_json
from utils.performance_monitor import get_performance_monitor


def load_camera_calibration(calibration_path: Optional[str] = None) -> tuple[np.ndarray, np.ndarray]:
    """Load camera calibration with fallback to placeholder values."""
    if calibration_path is None:
        default_path = os.path.join(os.path.dirname(__file__), "tags", "camera_calibration.json")
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

def normalize_yolo_model_path(model_name: str) -> str:
    """Ensure YOLO model path points to the models folder."""
    from utils.yolo_model_manager import get_yolo_model_path
    
    vision_dir = os.path.dirname(__file__)
    return get_yolo_model_path(model_name, vision_dir)

def benchmark_system(args):
    """Run system benchmark and performance analysis."""
    print("Running system benchmark...")
    
    # Load calibration
    camera_matrix, dist_coeffs = load_camera_calibration(args.calib)
    
    # Normalize model path
    try:
        normalized_model_path = normalize_yolo_model_path(args.yolo_model)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Test different configurations
    configs = [
        {"name": "Low Latency", "target_fps": 60, "skip_frames": 0, "low_latency": True},
        {"name": "Balanced", "target_fps": 30, "skip_frames": 1, "low_latency": True},
        {"name": "High Quality", "target_fps": 15, "skip_frames": 2, "low_latency": False},
    ]
    
    results = {}
    
    for config in configs:
        print(f"\nTesting {config['name']} configuration...")
        
        try:
            system = VisionSystem(
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                marker_length_m=0.03,
                yolo_model_path=normalized_model_path,
                yolo_device=args.yolo_device,
                yolo_conf=args.yolo_conf,
                yolo_iou=args.yolo_iou,
                target_fps=config["target_fps"],
                skip_frames=config["skip_frames"],
                camera_device_id=0,
                low_latency_mode=config["low_latency"],
                monitor_performance=args.monitor_performance,
            )
            
            # Run for 10 seconds
            start_time = time.time()
            frame_count = 0
            
            while time.time() - start_time < 10.0:
                ret, frame = system.camera.read()
                if ret and frame is not None:
                    result = system.process_frame(frame, mode=args.mode)
                    frame_count += 1
            
            # Collect performance stats
            perf_stats = get_performance_monitor().get_all_stats()
            cup_stats = system.cup_detector.get_performance_stats()
            camera_stats = system.camera.get_performance_stats()
            
            results[config["name"]] = {
                "frames_processed": frame_count,
                "duration": 10.0,
                "avg_fps": frame_count / 10.0,
                "performance_stats": perf_stats,
                "cup_stats": cup_stats,
                "camera_stats": camera_stats,
            }
            
            system.cleanup()
            
        except Exception as e:
            print(f"Error testing {config['name']}: {e}")
            results[config["name"]] = {"error": str(e)}
    
    # Print benchmark results
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    
    for name, result in results.items():
        if "error" in result:
            print(f"\n{name}: ERROR - {result['error']}")
            continue
        
        print(f"\n{name}:")
        print(f"  Frames processed: {result['frames_processed']}")
        print(f"  Average FPS: {result['avg_fps']:.1f}")
        print(f"  Cup detection FPS: {result['cup_stats'].get('current_fps', 0):.1f}")
        print(f"  Avg inference time: {result['cup_stats'].get('avg_inference_time_ms', 0):.1f}ms")
        print(f"  Camera drop rate: {result['camera_stats'].get('drop_rate_percent', 0):.1f}%")
    
    # Save detailed results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    benchmark_file = f"benchmark_results_{timestamp}.json"
    
    with open(benchmark_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nDetailed benchmark results saved to {benchmark_file}")
    
def main() -> None:
    parser = argparse.ArgumentParser(description="Vision System")
    parser.add_argument("--show", action="store_true", help="Display camera frames in a window")
    parser.add_argument("--yolo-model", type=str, default="yolo11s.pt", help="YOLO model name")
    parser.add_argument("--yolo-device", type=str, default=None, help="Device for YOLO (e.g., cuda:0)")
    parser.add_argument("--yolo-conf", type=float, default=0.35, help="YOLO confidence threshold")
    parser.add_argument("--yolo-iou", type=float, default=0.45, help="YOLO IoU threshold")
    parser.add_argument("--mode", choices=["cup", "tag", "combined"], default="cup", help="Detection mode")
    parser.add_argument("--calib", type=str, default=None, help="Path to calibration JSON")
    parser.add_argument("--target-fps", type=float, default=30.0, help="Target FPS for processing")
    parser.add_argument("--skip-frames", type=int, default=0, help="Skip N frames between processing")
    parser.add_argument("--low-latency", action="store_true", help="Enable low latency mode")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--camera-device", type=int, default=0, help="Camera device ID")
    parser.add_argument("--monitor-performance", action="store_true", help="Monitor performance", default=False)
    args = parser.parse_args()
    
    # Run benchmark if requested
    if args.benchmark:
        benchmark_system(args)
        return
    
    # Load calibration
    camera_matrix, dist_coeffs = load_camera_calibration(args.calib)
    
    # Normalize model path
    try:
        normalized_model_path = normalize_yolo_model_path(args.yolo_model)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Create vision system
    system = VisionSystem(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        marker_length_m=0.03,
        yolo_model_path=normalized_model_path,
        yolo_device=args.yolo_device,
        yolo_conf=args.yolo_conf,
        yolo_iou=args.yolo_iou,
        target_fps=args.target_fps,
        skip_frames=args.skip_frames,
        camera_device_id=args.camera_device,
        low_latency_mode=args.low_latency,
        monitor_performance=args.monitor_performance,
    )
    
    print(f"Starting vision system:")
    print(f"  Mode: {args.mode}")
    print(f"  Target FPS: {args.target_fps}")
    print(f"  Skip frames: {args.skip_frames}")
    print(f"  Low latency: {args.low_latency}")
    print(f"  YOLO device: {args.yolo_device or 'auto'}")
    print(f"  Camera device: {args.camera_device}")
    print("\nControls:")
    print("  'q' - Quit")
    print("  'p' - Print performance stats")
    print("  's' - Save performance report")
    print("  '+' - Increase target FPS")
    print("  '-' - Decrease target FPS")
    
    try:
        # Run continuous processing
        system.run_continuous_processing(
            mode=args.mode,
            show_window=args.show,
            imu_delta=np.eye(4, dtype=np.float64)  # Placeholder IMU
        )
    except KeyboardInterrupt:
        print("\nStopping vision system...")
    finally:
        system.cleanup()

if __name__ == "__main__":
    main()
