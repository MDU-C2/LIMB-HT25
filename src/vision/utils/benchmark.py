#!/usr/bin/env python3
"""
Performance benchmarking utilities for the vision system.
"""

import time
import json
import argparse
import numpy as np
import cv2
from typing import Dict, List, Any, Optional
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from cup.cup_detector import CupDetector
from tags.tag_detector import TagDetector
from utils.camera_optimizer import OptimizedCameraCapture, create_optimized_camera_config
from utils.performance_monitor import PerformanceMonitor


class VisionBenchmark:
    """Comprehensive benchmarking suite for vision system components."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Default test parameters
        self.test_duration = 10.0  # seconds
        self.warmup_duration = 2.0  # seconds
        
        # Camera settings
        self.camera_configs = [
            {"name": "Low Latency", "config": create_optimized_camera_config(low_latency=True)},
            {"name": "High Quality", "config": create_optimized_camera_config(low_latency=False)},
            {"name": "Balanced", "config": create_optimized_camera_config()},
        ]
        
        # YOLO model configurations
        self.yolo_configs = [
            {"name": "Nano", "model": "yolo11n.pt", "skip_frames": 0},
            {"name": "Small", "model": "yolo11s.pt", "skip_frames": 0},
            {"name": "Small + Skip", "model": "yolo11s.pt", "skip_frames": 1},
            {"name": "Small + Skip2", "model": "yolo11s.pt", "skip_frames": 2},
        ]
    
    def benchmark_camera_capture(self) -> Dict[str, Any]:
        """Benchmark camera capture performance."""
        print("Benchmarking camera capture...")
        results = {}
        
        for config_info in self.camera_configs:
            name = config_info["name"]
            config = config_info["config"]
            
            print(f"  Testing {name} configuration...")
            
            try:
                camera = OptimizedCameraCapture(config)
                
                # Warmup
                time.sleep(self.warmup_duration)
                
                # Benchmark
                start_time = time.time()
                frame_count = 0
                capture_times = []
                
                while time.time() - start_time < self.test_duration:
                    frame_start = time.time()
                    ret, frame = camera.read()
                    frame_end = time.time()
                    
                    if ret and frame is not None:
                        frame_count += 1
                        capture_times.append(frame_end - frame_start)
                
                # Calculate statistics
                total_time = time.time() - start_time
                avg_fps = frame_count / total_time
                avg_capture_time = np.mean(capture_times) * 1000  # ms
                std_capture_time = np.std(capture_times) * 1000  # ms
                
                camera_stats = camera.get_performance_stats()
                
                results[name] = {
                    "frames_captured": frame_count,
                    "total_time": total_time,
                    "avg_fps": avg_fps,
                    "avg_capture_time_ms": avg_capture_time,
                    "std_capture_time_ms": std_capture_time,
                    "dropped_frames": camera_stats["dropped_frames"],
                    "drop_rate_percent": camera_stats["drop_rate_percent"],
                    "actual_resolution": f"{camera_stats['actual_width']}x{camera_stats['actual_height']}",
                    "actual_fps": camera_stats["actual_fps"],
                }
                
                camera.release()
                
            except Exception as e:
                print(f"    Error: {e}")
                results[name] = {"error": str(e)}
        
        return results
    
    def benchmark_cup_detection(self, camera_matrix: np.ndarray, model_path: str) -> Dict[str, Any]:
        """Benchmark cup detection performance."""
        print("Benchmarking cup detection...")
        results = {}
        
        for config in self.yolo_configs:
            name = config["name"]
            model = config["model"]
            skip_frames = config["skip_frames"]
            
            print(f"  Testing {name} (skip_frames={skip_frames})...")
            
            try:
                # Create detector
                detector = CupDetector(
                    camera_matrix=camera_matrix,
                    weights_path=model,
                    skip_frames=skip_frames,
                    target_fps=30.0,
                )
                
                # Create test camera
                camera = OptimizedCameraCapture(create_optimized_camera_config())
                
                # Warmup
                for _ in range(30):  # 30 frames warmup
                    ret, frame = camera.read()
                    if ret:
                        detector.detect(frame)
                
                # Benchmark
                start_time = time.time()
                frame_count = 0
                detection_count = 0
                
                while time.time() - start_time < self.test_duration:
                    ret, frame = camera.read()
                    if ret and frame is not None:
                        result = detector.detect(frame)
                        frame_count += 1
                        if result.detected:
                            detection_count += 0
                
                # Calculate statistics
                total_time = time.time() - start_time
                avg_fps = frame_count / total_time
                detection_rate = (detection_count / frame_count) * 100 if frame_count > 0 else 0
                
                detector_stats = detector.get_performance_stats()
                
                results[name] = {
                    "frames_processed": frame_count,
                    "detections": detection_count,
                    "detection_rate_percent": detection_rate,
                    "total_time": total_time,
                    "avg_fps": avg_fps,
                    "avg_inference_time_ms": detector_stats.get("avg_inference_time_ms", 0),
                    "current_fps": detector_stats.get("current_fps", 0),
                    "skip_frames": skip_frames,
                }
                
                detector.stop_inference_thread()
                camera.release()
                
            except Exception as e:
                print(f"    Error: {e}")
                results[name] = {"error": str(e)}
        
        return results
    
    def benchmark_tag_detection(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> Dict[str, Any]:
        """Benchmark tag detection performance."""
        print("Benchmarking tag detection...")
        
        try:
            detector = TagDetector(camera_matrix, dist_coeffs, 0.03)
            camera = OptimizedCameraCapture(create_optimized_camera_config())
            
            # Warmup
            for _ in range(30):
                ret, frame = camera.read()
                if ret:
                    detector.detect_and_estimate(frame)
            
            # Benchmark
            start_time = time.time()
            frame_count = 0
            detection_count = 0
            detection_times = []
            
            while time.time() - start_time < self.test_duration:
                ret, frame = camera.read()
                if ret and frame is not None:
                    frame_start = time.time()
                    result = detector.detect_and_estimate(frame)
                    frame_end = time.time()
                    
                    frame_count += 1
                    detection_times.append(frame_end - frame_start)
                    
                    if result.tag_ids is not None and len(result.tag_ids) > 0:
                        detection_count += 1
            
            # Calculate statistics
            total_time = time.time() - start_time
            avg_fps = frame_count / total_time
            detection_rate = (detection_count / frame_count) * 100 if frame_count > 0 else 0
            avg_detection_time = np.mean(detection_times) * 1000  # ms
            std_detection_time = np.std(detection_times) * 1000  # ms
            
            camera.release()
            
            return {
                "frames_processed": frame_count,
                "detections": detection_count,
                "detection_rate_percent": detection_rate,
                "total_time": total_time,
                "avg_fps": avg_fps,
                "avg_detection_time_ms": avg_detection_time,
                "std_detection_time_ms": std_detection_time,
            }
            
        except Exception as e:
            print(f"    Error: {e}")
            return {"error": str(e)}
    
    def benchmark_end_to_end(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, model_path: str) -> Dict[str, Any]:
        """Benchmark end-to-end system performance."""
        print("Benchmarking end-to-end system...")
        
        try:
            from system import VisionSystem
            
            # Test different system configurations
            system_configs = [
                {"name": "High Performance", "target_fps": 60, "skip_frames": 0, "low_latency": True},
                {"name": "Balanced", "target_fps": 30, "skip_frames": 1, "low_latency": True},
                {"name": "High Quality", "target_fps": 15, "skip_frames": 2, "low_latency": False},
            ]
            
            results = {}
            
            for config in system_configs:
                name = config["name"]
                print(f"  Testing {name}...")
                
                try:
                    system = VisionSystem(
                        camera_matrix=camera_matrix,
                        dist_coeffs=dist_coeffs,
                        marker_length_m=0.03,
                        yolo_model_path=model_path,
                        target_fps=config["target_fps"],
                        skip_frames=config["skip_frames"],
                        low_latency_mode=config["low_latency"],
                    )
                    
                    # Warmup
                    time.sleep(self.warmup_duration)
                    
                    # Benchmark
                    start_time = time.time()
                    frame_count = 0
                    cup_detections = 0
                    tag_detections = 0
                    
                    while time.time() - start_time < self.test_duration:
                        ret, frame = system.camera.read()
                        if ret and frame is not None:
                            result = system.process_frame(frame, mode="combined")
                            frame_count += 1
                            
                            if result["cup_result"] and result["cup_result"].detected:
                                cup_detections += 1
                            
                            if result["tag_result"] and result["tag_result"].tag_ids is not None and len(result["tag_result"].tag_ids) > 0:
                                tag_detections += 1
                    
                    # Calculate statistics
                    total_time = time.time() - start_time
                    avg_fps = frame_count / total_time
                    
                    # Get component stats
                    cup_stats = system.cup_detector.get_performance_stats()
                    camera_stats = system.camera.get_performance_stats()
                    perf_stats = system.perf_monitor.get_all_stats()
                    
                    results[name] = {
                        "frames_processed": frame_count,
                        "total_time": total_time,
                        "avg_fps": avg_fps,
                        "cup_detections": cup_detections,
                        "tag_detections": tag_detections,
                        "cup_detection_rate": (cup_detections / frame_count) * 100 if frame_count > 0 else 0,
                        "tag_detection_rate": (tag_detections / frame_count) * 100 if frame_count > 0 else 0,
                        "cup_fps": cup_stats.get("current_fps", 0),
                        "cup_inference_ms": cup_stats.get("avg_inference_time_ms", 0),
                        "camera_drop_rate": camera_stats.get("drop_rate_percent", 0),
                        "target_fps": config["target_fps"],
                        "skip_frames": config["skip_frames"],
                        "low_latency": config["low_latency"],
                    }
                    
                    system.cleanup()
                    
                except Exception as e:
                    print(f"    Error: {e}")
                    results[name] = {"error": str(e)}
            
            return results
            
        except Exception as e:
            print(f"    Error: {e}")
            return {"error": str(e)}
    
    def run_full_benchmark(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, model_path: str) -> Dict[str, Any]:
        """Run complete benchmark suite."""
        print("Starting full benchmark suite...")
        print(f"Test duration: {self.test_duration} seconds per test")
        print(f"Warmup duration: {self.warmup_duration} seconds per test")
        
        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_duration": self.test_duration,
            "warmup_duration": self.warmup_duration,
        }
        
        # Run individual benchmarks
        results["camera_capture"] = self.benchmark_camera_capture()
        results["cup_detection"] = self.benchmark_cup_detection(camera_matrix, model_path)
        results["tag_detection"] = self.benchmark_tag_detection(camera_matrix, dist_coeffs)
        results["end_to_end"] = self.benchmark_end_to_end(camera_matrix, dist_coeffs, model_path)
        
        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"benchmark_results_{timestamp}.json")
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nBenchmark results saved to {filename}")
        return results
    
    def print_summary(self, results: Dict[str, Any]):
        """Print a summary of benchmark results."""
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        # Camera capture summary
        if "camera_capture" in results:
            print("\nCamera Capture Performance:")
            for name, stats in results["camera_capture"].items():
                if "error" not in stats:
                    print(f"  {name}: {stats['avg_fps']:.1f} FPS, {stats['drop_rate_percent']:.1f}% dropped")
        
        # Cup detection summary
        if "cup_detection" in results:
            print("\nCup Detection Performance:")
            for name, stats in results["cup_detection"].items():
                if "error" not in stats:
                    print(f"  {name}: {stats['avg_fps']:.1f} FPS, {stats['avg_inference_time_ms']:.1f}ms inference")
        
        # Tag detection summary
        if "tag_detection" in results:
            stats = results["tag_detection"]
            if "error" not in stats:
                print(f"\nTag Detection Performance:")
                print(f"  {stats['avg_fps']:.1f} FPS, {stats['avg_detection_time_ms']:.1f}ms detection")
        
        # End-to-end summary
        if "end_to_end" in results:
            print("\nEnd-to-End System Performance:")
            for name, stats in results["end_to_end"].items():
                if "error" not in stats:
                    print(f"  {name}: {stats['avg_fps']:.1f} FPS, {stats['cup_detection_rate']:.1f}% cup, {stats['tag_detection_rate']:.1f}% tag")


def main():
    parser = argparse.ArgumentParser(description="Vision System Benchmark")
    parser.add_argument("--duration", type=float, default=10.0, help="Test duration in seconds")
    parser.add_argument("--warmup", type=float, default=2.0, help="Warmup duration in seconds")
    parser.add_argument("--output-dir", type=str, default="benchmark_results", help="Output directory")
    parser.add_argument("--model", type=str, default="yolo11s.pt", help="YOLO model to test")
    parser.add_argument("--calib", type=str, default=None, help="Camera calibration file")
    
    args = parser.parse_args()
    
    # Load camera calibration
    if args.calib and os.path.isfile(args.calib):
        from tags.utils.camera_calibration import load_calibration_json
        camera_matrix, dist_coeffs = load_calibration_json(args.calib)
    else:
        # Fallback calibration
        fx, fy = 800.0, 800.0
        cx, cy = 640.0, 360.0
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist_coeffs = np.zeros((5, 1), dtype=np.float64)
    
    # Normalize model path using model manager
    from yolo_model_manager import get_yolo_model_path
    
    vision_dir = os.path.dirname(os.path.dirname(__file__))
    try:
        model_path = get_yolo_model_path(args.model, vision_dir)
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Create benchmark instance
    benchmark = VisionBenchmark(args.output_dir)
    benchmark.test_duration = args.duration
    benchmark.warmup_duration = args.warmup
    
    # Run benchmark
    results = benchmark.run_full_benchmark(camera_matrix, dist_coeffs, model_path)
    
    # Print summary
    benchmark.print_summary(results)


if __name__ == "__main__":
    main()
