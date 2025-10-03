from typing import Any, Dict, Optional, Tuple
import time
import threading
from queue import Queue, Empty


import cv2
import numpy as np

from tags import TagDetector
from cup import CupDetector
#from imu import IMUFusion
from visualization import calculate_camera_relative_coordinates
from utils.performance_monitor import get_performance_monitor, start_timer
from utils.camera_optimizer import OptimizedCameraCapture, create_optimized_camera_config


class VisionSystem:
    """
    Orchestrates tag detection, cup detection, and IMU fusion, returning structured output.
    Also provides performance monitoring and camera optimization.
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        marker_length_m: float,
        assumed_cup_diameter_m: float = 0.08,
        yolo_model_path: str = "yolo11s.pt",
        yolo_device: Optional[str] = None,
        yolo_conf: float = 0.35,
        yolo_iou: float = 0.45,
        target_fps: float = 30.0,
        skip_frames: int = 0,
        camera_device_id: int = 0,
        low_latency_mode: bool = True,
    ) -> None:

        # Initialize performance monitor
        self.perf_monitor = get_performance_monitor()
        self.perf_monitor.start_monitoring()
        
        # Store configuration
        self.target_fps = target_fps
        self.skip_frames = skip_frames
        self.frame_interval = 1.0 / target_fps
        
        # Initialize detectors
        self.tag_detector = TagDetector(camera_matrix, dist_coeffs, marker_length_m)
        
        self.cup_detector = CupDetector(
            camera_matrix=camera_matrix,
            weights_path=yolo_model_path,
            assumed_cup_diameter_m=assumed_cup_diameter_m,
            device=yolo_device,
            conf=yolo_conf,
            iou=yolo_iou,
            skip_frames=skip_frames,
            target_fps=target_fps,
        )
        
        #self.imu_fusion = IMUFusion()
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        
        # Initialize optimized camera
        camera_config = create_optimized_camera_config(
            device_id=camera_device_id,
            target_fps=int(target_fps),
            low_latency=low_latency_mode
        )
        self.camera = OptimizedCameraCapture(camera_config)

        # Processing state
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.running = False
        
        # Results caching
        self.last_results = {
            "tag_result": None,
            "tag_coordinates": None,
            "cup_result": None,
        }

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        #imu_delta_T_cam_from_arm: Optional[np.ndarray] = None,
        mode: str = "cup",
    ) -> Dict[str, Any]:
        """
        Process frame with performance optimizations.
        """
        with start_timer("total_frame_processing"):
            self.frame_count += 1
            current_time = time.time()

            # Check if we should process this frame based on target FPS
            time_since_last = current_time - self.last_frame_time
            if time_since_last < self.frame_interval:
                # Return cached results if we're ahead of schedule
                return self.last_results
            
            self.last_frame_time = current_time
            self.perf_monitor.record_frame()

            # Process based on mode
            tag_result = None
            tag_coordinates = None
            cup_result = None

            if mode in ["tag", "combined"]:
                with start_timer("tag_detection"):
                    tag_result = self.tag_detector.detect_and_estimate(frame_bgr)
                    if tag_result is not None:
                        tag_coordinates = calculate_camera_relative_coordinates(tag_result)

            if mode in ["cup", "combined"]:
                with start_timer("cup_detection"):
                    cup_result = self.cup_detector.detect(frame_bgr)
        
            
            # Cache results
            self.last_results = {
                "tag_result": tag_result,
                "tag_coordinates": tag_coordinates,
                "cup_result": cup_result,
            }
            
            return self.last_results

    def run_continuous_processing(
        self,
        mode: str = "cup",
        show_window: bool = True,
        #imu_delta: Optional[np.ndarray] = None
    ):
        """
        Run continuous processing with optimized performance.
        """
        self.running = True
        print(f"Starting vision processing (mode: {mode}, target FPS: {self.target_fps})")

        try:
            while self.running:
                # Read frame from optimized camera
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    time.sleep(0.001)  # Small delay to prevent busy waiting
                    continue
                
                # Process frame
                result = self.process_frame(frame, mode)
                
                # Display results
                if show_window:
                    display_frame = self._prepare_display_frame(frame, result, mode)
                    cv2.imshow("Vision", display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('p'):
                        self._print_performance_stats()
                    elif key == ord('s'):
                        self._save_performance_report()
        
                # Print periodic performance updates
                if self.frame_count % 100 == 0:
                    self._print_performance_summary()
        
        except KeyboardInterrupt:
            print("Stopping vision processing...")
        finally:
            self.cleanup()

    def _prepare_display_frame(self, frame: np.ndarray, result: Dict[str, Any], mode: str) -> np.ndarray:
        """Prepare frame for display with performance overlays."""
        display_frame = frame.copy()
        
        # Add performance overlay
        self._add_performance_overlay(display_frame)
        
        # Add detection visualizations based on mode
        if mode == "cup" and result["cup_result"]:
            from visualization import visualize_cup_detection
            display_frame = visualize_cup_detection(display_frame, result["cup_result"])
        
        elif mode == "tag" and result["tag_result"]:
            from visualization import visualize_tag_detection
            display_frame = visualize_tag_detection(display_frame, result["tag_result"], self.camera_matrix, self.dist_coeffs)
        
        elif mode == "combined":
            from visualization import visualize_combined_detection
            display_frame = visualize_combined_detection(display_frame, result["tag_result"], result["cup_result"], self.camera_matrix, self.dist_coeffs)
        
            
        
        return display_frame

    def _add_performance_overlay(self, frame: np.ndarray):
        """Add performance statistics overlay to frame."""
        system_fps = self.perf_monitor.get_system_fps()
        cup_stats = self.cup_detector.get_performance_stats()
        camera_stats = self.camera.get_performance_stats()
        
        
        # Prepare overlay text
        overlay_lines = [
            f"System FPS: {system_fps:.1f}",
            f"Target FPS: {self.target_fps:.1f}",
            f"Frame: {self.frame_count}",
            f"Cup FPS: {cup_stats.get('current_fps', 0):.1f}",
            f"Cup Inf: {cup_stats.get('avg_inference_time_ms', 0):.1f}ms",
            f"Camera: {camera_stats.get('avg_capture_time_ms', 0):.1f}ms",
            f"Drop Rate: {camera_stats.get('drop_rate_percent', 0):.1f}%",
        ]

        # Draw overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        margin = 10
        
        for i, line in enumerate(overlay_lines):
            y_pos = margin + (i + 1) * 20
            cv2.putText(frame, line, (margin, y_pos), font, scale, (0, 255, 0), thickness, cv2.LINE_AA)

    def _print_performance_summary(self):
        """Print a brief performance summary."""
        system_fps = self.perf_monitor.get_system_fps()
        cup_stats = self.cup_detector.get_performance_stats()
        camera_stats = self.camera.get_performance_stats()
        
        print(f"\nPerformance Summary (Frame {self.frame_count}):")
        print(f"  System FPS: {system_fps:.1f} / {self.target_fps:.1f}")
        print(f"  Cup Detection: {cup_stats.get('current_fps', 0):.1f} FPS, {cup_stats.get('avg_inference_time_ms', 0):.1f}ms")
        print(f"  Camera: {camera_stats.get('avg_capture_time_ms', 0):.1f}ms, {camera_stats.get('drop_rate_percent', 0):.1f}% dropped")

    def _print_performance_stats(self):
        """Print detailed performance statistics."""
        print("\n" + "="*60)
        print("DETAILED PERFORMANCE STATISTICS")
        print("="*60)
        
        # System stats
        all_stats = self.perf_monitor.get_all_stats()
        for name, stats in all_stats.items():
            print(f"\n{name}:")
            print(f"  Avg: {stats['avg_time_ms']:.2f}ms")
            print(f"  Min: {stats['min_time_ms']:.2f}ms")
            print(f"  Max: {stats['max_time_ms']:.2f}ms")
            print(f"  Calls/sec: {stats['calls_per_second']:.1f}")

        # Cup detector stats
        cup_stats = self.cup_detector.get_performance_stats()
        print(f"\nCup Detector:")
        for key, value in cup_stats.items():
            print(f"  {key}: {value}")
        
        # Camera stats
        camera_stats = self.camera.get_performance_stats()
        print(f"\nCamera:")
        for key, value in camera_stats.items():
            print(f"  {key}: {value}")
        
        print("="*60)

    def _save_performance_report(self):
        """Save performance report to file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"performance_report_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write(f"Performance Report - {timestamp}\n")
            f.write("="*60 + "\n")

            # System stats
            all_stats = self.perf_monitor.get_all_stats()
            for name, stats in all_stats.items():
                f.write(f"\n{name}:\n")
                for key, value in stats.items():
                    f.write(f"  {key}: {value}\n")
            
            # Cup detector stats
            cup_stats = self.cup_detector.get_performance_stats()
            f.write(f"\nCup Detector:\n")
            for key, value in cup_stats.items():
                f.write(f"  {key}: {value}\n")
            
            # Camera stats
            camera_stats = self.camera.get_performance_stats()
            f.write(f"\nCamera:\n")
            for key, value in camera_stats.items():
                f.write(f"  {key}: {value}\n")
        
        print(f"Performance report saved to {filename}")

    def adjust_performance_settings(self, target_fps: Optional[float] = None, skip_frames: Optional[int] = None):
        """Dynamically adjust performance settings."""
        if target_fps is not None:
            self.target_fps = target_fps
            self.frame_interval = 1.0 / target_fps
            self.cup_detector.set_target_fps(target_fps)
            print(f"Adjusted target FPS to {target_fps}")
        
        if skip_frames is not None:
            self.skip_frames = skip_frames
            self.cup_detector.set_skip_frames(skip_frames)
            print(f"Adjusted skip frames to {skip_frames}")

    def cleanup(self):
        """Clean up resources."""
        self.running = False
        self.cup_detector.stop_inference_thread()
        self.camera.release()
        self.perf_monitor.stop_monitoring()
        cv2.destroyAllWindows()
        print("Vision system cleanup completed")


