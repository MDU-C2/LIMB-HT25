import cv2
import numpy as np
import threading
import time
from typing import Optional, Tuple, Dict, Any
from queue import Queue, Empty
from dataclasses import dataclass


@dataclass
class CameraConfig:
    """Camera configuration for optimal performance."""
    device_id: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    buffer_size: int = 1  # Minimal buffer to reduce latency
    fourcc: str = 'MJPG'  # Compressed format for better performance
    exposure: Optional[int] = None  # Auto exposure
    gain: Optional[int] = None  # Auto gain
    brightness: Optional[int] = None
    contrast: Optional[int] = None
    saturation: Optional[int] = None


class OptimizedCameraCapture:
    """
    Optimized camera capture with:
    - Background frame capture
    - Configurable buffer management
    - Performance monitoring
    - Automatic parameter tuning
    """
    
    def __init__(self, config: CameraConfig):
        self.config = config
        self.cap = None
        self.frame_queue = Queue(maxsize=2)  # Small buffer for low latency
        self.capture_thread = None
        self.running = False
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        # Performance monitoring
        self.capture_times = []
        self.dropped_frames = 0
        self.total_frames = 0
        
        # Initialize camera
        self._initialize_camera()
        self._start_capture_thread()
    
    def _initialize_camera(self):
        """Initialize camera with optimized settings."""
        self.cap = cv2.VideoCapture(self.config.device_id)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.config.device_id}")
        
        # Set resolution and FPS
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        
        # Set buffer size (minimal for low latency)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
        
        # Set codec for compressed capture
        fourcc = cv2.VideoWriter_fourcc(*self.config.fourcc)
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        
        # Set exposure and gain if specified
        if self.config.exposure is not None:
            self.cap.set(cv2.CAP_PROP_EXPOSURE, self.config.exposure)
        if self.config.gain is not None:
            self.cap.set(cv2.CAP_PROP_GAIN, self.config.gain)
        if self.config.brightness is not None:
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.config.brightness)
        if self.config.contrast is not None:
            self.cap.set(cv2.CAP_PROP_CONTRAST, self.config.contrast)
        if self.config.saturation is not None:
            self.cap.set(cv2.CAP_PROP_SATURATION, self.config.saturation)
        
        # Verify settings
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")
    
    def _start_capture_thread(self):
        """Start background frame capture thread."""
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.capture_thread.start()
    
    def _capture_worker(self):
        """Background worker for continuous frame capture."""
        while self.running:
            start_time = time.time()
            
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("Failed to capture frame")
                time.sleep(0.01)  # Small delay to prevent busy waiting
                continue
            
            self.total_frames += 1
            
            # Update latest frame
            with self.frame_lock:
                self.latest_frame = frame.copy()
            
            # Try to add to queue (non-blocking)
            try:
                self.frame_queue.put_nowait(frame.copy())
            except:
                # Queue full, drop frame
                self.dropped_frames += 1
                try:
                    # Remove oldest frame and add new one
                    self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(frame.copy())
                except Empty:
                    pass
            
            # Record capture time
            capture_time = time.time() - start_time
            self.capture_times.append(capture_time)
            if len(self.capture_times) > 100:  # Keep last 100 measurements
                self.capture_times.pop(0)
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read latest frame (non-blocking).
        Returns (success, frame) tuple.
        """
        with self.frame_lock:
            if self.latest_frame is not None:
                return True, self.latest_frame.copy()
            return False, None
    
    def read_blocking(self, timeout: float = 0.1) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read frame with timeout (blocking).
        Returns (success, frame) tuple.
        """
        try:
            frame = self.frame_queue.get(timeout=timeout)
            return True, frame
        except Empty:
            return False, None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get camera performance statistics."""
        avg_capture_time = 0.0
        if self.capture_times:
            avg_capture_time = np.mean(self.capture_times) * 1000  # Convert to ms
        
        drop_rate = 0.0
        if self.total_frames > 0:
            drop_rate = (self.dropped_frames / self.total_frames) * 100
        
        return {
            'avg_capture_time_ms': avg_capture_time,
            'total_frames': self.total_frames,
            'dropped_frames': self.dropped_frames,
            'drop_rate_percent': drop_rate,
            'queue_size': self.frame_queue.qsize(),
            'actual_fps': self.cap.get(cv2.CAP_PROP_FPS) if self.cap else 0.0,
            'actual_width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self.cap else 0,
            'actual_height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self.cap else 0,
        }
    
    def adjust_settings_for_performance(self):
        """Automatically adjust camera settings for better performance."""
        if not self.cap:
            return
        
        # Try to reduce exposure time for faster capture
        current_exposure = self.cap.get(cv2.CAP_PROP_EXPOSURE)
        if current_exposure > 0:  # Manual exposure mode
            new_exposure = max(current_exposure * 0.8, -10)  # Reduce exposure
            self.cap.set(cv2.CAP_PROP_EXPOSURE, new_exposure)
            print(f"Adjusted exposure: {current_exposure} -> {new_exposure}")
        
        # Try to increase gain to compensate for reduced exposure
        current_gain = self.cap.get(cv2.CAP_PROP_GAIN)
        if current_gain < 100:  # Reasonable gain limit
            new_gain = min(current_gain * 1.2, 100)
            self.cap.set(cv2.CAP_PROP_GAIN, new_gain)
            print(f"Adjusted gain: {current_gain} -> {new_gain}")
    
    def release(self):
        """Release camera resources."""
        self.running = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        
        if self.cap:
            self.cap.release()
            self.cap = None


def create_optimized_camera_config(
    device_id: int = 0,
    target_fps: int = 30,
    resolution: Tuple[int, int] = (640, 480),
    low_latency: bool = True
) -> CameraConfig:
    """
    Create optimized camera configuration based on requirements.
    
    Args:
        device_id: Camera device ID
        target_fps: Target frames per second
        resolution: (width, height) tuple
        low_latency: Whether to prioritize low latency over quality
    
    Returns:
        Optimized CameraConfig
    """
    width, height = resolution
    
    if low_latency:
        # Optimize for low latency
        return CameraConfig(
            device_id=device_id,
            width=width,
            height=height,
            fps=target_fps,
            buffer_size=1,  # Minimal buffer
            fourcc='MJPG',  # Compressed format
            exposure=-6,    # Faster exposure
            gain=50,        # Higher gain
            brightness=50,
            contrast=50,
            saturation=50
        )
    else:
        # Optimize for quality
        return CameraConfig(
            device_id=device_id,
            width=width,
            height=height,
            fps=target_fps,
            buffer_size=2,  # Slightly larger buffer
            fourcc='YUYV',  # Uncompressed format
            exposure=None,  # Auto exposure
            gain=None,      # Auto gain
            brightness=None,
            contrast=None,
            saturation=None
        )


def benchmark_camera_settings(device_id: int = 0) -> Dict[str, Any]:
    """
    Benchmark different camera settings to find optimal configuration.
    
    Args:
        device_id: Camera device ID to test
    
    Returns:
        Dictionary with benchmark results
    """
    results = {}
    
    # Test different configurations
    configs = [
        ("Low Latency", create_optimized_camera_config(device_id, low_latency=True)),
        ("High Quality", create_optimized_camera_config(device_id, low_latency=False)),
        ("Balanced", CameraConfig(device_id=device_id, width=640, height=480, fps=30, buffer_size=1, fourcc='MJPG'))
    ]
    
    for name, config in configs:
        print(f"Testing {name} configuration...")
        
        try:
            camera = OptimizedCameraCapture(config)
            time.sleep(2.0)  # Let it stabilize
            
            stats = camera.get_performance_stats()
            results[name] = stats
            
            camera.release()
            
        except Exception as e:
            print(f"Error testing {name}: {e}")
            results[name] = {"error": str(e)}
    
    return results
