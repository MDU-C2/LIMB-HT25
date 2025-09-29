from dataclasses import dataclass
from typing import Optional, Tuple, List
import time
import threading
from queue import Queue, Empty
import numpy as np
import cv2
import os


@dataclass
class CupDetectionResult:
    detected: bool # Flag if cup is detected
    pixel_center: Optional[Tuple[int, int]] # Center of cup in pixels
    bounding_box: Optional[Tuple[int, int, int, int]]  # x, y, w, h
    distance_m: Optional[float] # Distance of cup from camera in meters
    inference_time_ms: Optional[float] = None  # Time taken for inference
    confidence: Optional[float] = None  # Detection confidence


class CupDetector:
    """
    YOLOv11 via Ultralytics API. Optimized for Jetson Orin; set device appropriately.
    Filters predictions to the 'cup' class and estimates distance from bbox width.
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        weights_path: str,
        assumed_cup_diameter_m: float = 0.08,
        conf: float = 0.35,
        iou: float = 0.45,
        device: Optional[str] = None,
        max_queue_size: int = 3,
        skip_frames: int = 0,  # Process every (skip_frames + 1) frame
        target_fps: float = 30.0,
    ) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Ultralytics is required for YOLOv11. Install with 'pip install ultralytics'."
            ) from e

        # Ensure model is in the correct location
        from utils.yolo_model_manager import get_yolo_model_path
        
        # Get the vision directory (parent of cup directory)
        vision_dir = os.path.dirname(os.path.dirname(__file__))
        model_name = os.path.basename(weights_path)
        self.model_path = get_yolo_model_path(model_name, vision_dir)

        self._YOLO = YOLO
        self.model = YOLO(self.model_path)
        self.conf = float(conf)
        self.iou = float(iou)
        self.device = device  # e.g., 'cuda:0' on Orin
        self.camera_matrix = camera_matrix
        self.fx = float(camera_matrix[0, 0])
        self.assumed_cup_diameter_m = assumed_cup_diameter_m

        # Performance optimization parameters
        self.max_queue_size = max_queue_size
        self.skip_frames = skip_frames
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        
        # Threading and async processing
        self.inference_queue = Queue(maxsize=max_queue_size)
        self.result_queue = Queue(maxsize=max_queue_size)
        self.inference_thread = None
        self.running = False

        # Performance monitoring
        self.frame_count = 0
        self.processed_count = 0
        self.total_inference_time = 0.0
        self.last_fps_time = time.time()
        self.current_fps = 0.0

        # Resolve cup class id from model names (if available), else fallback to 'cup'
        self._cup_class_ids: List[int] = []
        names = getattr(self.model.model, 'names', None)
        if isinstance(names, dict):
            for k, v in names.items():
                if str(v).lower() == 'cup': # Maybe add bottle class id here later
                    try:
                        self._cup_class_ids.append(int(k))
                    except Exception:
                        pass
        # Fallback to common COCO id for cup (41)
        if not self._cup_class_ids:
            self._cup_class_ids = [41]

        # Pre-allocate arrays for better performance
        self._last_result = CupDetectionResult(False, None, None, None)
        self._frame_buffer = None
        
        # Start inference thread
        self.start_inference_thread()

    def start_inference_thread(self):
        """Start the background inference thread."""
        if self.inference_thread is None or not self.inference_thread.is_alive():
            self.running = True
            self.inference_thread = threading.Thread(target=self._inference_worker, daemon=True)
            self.inference_thread.start()

    def stop_inference_thread(self):
        """Stop the background inference thread."""
        self.running = False
        if self.inference_thread and self.inference_thread.is_alive():
            self.inference_thread.join(timeout=1.0)

    def _inference_worker(self):
        """Background worker thread for YOLO inference."""
        while self.running:
            try:
                # Get frame from queue with timeout
                frame_data = self.inference_queue.get(timeout=0.1)
                if frame_data is None:  # Shutdown signal
                    break
                
                frame, frame_id = frame_data
                start_time = time.time()
                
                # Run inference
                try:
                    results = self.model(
                        source=frame,
                        conf=self.conf,
                        iou=self.iou,
                        device=self.device,
                        verbose=False,
                        stream=False,
                    )

                    inference_time = (time.time() - start_time) * 1000  # Convert to ms
                    
                    # Process results
                    result = self._process_yolo_results(results, inference_time)
                    result.frame_id = frame_id
                    
                    # Put result in queue
                    try:
                        self.result_queue.put_nowait(result)
                    except:
                        # Queue full, replace with latest result
                        try:
                            self.result_queue.get_nowait()
                            self.result_queue.put_nowait(result)
                        except Empty:
                            pass
                            
                except Exception as e:
                    print(f"Inference error: {e}")
                    # Put error result
                    error_result = CupDetectionResult(False, None, None, None, None)
                    error_result.frame_id = frame_id
                    try:
                        self.result_queue.put_nowait(error_result)
                    except:
                        pass
                
                self.inference_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                print(f"Inference worker error: {e}")
                continue

    def _process_yolo_results(self, results, inference_time_ms: float) -> CupDetectionResult:
        """Process YOLO results and return detection result."""
        if not results:
            return CupDetectionResult(False, None, None, None, inference_time_ms)
        
        result = results[0]
        boxes = getattr(result, 'boxes', None)
        if boxes is None or boxes.xyxy is None:
            return CupDetectionResult(False, None, None, None, inference_time_ms)
        
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else np.array(boxes.xyxy)
        cls = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, 'cls') else None
        confs = boxes.conf.cpu().numpy() if hasattr(boxes, 'conf') else None
        
        best_idx = -1
        best_conf = -1.0
        if cls is not None:
            for i, c in enumerate(cls):
                if c in self._cup_class_ids:
                    score = float(confs[i]) if confs is not None else 1.0
                    if score > best_conf:
                        best_conf = score
                        best_idx = i
        
        if best_idx < 0:
            return CupDetectionResult(False, None, None, None, inference_time_ms)
        
        x1, y1, x2, y2 = xyxy[best_idx]
        x = int(max(0, x1))
        y = int(max(0, y1))
        w = int(max(1, x2 - x1))
        h = int(max(1, y2 - y1))
        center = (int(x + w / 2), int(y + h / 2))
        
        distance_m = self.fx * self.assumed_cup_diameter_m / float(max(w, 1))
        return CupDetectionResult(True, center, (x, y, w, h), distance_m, inference_time_ms, best_conf)
        

    def detect(self, frame_bgr: np.ndarray) -> CupDetectionResult:
        """
        Detect cups in frame with optimized processing.
        Returns immediately with latest available result or cached result.
        """
        self.frame_count += 1
        
        # Check if we should process this frame
        should_process = (self.frame_count % (self.skip_frames + 1)) == 0
        
        if should_process:
            # Try to get latest result from queue
            try:
                result = self.result_queue.get_nowait()
                self._last_result = result
                self.processed_count += 1
                if result.inference_time_ms:
                    self.total_inference_time += result.inference_time_ms
            except Empty:
                pass  # Use cached result
            
            # Queue new frame for processing if queue has space
            try:
                self.inference_queue.put_nowait((frame_bgr.copy(), self.frame_count))
            except:
                pass  # Queue full, skip this frame
        
        # Update FPS calculation
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:  # Update every second
            self.current_fps = self.processed_count / (current_time - self.last_fps_time)
            self.last_fps_time = current_time
            self.processed_count = 0
        
        return self._last_result
    
    def get_performance_stats(self) -> dict:
        """Get current performance statistics."""
        avg_inference_time = 0.0
        if self.processed_count > 0:
            avg_inference_time = self.total_inference_time / self.processed_count
        
        return {
            'current_fps': self.current_fps,
            'target_fps': self.target_fps,
            'frame_count': self.frame_count,
            'processed_count': self.processed_count,
            'skip_frames': self.skip_frames,
            'avg_inference_time_ms': avg_inference_time,
            'queue_size': self.inference_queue.qsize(),
            'result_queue_size': self.result_queue.qsize(),
            'device': self.device,
        }
    
    def set_skip_frames(self, skip_frames: int):
        """Dynamically adjust frame skipping for performance tuning."""
        self.skip_frames = max(0, skip_frames)
    
    def set_target_fps(self, target_fps: float):
        """Dynamically adjust target FPS for performance tuning."""
        self.target_fps = max(1.0, target_fps)
    
    def __del__(self):
        """Cleanup on destruction."""
        self.stop_inference_thread()


