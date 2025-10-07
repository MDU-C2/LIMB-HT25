"""
Vision System Wrapper

Wrapper around the existing vision system to integrate it with the sensor management system.
Provides a clean interface for object detection and pose estimation.
"""

import cv2
import numpy as np
import time
from typing import Optional, Dict, Any, Tuple

class VisionSystemWrapper:
    """Wrapper for the existing vision system to integrate with sensor management."""
    
    def __init__(self):
        self.vision_system = None
        self.camera = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.running = False
        
        # Configuration
        self.camera_device_id = 0
        self.target_fps = 30.0
        
        print("VisionSystemWrapper initialized")
    
    def activate(self, camera_matrix: Optional[np.ndarray] = None, 
                 dist_coeffs: Optional[np.ndarray] = None) -> bool:
        """
        Activate vision system.
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: Camera distortion coefficients
            
        Returns:
            True if activation successful
        """
        if self.running:
            return True
        
        try:
            # Import vision system components
            from vision.system import VisionSystem
            from vision.tags.utils.camera_calibration import load_calibration_json
            
            # Load camera calibration
            if camera_matrix is None or dist_coeffs is None:
                try:
                    calibration_path = "vision/tags/utils/camera_calibration.json"
                    self.camera_matrix, self.dist_coeffs = load_calibration_json(calibration_path)
                    print(f"Loaded camera calibration from {calibration_path}")
                except:
                    # Fallback calibration
                    fx, fy = 800.0, 800.0
                    cx, cy = 320.0, 240.0
                    self.camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
                    self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)
                    print("Using fallback camera calibration")
            else:
                self.camera_matrix = camera_matrix
                self.dist_coeffs = dist_coeffs
                print("Using provided camera calibration")
            
            # Initialize vision system
            self.vision_system = VisionSystem(
                camera_matrix=self.camera_matrix,
                dist_coeffs=self.dist_coeffs,
                marker_length_m=0.03,
                assumed_cup_diameter_m=0.08,
                yolo_model_path="yolo11s.pt",
                target_fps=self.target_fps,
                skip_frames=1,
                camera_device_id=self.camera_device_id,
                low_latency_mode=True,
                monitor_performance=False
            )
            
            # Initialize camera
            self.camera = cv2.VideoCapture(self.camera_device_id)
            if not self.camera.isOpened():
                raise RuntimeError("Could not open camera")
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            self.running = True
            print(f"Vision system activated (camera {self.camera_device_id})")
            return True
            
        except Exception as e:
            print(f"Failed to activate vision system: {e}")
            return False
    
    def deactivate(self):
        """Deactivate vision system."""
        if not self.running:
            return
        
        self.running = False
        
        if self.camera:
            self.camera.release()
            self.camera = None
        
        if self.vision_system:
            self.vision_system.cleanup()
            self.vision_system = None
        
        print("Vision system deactivated")
    
    def detect_cup(self) -> Optional[Dict[str, Any]]:
        """
        Detect cup in current frame.
        
        Returns:
            Dictionary with cup detection results or None if no cup detected
        """
        if not self.running or not self.camera:
            return None
        
        try:
            # Read camera frame
            ret, frame = self.camera.read()
            if not ret:
                return None
            
            # Process frame with vision system
            result = self.vision_system.process_frame(frame, mode="cup")
            
            # Extract cup detection results
            if result and 'cup_3d' in result and result['cup_3d']['detected']:
                cup_data = result['cup_3d']
                return {
                    'detected': True,
                    'position_world': cup_data.get('position_world'),
                    'position_camera': cup_data.get('position_camera'),
                    'distance': cup_data.get('distance'),
                    'timestamp': time.time()
                }
            
            return None
            
        except Exception as e:
            print(f"Error in cup detection: {e}")
            return None
    
    def detect_hand_pose(self) -> Optional[Dict[str, Any]]:
        """
        Detect hand pose using ArUco markers.
        
        Returns:
            Dictionary with hand pose results or None if no pose detected
        """
        if not self.running or not self.camera:
            return None
        
        try:
            # Read camera frame
            ret, frame = self.camera.read()
            if not ret:
                return None
            
            # Process frame with vision system
            result = self.vision_system.process_frame(frame, mode="tag")
            
            # Extract hand pose results
            if result and 'hand_pose' in result and result['hand_pose']['detected']:
                hand_data = result['hand_pose']
                return {
                    'detected': True,
                    'position': hand_data.get('position'),
                    'orientation': hand_data.get('orientation'),
                    'transform_matrix': hand_data.get('T_WH'),
                    'source': hand_data.get('source'),
                    'timestamp': time.time()
                }
            
            return None
            
        except Exception as e:
            print(f"Error in hand pose detection: {e}")
            return None
    
    def detect_combined(self) -> Dict[str, Any]:
        """
        Detect both cup and hand pose in current frame.
        
        Returns:
            Dictionary with both cup and hand detection results
        """
        if not self.running or not self.camera:
            return {'cup': None, 'hand': None}
        
        try:
            # Read camera frame
            ret, frame = self.camera.read()
            if not ret:
                return {'cup': None, 'hand': None}
            
            # Process frame with vision system
            result = self.vision_system.process_frame(frame, mode="combined")
            
            # Extract results
            cup_result = None
            hand_result = None
            
            if result:
                # Extract cup data
                if 'cup_3d' in result and result['cup_3d']['detected']:
                    cup_data = result['cup_3d']
                    cup_result = {
                        'detected': True,
                        'position_world': cup_data.get('position_world'),
                        'position_camera': cup_data.get('position_camera'),
                        'distance': cup_data.get('distance')
                    }
                
                # Extract hand data
                if 'hand_pose' in result and result['hand_pose']['detected']:
                    hand_data = result['hand_pose']
                    hand_result = {
                        'detected': True,
                        'position': hand_data.get('position'),
                        'orientation': hand_data.get('orientation'),
                        'transform_matrix': hand_data.get('T_WH'),
                        'source': hand_data.get('source')
                    }
            
            return {
                'cup': cup_result,
                'hand': hand_result,
                'timestamp': time.time()
            }
            
        except Exception as e:
            print(f"Error in combined detection: {e}")
            return {'cup': None, 'hand': None}
    
    def get_latest_data(self) -> Optional[Dict[str, Any]]:
        """Get latest detection data (alias for detect_combined)."""
        return self.detect_combined()
    
    def get_camera_frame(self) -> Optional[np.ndarray]:
        """Get current camera frame."""
        if not self.running or not self.camera:
            return None
        
        try:
            ret, frame = self.camera.read()
            return frame if ret else None
        except Exception as e:
            print(f"Error getting camera frame: {e}")
            return None
    
    def set_camera_properties(self, width: int = 640, height: int = 480, fps: float = 30.0):
        """Set camera properties."""
        if self.camera:
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.camera.set(cv2.CAP_PROP_FPS, fps)
            self.target_fps = fps
    
    def get_status(self) -> Dict[str, Any]:
        """Get vision system status."""
        return {
            'running': self.running,
            'camera_device_id': self.camera_device_id,
            'target_fps': self.target_fps,
            'camera_matrix_available': self.camera_matrix is not None,
            'camera_opened': self.camera is not None and self.camera.isOpened() if self.camera else False
        }
    
    def cleanup(self):
        """Cleanup vision system resources."""
        self.deactivate()
        print("Vision system cleanup completed")
