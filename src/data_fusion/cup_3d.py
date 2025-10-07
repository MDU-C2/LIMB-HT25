"""Cup 3D position estimation from YOLO + depth.

Processes YOLO detection results and depth data to estimate cup position:
bbox to depth ROI to robust depth pooling to p_C^cup to p_W^cup
"""

from typing import Optional, Tuple, List
import numpy as np
import cv2

from .frames import FrameType, TransformManager
from .calibration import CalibrationManager


class Cup3DEstimator:
    """Estimates 3D cup position from YOLO detection and depth data."""
    
    def __init__(self, calibration_manager: CalibrationManager, camera_matrix: np.ndarray):
        """Initialize cup 3D estimator.
        
        Args:
            calibration_manager: Calibration manager with T_WC
            camera_matrix: 3x3 camera intrinsic matrix
        """
        self.calibration_manager = calibration_manager
        self.camera_matrix = camera_matrix
        self.transform_manager = calibration_manager.transform_manager
    
    def estimate_cup_3d(self, 
                       cup_detection_result: Optional[dict], 
                       depth_image: Optional[np.ndarray] = None,
                       depth_method: str = "median") -> Optional[np.ndarray]:
        """Estimate cup 3D position in world frame.
        
        Args:
            cup_detection_result: Result from cup detector (should contain bbox)
            depth_image: Depth image (optional, for future depth camera integration)
            depth_method: Method for depth estimation ("median", "mean", "center")
            
        Returns:
            3D position [x, y, z] in world frame or None if detection failed
        """
        if cup_detection_result is None:
            return None
        
        # Extract cup position in camera frame
        p_C_cup = self._estimate_cup_position_camera_frame(cup_detection_result, depth_image, depth_method)
        if p_C_cup is None:
            return None
        
        # Transform to world frame: p_W^cup = T_WC · p_C^cup
        p_W_cup = self.transform_manager.transform_point(p_C_cup, FrameType.CAMERA, FrameType.WORLD)
        
        return p_W_cup
    
    def _estimate_cup_position_camera_frame(self, 
                                          cup_detection_result: dict, 
                                          depth_image: Optional[np.ndarray] = None,
                                          depth_method: str = "median") -> Optional[np.ndarray]:
        """Estimate cup position in camera frame.
        
        Args:
            cup_detection_result: Result from cup detector
            depth_image: Depth image (optional)
            depth_method: Method for depth estimation
            
        Returns:
            3D position [x, y, z] in camera frame or None if estimation failed
        """
        # Get cup center from detection
        cup_center = self._extract_cup_center(cup_detection_result)
        if cup_center is None:
            return None
        
        # Estimate depth
        depth = self._estimate_depth(cup_detection_result, depth_image, depth_method)
        if depth is None:
            return None
        
        # Convert to 3D position in camera frame
        p_C_cup = self._pixel_to_3d(cup_center, depth)
        
        return p_C_cup
    
    def _extract_cup_center(self, cup_detection_result: dict) -> Optional[Tuple[int, int]]:
        """Extract cup center pixel coordinates from detection result.
        
        Args:
            cup_detection_result: Result from cup detector
            
        Returns:
            (x, y) pixel coordinates or None if not available
        """
        # Try different possible keys for the center
        center_keys = ['pixel_center', 'center', 'bbox_center', 'center_pixel']
        
        for key in center_keys:
            if key in cup_detection_result:
                center = cup_detection_result[key]
                if isinstance(center, (list, tuple)) and len(center) == 2:
                    return (int(center[0]), int(center[1]))
                elif isinstance(center, np.ndarray) and center.shape == (2,):
                    return (int(center[0]), int(center[1]))
        
        # Try to extract from bbox
        if 'bbox' in cup_detection_result:
            bbox = cup_detection_result['bbox']
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                return (int(center_x), int(center_y))
        
        print("Warning: Could not extract cup center from detection result")
        return None
    
    def _estimate_depth(self, 
                       cup_detection_result: dict, 
                       depth_image: Optional[np.ndarray] = None,
                       depth_method: str = "median") -> Optional[float]:
        """Estimate depth of cup.
        
        Args:
            cup_detection_result: Result from cup detector
            depth_image: Depth image (optional)
            depth_method: Method for depth estimation
            
        Returns:
            Depth in meters or None if estimation failed
        """
        # If depth image is available, use it
        if depth_image is not None:
            return self._estimate_depth_from_image(cup_detection_result, depth_image, depth_method)
        
        # Otherwise, try to use distance from cup detector
        distance_keys = ['distance_m', 'distance', 'depth', 'z_distance']
        
        for key in distance_keys:
            if key in cup_detection_result:
                distance = cup_detection_result[key]
                if isinstance(distance, (int, float)) and distance > 0:
                    return float(distance)
        
        # Fallback: assume reasonable depth
        print("Warning: No depth information available, using fallback depth")
        return 0.5  # 50cm fallback
    
    def _estimate_depth_from_image(self, 
                                 cup_detection_result: dict, 
                                 depth_image: np.ndarray,
                                 depth_method: str = "median") -> Optional[float]:
        """Estimate depth from depth image using ROI around cup.
        
        Args:
            cup_detection_result: Result from cup detector
            depth_image: Depth image
            depth_method: Method for depth estimation ("median", "mean", "center")
            
        Returns:
            Depth in meters or None if estimation failed
        """
        # Get cup bbox
        bbox = self._extract_bbox(cup_detection_result)
        if bbox is None:
            return None
        
        x1, y1, x2, y2 = bbox
        
        # Ensure bbox is within image bounds
        h, w = depth_image.shape[:2]
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))
        
        # Extract depth ROI
        depth_roi = depth_image[y1:y2, x1:x2]
        
        # Filter out invalid depth values (0 or NaN)
        valid_depths = depth_roi[(depth_roi > 0) & ~np.isnan(depth_roi)] # Maybe here we could remove points outside a certain distance from the camera
        
        if len(valid_depths) == 0:
            print("Warning: No valid depth values in cup ROI")
            return None
        
        # Apply depth estimation method
        if depth_method == "median":
            depth = np.median(valid_depths)
        elif depth_method == "mean":
            depth = np.mean(valid_depths)
        elif depth_method == "center":
            # Use depth at center of ROI
            center_y, center_x = depth_roi.shape[0] // 2, depth_roi.shape[1] // 2
            depth = depth_roi[center_y, center_x]
            if depth <= 0 or np.isnan(depth):
                depth = np.median(valid_depths)  # Fallback to median
        else:
            print(f"Warning: Unknown depth method '{depth_method}', using median")
            depth = np.median(valid_depths)
        
        return float(depth)
    
    def _extract_bbox(self, cup_detection_result: dict) -> Optional[Tuple[float, float, float, float]]:
        """Extract bounding box from cup detection result.
        
        Args:
            cup_detection_result: Result from cup detector
            
        Returns:
            (x1, y1, x2, y2) bounding box or None if not available
        """
        bbox_keys = ['bbox', 'bounding_box', 'box']
        
        for key in bbox_keys:
            if key in cup_detection_result:
                bbox = cup_detection_result[key]
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    return tuple(bbox)
                elif isinstance(bbox, np.ndarray) and bbox.shape == (4,):
                    return tuple(bbox)
        
        print("Warning: Could not extract bbox from cup detection result")
        return None
    
    def _pixel_to_3d(self, pixel: Tuple[int, int], depth: float) -> np.ndarray:
        """Convert pixel coordinates and depth to 3D position in camera frame.
        
        Args:
            pixel: (x, y) pixel coordinates
            depth: Depth in meters
            
        Returns:
            3D position [x, y, z] in camera frame
        """
        u, v = pixel
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        # Convert to 3D coordinates
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        
        return np.array([x, y, z], dtype=np.float64)
    
    def get_cup_position_camera_frame(self, 
                                    cup_detection_result: Optional[dict], 
                                    depth_image: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Get cup position in camera frame only.
        
        Args:
            cup_detection_result: Result from cup detector
            depth_image: Depth image (optional)
            
        Returns:
            3D position [x, y, z] in camera frame or None if detection failed
        """
        return self._estimate_cup_position_camera_frame(cup_detection_result, depth_image)
