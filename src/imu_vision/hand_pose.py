"""Hand pose estimation from ArUco detection.

Converts ArUco detection results to hand pose in world frame:
T_CH_meas (camera to hand) to T_WH (world to hand)
"""

from typing import Optional, Dict, Any
import numpy as np

from .frames import FrameType, TransformManager
from .calibration import CalibrationManager


class HandPoseEstimator:
    """Estimates hand pose from ArUco detection."""
    
    def __init__(self, calibration_manager: CalibrationManager):
        """Initialize hand pose estimator.
        
        Args:
            calibration_manager: Calibration manager with T_WC and T_CH_fixed
        """
        self.calibration_manager = calibration_manager
        self.transform_manager = calibration_manager.transform_manager
    
    def estimate_hand_pose(self, tag_detection_result) -> Optional[np.ndarray]:
        """Estimate hand pose in world frame from ArUco detection.
        
        Args:
            tag_detection_result: TagDetectionResult object or dict from tag detector
            
        Returns:
            4x4 transformation matrix T_WH (world to hand) or None if detection failed
        """
        if tag_detection_result is None:
            return None
        
        # Extract T_CH_meas from tag detection
        T_CH_meas = self._extract_tag_transform(tag_detection_result)
        if T_CH_meas is None:
            return None
        
        # Get T_WC (camera to world) from calibration
        T_WC = self.transform_manager.get_transform(FrameType.CAMERA, FrameType.WORLD)
        if T_WC is None:
            print("Warning: Camera to world transform not available")
            return None
        
        # Convert to world frame: T_WH = T_WC * T_CH_meas
        T_WH = T_WC @ T_CH_meas
        
        return T_WH
    
    def _extract_tag_transform(self, tag_detection_result) -> Optional[np.ndarray]:
        """Extract 4x4 transform from tag detection result.
        
        Args:
            tag_detection_result: TagDetectionResult object or dict from tag detector
            
        Returns:
            4x4 transformation matrix or None if not available
        """
        # Handle new TagDetectionResult object
        if hasattr(tag_detection_result, 'transforms') and tag_detection_result.transforms is not None:
            # Use the first detected tag's transform
            if len(tag_detection_result.transforms) > 0:
                return tag_detection_result.transforms[0]
        
        # Handle legacy dict format
        if isinstance(tag_detection_result, dict):
            # Try different possible keys for the transform
            transform_keys = ['transform_cam_from_tag', 'transform', 'pose', 'T_cam_from_tag']
            
            for key in transform_keys:
                if key in tag_detection_result:
                    transform = tag_detection_result[key]
                    if isinstance(transform, np.ndarray) and transform.shape == (4, 4):
                        return transform
                    elif isinstance(transform, list) and len(transform) == 4 and len(transform[0]) == 4:
                        return np.array(transform, dtype=np.float64)
            
            # If no direct transform, try to construct from position and orientation
            if 'position' in tag_detection_result and 'orientation' in tag_detection_result:
                return self._construct_transform_from_pose(tag_detection_result)
        
        print("Warning: Could not extract transform from tag detection result")
        return None
    
    def _construct_transform_from_pose(self, tag_detection_result: Dict[str, Any]) -> Optional[np.ndarray]:
        """Construct 4x4 transform from position and orientation.
        
        Args:
            tag_detection_result: Result containing position and orientation
            
        Returns:
            4x4 transformation matrix or None if construction failed
        """
        try:
            position = tag_detection_result['position']
            orientation = tag_detection_result['orientation']
            
            # Convert to numpy arrays
            if isinstance(position, (list, tuple)):
                pos = np.array(position, dtype=np.float64)
            else:
                pos = position
            
            if isinstance(orientation, (list, tuple)):
                orient = np.array(orientation, dtype=np.float64)
            else:
                orient = orientation
            
            # Create rotation matrix from orientation
            if orient.shape == (3,):  # Euler angles (roll, pitch, yaw)
                R = self._euler_to_rotation_matrix(orient)
            elif orient.shape == (4,):  # Quaternion (w, x, y, z)
                R = self._quaternion_to_rotation_matrix(orient)
            elif orient.shape == (3, 3):  # Already a rotation matrix
                R = orient
            else:
                print(f"Warning: Unknown orientation format with shape {orient.shape}")
                return None
            
            # Construct 4x4 transform
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = pos
            
            return T
            
        except Exception as e:
            print(f"Warning: Could not construct transform from pose: {e}")
            return None
    
    def _euler_to_rotation_matrix(self, euler_angles: np.ndarray) -> np.ndarray:
        """Convert Euler angles (roll, pitch, yaw) to rotation matrix.
        
        Args:
            euler_angles: [roll, pitch, yaw] in radians
            
        Returns:
            3x3 rotation matrix
        """
        roll, pitch, yaw = euler_angles
        
        # Rotation matrices
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(roll), -np.sin(roll)],
                        [0, np.sin(roll), np.cos(roll)]])
        
        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])
        
        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])
        
        # Combined rotation: R = R_z * R_y * R_x
        return R_z @ R_y @ R_x
    
    def _quaternion_to_rotation_matrix(self, quaternion: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix.
        
        Args:
            quaternion: [w, x, y, z] quaternion
            
        Returns:
            3x3 rotation matrix
        """
        w, x, y, z = quaternion
        
        # Normalize quaternion
        norm = np.sqrt(w*w + x*x + y*y + z*z)
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
        
        # Convert to rotation matrix
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
        
        return R
    
    def get_hand_position(self, tag_detection_result) -> Optional[np.ndarray]:
        """Get hand position in world frame.
        
        Args:
            tag_detection_result: TagDetectionResult object or dict from tag detector
            
        Returns:
            3D position [x, y, z] in world frame or None if detection failed
        """
        T_WH = self.estimate_hand_pose(tag_detection_result)
        if T_WH is None:
            return None
        
        return T_WH[:3, 3]
    
    def get_hand_orientation(self, tag_detection_result) -> Optional[np.ndarray]:
        """Get hand orientation in world frame.
        
        Args:
            tag_detection_result: TagDetectionResult object or dict from tag detector
            
        Returns:
            3x3 rotation matrix in world frame or None if detection failed
        """
        T_WH = self.estimate_hand_pose(tag_detection_result)
        if T_WH is None:
            return None
        
        return T_WH[:3, :3]
