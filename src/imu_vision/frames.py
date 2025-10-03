"""Frame definitions and transform management for the fiducial + depth system.

Frames:
- W: robot base/shoulder frame (world frame)
- C: camera frame (rigidly fixed to W)
- H: hand/tool frame (ArUco board on the hand)
"""

from enum import Enum
from typing import Optional
import numpy as np


class FrameType(Enum):
    """Frame identifiers for the system."""
    WORLD = "W"      # robot base/shoulder frame
    CAMERA = "C"     # camera frame
    HAND = "H"       # hand/tool frame


class TransformManager:
    """Manages transforms between different frames."""
    
    def __init__(self):
        self._transforms = {}
    
    def set_transform(self, from_frame: FrameType, to_frame: FrameType, transform: np.ndarray) -> None:
        """Set a 4x4 transform matrix from one frame to another.
        
        Args:
            from_frame: Source frame
            to_frame: Target frame  
            transform: 4x4 homogeneous transformation matrix
        """
        key = (from_frame, to_frame)
        self._transforms[key] = transform.copy()
        
        # Also store the inverse
        inv_key = (to_frame, from_frame)
        self._transforms[inv_key] = np.linalg.inv(transform)
    
    def get_transform(self, from_frame: FrameType, to_frame: FrameType) -> Optional[np.ndarray]:
        """Get transform from one frame to another.
        
        Args:
            from_frame: Source frame
            to_frame: Target frame
            
        Returns:
            4x4 transformation matrix or None if not available
        """
        key = (from_frame, to_frame)
        return self._transforms.get(key)
    
    def transform_point(self, point: np.ndarray, from_frame: FrameType, to_frame: FrameType) -> Optional[np.ndarray]:
        """Transform a 3D point from one frame to another.
        
        Args:
            point: 3D point as numpy array [x, y, z]
            from_frame: Source frame
            to_frame: Target frame
            
        Returns:
            Transformed 3D point or None if transform not available
        """
        transform = self.get_transform(from_frame, to_frame)
        if transform is None:
            return None
        
        # Convert point to homogeneous coordinates
        point_homo = np.append(point, 1.0)
        transformed_homo = transform @ point_homo
        return transformed_homo[:3]
    
    def transform_pose(self, pose: np.ndarray, from_frame: FrameType, to_frame: FrameType) -> Optional[np.ndarray]:
        """Transform a 4x4 pose from one frame to another.
        
        Args:
            pose: 4x4 homogeneous transformation matrix
            from_frame: Source frame
            to_frame: Target frame
            
        Returns:
            Transformed 4x4 pose or None if transform not available
        """
        from_transform = self.get_transform(from_frame, to_frame)
        if from_transform is None:
            return None
        
        return from_transform @ pose
    
    def has_transform(self, from_frame: FrameType, to_frame: FrameType) -> bool:
        """Check if transform between frames is available.
        
        Args:
            from_frame: Source frame
            to_frame: Target frame
            
        Returns:
            True if transform is available
        """
        return (from_frame, to_frame) in self._transforms
