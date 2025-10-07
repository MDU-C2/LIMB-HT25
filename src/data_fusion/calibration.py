"""Calibration management for the fiducial + depth system.

Handles:
- T_WC: camera to world transform (constant, calibrated once)
- T_CH_fixed: camera to hand tag transform when tag is seen (known from tag size/layout)
"""

import json
import os
from typing import Optional, Dict, Any
import numpy as np

from .frames import FrameType, TransformManager


class CalibrationManager:
    """Manages calibration data for the system."""
    
    def __init__(self, calibration_file: Optional[str] = None):
        """Initialize calibration manager.
        
        Args:
            calibration_file: Path to calibration JSON file. If None, uses default location.
        """
        self.calibration_file = calibration_file or self._get_default_calibration_path()
        self.transform_manager = TransformManager()
        self._load_calibration()
    
    def _get_default_calibration_path(self) -> str:
        """Get default calibration file path."""
        # Get the directory of this file (src/vision/imu_vision/)
        imu_vision_dir = os.path.dirname(__file__)
        return os.path.join(imu_vision_dir, "calibration.json")
    
    def _load_calibration(self) -> None:
        """Load calibration data from file."""
        if os.path.exists(self.calibration_file):
            try:
                with open(self.calibration_file, 'r') as f:
                    data = json.load(f)
                
                # Load T_WC (camera toworld)
                if 'T_WC' in data:
                    T_WC = np.array(data['T_WC'], dtype=np.float64)
                    self.transform_manager.set_transform(FrameType.CAMERA, FrameType.WORLD, T_WC)
                
                # Load T_CH_fixed (camera to hand tag when seen)
                if 'T_CH_fixed' in data:
                    T_CH_fixed = np.array(data['T_CH_fixed'], dtype=np.float64)
                    self.transform_manager.set_transform(FrameType.CAMERA, FrameType.HAND, T_CH_fixed)
                
                print(f"Loaded calibration from {self.calibration_file}")
                
            except Exception as e:
                print(f"Warning: Could not load calibration from {self.calibration_file}: {e}")
                self._create_default_calibration()
        else:
            print(f"Calibration file {self.calibration_file} not found. Creating default calibration.")
            self._create_default_calibration()
    
    def _create_default_calibration(self) -> None:
        """Create default calibration (identity transforms)."""
        # Default: camera is at origin of world frame
        T_WC = np.eye(4, dtype=np.float64)
        self.transform_manager.set_transform(FrameType.CAMERA, FrameType.WORLD, T_WC)
        
        # Default: hand tag is 0.1m in front of camera
        T_CH_fixed = np.eye(4, dtype=np.float64)
        T_CH_fixed[2, 3] = 0.1  # 10cm in front
        self.transform_manager.set_transform(FrameType.CAMERA, FrameType.HAND, T_CH_fixed)
        
        print("Using default calibration (camera at world origin, hand 10cm in front)")
    
    def save_calibration(self) -> None:
        """Save current calibration to file."""
        data = {}
        
        # Save T_WC
        T_WC = self.transform_manager.get_transform(FrameType.CAMERA, FrameType.WORLD)
        if T_WC is not None:
            data['T_WC'] = T_WC.tolist()
        
        # Save T_CH_fixed
        T_CH_fixed = self.transform_manager.get_transform(FrameType.CAMERA, FrameType.HAND)
        if T_CH_fixed is not None:
            data['T_CH_fixed'] = T_CH_fixed.tolist()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.calibration_file), exist_ok=True)
        
        with open(self.calibration_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved calibration to {self.calibration_file}")
    
    def set_camera_to_world_transform(self, T_WC: np.ndarray) -> None:
        """Set the camera→world transform.
        
        Args:
            T_WC: 4x4 transformation matrix from camera to world frame
        """
        self.transform_manager.set_transform(FrameType.CAMERA, FrameType.WORLD, T_WC)
        print("Updated camera to world transform")
    
    def set_hand_tag_transform(self, T_CH_fixed: np.ndarray) -> None:
        """Set the camera→hand tag transform (when tag is seen).
        
        Args:
            T_CH_fixed: 4x4 transformation matrix from camera to hand tag frame
        """
        self.transform_manager.set_transform(FrameType.CAMERA, FrameType.HAND, T_CH_fixed)
        print("Updated camera to hand tag transform")
    
    def get_camera_to_world_transform(self) -> Optional[np.ndarray]:
        """Get the camera→world transform."""
        return self.transform_manager.get_transform(FrameType.CAMERA, FrameType.WORLD)
    
    def get_hand_tag_transform(self) -> Optional[np.ndarray]:
        """Get the camera→hand tag transform."""
        return self.transform_manager.get_transform(FrameType.CAMERA, FrameType.HAND)
    
    def is_calibrated(self) -> bool:
        """Check if system is calibrated."""
        return (self.transform_manager.has_transform(FrameType.CAMERA, FrameType.WORLD) and
                self.transform_manager.has_transform(FrameType.CAMERA, FrameType.HAND))
    
    def get_calibration_info(self) -> Dict[str, Any]:
        """Get calibration information for debugging."""
        info = {
            'calibration_file': self.calibration_file,
            'is_calibrated': self.is_calibrated(),
            'has_T_WC': self.transform_manager.has_transform(FrameType.CAMERA, FrameType.WORLD),
            'has_T_CH_fixed': self.transform_manager.has_transform(FrameType.CAMERA, FrameType.HAND)
        }
        
        if info['has_T_WC']:
            T_WC = self.get_camera_to_world_transform()
            info['T_WC'] = T_WC.tolist()
        
        if info['has_T_CH_fixed']:
            T_CH_fixed = self.get_hand_tag_transform()
            info['T_CH_fixed'] = T_CH_fixed.tolist()
        
        return info
