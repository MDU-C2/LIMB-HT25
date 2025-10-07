"""Relative pose calculation for control.

Calculates relative poses and positions for robot control:
- T_HW: hand to world transform (inverse of T_WH)
- p_H^cup: cup position in hand frame
- Control errors for reach/grasp controller
"""

from typing import Optional, Dict, Any
import numpy as np

from .frames import FrameType, TransformManager
from .calibration import CalibrationManager


class RelativePoseCalculator:
    """Calculates relative poses and positions for robot control."""
    
    def __init__(self, calibration_manager: CalibrationManager):
        """Initialize relative pose calculator.
        
        Args:
            calibration_manager: Calibration manager with transforms
        """
        self.calibration_manager = calibration_manager
        self.transform_manager = calibration_manager.transform_manager
    
    def calculate_hand_to_world_transform(self, T_WH: np.ndarray) -> np.ndarray:
        """Calculate hand to world transform from world to hand transform.
        
        Args:
            T_WH: 4x4 world to hand transform matrix
            
        Returns:
            4x4 hand to world transform matrix (T_HW = T_WH^(-1))
        """
        return np.linalg.inv(T_WH)
    
    def calculate_cup_position_in_hand_frame(self, 
                                           p_W_cup: np.ndarray, 
                                           T_WH: np.ndarray) -> np.ndarray:
        """Calculate cup position in hand frame.
        
        Args:
            p_W_cup: Cup position in world frame [x, y, z]
            T_WH: World→hand transform matrix
            
        Returns:
            Cup position in hand frame [x, y, z]
        """
        # Convert to homogeneous coordinates
        p_W_cup_homo = np.append(p_W_cup, 1.0)
        
        # Transform to hand frame: p_H^cup = T_HW · p_W^cup
        T_HW = self.calculate_hand_to_world_transform(T_WH)
        p_H_cup_homo = T_HW @ p_W_cup_homo
        
        return p_H_cup_homo[:3]
    
    def calculate_control_errors(self, 
                               p_H_cup: np.ndarray, 
                               target_position: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Calculate control errors for reach/grasp controller.
        
        Args:
            p_H_cup: Cup position in hand frame [x, y, z]
            target_position: Target position in hand frame [x, y, z] (optional)
            
        Returns:
            Dictionary containing control errors and information
        """
        errors = {
            'cup_position_hand_frame': p_H_cup.tolist(),
            'distance_to_cup': float(np.linalg.norm(p_H_cup)),
            'x_error': float(p_H_cup[0]),
            'y_error': float(p_H_cup[1]),
            'z_error': float(p_H_cup[2])
        }
        
        # Calculate errors relative to target if provided
        if target_position is not None:
            target = np.array(target_position, dtype=np.float64)
            position_error = p_H_cup - target
            
            errors.update({
                'target_position': target.tolist(),
                'position_error': position_error.tolist(),
                'position_error_magnitude': float(np.linalg.norm(position_error)),
                'x_error_from_target': float(position_error[0]),
                'y_error_from_target': float(position_error[1]),
                'z_error_from_target': float(position_error[2])
            })
        
        return errors
    
    def calculate_grasp_approach_vector(self, 
                                      p_H_cup: np.ndarray, 
                                      approach_distance: float = 0.05) -> Dict[str, Any]:
        """Calculate approach vector for grasping.
        
        Args:
            p_H_cup: Cup position in hand frame [x, y, z]
            approach_distance: Distance to maintain during approach (meters)
            
        Returns:
            Dictionary containing approach vector and target position
        """
        # Calculate unit vector from hand to cup
        distance = np.linalg.norm(p_H_cup)
        if distance < 1e-6:  # Avoid division by zero
            approach_vector = np.array([0, 0, 1])  # Default forward direction
        else:
            approach_vector = p_H_cup / distance
        
        # Calculate target position (approach_distance away from cup)
        target_position = p_H_cup - approach_vector * approach_distance
        
        return {
            'approach_vector': approach_vector.tolist(),
            'target_position': target_position.tolist(),
            'current_distance': float(distance),
            'approach_distance': approach_distance
        }
    
    def calculate_orientation_error(self, 
                                  R_WH: np.ndarray, 
                                  target_orientation: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Calculate orientation error for control.
        
        Args:
            R_WH: 3x3 rotation matrix (world→hand)
            target_orientation: Target 3x3 rotation matrix (optional)
            
        Returns:
            Dictionary containing orientation errors
        """
        # Convert rotation matrix to Euler angles for easier interpretation
        euler_angles = self._rotation_matrix_to_euler(R_WH)
        
        orientation_info = {
            'current_orientation_euler': euler_angles.tolist(),
            'current_orientation_matrix': R_WH.tolist()
        }
        
        if target_orientation is not None:
            target_euler = self._rotation_matrix_to_euler(target_orientation)
            orientation_error = euler_angles - target_euler
            
            # Normalize angles to [-π, π]
            orientation_error = np.arctan2(np.sin(orientation_error), np.cos(orientation_error))
            
            orientation_info.update({
                'target_orientation_euler': target_euler.tolist(),
                'orientation_error_euler': orientation_error.tolist(),
                'orientation_error_magnitude': float(np.linalg.norm(orientation_error))
            })
        
        return orientation_info
    
    def _rotation_matrix_to_euler(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to Euler angles (roll, pitch, yaw).
        
        Args:
            R: 3x3 rotation matrix
            
        Returns:
            [roll, pitch, yaw] in radians
        """
        # ZYX Euler angles (yaw-pitch-roll)
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        
        singular = sy < 1e-6
        
        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0
        
        return np.array([roll, pitch, yaw])
    
    def get_complete_control_info(self, 
                                T_WH: np.ndarray, 
                                p_W_cup: np.ndarray,
                                target_position: Optional[np.ndarray] = None,
                                target_orientation: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Get complete control information for robot control.
        
        Args:
            T_WH: World→hand transform matrix
            p_W_cup: Cup position in world frame
            target_position: Target position in hand frame (optional)
            target_orientation: Target orientation matrix (optional)
            
        Returns:
            Dictionary containing all control information
        """
        # Calculate hand→world transform
        T_HW = self.calculate_hand_to_world_transform(T_WH)
        
        # Calculate cup position in hand frame
        p_H_cup = self.calculate_cup_position_in_hand_frame(p_W_cup, T_WH)
        
        # Calculate control errors
        control_errors = self.calculate_control_errors(p_H_cup, target_position)
        
        # Calculate grasp approach vector
        grasp_info = self.calculate_grasp_approach_vector(p_H_cup)
        
        # Calculate orientation errors
        R_WH = T_WH[:3, :3]
        orientation_info = self.calculate_orientation_error(R_WH, target_orientation)
        
        return {
            'hand_pose_world': {
                'position': T_WH[:3, 3].tolist(),
                'orientation_matrix': R_WH.tolist(),
                'orientation_euler': orientation_info['current_orientation_euler']
            },
            'cup_position': {
                'world_frame': p_W_cup.tolist(),
                'hand_frame': p_H_cup.tolist()
            },
            'control_errors': control_errors,
            'grasp_approach': grasp_info,
            'orientation_control': orientation_info,
            'transforms': {
                'T_WH': T_WH.tolist(),
                'T_HW': T_HW.tolist()
            }
        }
