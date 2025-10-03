"""Main fusion system that orchestrates all components.

This is the main interface for the fiducial + depth system that coordinates:
- Hand pose estimation from ArUco
- Cup 3D position from YOLO + depth  
- Relative pose calculation for control
- Optional IMU smoothing
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np
import time

from .calibration import CalibrationManager
from .hand_pose import HandPoseEstimator
from .cup_3d import Cup3DEstimator
from .relative_pose import RelativePoseCalculator
from .smoothing import IMUSmoother, IMUData, IMUValidator


class FiducialDepthSystem:
    """Main fusion system for fiducial + depth approach."""
    
    def __init__(self, 
                 camera_matrix: np.ndarray,
                 calibration_file: Optional[str] = None,
                 enable_imu_smoothing: bool = False,
                 enable_imu_validation: bool = True,
                 smoothing_method: str = "complementary",
                 validation_thresholds: Optional[Dict[str, float]] = None):
        """Initialize the fiducial depth system.
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix
            calibration_file: Path to calibration file (optional)
            enable_imu_smoothing: Whether to enable IMU smoothing
            enable_imu_validation: Whether to enable IMU validation
            smoothing_method: IMU smoothing method ("complementary", "ekf", "hold")
            validation_thresholds: Custom validation thresholds
        """
        self.camera_matrix = camera_matrix
        self.enable_imu_smoothing = enable_imu_smoothing
        self.enable_imu_validation = enable_imu_validation
        
        # Initialize components
        self.calibration_manager = CalibrationManager(calibration_file)
        self.hand_pose_estimator = HandPoseEstimator(self.calibration_manager)
        self.cup_3d_estimator = Cup3DEstimator(self.calibration_manager, camera_matrix)
        self.relative_pose_calculator = RelativePoseCalculator(self.calibration_manager)
        
        # Initialize IMU smoother if enabled
        self.imu_smoother = None
        if enable_imu_smoothing:
            self.imu_smoother = IMUSmoother(smoothing_method)
        
        # Initialize IMU validator if enabled
        self.imu_validator = None
        if enable_imu_validation:
            self.imu_validator = IMUValidator(validation_thresholds)
        
        # State tracking
        self.last_hand_pose = None
        self.last_cup_position = None
        self.last_update_time = None
        
        print(f"FiducialDepthSystem initialized with IMU smoothing: {enable_imu_smoothing}, IMU validation: {enable_imu_validation}")
    
    def process_frame(self, 
                     tag_detection_result: Optional[Dict[str, Any]] = None,
                     cup_detection_result: Optional[Dict[str, Any]] = None,
                     depth_image: Optional[np.ndarray] = None,
                     imu_data: Optional[IMUData] = None) -> Dict[str, Any]:
        """Process a single frame with tag and cup detection results.
        
        Args:
            tag_detection_result: Result from tag detector
            cup_detection_result: Result from cup detector
            depth_image: Depth image (optional)
            imu_data: IMU data (optional, for smoothing)
            
        Returns:
            Dictionary containing all processing results
        """
        current_time = time.time()
        
        # 1. Estimate hand pose from ArUco detection
        hand_pose_result = self._process_hand_pose(tag_detection_result, current_time, imu_data)
        
        # 2. Estimate cup 3D position
        cup_3d_result = self._process_cup_3d(cup_detection_result, depth_image)
        
        # 3. Calculate relative poses and control information
        control_result = self._process_control_info(hand_pose_result, cup_3d_result)
        
        # 4. Update state
        self.last_update_time = current_time
        
        # Compile results
        result = {
            'timestamp': current_time,
            'hand_pose': hand_pose_result,
            'cup_3d': cup_3d_result,
            'control': control_result,
            'system_info': self._get_system_info()
        }
        
        return result
    
    def _process_hand_pose(self, 
                          tag_detection_result: Optional[Dict[str, Any]], 
                          current_time: float,
                          imu_data: Optional[IMUData]) -> Dict[str, Any]:
        """Process hand pose estimation with IMU validation."""
        result = {
            'detected': False,
            'T_WH': None,
            'position': None,
            'orientation': None,
            'source': None,
            'validation': None
        }
        
        # Try to get pose from ArUco detection
        T_WH_vision = self.hand_pose_estimator.estimate_hand_pose(tag_detection_result)
        
        if T_WH_vision is not None:
            # Vision-based pose available
            validation_result = None
            
            # Validate with IMU if available
            if self.imu_validator is not None and imu_data is not None:
                validation_result = self.imu_validator.validate_pose(T_WH_vision, imu_data, current_time)
                
                # Log validation issues
                if not validation_result['is_valid']:
                    print(f"⚠️  IMU Validation Failed: {validation_result['issues']}")
                elif validation_result['confidence'] < 0.8:
                    print(f"⚠️  Low IMU Validation Confidence: {validation_result['confidence']:.2f}")
            
            result.update({
                'detected': True,
                'T_WH': T_WH_vision,
                'position': T_WH_vision[:3, 3].tolist(),
                'orientation': T_WH_vision[:3, :3].tolist(),
                'source': 'vision',
                'validation': validation_result
            })
            
            # Update IMU smoother with vision correction (if smoothing enabled)
            if self.imu_smoother is not None:
                self.imu_smoother.update_with_vision(T_WH_vision, current_time)
            
            self.last_hand_pose = T_WH_vision
            
        elif self.imu_smoother is not None and imu_data is not None:
            # Try IMU prediction when vision is not available
            dt = current_time - self.last_update_time if self.last_update_time else 0.0
            T_WH_imu = self.imu_smoother.predict_with_imu(imu_data, dt)
            
            if T_WH_imu is not None:
                result.update({
                    'detected': True,
                    'T_WH': T_WH_imu,
                    'position': T_WH_imu[:3, 3].tolist(),
                    'orientation': T_WH_imu[:3, :3].tolist(),
                    'source': 'imu_prediction',
                    'validation': None  # IMU prediction doesn't need validation
                })
                
                self.last_hand_pose = T_WH_imu
        
        return result
    
    def _process_cup_3d(self, 
                       cup_detection_result: Optional[Dict[str, Any]], 
                       depth_image: Optional[np.ndarray]) -> Dict[str, Any]:
        """Process cup 3D position estimation."""
        result = {
            'detected': False,
            'position_world': None,
            'position_camera': None,
            'distance': None
        }
        
        # Estimate cup 3D position
        p_W_cup = self.cup_3d_estimator.estimate_cup_3d(cup_detection_result, depth_image)
        
        if p_W_cup is not None:
            # Get camera frame position for reference
            p_C_cup = self.cup_3d_estimator.get_cup_position_camera_frame(cup_detection_result, depth_image)
            
            result.update({
                'detected': True,
                'position_world': p_W_cup.tolist(),
                'position_camera': p_C_cup.tolist() if p_C_cup is not None else None,
                'distance': float(np.linalg.norm(p_W_cup))
            })
            
            self.last_cup_position = p_W_cup
        
        return result
    
    def _process_control_info(self, 
                            hand_pose_result: Dict[str, Any], 
                            cup_3d_result: Dict[str, Any]) -> Dict[str, Any]:
        """Process control information and relative poses."""
        result = {
            'available': False,
            'hand_to_cup_vector': None,
            'control_errors': None,
            'grasp_approach': None
        }
        
        # Check if both hand pose and cup position are available
        if (hand_pose_result['detected'] and 
            cup_3d_result['detected'] and 
            hand_pose_result['T_WH'] is not None and 
            cup_3d_result['position_world'] is not None):
            
            T_WH = np.array(hand_pose_result['T_WH'])
            p_W_cup = np.array(cup_3d_result['position_world'])
            
            # Calculate control information
            control_info = self.relative_pose_calculator.get_complete_control_info(T_WH, p_W_cup)
            
            result.update({
                'available': True,
                'hand_to_cup_vector': control_info['cup_position']['hand_frame'],
                'control_errors': control_info['control_errors'],
                'grasp_approach': control_info['grasp_approach'],
                'complete_info': control_info
            })
        
        return result
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for debugging."""
        info = {
            'calibrated': self.calibration_manager.is_calibrated(),
            'imu_smoothing_enabled': self.enable_imu_smoothing,
            'imu_validation_enabled': self.enable_imu_validation,
            'last_update_time': self.last_update_time
        }
        
        if self.imu_smoother is not None:
            info['imu_smoothing'] = self.imu_smoother.get_smoothing_info()
        
        if self.imu_validator is not None:
            info['imu_validation'] = self.imu_validator.get_validation_info()
        
        info['calibration'] = self.calibration_manager.get_calibration_info()
        
        return info
    
    def set_calibration(self, T_WC: np.ndarray, T_CH_fixed: np.ndarray) -> None:
        """Set calibration transforms.
        
        Args:
            T_WC: Camera→world transform
            T_CH_fixed: Camera→hand tag transform
        """
        self.calibration_manager.set_camera_to_world_transform(T_WC)
        self.calibration_manager.set_hand_tag_transform(T_CH_fixed)
        self.calibration_manager.save_calibration()
    
    def get_control_command(self, 
                          target_position: Optional[np.ndarray] = None,
                          target_orientation: Optional[np.ndarray] = None) -> Optional[Dict[str, Any]]:
        """Get control command for robot control.
        
        Args:
            target_position: Target position in hand frame (optional)
            target_orientation: Target orientation matrix (optional)
            
        Returns:
            Control command dictionary or None if not available
        """
        if (self.last_hand_pose is None or 
            self.last_cup_position is None):
            return None
        
        # Get complete control information
        control_info = self.relative_pose_calculator.get_complete_control_info(
            self.last_hand_pose, 
            self.last_cup_position,
            target_position,
            target_orientation
        )
        
        # Extract control command
        command = {
            'hand_pose': control_info['hand_pose_world'],
            'cup_position': control_info['cup_position'],
            'position_error': control_info['control_errors'],
            'approach_vector': control_info['grasp_approach']['approach_vector'],
            'target_position': control_info['grasp_approach']['target_position'],
            'distance_to_cup': control_info['control_errors']['distance_to_cup']
        }
        
        return command
    
    def reset_system(self) -> None:
        """Reset system state."""
        self.last_hand_pose = None
        self.last_cup_position = None
        self.last_update_time = None
        
        if self.imu_smoother is not None:
            self.imu_smoother.is_tracking = False
            self.imu_smoother.last_vision_pose = None
            self.imu_smoother.last_vision_timestamp = None
        
        print("System state reset")
