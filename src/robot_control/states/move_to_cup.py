"""
Move to Cup Action State

Implements the "MOVE to cup" action from the flowchart.
Uses vision system to locate cup, IMU for smooth movement, and piezo for contact detection.
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from .base_state import BaseState

class MoveToCupState(BaseState):
    """Move robot hand to cup position using vision, IMU, and piezo sensors."""
    
    def __init__(self, sensor_manager, robot_arm):
        super().__init__(sensor_manager, robot_arm)
        self.approach_distance = 0.05  # 5cm approach distance
        self.max_attempts = 3
        self.attempt_count = 0
    
    def execute(self) -> Tuple[bool, Optional[str]]:
        """Execute move to cup action."""
        self.pre_execute()
        
        try:
            success = self._move_to_cup()
            self.post_execute(success)
            
            if success:
                # Let LSTM determine next state, don't specify next state
                return True, None
            else:
                self.attempt_count += 1
                if self.attempt_count >= self.max_attempts:
                    print(f"Failed to move to cup after {self.max_attempts} attempts")
                    return False, None  # Stay in current state for retry
                else:
                    print(f"Retrying move to cup (attempt {self.attempt_count + 1})")
                    return False, None  # Stay in current state
                    
        except Exception as e:
            print(f"Error in MoveToCupState: {e}")
            self.post_execute(False)
            return False, None  # Stay in current state for retry
    
    def _move_to_cup(self) -> bool:
        """
        Move to cup using IMU direction/speed + vision target.
        LSTM already determined we should "move" (gatekeeper decision).
        Now use IMU + Vision to determine HOW to move.
        """
        print("Executing move to cup with IMU/Vision guidance...")
        
        # Get execution guidance from IMU and Vision systems
        guidance = self._get_execution_guidance()
        
        # Check if we have vision target
        if not guidance['vision_detected'] or not guidance['vision_target']:
            print("No vision target available for movement")
            return False
        
        target_position = guidance['vision_target']
        print(f"Vision target position: {target_position}")
        
        # Get IMU guidance for movement direction only
        if guidance['imu_direction'] is not None:
            print(f"Using IMU direction guidance: {guidance['imu_direction']}")
            print(f"Using hard-coded speed: {guidance['hardcoded_speed']}")
            
            # Apply IMU-based movement guidance (direction only)
            target_position = self._apply_imu_direction_guidance(target_position, guidance)
        else:
            print("No IMU direction available, using direct vision target")
        
        # Execute movement with IMU direction and hard-coded speed
        print(f"Moving robot arm to position: {target_position}")
        movement_success = self.robot_arm.move_with_direction_guidance(
            target_position,
            direction=guidance['imu_direction'],
            speed=guidance['hardcoded_speed']
        )
        
        if not movement_success:
            print("Robot arm movement failed")
            return False
        
        # Check for contact using piezo sensor
        piezo_data = self.get_sensor_data('piezo')
        if piezo_data and piezo_data.get('contact_detected'):
            print("Contact detected by piezo sensor - approaching too close")
            # Back off slightly
            backup_position = self._calculate_backup_position(target_position, guidance['vision_target'])
            self.robot_arm.move_to_position(backup_position)
            return True
        else:
            print("No contact detected, movement completed")
            return True
    
    def _get_execution_guidance(self) -> Dict[str, Any]:
        """Get execution guidance from IMU and Vision systems (direction only, speed hard-coded)."""
        guidance = {
            'imu_direction': None,
            'hardcoded_speed': 0.1,  # Hard-coded movement speed
            'vision_target': None,
            'vision_detected': False
        }
        
        try:
            # Get IMU data for movement direction only
            imu_data = self.get_sensor_data('imu')
            if imu_data and hasattr(imu_data, 'angular_velocity'):
                guidance['imu_direction'] = imu_data.angular_velocity
                print(f"IMU guidance: direction={guidance['imu_direction']}")
                print(f"Using hard-coded speed: {guidance['hardcoded_speed']}")
            
            # Get Vision data for target position
            vision_data = self.get_sensor_data('vision')
            if vision_data and vision_data.get('cup', {}).get('detected'):
                guidance['vision_target'] = vision_data['cup']['position_world']
                guidance['vision_detected'] = True
                print(f"Vision guidance: target={guidance['vision_target']}")
            else:
                print("Vision guidance: No target detected")
                
        except Exception as e:
            print(f"Error getting execution guidance: {e}")
        
        return guidance
    
    def _apply_imu_direction_guidance(self, target_pos: List[float], guidance: Dict[str, Any]) -> List[float]:
        """Apply IMU direction guidance to target position (speed is hard-coded)."""
        try:
            if guidance['imu_direction'] is None:
                return target_pos
            
            # Apply IMU direction only (speed is hard-coded)
            imu_direction = np.array(guidance['imu_direction'])
            hardcoded_speed = guidance['hardcoded_speed']
            
            # Scale the direction by hard-coded speed and apply to target
            movement_offset = imu_direction * hardcoded_speed * 0.01  # Scale factor
            
            guided_position = [
                target_pos[0] + movement_offset[0],
                target_pos[1] + movement_offset[1], 
                target_pos[2] + movement_offset[2]
            ]
            
            print(f"Applied IMU direction guidance: {target_pos} -> {guided_position}")
            print(f"Direction: {imu_direction}, Hard-coded speed: {hardcoded_speed}")
            return guided_position
            
        except Exception as e:
            print(f"Error applying IMU direction guidance: {e}")
            return target_pos
    
    def _calculate_backup_position(self, current_pos: List[float], cup_pos: List[float]) -> List[float]:
        """Calculate backup position to avoid contact."""
        # Move back along the approach vector
        approach_vector = np.array(cup_pos) - np.array(current_pos)
        approach_vector = approach_vector / np.linalg.norm(approach_vector)
        
        backup_distance = self.approach_distance
        backup_position = np.array(current_pos) - approach_vector * backup_distance
        
        return backup_position.tolist()
    
    def get_required_sensors(self) -> List[str]:
        """Get sensors required for this action state."""
        return ['vision', 'imu', 'piezo']
    
    def get_status(self) -> dict:
        """Get current status of this action state."""
        return {
            'attempt_count': self.attempt_count,
            'max_attempts': self.max_attempts,
            'approach_distance': self.approach_distance,
            'execution_time': self.get_execution_time()
        }
