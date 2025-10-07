"""
Move to Cup Action State

Implements the "MOVE to cup" action from the flowchart.
Uses vision system to locate cup, IMU for smooth movement, and piezo for contact detection.
"""

import numpy as np
from typing import Tuple, Optional, List
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
                return True, "WAIT_to_grab_cup"
            else:
                self.attempt_count += 1
                if self.attempt_count >= self.max_attempts:
                    print(f"Failed to move to cup after {self.max_attempts} attempts")
                    return False, "WAIT_for_move_to_cup"  # Reset to start
                else:
                    print(f"Retrying move to cup (attempt {self.attempt_count + 1})")
                    return False, None  # Stay in current state
                    
        except Exception as e:
            print(f"Error in MoveToCupState: {e}")
            self.post_execute(False)
            return False, "WAIT_for_move_to_cup"
    
    def _move_to_cup(self) -> bool:
        """Perform the actual movement to cup."""
        # Get cup position from vision
        cup_data = self.get_sensor_data('vision')
        if not cup_data or not cup_data.get('cup', {}).get('detected'):
            print("No cup detected by vision system")
            return False
        
        cup_position = cup_data['cup']['position_world']
        if cup_position is None:
            print("Cup position not available")
            return False
        
        print(f"Cup detected at position: {cup_position}")
        
        # Get current hand orientation from IMU
        imu_data = self.get_sensor_data('imu')
        if imu_data:
            # Apply IMU-based motion compensation for smooth movement
            target_position = self._apply_imu_compensation(cup_position, imu_data)
            print("Applied IMU motion compensation")
        else:
            target_position = cup_position
            print("No IMU data available, using direct cup position")
        
        # Move robot arm towards cup
        print(f"Moving robot arm to position: {target_position}")
        movement_success = self.robot_arm.move_to_position(target_position)
        
        if not movement_success:
            print("Robot arm movement failed")
            return False
        
        # Check for contact using piezo sensor
        piezo_data = self.get_sensor_data('piezo')
        if piezo_data and piezo_data.get('contact_detected'):
            print("Contact detected by piezo sensor - approaching too close")
            # Back off slightly
            backup_position = self._calculate_backup_position(target_position, cup_position)
            self.robot_arm.move_to_position(backup_position)
            return True
        else:
            print("No contact detected, movement completed")
            return True
    
    def _apply_imu_compensation(self, target_pos: List[float], imu_data) -> List[float]:
        """Apply IMU-based motion compensation for smooth movement."""
        try:
            # Get angular velocity from IMU
            angular_velocity = imu_data.angular_velocity
            
            # Calculate compensation based on angular velocity
            # Higher angular velocity = more compensation needed
            compensation_factor = 0.01  # Tune this parameter
            compensation = angular_velocity * compensation_factor
            
            # Apply compensation to target position
            compensated_position = [
                target_pos[0] + compensation[0],
                target_pos[1] + compensation[1], 
                target_pos[2] + compensation[2]
            ]
            
            return compensated_position
            
        except Exception as e:
            print(f"Error applying IMU compensation: {e}")
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
