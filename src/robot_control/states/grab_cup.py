"""
Grab Cup Action State

Implements the "GRAB cup" action from the flowchart.
Uses pressure and slip sensors for grip feedback during gripping phase.
LSTM already determined we should "grip" (gatekeeper decision).
"""

from typing import Tuple, Optional, List, Dict, Any
import time
from .base_state import BaseState

class GrabCupState(BaseState):
    """Grab cup using pressure and slip sensors for grip feedback."""
    
    def __init__(self, sensor_manager, robot_arm):
        super().__init__(sensor_manager, robot_arm)
        self.max_grip_force = 5.0  # Maximum grip force
        self.min_grip_force = 0.5  # Minimum grip force
        self.grip_timeout = 10.0   # Maximum time to achieve grip
        self.grip_start_time = None
    
    def execute(self) -> Tuple[bool, Optional[str]]:
        """Execute grab cup action with pressure and slip feedback."""
        self.pre_execute()
        
        try:
            print("Executing grab cup with pressure/slip feedback...")
            
            # LSTM already determined we should "grip" (gatekeeper decision)
            # Now use pressure and slip sensors to ensure adequate grip
            
            success = self._grab_cup_with_feedback()
            self.post_execute(success)
            
            if success:
                # Let LSTM determine next state
                return True, None
            else:
                return False, None  # Stay in current state for retry
                
        except Exception as e:
            print(f"Error in GrabCupState: {e}")
            self.post_execute(False)
            return False, None
    
    def _grab_cup_with_feedback(self) -> bool:
        """Grab cup using pressure and slip sensor feedback."""
        print("Starting grip process with sensor feedback...")
        self.grip_start_time = time.time()
        
        # Start gripping
        self.robot_arm.start_gripping()
        
        # Monitor grip with pressure and slip sensors
        while not self._is_grip_adequate():
            # Check for timeout
            if self.is_timeout():
                print("Grip timeout - failed to achieve adequate grip")
                return False
            
            # Get grip feedback from sensors
            feedback = self._get_grip_feedback()
            
            # Adjust grip force based on sensor feedback
            if feedback['slip_detected']:
                print("Slip detected - increasing grip force")
                self.robot_arm.increase_grip_force()
            elif feedback['pressure_force'] and feedback['pressure_force'] > self.max_grip_force:
                print("Excessive pressure - decreasing grip force")
                self.robot_arm.decrease_grip_force()
            elif feedback['pressure_force'] and feedback['pressure_force'] < self.min_grip_force:
                print("Insufficient pressure - increasing grip force")
                self.robot_arm.increase_grip_force()
            
            # Small delay to allow sensor updates
            time.sleep(0.1)
        
        print("Adequate grip achieved!")
        return True
    
    def _get_grip_feedback(self) -> Dict[str, Any]:
        """Get grip feedback from Pressure and Slip sensors."""
        feedback = {
            'pressure_force': None,
            'slip_detected': False,
            'grip_adequate': False
        }
        
        try:
            # Get pressure sensor data
            pressure_data = self.get_sensor_data('pressure')
            if pressure_data:
                feedback['pressure_force'] = pressure_data.get('force', 0.0)
                print(f"Pressure feedback: force={feedback['pressure_force']}")
            
            # Get slip sensor data
            slip_data = self.get_sensor_data('slip')
            if slip_data:
                feedback['slip_detected'] = slip_data.get('slipping_detected', False)
                print(f"Slip feedback: slipping={feedback['slip_detected']}")
            
            # Determine if grip is adequate
            feedback['grip_adequate'] = self._evaluate_grip_adequacy(feedback)
            
        except Exception as e:
            print(f"Error getting grip feedback: {e}")
        
        return feedback
    
    def _evaluate_grip_adequacy(self, feedback: Dict[str, Any]) -> bool:
        """Evaluate if the current grip is adequate based on sensor feedback."""
        # Check if we have sufficient pressure
        has_sufficient_pressure = (
            feedback.get('pressure_force', 0) >= self.min_grip_force and
            feedback.get('pressure_force', 0) <= self.max_grip_force
        )
        
        # Check if object is not slipping
        no_slip = not feedback.get('slip_detected', True)
        
        # Grip is adequate if we have proper pressure and no slip
        adequate = has_sufficient_pressure and no_slip
        
        if adequate:
            print("Grip feedback: Adequate grip detected")
        else:
            print(f"Grip feedback: Inadequate grip (pressure: {has_sufficient_pressure}, no_slip: {no_slip})")
        
        return adequate
    
    def _is_grip_adequate(self) -> bool:
        """Check if current grip is adequate using sensor feedback."""
        feedback = self._get_grip_feedback()
        return feedback['grip_adequate']
    
    def get_required_sensors(self) -> List[str]:
        """Get sensors required for this action state."""
        return ['pressure', 'slip']
