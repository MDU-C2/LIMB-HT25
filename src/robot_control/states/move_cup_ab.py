"""
Move Cup A->B Action State

Implements the "MOVE cup A->B" action from the flowchart.
Uses vision system, pressure sensor, slip sensor, and IMU for stable cup transport.
"""

from typing import Tuple, Optional, List
from .base_state import BaseState

class MoveCupABState(BaseState):
    """Move cup from position A to position B."""
    
    def execute(self) -> Tuple[bool, Optional[str]]:
        """Execute move cup A->B action."""
        self.pre_execute()
        
        try:
            print("Moving cup from A to B...")
            success = self._move_cup_ab()
            self.post_execute(success)
            
            if success:
                return True, "WAIT_to_place_down_cup"
            else:
                return False, "WAIT_to_move_cup"
                
        except Exception as e:
            print(f"Error in MoveCupABState: {e}")
            self.post_execute(False)
            return False, "WAIT_to_move_cup"
    
    def _move_cup_ab(self) -> bool:
        """Perform the actual cup movement."""
        # TODO: Implement actual movement logic
        return True
    
    def get_required_sensors(self) -> List[str]:
        """Get sensors required for this action state."""
        return ['vision', 'pressure', 'slip', 'imu']
