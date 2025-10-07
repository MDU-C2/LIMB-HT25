"""
Move Back Hand Action State

Implements the "MOVE back hand" action from the flowchart.
Uses vision system, IMU, and piezo sensor for safe hand retraction.
"""

from typing import Tuple, Optional, List
from .base_state import BaseState

class MoveBackHandState(BaseState):
    """Move robot hand back to safe position."""
    
    def execute(self) -> Tuple[bool, Optional[str]]:
        """Execute move back hand action."""
        self.pre_execute()
        
        try:
            print("Moving hand back...")
            success = self._move_back_hand()
            self.post_execute(success)
            
            if success:
                return True, "WAIT_for_move_to_cup"  # Loop back for continuous operation
            else:
                return False, "WAIT_for_move_back_hand"
                
        except Exception as e:
            print(f"Error in MoveBackHandState: {e}")
            self.post_execute(False)
            return False, "WAIT_for_move_back_hand"
    
    def _move_back_hand(self) -> bool:
        """Perform the actual hand movement back."""
        # TODO: Implement actual movement logic
        return True
    
    def get_required_sensors(self) -> List[str]:
        """Get sensors required for this action state."""
        return ['vision', 'imu', 'piezo']
