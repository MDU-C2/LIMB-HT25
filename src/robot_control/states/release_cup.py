"""
Release Cup Action State

Implements the "RELEASE cup" action from the flowchart.
Uses vision system, pressure sensor, and slip sensor for controlled release.
"""

from typing import Tuple, Optional, List
from .base_state import BaseState

class ReleaseCupState(BaseState):
    """Release cup from gripper."""
    
    def execute(self) -> Tuple[bool, Optional[str]]:
        """Execute release cup action."""
        self.pre_execute()
        
        try:
            print("Releasing cup...")
            success = self._release_cup()
            self.post_execute(success)
            
            if success:
                return True, "WAIT_for_move_back_hand"
            else:
                return False, "WAIT_for_release_cup"
                
        except Exception as e:
            print(f"Error in ReleaseCupState: {e}")
            self.post_execute(False)
            return False, "WAIT_for_release_cup"
    
    def _release_cup(self) -> bool:
        """Perform the actual cup release."""
        # TODO: Implement actual release logic
        return True
    
    def get_required_sensors(self) -> List[str]:
        """Get sensors required for this action state."""
        return ['vision', 'pressure', 'slip']
