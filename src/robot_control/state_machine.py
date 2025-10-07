"""
State Machine for Robotic Manipulation

Implements the state machine for cup manipulation based on the flowchart.
Manages transitions between waiting states and action states.
"""

from typing import Dict, List, Optional, Tuple, Any
import time
import threading

class StateMachine:
    """Main state machine for robotic manipulation tasks."""
    
    def __init__(self, sensor_manager, robot_arm):
        """
        Initialize state machine.
        
        Args:
            sensor_manager: SensorManager instance
            robot_arm: RobotArm instance
        """
        self.sensor_manager = sensor_manager
        self.robot_arm = robot_arm
        
        # State definitions
        self.waiting_states = [
            "WAIT_for_move_to_cup",
            "WAIT_to_grab_cup", 
            "WAIT_to_lift_cup",
            "WAIT_to_move_cup",
            "WAIT_to_place_down_cup",
            "WAIT_for_release_cup",
            "WAIT_for_move_back_hand"
        ]
        
        self.action_states = [
            "MOVE_to_cup",
            "GRAB_cup",
            "LIFT_cup", 
            "MOVE_cup_ab",
            "PLACE_DOWN_cup",
            "RELEASE_cup",
            "MOVE_back_hand"
        ]
        
        # State transitions
        self.state_transitions = {
            "WAIT_for_move_to_cup": "MOVE_to_cup",
            "WAIT_to_grab_cup": "GRAB_cup",
            "WAIT_to_lift_cup": "LIFT_cup", 
            "WAIT_to_move_cup": "MOVE_cup_ab",
            "WAIT_to_place_down_cup": "PLACE_DOWN_cup",
            "WAIT_for_release_cup": "RELEASE_cup",
            "WAIT_for_move_back_hand": "MOVE_back_hand"
        }
        
        # Initialize state machine
        self.current_state = "WAIT_for_move_to_cup"
        self.running = False
        self.state_history = []
        self.start_time = None
        
        # Import action states
        self._initialize_action_states()
        
        print("StateMachine initialized")
        print(f"  Initial state: {self.current_state}")
    
    def _initialize_action_states(self):
        """Initialize action state classes."""
        try:
            from .states.move_to_cup import MoveToCupState
            from .states.grab_cup import GrabCupState
            from .states.lift_cup import LiftCupState
            from .states.move_cup_ab import MoveCupABState
            from .states.place_down_cup import PlaceDownCupState
            from .states.release_cup import ReleaseCupState
            from .states.move_back_hand import MoveBackHandState
            
            self.action_state_classes = {
                "MOVE_to_cup": MoveToCupState,
                "GRAB_cup": GrabCupState,
                "LIFT_cup": LiftCupState,
                "MOVE_cup_ab": MoveCupABState,
                "PLACE_DOWN_cup": PlaceDownCupState,
                "RELEASE_cup": ReleaseCupState,
                "MOVE_back_hand": MoveBackHandState
            }
            print("✓ Action states initialized")
            
        except ImportError as e:
            print(f"Warning: Could not import action states: {e}")
            self.action_state_classes = {}
    
    def run(self):
        """Run the state machine."""
        self.running = True
        self.start_time = time.time()
        
        print(f"Starting state machine execution...")
        print(f"Initial state: {self.current_state}")
        
        try:
            while self.running and self.current_state != "END":
                self._execute_current_state()
                time.sleep(0.01)  # Small delay to prevent busy waiting
                
        except KeyboardInterrupt:
            print("\nState machine interrupted by user")
        except Exception as e:
            print(f"Error in state machine: {e}")
        finally:
            self.running = False
            print("State machine stopped")
    
    def _execute_current_state(self):
        """Execute the current state."""
        if self.current_state.startswith("WAIT_"):
            self._execute_waiting_state()
        else:
            self._execute_action_state()
    
    def _execute_waiting_state(self):
        """Execute waiting state - wait for intention/trigger."""
        # For now, immediately transition to action state
        # In a real system, this would wait for external triggers
        
        next_action = self.state_transitions.get(self.current_state)
        if next_action:
            print(f"Transitioning from {self.current_state} to {next_action}")
            self._transition_to(next_action)
        else:
            print(f"Warning: No transition defined for {self.current_state}")
            self.running = False
    
    def _execute_action_state(self):
        """Execute action state."""
        print(f"Executing action state: {self.current_state}")
        
        # Activate required sensors for this state
        success = self.sensor_manager.activate_sensors_for_state(self.current_state)
        if not success:
            print(f"Warning: Failed to activate all sensors for {self.current_state}")
        
        # Execute the action
        action_success, next_state = self._run_action_state()
        
        # Transition based on result
        if action_success:
            print(f"{self.current_state} completed successfully")
            if next_state:
                self._transition_to(next_state)
            else:
                # Determine next state based on current state
                next_state = self._get_next_state()
                self._transition_to(next_state)
        else:
            print(f"{self.current_state} failed, retrying...")
            # Stay in current state or go back to waiting state
            waiting_state = f"WAIT_{self.current_state.lower()}"
            if waiting_state in self.waiting_states:
                self._transition_to(waiting_state)
    
    def _run_action_state(self) -> Tuple[bool, Optional[str]]:
        """Run the specific action state."""
        if self.current_state not in self.action_state_classes:
            print(f"Error: No action class defined for {self.current_state}")
            return False, None
        
        try:
            # Create action state instance
            action_class = self.action_state_classes[self.current_state]
            action_instance = action_class(self.sensor_manager, self.robot_arm)
            
            # Execute action
            return action_instance.execute()
            
        except Exception as e:
            print(f"Error executing {self.current_state}: {e}")
            return False, None
    
    def _get_next_state(self) -> str:
        """Determine next state based on current state."""
        next_state_map = {
            "MOVE_to_cup": "WAIT_to_grab_cup",
            "GRAB_cup": "WAIT_to_lift_cup",
            "LIFT_cup": "WAIT_to_move_cup",
            "MOVE_cup_ab": "WAIT_to_place_down_cup", 
            "PLACE_DOWN_cup": "WAIT_for_release_cup",
            "RELEASE_cup": "WAIT_for_move_back_hand",
            "MOVE_back_hand": "WAIT_for_move_to_cup"  # Loop back for continuous operation
        }
        
        return next_state_map.get(self.current_state, "END")
    
    def _transition_to(self, new_state: str):
        """Transition to a new state."""
        if new_state not in self.waiting_states and new_state not in self.action_states and new_state != "END":
            print(f"Warning: Unknown state '{new_state}'")
            return
        
        # Record state transition
        self.state_history.append({
            'from_state': self.current_state,
            'to_state': new_state,
            'timestamp': time.time(),
            'elapsed_time': time.time() - (self.start_time or time.time())
        })
        
        old_state = self.current_state
        self.current_state = new_state
        
        print(f"State transition: {old_state} → {new_state}")
    
    def get_current_state(self) -> str:
        """Get current state."""
        return self.current_state
    
    def get_state_history(self) -> List[Dict[str, Any]]:
        """Get state transition history."""
        return self.state_history
    
    def get_status(self) -> Dict[str, Any]:
        """Get state machine status."""
        return {
            'current_state': self.current_state,
            'running': self.running,
            'start_time': self.start_time,
            'elapsed_time': time.time() - (self.start_time or time.time()),
            'state_count': len(self.state_history),
            'available_states': {
                'waiting': self.waiting_states,
                'action': self.action_states
            }
        }
    
    def stop(self):
        """Stop the state machine."""
        self.running = False
        print("State machine stop requested")
    
    def reset(self):
        """Reset state machine to initial state."""
        self.current_state = "WAIT_for_move_to_cup"
        self.state_history = []
        self.start_time = None
        print("State machine reset to initial state")
    
    def force_transition(self, new_state: str):
        """Force transition to a specific state (for testing/debugging)."""
        print(f"Forcing transition to {new_state}")
        self._transition_to(new_state)
