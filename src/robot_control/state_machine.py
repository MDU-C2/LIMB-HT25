"""
State Machine for Robotic Manipulation

Implements the state machine for cup manipulation based on the flowchart.
Manages transitions between action states based on LSTM classified intentions.
"""

from typing import Dict, List, Optional, Tuple, Any
import time
import threading
import numpy as np

class StateMachine:
    """Main state machine for robotic manipulation tasks."""
    
    def __init__(self, sensor_manager, robot_arm, lstm_classifier=None):
        """
        Initialize state machine.
        
        Args:
            sensor_manager: SensorManager instance
            robot_arm: RobotArm instance
            lstm_classifier: LSTM classifier for intention recognition (optional)
        """
        self.sensor_manager = sensor_manager
        self.robot_arm = robot_arm
        self.lstm_classifier = lstm_classifier
        
        # Action states only (no waiting states)
        self.action_states = [
            "MOVE_to_cup",
            "GRAB_cup",
            "LIFT_cup", 
            "MOVE_cup_ab",
            "PLACE_DOWN_cup",
            "RELEASE_cup",
            "MOVE_back_hand"
        ]
        
        # LSTM intention classifications
        self.intention_classes = [
            "rest",
            "grip", 
            "move"
        ]
        
        # Map LSTM intentions to action states
        # This mapping determines which state to transition to based on intention
        self.intention_to_state = {
            "rest": "MOVE_back_hand",  # Rest intention -> move hand back
            "grip": "GRAB_cup",         # Grip intention -> grab cup
            "move": "MOVE_to_cup"       # Move intention -> move to cup
        }
        
        # Initialize state machine
        self.current_state = "MOVE_to_cup"  # Start with first action
        self.running = False
        self.state_history = []
        self.start_time = None
        self.last_intention = None
        
        # Import action states
        self._initialize_action_states()
        
        print("StateMachine initialized")
        print(f"  Initial state: {self.current_state}")
        print(f"  LSTM classifier: {'Available' if self.lstm_classifier else 'Not provided'}")
    
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
        print(f"Executing action state: {self.current_state}")
        
        # Activate required sensors for this state
        success = self.sensor_manager.activate_sensors_for_state(self.current_state)
        if not success:
            print(f"Warning: Failed to activate all sensors for {self.current_state}")
        
        # Execute the action
        action_success, next_state = self._run_action_state()
        
        # Transition based on result and LSTM classification
        if action_success:
            print(f"{self.current_state} completed successfully")
            
            # Get next state from LSTM classification or action state
            if next_state:
                self._transition_to(next_state)
            else:
                # Use LSTM to determine next state
                next_state = self._get_next_state_from_lstm()
                self._transition_to(next_state)
        else:
            print(f"{self.current_state} failed, retrying...")
            # Stay in current state for retry
            # Could also implement fallback logic here
    
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
    
    def _get_next_state_from_lstm(self) -> str:
        """
        LSTM acts as gatekeeper - determines WHAT the robot should do.
        Returns the high-level state based on classified intention.
        """
        if not self.lstm_classifier:
            print("Warning: No LSTM classifier available, using fallback logic")
            return self._get_fallback_next_state()
        
        try:
            # Get sensor data for LSTM classification
            sensor_data = self.sensor_manager.get_all_active_data()
            
            # LSTM classifies the high-level intention (gatekeeper role)
            intention = self.lstm_classifier.classify_intention(sensor_data)
            self.last_intention = intention
            
            print(f"LSTM gatekeeper classified intention: {intention}")
            print(f"  -> Determining high-level action: {self.intention_to_state.get(intention, 'END')}")
            
            # Map intention to state (WHAT to do)
            next_state = self.intention_to_state.get(intention, "END")
            
            if next_state == "END":
                print("LSTM gatekeeper determined end sequence")
                return "END"
            
            return next_state
            
        except Exception as e:
            print(f"Error in LSTM gatekeeper classification: {e}")
            return self._get_fallback_next_state()
    
    def _get_fallback_next_state(self) -> str:
        """Fallback logic when LSTM is not available."""
        # Simple sequential fallback when LSTM is not available
        fallback_map = {
            "MOVE_to_cup": "GRAB_cup",
            "GRAB_cup": "LIFT_cup",
            "LIFT_cup": "MOVE_cup_ab",
            "MOVE_cup_ab": "PLACE_DOWN_cup", 
            "PLACE_DOWN_cup": "RELEASE_cup",
            "RELEASE_cup": "MOVE_back_hand",
            "MOVE_back_hand": "MOVE_to_cup"  # Loop back for continuous operation
        }
        
        return fallback_map.get(self.current_state, "END")
    
    def _transition_to(self, new_state: str):
        """Transition to a new state."""
        if new_state not in self.action_states and new_state != "END":
            print(f"Warning: Unknown state '{new_state}'")
            return
        
        # Record state transition
        self.state_history.append({
            'from_state': self.current_state,
            'to_state': new_state,
            'timestamp': time.time(),
            'elapsed_time': time.time() - (self.start_time or time.time()),
            'lstm_intention': self.last_intention
        })
        
        old_state = self.current_state
        self.current_state = new_state
        
        print(f"State transition: {old_state} → {new_state}")
        if self.last_intention:
            print(f"  LSTM intention: {self.last_intention}")
    
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
            'last_intention': self.last_intention,
            'lstm_available': self.lstm_classifier is not None,
            'available_states': {
                'action': self.action_states
            },
            'intention_classes': self.intention_classes
        }
    
    def stop(self):
        """Stop the state machine."""
        self.running = False
        print("State machine stop requested")
    
    def reset(self):
        """Reset state machine to initial state."""
        self.current_state = "MOVE_to_cup"
        self.state_history = []
        self.start_time = None
        self.last_intention = None
        print("State machine reset to initial state")
    
    def force_transition(self, new_state: str):
        """Force transition to a specific state (for testing/debugging)."""
        print(f"Forcing transition to {new_state}")
        self._transition_to(new_state)
    
    def set_lstm_classifier(self, lstm_classifier):
        """Set or update the LSTM classifier."""
        self.lstm_classifier = lstm_classifier
        print(f"LSTM classifier {'updated' if self.lstm_classifier else 'removed'}")
    
    def get_last_intention(self) -> Optional[str]:
        """Get the last classified intention."""
        return self.last_intention
    
    def get_execution_guidance(self) -> Dict[str, Any]:
        """
        Get execution guidance from IMU and Vision systems.
        Returns direction information for movement execution (speed is hard-coded).
        """
        guidance = {
            'imu_direction': None,
            'hardcoded_speed': 0.1,  # Hard-coded movement speed
            'vision_target': None,
            'vision_detected': False
        }
        
        try:
            # Get IMU data for movement direction only
            imu_data = self.sensor_manager.get_sensor_data('imu')
            if imu_data:
                if hasattr(imu_data, 'angular_velocity'):
                    guidance['imu_direction'] = imu_data.angular_velocity
                    print(f"IMU guidance: direction={guidance['imu_direction']}")
                    print(f"Using hard-coded speed: {guidance['hardcoded_speed']}")
            
            # Get Vision data for target position
            vision_data = self.sensor_manager.get_sensor_data('vision')
            if vision_data and vision_data.get('cup', {}).get('detected'):
                guidance['vision_target'] = vision_data['cup']['position_world']
                guidance['vision_detected'] = True
                print(f"Vision guidance: target={guidance['vision_target']}")
            else:
                print("Vision guidance: No target detected")
                
        except Exception as e:
            print(f"Error getting execution guidance: {e}")
        
        return guidance
    
    def get_grip_feedback(self) -> Dict[str, Any]:
        """
        Get grip feedback from Pressure and Slip sensors.
        Returns force and slip information for grip control.
        """
        feedback = {
            'pressure_force': None,
            'slip_detected': False,
            'grip_adequate': False
        }
        
        try:
            # Get pressure sensor data
            pressure_data = self.sensor_manager.get_sensor_data('pressure')
            if pressure_data:
                feedback['pressure_force'] = pressure_data.get('force', 0.0)
                print(f"Pressure feedback: force={feedback['pressure_force']}")
            
            # Get slip sensor data
            slip_data = self.sensor_manager.get_sensor_data('slip')
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
        # Check if we have minimum pressure
        min_pressure = 0.1  # Minimum grip force threshold
        has_pressure = feedback.get('pressure_force', 0) > min_pressure
        
        # Check if object is not slipping
        no_slip = not feedback.get('slip_detected', True)
        
        # Grip is adequate if we have pressure and no slip
        adequate = has_pressure and no_slip
        
        if adequate:
            print("Grip feedback: Adequate grip detected")
        else:
            print(f"Grip feedback: Inadequate grip (pressure: {has_pressure}, no_slip: {no_slip})")
        
        return adequate
