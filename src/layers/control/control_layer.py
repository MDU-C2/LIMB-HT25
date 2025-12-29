from multiprocessing import Process, Event
import time
from hardware.can.can_message_parser import CANMessageParser
from shared.queues import DataQueue
from typing import Optional, Dict, Tuple, List

# Questions:
# 1. Is it a good idea to have the control rate of the system?


class ControlLayer(Process):
    """
    Control layer: decision making, control signals.
    
    This layer computes arm commands based on processed sensor data and ML predictions.
    Note: Gripper control is handled locally on ESP32, not in this layer.
    We only receive gripper_status via CAN to monitor the current gripper state.
    """

    def __init__(self, input_queue: DataQueue, can_interface, control_rate, config: Optional[Dict] = None):
        super().__init__(name="ControlLayer")
        self.input_queue = input_queue
        self.can_interface = can_interface
        self.control_rate = control_rate
        self.control_period = 1.0 / control_rate
        self.running = Event()
        self.can_parser = CANMessageParser()

        # Load config or use defaults
        if config is None:
            config = {}
        
        thresholds_config = config.get("thresholds", {})
        motor_primitives_config = config.get("motor_primitives", {})
        workspace_limits_config = config.get("workspace_limits", {})
        joint_limits_config = config.get("joint_limits", [])

        # State machine
        self.STATES = {"Waiting": "Waiting for LSTM grip intention", "Gripping": "Cup gripped, waiting for stable grip + move intention", "Carrying": "Handling motors based on human IMU"}
        self.current_state = "Waiting"
        self.state_history = []
        self.state_entry_time = time.time()
        self.last_cls = None
        self.grip_command_sent = False
        
        # Configurable parameters
        self.conf_threshold = config.get("conf_threshold", 0.5)  # LSTM confidence threshold
        self.hand_cup_distance_threshold = thresholds_config.get("hand_cup_distance", 0.2)  # meters
        self.placement_distance_threshold = thresholds_config.get("placement_distance", 0.1)  # meters
        self.motor_step_size = motor_primitives_config.get("step_size", 0.05)  # meters
        
        # Workspace limits (in meters)
        self.workspace_limits = {
            "x": workspace_limits_config.get("x", [-0.5, 0.5]),
            "y": workspace_limits_config.get("y", [-0.5, 0.5]),
            "z": workspace_limits_config.get("z", [0.0, 0.8])
        }
        
        # Joint limits (in radians)
        if joint_limits_config:
            self.joint_limits = joint_limits_config
        else:
            # Default joint limits
            self.joint_limits = [
                (-3.14, 3.14),  # Joint 1: ±180 degrees
                (-1.57, 1.57),  # Joint 2: ±90 degrees
                (-3.14, 3.14),  # Joint 3: ±180 degrees
                (-1.57, 1.57),  # Joint 4: ±90 degrees
                (-3.14, 3.14),  # Joint 5: ±180 degrees
            ]

    def run(self):
        """Main process loop - runs at control rate (Hz)"""
        self.running.set()

        while self.running.is_set():
            # Get packet (non-blocking with timeout)
            cycle_start = time.time()

            # 1. Get latest processed packet
            packet = self._get_latest_packet()

            if packet is not None:
                # 2. Decide what to do (contol logic)
                commands = self._compute_commands(packet)

                # 3. Sned commands to actuators via CAN
                if commands:
                    self._send_commands(commands)

            # 4. Maintain control rate (sleep to hit target freq)
            elapsed = time.time() - cycle_start
            sleep_time = max(0, self.control_period - elapsed)
            if sleep_time > 0: # Why 0.001?
                time.sleep(sleep_time)

    def stop(self):
        """Stop the process"""
        self.running.clear() # Clear the event to signal the process to stop

    def _get_latest_packet(self):
        """
        Get the most recent packet from queue (drop old ones if queue backed up)

        This method drains the queue to get the latest packet, prioritizing fresh data for real-time control.
        """
        from shared.models.packet import DataPacket


        latest_packet = None
        max_age_ms = 100.0 # Max package age in ms
        
        while not self.input_queue.empty():
            try:
                packet = self.input_queue.get_nowait()
                time.sleep(0.005)

                # Update packet age
                packet.update_age()

                # Skip stale packets
                if packet.is_stale(max_age_ms=max_age_ms):
                    continue

                latest_packet = packet
            except:
                break # Queue empty or other error

        return latest_packet

    def _compute_commands(self, packet):
        """
        Compute actuator commands based on processed packet data.

        This is where the control logic lives. It takes the processed packet (with ML predictions, sensor data, motor states)
        and decides what the actuators should do.
        
        Note: Only arm commands are computed here. Gripper control is handled locally on ESP32,
        and we only receive gripper_status via CAN to monitor the current gripper state.
        """
        commands = {}

        # ------------------------------ 0. Update state machine ------------------------------
        self._update_state_machine(packet)
        
        # -------------------------------- 1. Compute state-specific commands --------------------------------
        arm_command = None

        if self.current_state == "Waiting":
            arm_command = self._compute_waiting_state_command(packet)
        elif self.current_state == "Gripping":
            arm_command = self._compute_gripping_state_command(packet)
        elif self.current_state == "Carrying":
            arm_command = self._compute_carrying_state_command(packet)
        
        if arm_command:
            joint_positions = arm_command.get("joint_positions")
            if joint_positions:
                commands["arm"] = {"joint_positions": joint_positions}

        # ------------------------------ 2. Apply safety limits ------------------------------
        motors = packet.motors
        commands = self._apply_safety_limits(commands, motors)
        
        return commands if commands else None

    def _send_commands(self, commands):
        """
        Send the release command to ESP32 via CAN
        """
        if not commands:
            return

        # Send arm command if present
        if "arm" in commands:
            arm_cmd = commands["arm"]
            joint_positions = arm_cmd.get("joint_positions")

            if joint_positions:
                encoded = self._encode_arm_command(joint_positions)
                if encoded:
                    can_id, data = encoded
                    success = self.can_interface.send(can_id, data)
                    if success:
                        print(f"[Control Layer] Sent arm command: {joint_positions}")
                    else:
                        print(f"[Control Layer] Failed to send arm command")

                else:
                    print(f"[Control Layer] Failed to encode arm command")

        # Send gripper command if present (for release command)
        if "gripper" in commands:
            gripper_cmd = commands["gripper"]
            action = gripper_cmd.get("action")
            force = gripper_cmd.get("force", 0.0)

            if action:
                encoded = self._encode_gripper_command(action, force)
                if encoded:
                    can_id, data = encoded
                    success = self.can_interface.send(can_id, data)
                    if success:
                        print(f"[Control Layer] Sent gripper command: {action} with force {force}")
                    else:
                        print(f"[Control Layer] Failed to send gripper command")
                else:
                    print(f"[Control Layer] Failed to encode gripper command")

    def _apply_safety_limits(self, commands, motors):
        """
        Apply safety checks based on commands and motor states.

        This method takes the commands and the current motor states and decides if the commands are safe to send to the actuators.
        
        Args:
            commands: Dict with actuator commands
            motors: MotorState object with current motor states
        
        Returns:
            Dict with validated commands (may be modified or filtered), or None if commands are unsafe
        """
        if not commands:
            return None
        
        if "arm" in commands:
            arm_cmd = commands["arm"]
            joint_positions = arm_cmd.get("joint_positions")

            if joint_positions:
                if not self._validate_joint_positions(joint_positions):
                    print(f"[Control Layer] Arm commands rejected: joint positions out of limits")
                    commands.pop("arm")

                # TODO: Add velocity limits check if tracking velocities
                # TODO: Add collision avoidance check
                # TODO: Add emergency stop conditions

        if "gripper" in commands:
            gripper_cmd = commands["gripper"]
            force = gripper_cmd.get("force", 0.0)

            if force < 0.0 or force > 1.0:
                print(f"[Control Layer] Gripper force {force} out of range [0.0, 1.0], clamping...")
                gripper_cmd["force"] = max(0.0, min(1.0, force))

        return commands if commands else None

    def _get_current_state(self):
        """Return the curren state"""
        return self.current_state

    def _transition_to_state(self, new_state: str, reason: str = ""):
        """Handle state transition with logging"""

        #TODO: Check if "reason" is needed and what it is for
        
        if new_state not in self.STATES:
            print(f"Warning: Unknown state '{new_state}'")
            return
        
        if new_state == self.current_state:
            return # No transition needed

        old_state = self.current_state
        transition_time = time.time()

        # Record transition
        self.state_history.append({
            "from_state": old_state,
            "to_state": new_state,
            "timestamp": transition_time,
            "reason": reason,
            "time_in_state": transition_time - self.state_entry_time
        })

        # Update state
        self.current_state = new_state
        self.state_entry_time = transition_time

        # Reset state_specific flags
        if new_state == "Waiting":
            self.grip_command_sent = False
        elif new_state == "Gripping":
            pass
        elif new_state == "Carrying":
            pass

        self._log_state_transition(old_state, new_state, reason)

    def _update_state_machine(self, packet):
        """Main state update logic (called from _compute_commands)"""
        
        # Extract data from packet
        ml_prediction_dict = packet.metadata.get("ml_prediction")
        grip_intention = self._interpret_intention(ml_prediction_dict) if ml_prediction_dict else None
        movement_intention = packet.metadata.get("movement_intention")

        sensors = packet.sensors
        vision_data = sensors.vision if sensors else None
        pressure = sensors.pressure if sensors else None

        motors = packet.motors
        gripper_state = motors.gripper_state if motors else None
        
        apriltag_pose = vision_data.get("apriltag_pose") if vision_data else None

        # State-specific transition logic
        if self.current_state == "Waiting":
            # Transition: Waiting -> Gripping
            # Requires: LSTM grip intention + vision validation (hand-cup distance)
            if grip_intention == "grip":

                fused_arm_pose = packet.metadata.get("fused_arm_pose")
                hand_position = None
                hand_pos_m = None
                if fused_arm_pose and "position" in fused_arm_pose:
                    hand_position = fused_arm_pose["position"] # [x, y, z] in mm, convert to meter

                if hand_position:
                    import numpy as np
                    hand_pos_m = np.array(hand_position) / 1000.0 if max(hand_position) > 10 else np.array(hand_position)
                    #hand_pos_m = hand_pos_m.astype(np.float32) # TODO: Check if this is needed

                if self._check_hand_cup_distance(vision_data, hand_pos_m):
                    self._transition_to_state("Gripping", "Hold intention + vision validation")
        
        elif self.current_state == "Gripping":
            # Transition: Gripping -> Carrying
            # Requires: Stable grip + vision validation (placement)
            if self._check_stable_grip(gripper_state):
                if movement_intention and movement_intention.get("direction"):
                    self._transition_to_state("Carrying", "Stable grip + movement intention")

        elif self.current_state == "Carrying":
            # Transition: Carrying -> Waiting
            # Requires: Movement intention + vision validation (placement)
            if grip_intention == "rest":
                if self._check_placement_valid(vision_data, apriltag_pose):
                    self._send_release_command()
                    self._transition_to_state("Waiting", "Release intention + vision validation")

    def _log_state_transition(self, from_state: str, to_state: str, reason):
        """Log transitions for debugging"""
        print(f"[State Machine] {from_state} -> {to_state}")
        if reason:
            print(f"  Reason: {reason}")
        print(f"  Time: {time.strftime('%H:%M:%S', time.localtime())}")

    def _interpret_intention(self, ml_prediction_dict) -> Optional[str]:
        """Interpret ML prediction to determine the next state"""
        
        # Extract class and confidence
        cls = ml_prediction_dict.get("class")
        conf = ml_prediction_dict.get("confidence")

        # Apply conf threshold
        if conf is None or conf < self.conf_threshold:
            return None

        if cls == 0:
            return "rest"
        elif cls == 1:
            return "grip"
        
        return None

    def _check_hand_cup_distance(self, vision_data, hand_position=None) -> bool:
        """
        Check if the hand is close to the cup.
        Hand position is in meters.
        
        Called in Waiting state before transition to Gripping state.
        """

        import numpy as np

        if not vision_data:
            return False

        threshold = self.hand_cup_distance_threshold

        cup_position = None
        cup_detection = vision_data.get("cup_detections", [])

        if cup_detection:
            # Try to get position from first cup detection
            cup_det = cup_detection[0] if isinstance(cup_detection, list) else cup_detection 
            
            if isinstance(cup_det, dict):
                cup_position = cup_det.get("position")
            elif hasattr(cup_det, "position"):
                cup_position = cup_det.position

        if cup_position is None:
            return False # no cup detected

       # Convert to numpy array if needed
        if not isinstance(cup_position, np.ndarray):
            cup_position = np.array(cup_position)
        
        # Use hand position if provided, otherwise return False
        if hand_position is None:
            return False
        
        # Convert hand position to numpy array if needed
        if not isinstance(hand_position, np.ndarray):
            hand_position = np.array(hand_position)
        
        # Calculate Euclidean distance
        distance = np.linalg.norm(cup_position - hand_position)
        
        # Check threshold
        return distance < threshold

    def _check_placement_valid(self, vision_data, apriltag_pose) -> bool:
        """Check if the placement is valid. Called in Carrying state before releasing cup."""

        import numpy as np

        if not vision_data or not apriltag_pose:
            return False

        target_position = None
        if isinstance(apriltag_pose, dict):
            target_position = apriltag_pose.get("position")
        
        if target_position is None:
            return False

        if not isinstance(target_position, np.ndarray):
            target_position = np.array(target_position)

        # Extract cup position from cup_detections
        cup_position = None
        cup_detection = vision_data.get("cup_detections", [])

        if cup_detection:
            cup_det = cup_detection[0] if isinstance(cup_detection, list) else cup_detection 
            if isinstance(cup_det, dict):
                cup_position = cup_det.get("position")
            elif hasattr(cup_det, "position"):
                cup_position = cup_det.position

        if cup_position is None:
            return False # no cup detected

        if not isinstance(cup_position, np.ndarray):
            cup_position = np.array(cup_position)

        placement_threshold = self.placement_distance_threshold
        distance = np.linalg.norm(cup_position - target_position)

        return distance < placement_threshold

    def _check_stable_grip(self, gripper_state) -> bool:
        """Check if the grip is stable. Called in Gripping state before transition to Carrying state."""

        # Check if gripper state indicates stable grip, ESP32 processes data and sends back a flag.
        if gripper_state:
            # Format {"open": bool, "force": float, "stable": bool}
            if gripper_state.get("stable"):
                return True
            else:
                return False
                
        return False # Default to False if gripper state is not available

    def _get_movement_direction(self, movement_intention) -> Optional[str]:
        """Get the movement direction based on the intention from the user"""
        
        if not movement_intention:
            return None
        
        # Extract direction and conf
        # Format: {'direction': str, 'acceleration': [ax, ay, az], magnitude: float, confidence: float, timestamp: float}
        direction = movement_intention.get("direction")
        if not direction: 
            return None

        if direction.lower() in ["forward", "backward", "left", "right", "up", "down"]:
            return direction.lower()

        return None

    def _get_motor_primitives(self, direction: str, current_pose, movement_intention) -> Optional[Dict]:
        """Get the motor primitives based on the direction and the intention from the user"""
        import numpy as np

        if not direction or not current_pose:
            return None

        # Extract current position from current_pose
        # Format: {"postion": [x, y, z], "orientation_euler": [roll, pitch, yaw], ...}
        current_position = current_pose.get("position")
        if not current_position:
            return None

        # Convert to numpy array if needed
        if not isinstance(current_position, np.ndarray):
            current_position = np.array(current_position)

        # Define simple motor primitives (fixed offsets in meters)
        # These are simple movements - each direction has a fixed step size
        step_size = self.motor_step_size

        # Scale by confidence if available?
        offset = np.array([0.0, 0.0, 0.0])

        if direction == "up":
            offset = np.array([0.0, 0.0, step_size]) # Move up in Z
        elif direction == "down":
            offset = np.array([0.0, 0.0, -step_size]) # Move down in Z
        elif direction == "forward":
            offset = np.array([step_size, 0.0, 0.0]) # Move forward in X
        elif direction == "backward":
            offset = np.array([-step_size, 0.0, 0.0]) # Move backward in X
        elif direction == "left":
            offset = np.array([0.0, step_size, 0.0]) # Move left in Y
        elif direction == "right":
            offset = np.array([0.0, -step_size, 0.0]) # Move right in Y
        else:
            return None

        # Calculate target position
        target_position = current_position + offset

        # Return target pose (keep orientation the same for now)
        return {
            "position": target_position.tolist(),
            "orientation": current_pose.get("orientation"), # Keep same orientation for now
            "direction": direction,
            "step_size": float(step_size)
        }

    def _compute_waiting_state_command(self, packet) -> Optional[Dict]:
        """
        Compute the command for the waiting state
        
        In Waiting state, the arm should move towards the cup when grip intention is detected.
        Uses vision data to locate the cup and moves the arm to approach it.
        """
        import numpy as np

        sensors = packet.sensors
        vision_data = sensors.vision if sensors else None
        
        if not vision_data:
            return None

        cup_detections = vision_data.get("cup_detections", [])
        if not cup_detections:
            return None

        # Get first cup detection
        cup_det = cup_detections[0] if isinstance(cup_detections, list) else cup_detections

        # Extract cup position
        cup_position = None
        if isinstance(cup_det, dict):
            cup_position = cup_det.get("position")
        elif hasattr(cup_det, "position"):
            cup_position = cup_det.position

        if cup_position is None:
            return None

        # Convert to numpy array if needed
        if not isinstance(cup_position, np.ndarray):
            cup_position = np.array(cup_position)

        # Get current arm pose
        current_pose = self._get_current_arm_pose(packet)
        if current_pose is None:
            return None

        current_position = current_pose.get("position")
        if current_position is None:
            return None

        if not isinstance(current_position, np.ndarray):
            current_position = np.array(current_position)

        # Calculate approach position (offset from cup approach from above)
        approach_offset = np.array([0.0, 0.0, 0.15]) # 15 cm above the cup
        target_position = cup_position + approach_offset

        # Validate target pose
        target_pose = {
            "position": target_position.tolist(),
            "orientation": current_pose.get("orientation"), # Keep the current orientation
        }

        if not self._validate_target_pose(target_pose):
            return None

        # Compute joint positions using inverse kinematics
        joint_positions = self._inverse_kinematics(target_pose)
        if joint_positions is None:
            return None

        return {
            "joint_positions": joint_positions,
            "target_pose": target_pose,
        }
        
    def _compute_gripping_state_command(self, packet) -> Optional[Dict]:
        """
        Compute the command for the gripping state
        
        In gripping state, the arm should maintain the position while the gripper closes.
        Small adjustmenst may be needed to maintain the alignemtn with the cup.

        The gripper is handled by ESP32, so we just hold position here.
        """
        
        motors = packet.motors
        if motors and motors.joint_positions is not None:
            return {
                "joint_positions": motors.joint_positions.tolist(),
                "maintain_position": True
            }

        return None

    def _compute_carrying_state_command(self, packet) -> Optional[Dict]:
        """
        Compute the command for the carrying state

        In carrying state, the arm moves based on human IMU movement intention.
        Uses motor primitives to translate movement direction into arm motion.
        """
        movement_intention = packet.metadata.get("movement_intention")
        if not movement_intention:
            return None

        direction = self._get_movement_direction(movement_intention)
        if not direction:
            return None
        
        current_pose = self._get_current_arm_pose(packet)
        if not current_pose:
            return None

        target_pose = self._get_motor_primitives(direction, current_pose, movement_intention)
        if not target_pose:
            return None

        if not self._validate_target_pose(target_pose):
            return None
        
        # Compute joint positions using inverse kinematics
        joint_positions = self._inverse_kinematics(target_pose)
        if joint_positions is None:
            return None
        
        return {
            "joint_positions": joint_positions,
            "target_pose": target_pose,
            "direction": direction
        }

    def _encode_gripper_command(self, action: str, force: float) -> Tuple[int, bytes]:
        """Encode the gripper command"""
        # Convert actio string to byte: 0 = open, 1 = close
        action_byte = 1 if action.lower() == "close" else 0

        # Clamp force to valid range
        force = max(0.0, min(1.0, force))

        # Use CANMessageParser to encode
        result = self.can_parser.encode("gripper_command", {"action": action_byte, "force": force})
        return result
    
    def _encode_arm_command(self, joint_positions: List[float]) -> Tuple[int, bytes]:
        """Encode the arm command"""
        if not joint_positions or len(joint_positions) != 5:
            return None

        # Use CANMessageParser to encode
        result = self.can_parser.encode("arm_command", {"joint_positions": joint_positions})
        return result

    def _send_release_command(self):
        """Send the release command via CAN"""
        # Encode and send gripper open command
        encoded = self._encode_gripper_command("open", 0.0)
        if encoded:
            can_id, data = encoded
            success = self.can_interface.send(can_id, data)
            if success:
                self.grip_command_sent = False
                print(f"[Control Layer] Release command sent via CAN")
            else:
                print(f"[Control Layer] Failed to send release command")
        else:
            print(f"[Control Layer] Failed to encode release command")
   
    # --- Helper methods ----

    def _forward_kinematics(self, joint_positions: List[float]) -> Dict:
        """
        Compute the forward kinematics for the arm.
        
        Args:
            joint_positions: List of 5 joint angles in radians
        
        Returns:
            Dict with 'position' and 'orientation', or None if computation fails
        """
        # TODO: Implement forward kinematics based on robot arm DH parameters
        # This is a placeholder - needs to be implemented with actual robot arm model
        # Example structure:
        #   - Use DH parameters or transformation matrices
        #   - Compute end-effector position [x, y, z]
        #   - Compute end-effector orientation (quaternion or euler angles)
        
        print("[Control Layer] Forward kinematics not yet implemented")
        return None

    def _inverse_kinematics(self, target_pose: Dict) -> List[float]:
        """
        Compute the inverse kinematics for the arm.
        
        Args:
            target_pose: Dict with 'position' and optionally 'orientation'
                Format: {'position': [x, y, z], 'orientation': ...}
        
        Returns:
            List of 5 joint angles in radians, or None if no solution found
        """
        # TODO: Implement inverse kinematics based on robot arm DH parameters
        # This is a placeholder - needs to be implemented with actual robot arm model
        # Example approaches:
        #   - Analytical IK if available
        #   - Numerical IK (e.g., Jacobian-based iterative method)
        #   - Use existing IK library (e.g., ikpy, pybullet)
        
        print("[Control Layer] Inverse kinematics not yet implemented")
        return None

    def _get_current_arm_pose(self, packet) -> Dict:
        """Get the current arm pose from the packet"""
        import numpy as np

        fused_arm_pose = packet.metadata.get("fused_arm_pose")
        if fused_arm_pose:

            position = fused_arm_pose.get("position")
            orientation = fused_arm_pose.get("orientation") # Pitch, roll, yaw is orientation_euler but use quarternion now
            if position:
                return {
                    "position": np.array(position) if not isinstance(position, np.ndarray) else position,
                    "orientation": orientation,
                    "source": "fused_arm_pose"
                }

        # Implemented a secondary source of arm pose using forward kinematics from current joint positions.
        motors = packet.motors
        if motors and motors.joint_positions is not None:
            current_pose = self._forward_kinematics(motors.joint_positions.tolist())
            if current_pose:
                return {
                    **current_pose,
                    "source": "forward_kinematics"
                }
        
        return None

    def _validate_joint_positions(self, joint_positions: List[float]) -> bool:
        """Validate the joint positions"""
        if not joint_positions or len(joint_positions) != 5:
            return False

        # Use configured joint limits
        for i, (pos, (min_val, max_val)) in enumerate(zip(joint_positions, self.joint_limits)):
            if not (min_val <= pos <= max_val):
                print(f"[Control Layer] Joint {i+1} position {pos} is out of range [{min_val}, {max_val}]")
                return False

        return True

    def _validate_target_pose(self, target_pose: Dict) -> bool:
        """Validate the target pose"""
        import numpy as np

        if not target_pose:
            return False

        position = target_pose.get("position")
        if position is None:
            return False
        
        if not isinstance(position, np.ndarray):
            position = np.array(position)

        if len(position) != 3:
            return False

        # Check positions is within workspace
        x, y, z = position[0], position[1], position[2]

        if not (self.workspace_limits["x"][0] <= x <= self.workspace_limits["x"][1]):
            print(f"[Control Layer] Target position X {x} is out of range [{self.workspace_limits['x'][0]}, {self.workspace_limits['x'][1]}]")
            return False
        
        if not (self.workspace_limits["y"][0] <= y <= self.workspace_limits["y"][1]):
            print(f"[Control Layer] Target position Y {y} is out of range [{self.workspace_limits['y'][0]}, {self.workspace_limits['y'][1]}]")
            return False
        
        if not (self.workspace_limits["z"][0] <= z <= self.workspace_limits["z"][1]):
            print(f"[Control Layer] Target position Z {z} is out of range [{self.workspace_limits['z'][0]}, {self.workspace_limits['z'][1]}]")
            return False
        
        return True