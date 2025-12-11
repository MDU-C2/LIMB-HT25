from multiprocessing import Process, Event
import time
from shared.queues import DataQueue

# Questions:
# 1. Is it a good idea to have the control rate of the system?


class ControlLayer(Process):
    """
    Control layer: decision making, control signals.
    
    This layer computes arm commands based on processed sensor data and ML predictions.
    Note: Gripper control is handled locally on ESP32, not in this layer.
    We only receive gripper_status via CAN to monitor the current gripper state.
    """

    def __init__(self, input_queue: DataQueue, can_interface, control_rate):
        super().__init__(name="ControlLayer")
        self.input_queue = input_queue
        self.can_interface = can_interface
        self.control_rate = control_rate
        self.control_period = 1.0 / control_rate
        self.running = Event()

    def run(self):
        """Main process loop - runs at control rate (Hz)"""
        self.running.set()

        while self.running.is_set():
            # Get packet (non-blocking with timeout)
            cycle_start = time.time()

            # 1. Get latest processed packet
            packet = self._get_latest_packet() # TODO: Implement this function

            if packet is not None:
                # 2. Decide what to do (contol logic)
                commands = self._compute_commands(packet) # TODO: Implement this function

                # 3. Sned commands to actuators via CAN
                if commands:
                    self._send_commands(commands) # TODO: Implement this function

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

        # Drain queue to get latest packet (drop old ones)
        while not self.input_queue.empty():
            try:
                packet = self.input_queue.get_nowait()

                # Update packet age
                packet.update_age()

                # Skip stale packets
                if packet.is_stale(max_age_ms=max_age_ms):
                    continue

                latest_packet = packet
            except:
                break # Queue empty or other error

        # TODO: Could add logging to track dropped packets.
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

        # ------------------------------ 1. Extract ML prediction (from processing layer) ------------------------------
        # ML prediction is a dict with keys: 'class', 'probabilities', 'confidence', 'timestamp'
        ml_prediction_dict = packet.metadata.get("ml_prediction")
        ml_prediction_class = ml_prediction_dict.get("class") if ml_prediction_dict else None
        ml_confidence = ml_prediction_dict.get("confidence", 0.0) if ml_prediction_dict else 0.0

        # ------------------------------ 2. Extract movement intention (from processing layer) ------------------------------
        # Movement intention is a dict with keys: 'direction', 'acceleration', 'magnitude', 'confidence', 'timestamp'
        movement_intention = packet.metadata.get("movement_intention")

        # ------------------------------ 3. Extract fused arm pose (from processing layer, if available) ------------------------------
        # Fused arm pose from IMU-vision fusion (if processing layer calls _process_imu_vision_fusion)
        fused_arm_pose = packet.metadata.get("fused_arm_pose")

        # ------------------------------ 4. Extract sensor data ------------------------------
        sensors = packet.sensors

        # Vision data
        vision_data = sensors.vision if sensors else None
        cup_detections = vision_data.get("cup_detections", []) if vision_data else []
        apriltag_pose = vision_data.get("apriltag_pose") if vision_data else None

        # Tactile sensors
        # Pressure: List[float] with 5 values [thumb, index, middle, ring, little]
        pressure = sensors.pressure if sensors else None
        
        # Piezo in HumanDataWindow
        # Access from human_data if needed (time-series data)
        piezo_data = None
        if packet.human_data and packet.human_data.piezo is not None:
            # Get latest piezo value (last element in the window)
            piezo_array = packet.human_data.piezo
            if len(piezo_array) > 0:
                piezo_data = float(piezo_array[-1])
            
        # ------------------------------ 5. Extract motor states ------------------------------
        motors = packet.motors
        current_joint_positions = motors.joint_positions if motors else None
        current_gripper_state = motors.gripper_state if motors else None
        # Note: Gripper control is handled locally on ESP32, not on AGX.
        # We only receive gripper_status via CAN to know the current state.

        # ------------------------------ 6. Control logic - Arm control ------------------------------
        arm_command = self._compute_arm_command(
            cup_detections=cup_detections,
            apriltag_pose=apriltag_pose,
            fused_arm_pose=fused_arm_pose,  # Fused pose from IMU-vision fusion
            movement_intention=movement_intention,  # Human movement intention
            current_joint_positions=current_joint_positions,
            ml_prediction=ml_prediction_class
        )
        if arm_command:
            commands["arm"] = arm_command

        # ------------------------------ 7. Safety checks ------------------------------
        commands = self._apply_safety_limits(commands, motors)
        
        return commands if commands else None

    def _send_commands(self, commands):
        """
        Send actuator commands to ESP32 via CAN interface.
        
        Note: Only arm commands are sent. Gripper control is handled locally on ESP32.
        
        Args:
            commands: Dict with commands, e.g.:
                {'arm': {'joint_positions': [j1, j2, j3, j4, j5]}}
        """
        # TODO: Implement CAN message encoding and sending
        # Use CANMessageParser.encode() to encode arm_command messages
        # Use can_interface.send() to send messages
        pass

    def _compute_arm_command(self, cup_detections, apriltag_pose, fused_arm_pose, movement_intention, current_joint_positions, ml_prediction):
        """
        Compute arm command based on cup detections, apriltag pose, fused arm pose, movement intention, 
        current joint positions, and ML prediction.

        Args:
            cup_detections: List of cup detection objects
            apriltag_pose: Dict with AprilTag pose data or None
            fused_arm_pose: Dict with fused pose from IMU-vision fusion or None
                Format: {'position': [x, y, z], 'orientation_euler': [roll, pitch, yaw], ...}
            movement_intention: Dict with human movement intention or None
                Format: {'direction': str, 'acceleration': [ax, ay, az], 'confidence': float, ...}
            current_joint_positions: np.ndarray with current joint positions (5 joints) or None
            ml_prediction: ML prediction class (int or None)
        
        Returns:
            Dict with arm command, e.g.:
                {'joint_positions': [j1, j2, j3, j4, j5]}  # Target joint angles in radians
            or None if no command should be sent
        """
        command = None
        # TODO: Implement this function
        # Example logic:
        # - Use cup_detections to find target cup position
        # - Use fused_arm_pose for current arm pose estimate
        # - Use movement_intention to interpret human intent (up, down, forward, etc.)
        # - Use current_joint_positions for current state
        # - Compute target joint positions using inverse kinematics or trajectory planning
        return command

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
        
        # TODO: Implement safety checks:
        # - Joint position limits (min/max angles)
        # - Joint velocity limits (if we track velocities)
        # - Collision avoidance
        # - Emergency stop conditions
        # Note: Gripper safety is handled on ESP32
        
        return commands