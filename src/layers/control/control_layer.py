from multiprocessing import Process, Event
import time
from shared.queues import DataQueue

# Questions:
# 1. Is it a good idea to have the control rate of the system?


class ControlLayer(Process):
    """Control layer: decision making, control signals."""

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
        """
        commands = {}

        # ------------------------------ 1. Extract ML prediction (from processing layer) ------------------------------
        ml_prediction = packet.metadata.get("ml_prediction")
        ml_confidence = packet.metadata.get("ml_confidence", 0.0)

        # ------------------------------ 2. Extract sensor data ------------------------------
        sensors = packet.sensors

        # Vision data
        vision_data = sensors.vision if sensors else None
        cup_detections = vision_data.get("cup_detections", []) if sensors else None
        apriltag_pose = vision_data.get("apriltag_pose") if sensors else None

        # Tactile sensors
        pressure = sensors.pressure if sensors else None
        piezo = sensors.piezo if sensors else None
            
        # ------------------------------ 3. Extract motor states ------------------------------
        motors = packet.motors
        current_joint_positions = motors.joint_positions if motors else None
        current_gripper_state = motors.gripper_state if motors else None

        # ------------------------------ 4. Control logic - Gripper control ------------------------------
        gripper_command = self._compute_gripper_command(
            ml_prediction=ml_prediction,
            ml_confidence=ml_confidence,
            pressure=pressure
        )
        if gripper_command:
            commands["gripper"] = gripper_command

        # ------------------------------ 5. Control logic - Arm control ------------------------------
        arm_command = self._compute_arm_command(
            cup_detections=cup_detections,
            apriltag_pose=apriltag_pose,
            current_joint_positions=current_joint_positions,
            ml_prediction=ml_prediction
        )
        if arm_command:
            commands["arm"] = arm_command

        # ------------------------------ 6. Safety checks ------------------------------
        commands = self._apply_safety_limits(commands, motors)
        
        return commands if commands else None


    def _send_commands(self, commands):
        pass


    def _compute_gripper_command(self, ml_prediction, ml_confidence, pressure):
        """
        Compute gripper command based on ML prediction, confidence, and pressure data.

        This method takes the ML prediction, confidence, and pressure data and decides what the gripper should do.
        """
        command = None
        # TODO: Implement this function
        return command

    def _compute_arm_command(self, cup_detections, apriltag_pose, current_joint_positions, ml_prediction):
        """
        Compute arm command based on cup detections, apriltag pose, current joint positions, and ML prediction.

        This method takes the cup detections, apriltag pose, current joint positions, and ML prediction and decides what the arm should do.
        """
        command = None
        # TODO: Implement this function
        return command

    def _apply_safety_limits(self, commands, motors):
        """
        Compute safety checks based on commands and motor states.

        This method takes the commands and the current motor states and decides if the commands are safe to send to the actuators.
        """
        return True