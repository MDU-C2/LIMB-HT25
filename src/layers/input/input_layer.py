
from multiprocessing import Process, Event
from hardware.can.can_interface import CANInterface
from hardware.ble.ble_interface import BLEInterface
from .window_buffer import WindowBuffer
from .packet_builder import PacketBuilder
from shared.queues import DataQueue
import time
import numpy as np

# The difference between threading and multiprocessing:
# Multiprocessing create separate OS processes, while threading creates separate threads within the same process.

class InputLayer(Process):
    """Input layer process: reads CAN, BLE and builds packets"""

    def __init__(self, can_interface: CANInterface,
                    ble_interface: BLEInterface, 
                    output_queue: DataQueue, 
                    window_size: int = 100, 
                    sample_rate: float = 100.0,
                    vision_source = None):

        super().__init__(name="InputLayer")
        self.running = Event() # Event to signal the process to stop
        self.can_interface = can_interface
        self.ble_interface = ble_interface
        self.window_buffer = WindowBuffer(window_size)
        self.packet_builder = PacketBuilder(sequence_start=0, vision_source=vision_source)
        self.sample_rate = sample_rate
        self.output_queue = output_queue
        self.vision_source = vision_source

        # Store latest sensor values (from CAN)
        # Pressure: dictionary to accumulate individual finger values
        # Keys: 'thumb', 'index', 'middle', 'ring', 'pinky'
        self.pressure_values = {}  # Dict[str, float] - individual finger pressure values
        self.latest_pressure = None      # List[float] - 5 finger values [thumb, index, middle, ring, pinky]
        self.latest_motor_state = None   # MotorState object
        self.latest_robot_imu = None     # Dict[str, np.ndarray] - {'data': [ax, ay, az, wx, wy, wz]}
        
        # Buffer for robot IMU data (gyro and accel come separately)
        # Format: {source: {"gyro": [gx, gy, gz], "accel": [ax, ay, az]}}
        self.robot_imu_buffer = {}  # Store latest gyro/accel by source (e.g., "robot_hand")
        
        # Track gripper state (from actuation commands we send)
        self.gripper_state = {"open": True, "force": 0.0}

    def run(self):
        """Main process loop"""
        
        self.running.set() # Set the event to signal the process to start
        self.can_interface.start() # Start the can interface
        self.ble_interface.start() # Start the ble interface

        while self.running.is_set():
            
            # Read BLE data (EMG, IMU, piezo)
            ble_data = self.ble_interface.read()
            for sample in ble_data:
                if sample.message_type == "EMG":
                    self.window_buffer.add_emg(sample.data["channels"], sample.timestamp)
                elif sample.message_type == "IMU":
                    self.window_buffer.add_imu(sample.data["data"], sample.timestamp)
                elif sample.message_type == "piezo":
                    # Add piezo to window buffer
                    piezo_value = sample.data.get("value")
                    if piezo_value is not None:
                        self.window_buffer.add_piezo(piezo_value, sample.timestamp)

            # Read CAN messages (non-blocking)
            can_messages = self.can_interface.read()
            for msg in can_messages: 
                # Skip messages with errors
                if not msg.parsed_data or "error" in msg.parsed_data:
                    continue
                
                msg_type = msg.message_type
                if not msg_type:
                    continue

                # Handle robot IMU messages (separate gyro and accel)
                if msg_type.endswith("_imu_gyro") or msg_type.endswith("_imu_accel"):
                    source = msg.parsed_data.get("source", "unknown")
                    if source not in self.robot_imu_buffer:
                        self.robot_imu_buffer[source] = {}
                    
                    if msg_type.endswith("_imu_gyro"):
                        self.robot_imu_buffer[source]["gyro"] = msg.parsed_data["data"]
                    elif msg_type.endswith("_imu_accel"):
                        self.robot_imu_buffer[source]["accel"] = msg.parsed_data["data"]
                    
                    # Combine gyro + accel when both are available (for robot hand IMU)
                    # Robot hand IMU is used for fusion, so prioritize that
                    if source == "robot_hand" and "gyro" in self.robot_imu_buffer[source] and "accel" in self.robot_imu_buffer[source]:
                        gyro = self.robot_imu_buffer[source]["gyro"]
                        accel = self.robot_imu_buffer[source]["accel"]
                        # Format: [ax, ay, az, wx, wy, wz] - accel first, then gyro
                        combined = accel + gyro
                        self.latest_robot_imu = {"data": combined, "timestamp": msg.timestamp}

                # Handle pressure sensor messages
                elif msg_type.endswith("_pressure"):
                    finger_name = msg.parsed_data.get("finger")
                    pressure_value = msg.parsed_data.get("value")
                    if finger_name and pressure_value is not None:
                        self.pressure_values[finger_name] = pressure_value
                        
                        # Update latest_pressure list when we have all 5 fingers
                        # Order: [thumb, index, middle, ring, pinky]
                        finger_order = ['thumb', 'index', 'middle', 'ring', 'pinky']
                        if all(f in self.pressure_values for f in finger_order):
                            self.latest_pressure = [self.pressure_values[f] for f in finger_order]

                # Handle potentiometer messages (could be used for joint positions)
                # Note: Potentiometers may not directly map to joint positions, but can use them
                # as a fallback if motor_status is not available
                elif "potentiometer" in msg_type:
                    # Potentiometers give position feedback for specific joints
                    # Could potentially use these to reconstruct joint positions
                    # For now, skip them as they may not map directly to 5-joint model
                    pass
                
                # Update motor_state with gripper state (from our tracked state)
                if self.latest_motor_state:
                    self.latest_motor_state.gripper_state = self.gripper_state.copy()
                else:
                    # Create default motor state if it doesn't exist
                    from shared.models.packet import MotorState
                    self.latest_motor_state = MotorState(
                        joint_positions=np.array([0.0] * 5),  # Default positions
                        gripper_state=self.gripper_state.copy(),
                        timestamp=time.time()
                    )

            # Create packet (only when window buffer is full)
            if self.window_buffer.is_full():

                # Update vision system to poll queues and get latest detections and pose estimates
                if self.vision_source is not None:
                    self.vision_source.update()    


                packet = self.packet_builder.build(
                    self.window_buffer,
                    self.sample_rate,
                    latest_pressure=self.latest_pressure,
                    latest_motor_state=self.latest_motor_state,
                    latest_robot_imu=self.latest_robot_imu
                )
                
                # Send packet to the next layer via an async queue
                self.output_queue.put(packet)
            time.sleep(0.001)

    def stop(self):
        """Stop the process"""
        self.running.clear() # Clear the event to signal the process to stop
        self.can_interface.stop() # Stop the CAN interface
        self.window_buffer.clear() # Clear the window buffer
        

