
from multiprocessing import Process, Event
from hardware.can.can_interface import CANInterface
from hardware.ble.ble_interface import BLEInterface
from window_buffer import WindowBuffer
from packet_builder import PacketBuilder
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
        # Keys: 'thumb', 'index', 'middle', 'ring', 'little'
        self.pressure_values = {}  # Dict[str, float] - individual finger pressure values
        self.latest_pressure = None      # List[float] - 5 finger values [thumb, index, middle, ring, little]
        self.latest_motor_state = None   # MotorState object
        self.latest_robot_imu = None     # Dict[str, np.ndarray] - {'data': [ax, ay, az, wx, wy, wz]}

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
            can_messages = self.can_interface.read() # Maybe read all? Is it a list or a dict?
            for msg in can_messages: 
                # TODO: EMG and IMU and other sensors have different sample rates, do we need to handle this somehow?

                if msg.message_type == "IMU":  # IMU from robot
                    # CAN parser returns: {'data': [ax, ay, az, wx, wy, wz]}
                    if msg.parsed_data and "data" in msg.parsed_data:
                        self.latest_robot_imu = {"data": msg.parsed_data["data"], "timestamp": msg.timestamp}

                elif msg.message_type.startswith("pressure_"):
                    # Handle individual finger pressure messages
                    # CAN parser returns: {'value': float, 'finger': 'thumb'|'index'|'middle'|'ring'|'little'}
                    if msg.parsed_data and "value" in msg.parsed_data and "finger" in msg.parsed_data:
                        finger_name = msg.parsed_data["finger"]
                        pressure_value = msg.parsed_data["value"]
                        self.pressure_values[finger_name] = pressure_value
                        
                        # Update latest_pressure list when we have all 5 fingers
                        # Order: [thumb, index, middle, ring, little]
                        finger_order = ['thumb', 'index', 'middle', 'ring', 'little']
                        if all(f in self.pressure_values for f in finger_order):
                            self.latest_pressure = [self.pressure_values[f] for f in finger_order]

                elif msg.message_type == "motor_status":
                    # Store latest motor state
                    # CAN parser returns: {'joint_positions': [j1, j2, j3, j4, j5]} # TODO: Clarify what joint correspond to what part of the arm
                    if msg.parsed_data:
                        from shared.models.packet import MotorState
                        positions = msg.parsed_data.get("joint_positions", [])
                        # Ensure we have 5 joint positions
                        if len(positions) == 5:
                            self.latest_motor_state = MotorState(
                                joint_positions=np.array(positions),
                                gripper_state={},  # Will be updated from gripper_status
                                timestamp=msg.timestamp
                            )

                elif msg.message_type == "gripper_status":
                    # Update gripper state in motor_state
                    # CAN parser returns: {'state': int, 'force': float}
                    if self.latest_motor_state and msg.parsed_data:
                        self.latest_motor_state.gripper_state = {
                            "open": msg.parsed_data.get("state", 0) == 1,
                            "force": msg.parsed_data.get("force", 0.0)
                        }
                    elif msg.parsed_data:
                        # If no motor_state exists yet, create one with default joint positions
                        from shared.models.packet import MotorState
                        self.latest_motor_state = MotorState(
                            joint_positions=np.array([0.0] * 5),
                            gripper_state={
                                "open": msg.parsed_data.get("state", 0) == 1,
                                "force": msg.parsed_data.get("force", 0.0)
                            },
                            timestamp=msg.timestamp
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
        

