
from multiprocessing import Process, Event
from hardware.can.can_interface import CANInterface
from hardware.ble.ble_interface import BLEInterface
from window_buffer import WindowBuffer
from packet_builder import PacketBuilder
from shared.queues import DataQueue
import time

# The difference between threading and multiprocessing:
# Multiprocessing create separate OS processes, while threading creates separate threads within the same process.

class InputLayer(Process):
    """Input layer process: reads CAN and builds packets"""

    def __init__(self, can_interface: CANInterface,
                    ble_interface: BLEInterface, 
                    output_queue: DataQueue, 
                    window_size: int = 100, 
                    sample_rate: float = 100.0,
                    vision_source = None,
                    pressure_source = None,
                    piezo_source = None,
                    motor_state_source = None):

        super().__init__(name="InputLayer")
        self.running = Event() # Event to signal the process to stop
        self.can_interface = can_interface
        self.ble_interface = ble_interface
        self.window_buffer = WindowBuffer(window_size)
        self.packet_builder = PacketBuilder(sequence_start=0)
        self.sample_rate = sample_rate # Do we need this?
        self.output_queue = output_queue

        self.packet_builder.set_sensor_sources(
            vision_source=vision_source, 
            pressure_source=pressure_source, 
            piezo_source=piezo_source)

        self.packet_builder.set_motor_state_source(motor_state_source)

    def run(self):
        """Main process loop"""
        
        self.running.set() # Set the event to signal the process to start
        self.can_interface.start() # Start the can interface
        self.ble_interface.start() # Start the ble interface

        while self.running.is_set():
            
            # Read BLE data (EMG, IMU, piezo)
            ble_data = self.ble_interface.read() # TODO: Implement this
            for sample in ble_data:
                if sample.message_type == "EMG":
                    self.window_buffer.add_emg(sample.data["channels"], sample.timestamp)
                elif sample.message_type == "IMU":
                    self.window_buffer.add_imu(sample.data["data"], sample.timestamp)
                elif sample.message_type == "piezo":
                    self.window_buffer.add_piezo(sample.data["data"], sample.timestamp)

            # Read CAN messages (non-blocking)
            can_messages = self.can_interface.read() # Maybe read all? Is it a list or a dict?
            for msg in can_messages:
                # Update window buffer with new data
                # To add functions in each branch, to the window buffer?

                # TODO: EMG and IMU and other sensors have different sample rates, do we need to handle this somehow?

                if msg.message_type == "IMU":
                    self.window_buffer.add_imu(msg.data["data"], msg.timestamp)

                elif msg.message_type == "pressure":
                    pass # TODO: Implement this

                elif msg.message_type == "motor_status":
                    pass

                elif msg.message_type == "gripper_status":
                    pass

            # Create packet (only when window buffer is full)
            if self.window_buffer.is_full():
                packet = self.packet_builder.build(self.window_buffer, self.sample_rate)
                
                # Send packet to the next layer via an async queue
                self.output_queue.put(packet)
            time.sleep(0.001)

    def stop(self):
        """Stop the process"""
        self.running.clear() # Clear the event to signal the process to stop
        self.can_interface.stop() # Stop the CAN interface
        self.window_buffer.clear() # Clear the window buffer
        

