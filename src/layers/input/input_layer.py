
from multiprocessing import Process, Event
from XXXXX import CanInterface
from window_buffer import WindowBuffer
from packet_builder import PacketBuilder
import time

# The difference between threading and multiprocessing:
# Multiprocessing create separate OS processes, while threading creates separate threads within the same process.

class InputLayer(Process):
    """Input layer process: reads CAN and builds packets"""

    def __init__(self, can_interface: CanInterface, output_queue, window_size: int = 100, sample_rate: float = 100.0):

        self.running = Event() # Event to signal the process to stop
        self.can_interface = can_interface
        self.window_buffer = WindowBuffer(window_size)
        self.packet_builder = PacketBuilder()
        self.sample_rate = sample_rate
        self.output_queue = output_queue # How to implement this queue? Use existing from library or make own class?

    def run(self):
        """Main process loop"""
        
        self.running.set() # Set the event to signal the process to start
        self.can_interface.start() # Start the can interface

        while self.running.is_set():

            # Read CAN messages (non-blocking)
            can_messages = self.can_interface.read() # Maybe read all? Is it a list or a dict?
            
            # Process CAN messages
            for msg in can_messages:
                # Update window buffer with new data
                # To add functions in each branch, to the window buffer?

                if msg.type == "EMG":
                    pass

                elif msg.type == "IMU":
                    pass

                elif msg.type == "piezo":
                    pass

                elif msg.type == "potentiometer": # Do we need this?
                    pass

            # Create packet (only when window buffer is full)
            if self.window_buffer.is_full():
                packet = self.packet_builder.build() # TODO: Implement this function and Packet stuff
                
                # Send packet to the next layer via an async queue
                self.output_queue.put(packet)
            time.sleep(0.001)

    def stop(self):
        """Stop the process"""
        self.running.clear() # Clear the event to signal the process to stop

