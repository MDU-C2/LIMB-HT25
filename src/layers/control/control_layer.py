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
        pass

    def _compute_commands(self, packet):
        pass

    def _send_commands(self, commands):
        pass
        