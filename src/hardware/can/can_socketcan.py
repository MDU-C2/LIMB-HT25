
import can
import time
from typing import List, Optional
from .can_interface import CANInterface, CANMessage
from .can_message_parser import CANMessageParser

# TODO: Add requirements file for python-can

class SocketCANInterface(CANInterface):
    """
    CAN interface using Linux SocketCAN (for Jetson Orin).

    Uses python-can library which provides SocketCAN support for Python.
    """

    def __init__(self, interface: str = "can0", bitrate: int = 500000, timeout: float = 0.1):
        self.interface = interface
        self.bitrate = bitrate
        self.timeout = timeout
        self.bus: Optional[can.Bus] = None
        self.running = False
        self.message_parser = CANMessageParser()

        # TODO: Do we need this?
        self.rx_count = 0
        self.tx_count = 0
        self.error_count = 0

    def start(self) -> bool:
        """Start the CAN bus."""
        try:
            self.bus = can.Bus(
                interface="socketcan",
                channel=self.interface,
                bitrate=self.bitrate,
                receive_own_messages=False
            )
            self.running = True
            print(f"CAN interface {self.interface} started at {self.bitrate} bps.")
            return True
        except Exception as e:
            print(f"Failed to start CAN interface: {e}")
            self.running = False
            return False

    def stop(self) -> bool:
        """Stop the CAN bus."""
        try: 
            if self.bus:
                self.bus.shutdown()
            self.running = False
            print(f"CAN interface {self.interface} stopped.")
            return True
        except Exception as e:
            print(f"Failed to stop CAN interface: {e}")
            # TODO: Do I need to add self.running = False here?
            return False

    def read(self, timeout: Optional[float] = None) -> List[CANMessage]:
        """Read messages from the CAN bus (non-blocking or with timeout)"""
        if not self.running or not self.bus:
            return []

        messages = []
        read_timeout = timeout if timeout is not None else self.timeout

        try:
            # Read all available messages (non-blocking)
            while True:
                message = self.bus.recv(timeout=read_timeout)
                if message is None:
                    break
                    
                # Convert python-can message to our CANMessage format
                can_msg = CANMessage(
                    can_id=message.arbitration_id,
                    data=bytes(message.data),
                    timestamp=message.timestamp or time.time()
                )

                # Parse message based on its ID
                parsed = self.message_parser.parse(can_msg)
                # Store parsed data as attributes (using setattr for compatibility)
                setattr(can_msg, 'message_type', parsed.get("type"))
                setattr(can_msg, 'parsed_data', parsed.get("data"))

                messages.append(can_msg)
                self.rx_count += 1

                # Limit number of messages to read to avoid blocking?
                if len(messages) >= 100:
                    break
                
        except can.CanError as e:
            self.error_count += 1
            print(f"CAN read error: {e}")
        except Exception as e:
            self.error_count += 1
            print(f"Unexpected error in CAN read: {e}")
        
        return messages


    def send(self, can_id: int, data: bytes) -> bool:
        """Send a message on the CAN bus."""
        if not self.running or not self.bus:
            return False

        if len(data) > 8:
            print(f"Warning: CAN data too long ({len(data)} bytes), truncating")
            data = data[:8]

        try:
            message = can.Message(arbitration_id=can_id, data=data)
            self.bus.send(message)
            self.tx_count += 1
            return True

        except can.CanError as e:
            self.error_count += 1
            print(f"CAN send error: {e}")
            return False
        except Exception as e:
            self.error_count += 1
            print(f"Unexpected error in CAN send: {e}")
            return False
        

    def is_running(self) -> bool:
        """Check if the CAN interface is running."""
        return self.running
        
    def get_statistics(self) -> dict:
        """Get CAN bus statistics."""
        return {
            "rx_count": self.rx_count,
            "tx_count": self.tx_count,
            "error_count": self.error_count,
            "running": self.running,
            "interface": self.interface,
            "bitrate": self.bitrate
        }