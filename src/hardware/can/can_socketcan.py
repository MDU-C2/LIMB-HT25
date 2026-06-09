
import can
import time
from types import TracebackType
from typing import List, Optional
from typing_extensions import Self

from .can_interface import CANInterface, CANMessage
from .can_message_parser import CANMessageParser

class SocketCANInterface(CANInterface):
    """
    CAN interface using Linux SocketCAN (for Jetson Orin).

    Uses python-can library which provides SocketCAN support for Python.
    """

    def __init__(self, interface: str = "can0", bitrate: int = 1000000, timeout: float = 0.1):
        self.interface = interface
        self.bitrate = bitrate
        self.timeout = timeout
        self.bus: Optional[can.Bus] = None
        self.running = False
        self.message_parser = CANMessageParser()

        self.rx_count = 0
        self.tx_count = 0
        self.error_count = 0

    def __enter__(self) -> Self:
        """Automatically start the CAN bus when using a `with` statement."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Automatically stop the CAN bus when exiting a `with` statement."""
        self.stop()

    def start(self) -> bool:
        """Start the CAN bus with standard 11-bit CAN IDs only."""
        try:
            self.bus = can.Bus(
                interface="socketcan",
                channel=self.interface,
                bitrate=self.bitrate,
                receive_own_messages=False,
                # Explicitly disable extended CAN IDs (29-bit)
                # Only use standard 11-bit CAN IDs (0x000 to 0x7FF)
                can_filters=None  # Accept all standard IDs
            )
            self.running = True
            print(f"CAN interface {self.interface} started at {self.bitrate} bps (standard 11-bit IDs only).")
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

    # Receive and ignore all messages that are waiting in the CAN RX buffer.
    def flush_rx(self) -> None:
        if not self.running or not self.bus:
            return
        while self.bus.recv(timeout=0) is not None:
            pass

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
                    
                # Only process standard 11-bit CAN IDs (ignore extended IDs)
                if message.is_extended_id:
                    # Skip extended CAN messages
                    continue
                
                # Convert python-can message to our CANMessage format
                can_msg = CANMessage(
                    can_id=message.arbitration_id,
                    data=bytes(message.data),
                    timestamp=message.timestamp or time.time()
                )

                # Parse message based on its ID
                parsed = self.message_parser.parse(can_msg)
                # Store parsed data as attributes (using setattr for compatibility)
                setattr(can_msg, 'message_type', parsed.get("message_type"))
                setattr(can_msg, 'parsed_data', parsed.get("parsed_data"))

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
        """Send a message on the CAN bus using standard 11-bit CAN IDs only."""
        if not self.running or not self.bus:
            return False

        if len(data) > 8:
            print(f"Warning: CAN data too long ({len(data)} bytes), truncating")
            data = data[:8]

        # Ensure CAN ID is within standard 11-bit range (0x000 to 0x7FF)
        if can_id > 0x7FF:
            print(f"Warning: CAN ID 0x{can_id:03X} exceeds 11-bit standard range, masking to 0x{can_id & 0x7FF:03X}")
            can_id = can_id & 0x7FF

        try:
            # Explicitly set is_extended_id=False to ensure standard 11-bit CAN IDs
            message = can.Message(
                arbitration_id=can_id,
                data=data,
                is_extended_id=False
            )
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
