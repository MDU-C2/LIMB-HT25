from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass
import time 


@dataclass
class CANMessage:
    """Represents a CAN message"""
    can_id: int
    data: bytes
    timestamp: float

class CANInterface(ABC):
    pass

    @abstractmethod
    def start(self) -> bool:
        """Start the CAN interface."""
        pass

    @abstractmethod
    def stop(self) -> bool:
        """Stop the CAN interface."""
        pass

    @abstractmethod
    def read(self, timeout: Optional[float] = None) -> List[CANMessage]:
        """Read messages from the CAN interface (non-blocking or with timeout)"""
        pass

    @abstractmethod
    def send(self, can_id: int, data: bytes) -> bool:
        """Send a message on the CAN interface."""
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """Check if the CAN interface is running."""
        pass
