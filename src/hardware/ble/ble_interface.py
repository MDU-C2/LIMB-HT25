

from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass
import time

@dataclass
class BLESample:
    """Represents a single BLE sample."""
    message_type: str
    data: dict
    timestamp: float

class BLEInterface(ABC):
    """Abstract base class for BLE interfaces"""

    @abstractmethod
    def start(self) -> bool:
        """Start the BLE interface and connect to peripheral"""
        pass

    @abstractmethod
    def stop(self) -> bool:
        """Stop the BLE interface and disconnect from peripheral"""
        pass

    @abstractmethod
    def read(self, timeout: Optional[float] = None) -> List[BLESample]:
        """Read samples from BLE (non-blocking, returns samples)."""
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """Check if the BLE interface is running"""
        pass

