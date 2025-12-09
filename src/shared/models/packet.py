

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
import time

@dataclass
class MotorState:
    """
    Current state of the motors/actuators

    This is the "feedback" from the robot. It describes where the motors are now (and not where we want them to be).
    """
    joint_positions: np.ndarray # Shape: (5,) - 5 joint angles in radians
    gripper_state: Dict[str, Any] # {"open": bool, "force": float} - Whether the gripper is open and the force applied
    timestamp: float = field(default_factory=time.time)

@dataclass
class HumanDataWindow:
    """
    Time-series Window for human data (EMG/IMU/Piezo)
    """
    emg: np.ndarray         # Shape: (window_size, num_channels)
    imu: np.ndarray         # Shape: (window_size, 6)
    piezo: np.ndarray       # Shape: (window_size,) - piezo sensor values
    timestamp_start: float  # When window started (seconds since epoch)
    timestamp_end: float    # When window ended
    sample_rate: float      # In Hz

@dataclass
class SensorSnapshot:
    """
    Latest snapshot of sensor states (single readings, not time-series)

    These are "point-in-time" readings. The most recent value from each sensor at the moment the packet was created.
    """
    vision: Optional[Dict[str, Any]] = None                 # TODO: Define what vision data looks like
    pressure: Optional[List[float]] = None                  # 5 values: [thumb, index, middle, ring, little]
    timestamp: float = field(default_factory=time.time)     # When snapshot was taken

@dataclass
class DataPacket:
    """
    Complete packet that flows between layers

    This is the structure that carries all data from input -> processing -> output.
    """

    # Header information
    sequence_id: int = 0                                # Unique packet ID (increments with each packet)
    timestamp: float = field(default_factory=time.time) # When packet was created
    packet_age_ms: float = 0.0                          # How old oacket is (updated when received)

    # Core data
    human_data: Optional[HumanDataWindow] = None                                # Time-series Window (EMG/IMU/Piezo)
    sensors: Optional[SensorSnapshot] = field(default_factory=SensorSnapshot)   # Latest sensor readings
    motors: Optional[MotorState] = None                                         # Current motor states

    # Meta data
    metadata: Dict[str, Any] = field(default_factory=dict)


    # ----- Methods -----
    def is_stale(self, max_age_ms: float = 100.0) -> bool:
        """Check if packet is too old for real-time processing."""
        return self.packet_age_ms > max_age_ms

    def update_age(self):
        """Update the packet age based on the current time."""
        self.packet_age_ms = (time.time() - self.timestamp) * 1000.0 # Convert to milliseconds