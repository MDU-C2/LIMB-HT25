
import time
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

from shared.models.packet import (
    DataPacket,
    HumanDataWindow,
    SensorSnapshot,
    MotorState
)

"""
The packet builder should:
- Take window data from WindowBuffer
- Collect latest snapshots from sensors (vision, pressure, piezo, motors, emg, IMU)
- Construct a packet (DataPacket class) with all this information
- Handle timestamps and sequence IDs
"""


class PacketBuilder:
    """Builds packets from window buffer and latest sensor snapshots"""

    def __init__(self, sequence_start: int):
        self.sequence_id = sequence_start
        self.sensor_sources = {}            # Will be set by input layer
        self.motor_state_sources = None     # Will be set by input layer

    def set_sensor_sources(self, vision_source=None, pressure_source=None, piezo_source=None):

        """
        Set sensor sources for packet building.
        """
        self.sensor_sources = {
            "vision": vision_source,
            "pressure": pressure_source,
            "piezo": piezo_source
        }

    def set_motor_state_source(self, motor_source):

        """
        Set motor state source for packet building.
        """
        self.motor_state_sources = motor_source

    def build(self, window_buffer, sample_rate: float = 100.0) -> DataPacket:
        """Builds a complete packet from window buffer and latest sensor snapshots"""
        
        window_data = window_buffer.get_window() # Build human data window # TODO: Implement this function in window buffer
        human_data = self._build_human_data_window(window_data, sample_rate) # TODO: Implement this function
        sensor_snapshots = self.get_latest_sensors() # Get latest sensors snapshots # TODO: Implement this function
        motor_state = self.get_latest_motors()  # Get latest motor state # TODO: Implement this function
        
        # Build packet
        packet = DataPacket(
            sequence_id=self.sequence_id,
            timestamp=time.time(),
            packet_age_ms=0.0,
            human_data=human_data,
            sensor_snapshots=sensor_snapshots,
            motor_state=motor_state
            metadata={
                "sample_rate": sample_rate,
                "window_size": window_buffer.window_size,
                "build_time": time.time()
            }
        )
        self.sequence_id += 1 # Increment the sequence ID
        return packet


    def _build_human_data_window(self, window_data, sample_rate: float) -> HumanDataWindow:
        pass

    # Decide how self.moto_state_source looks like and then can remove some code
    def _get_latest_motors(self) -> MotorState:
        """
        Get latest motor state.
        """
        if self.motor_state_sources is None:
            return None

        try:
            # Get motor state from source
            if hasattr(self.motor_state_sources, "get_state"):
                state = self.motor_state_sources.get_state()
            elif callable(self.motor_state_sources):
                state = self.motor_state_sources()
            else:
                state = self.motor_state_source

            if isinstance(state, MotorState):
                return state
            elif isinstance(state, dict):
                return MotorState(
                    joint_positions=np.array(state.get("joint_positions", [])),
                    joint_velocities=np.array(state.get("joint_velocities", [])),
                    gripper_state=state.get("gripper_state", {}),
                    timestamp=state.get("timestamp", time.time())
            else:
            # Default empty state
            return MotorState(
                joint_positions=np.array([]),
                joint_velocities=np.array([]),
                gripper_state={},
                timestamp=time.time()
            )
        except Exception as e:
            print(f"Warning: Failed to get motor state: {e}")
            return None

    def _get_latest_sensors(self) -> SensorSnapshot:
        """
        Get latest sensors snapshots.
        """
        snapshot = SensorSnapshot()
        snapshot.timestamp = time.time()

        # Vision data
        if self.sensor_sources.get("vision"):
            snapshot.vision = self._get_vision_data()
        
        # Pressure sensor
        if self.sensor_sources.get("pressure"):
            snapshot.pressure = self._get_pressure_data()
        
        # Piezo sensor
        if self.sensor_sources.get("piezo"):
            snapshot.piezo = self._get_piezo_data()

        return snapshot

    def _get_vision_data(self) -> Dict[str, Any]:
        """
        Get the latest vision data (cup detections and AprilTag detections).
        """
        pass
    def _get_pressure_data(self) -> float:
        """
        Get the latest pressure data.
        """
        #TODO: Implement for each finger (and palm?)
        pass

    def _get_piezo_data(self) -> float:
        """
        Get the latest piezo data. 
        """
        pass