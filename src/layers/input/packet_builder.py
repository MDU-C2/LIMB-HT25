
import time
import numpy as np
from typing import Optional, Dict, Any, List
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

    def __init__(self, sequence_start: int, vision_source=None):
        """
        Initialize packet builder.
        
        Args:
            sequence_start: Starting sequence ID for packets
            vision_source: Vision system object (optional) - only sensor that needs separate interface
        """
        self.sequence_id = sequence_start
        self.vision_source = vision_source

    def build(self, 
              window_buffer, 
              sample_rate: float = 100.0,
              latest_pressure: Optional[List[float]] = None,
              latest_motor_state: Optional[MotorState] = None,
              latest_robot_imu: Optional[Dict[str, np.ndarray]] = None) -> DataPacket:
        """
        Builds a complete packet from window buffer and latest sensor snapshots.
        
        Args:
            window_buffer: WindowBuffer object with EMG/IMU/piezo data
            sample_rate: Sample rate in Hz
            latest_pressure: Latest pressure readings [thumb, index, middle, ring, pinky] (from CAN)
            latest_motor_state: Latest motor state (from CAN)
        
        Returns:
            DataPacket with all sensor data
        """
        
        window_data = window_buffer.get_window()
        human_data = self._build_human_data_window(window_data, sample_rate)
        sensor_snapshot = self._build_sensor_snapshot(
            latest_pressure=latest_pressure,
            latest_robot_imu=latest_robot_imu
        )
        
        # Build packet
        packet = DataPacket(
            sequence_id=self.sequence_id,
            timestamp=time.time(),
            packet_age_ms=0.0,
            human_data=human_data,
            sensors=sensor_snapshot,
            motors=latest_motor_state,
            metadata={
                "sample_rate": sample_rate,
                "window_size": window_buffer.window_size,
                "build_time": time.time()
            }
        )
        self.sequence_id += 1  # Increment the sequence ID
        return packet


    def _build_human_data_window(self, window_data, sample_rate: float) -> HumanDataWindow:
        """
        Build HumanDataWindow from window buffer data.
        
        Extracts EMG, IMU, and piezo arrays, timestamps, and creates a HumanDataWindow object
        with proper shapes matching the EMG processing pipeline expectations.
        """
        # Extract EMG, IMU, and piezo arrays
        emg_data = window_data.get('emg')
        imu_data = window_data.get('imu')
        piezo_data = window_data.get('piezo')
        
        # Ensure they are numpy arrays
        if not isinstance(emg_data, np.ndarray):
            emg_data = np.array(emg_data)
        if not isinstance(imu_data, np.ndarray):
            imu_data = np.array(imu_data)
        if not isinstance(piezo_data, np.ndarray):
            piezo_data = np.array(piezo_data)
        
        # Extract timestamps
        timestamps = window_data.get('timestamps')
        timestamp_start = window_data.get('timestamp_start')
        timestamp_end = window_data.get('timestamp_end')
        
        # Validate and ensure proper shapes
        # EMG should be (window_size, num_channels)
        if emg_data.ndim == 1:
            # If 1D, reshape to (window_size, 1)
            emg_data = emg_data.reshape(-1, 1)
        
        # IMU should be (window_size, 6) - [ax, ay, az, wx, wy, wz]
        if imu_data.ndim == 1:
            # If 1D, tile it (shouldn't happen if window is full)
            imu_data = np.tile(imu_data, (len(timestamps) if timestamps is not None else 100, 1))
        elif imu_data.ndim == 2:
            # Ensure shape is (window_size, 6)
            if imu_data.shape[1] != 6:
                if imu_data.shape[1] < 6:
                    # Pad with zeros
                    padding = np.zeros((imu_data.shape[0], 6 - imu_data.shape[1]))
                    imu_data = np.hstack([imu_data, padding])
                else:
                    # Truncate to 6 values
                    imu_data = imu_data[:, :6]
        
        # Piezo should be (window_size,) - 1D array
        if piezo_data.ndim == 0:
            # Scalar value, convert to 1D array
            piezo_data = np.array([piezo_data])
        elif piezo_data.ndim == 2:
            # If 2D, flatten or take first column
            if piezo_data.shape[1] == 1:
                piezo_data = piezo_data.flatten()
            else:
                piezo_data = piezo_data[:, 0]
        
        # Ensure piezo has correct length
        if len(piezo_data) != emg_data.shape[0]:
            if len(piezo_data) > emg_data.shape[0]:
                piezo_data = piezo_data[:emg_data.shape[0]]
            else:
                # Pad with last value or zeros
                padding = np.full(emg_data.shape[0] - len(piezo_data), piezo_data[-1] if len(piezo_data) > 0 else 0.0)
                piezo_data = np.concatenate([piezo_data, padding])
        
        # Validate timestamps
        if timestamps is None or len(timestamps) == 0:
            # Fallback: calculate from sample rate
            current_time = time.time()
            window_duration = emg_data.shape[0] / sample_rate if emg_data.shape[0] > 0 else 0.0
            timestamp_start = current_time - window_duration
            timestamp_end = current_time
        else:
            # Use provided timestamps
            if timestamp_start is None:
                timestamp_start = timestamps[0] if len(timestamps) > 0 else time.time()
            if timestamp_end is None:
                timestamp_end = timestamps[-1] if len(timestamps) > 0 else time.time()
        
        # Create HumanDataWindow object
        human_data_window = HumanDataWindow(
            emg=emg_data,
            imu=imu_data,
            piezo=piezo_data,
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            sample_rate=sample_rate
        )
        
        return human_data_window

    def _build_sensor_snapshot(self,
                                latest_pressure: Optional[List[float]] = None,
                                latest_robot_imu: Optional[Dict[str, np.ndarray]] = None) -> SensorSnapshot:
        """
        Build sensor snapshot from direct values (from CAN).
        
        Args:
            latest_pressure: Latest pressure readings [thumb, index, middle, ring, pinky] (from CAN)
            latest_robot_imu: Latest robot IMU data (from CAN)
        
        Returns:
            SensorSnapshot object
        """
        snapshot = SensorSnapshot()
        snapshot.timestamp = time.time()

        # Vision data (only sensor that needs separate interface)
        snapshot.vision = self._get_vision_data()
        
        # Pressure sensor (from CAN)
        if latest_pressure is not None:
            # Validate it's a list with 5 values
            if isinstance(latest_pressure, list) and len(latest_pressure) == 5:
                snapshot.pressure = latest_pressure
            else:
                print(f"Warning: Invalid pressure data format. Expected list of 5 floats, got {type(latest_pressure)}")

        # Robot IMU (from CAN)
        if latest_robot_imu is not None:
            snapshot.robot_imu = latest_robot_imu

        return snapshot

    def _get_vision_data(self) -> Optional[Dict[str, Any]]:
        """
        Get the latest vision data (cup detections and AprilTag detections).
        
        Returns:
            Dictionary with:
                - 'cup_detections': List of cup detection objects
                - 'apriltag_pose': Dict with AprilTag pose data
            or None if vision_source not available
        """
        if self.vision_source is None:
            return None
        
        try:
            vision_data = {}
            
            # Get cup detections
            if hasattr(self.vision_source, 'latest_cup_detections'):
                vision_data['cup_detections'] = self.vision_source.latest_cup_detections
            elif hasattr(self.vision_source, 'get_latest_cup_detections'):
                vision_data['cup_detections'] = self.vision_source.get_latest_cup_detections()
            
            # Get AprilTag pose
            if hasattr(self.vision_source, 'get_latest_pose'):
                apriltag_pose = self.vision_source.get_latest_pose()
                if apriltag_pose:
                    vision_data['apriltag_pose'] = apriltag_pose
            
            return vision_data if vision_data else None
            
        except Exception as e:
            print(f"Warning: Failed to get vision data: {e}")
            return None