"""
IMU data structures for the robotic arm system.

This module defines data structures for IMU readings from 4 IMUs:
- 2 IMUs on the robotic arm (forearm and upper arm)
- 2 IMUs on the EMG band/cuff (forearm and upper arm)
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from enum import Enum


class IMULocation(Enum):
    """Enumeration of IMU locations in the system."""
    ROBOT_FOREARM = "robot_forearm"
    ROBOT_UPPERARM = "robot_upperarm"
    EMG_FOREARM = "emg_forearm"
    EMG_UPPERARM = "emg_upperarm"


@dataclass
class IMUReading:
    """Raw IMU sensor reading containing accelerometer, gyroscope, and magnetometer data."""
    
    # Accelerometer data (m/s²)
    accel_x: float
    accel_y: float
    accel_z: float
    
    # Gyroscope data (rad/s)
    gyro_x: float
    gyro_y: float
    gyro_z: float
    
    # Magnetometer data (μT)
    mag_x: float
    mag_y: float
    mag_z: float
    
    # Timestamp (seconds since epoch)
    timestamp: float
    
    # IMU location identifier
    location: IMULocation
    
    # Temperature (optional, in Celsius)
    temperature: Optional[float] = None
    
    def to_numpy(self) -> np.ndarray:
        """Convert IMU reading to numpy array [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z]."""
        return np.array([
            self.accel_x, self.accel_y, self.accel_z,
            self.gyro_x, self.gyro_y, self.gyro_z,
            self.mag_x, self.mag_y, self.mag_z
        ])
    
    @classmethod
    def from_numpy(cls, data: np.ndarray, timestamp: float, location: IMULocation, temperature: Optional[float] = None) -> 'IMUReading':
        """Create IMUReading from numpy array."""
        if len(data) != 9:
            raise ValueError("Data array must have 9 elements: [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z]")
        
        return cls(
            accel_x=data[0], accel_y=data[1], accel_z=data[2],
            gyro_x=data[3], gyro_y=data[4], gyro_z=data[5],
            mag_x=data[6], mag_y=data[7], mag_z=data[8],
            timestamp=timestamp,
            location=location,
            temperature=temperature
        )


@dataclass
class IMUCalibration:
    """Calibration parameters for an IMU sensor."""
    
    # Accelerometer bias (m/s²)
    accel_bias: np.ndarray  # [x, y, z]
    
    # Gyroscope bias (rad/s)
    gyro_bias: np.ndarray  # [x, y, z]
    
    # Magnetometer bias (μT)
    mag_bias: np.ndarray  # [x, y, z]
    
    # Accelerometer scale factors
    accel_scale: np.ndarray  # [x, y, z]
    
    # Gyroscope scale factors
    gyro_scale: np.ndarray  # [x, y, z]
    
    # Magnetometer scale factors
    mag_scale: np.ndarray  # [x, y, z]
    
    # Accelerometer noise variance
    accel_noise_var: float
    
    # Gyroscope noise variance
    gyro_noise_var: float
    
    # Magnetometer noise variance
    mag_noise_var: float
    
    def __post_init__(self):
        """Ensure all arrays are numpy arrays."""
        self.accel_bias = np.asarray(self.accel_bias)
        self.gyro_bias = np.asarray(self.gyro_bias)
        self.mag_bias = np.asarray(self.mag_bias)
        self.accel_scale = np.asarray(self.accel_scale)
        self.gyro_scale = np.asarray(self.gyro_scale)
        self.mag_scale = np.asarray(self.mag_scale)


@dataclass
class ProcessedIMUReading:
    """Processed IMU reading with calibrated and filtered data."""
    
    # Calibrated accelerometer data (m/s²)
    accel: np.ndarray  # [x, y, z]
    
    # Calibrated gyroscope data (rad/s)
    gyro: np.ndarray  # [x, y, z]
    
    # Calibrated magnetometer data (μT)
    mag: np.ndarray  # [x, y, z]
    
    # Timestamp (seconds since epoch)
    timestamp: float
    
    # IMU location identifier
    location: IMULocation
    
    # Temperature (optional, in Celsius)
    temperature: Optional[float] = None
    
    # Estimated orientation quaternion [w, x, y, z] (optional)
    orientation: Optional[np.ndarray] = None
    
    # Estimated angular velocity magnitude (rad/s)
    angular_velocity_magnitude: Optional[float] = None
    
    # Estimated linear acceleration magnitude (m/s²)
    linear_acceleration_magnitude: Optional[float] = None


@dataclass
class IMUDataCollection:
    """Collection of IMU readings from all 4 IMUs at a specific timestamp."""
    
    # Readings from all 4 IMUs
    robot_forearm: Optional[ProcessedIMUReading] = None
    robot_upperarm: Optional[ProcessedIMUReading] = None
    emg_forearm: Optional[ProcessedIMUReading] = None
    emg_upperarm: Optional[ProcessedIMUReading] = None
    
    # Collection timestamp
    timestamp: float = 0.0
    
    def get_reading(self, location: IMULocation) -> Optional[ProcessedIMUReading]:
        """Get reading for a specific IMU location."""
        if location == IMULocation.ROBOT_FOREARM:
            return self.robot_forearm
        elif location == IMULocation.ROBOT_UPPERARM:
            return self.robot_upperarm
        elif location == IMULocation.EMG_FOREARM:
            return self.emg_forearm
        elif location == IMULocation.EMG_UPPERARM:
            return self.emg_upperarm
        else:
            return None
    
    def set_reading(self, location: IMULocation, reading: ProcessedIMUReading) -> None:
        """Set reading for a specific IMU location."""
        if location == IMULocation.ROBOT_FOREARM:
            self.robot_forearm = reading
        elif location == IMULocation.ROBOT_UPPERARM:
            self.robot_upperarm = reading
        elif location == IMULocation.EMG_FOREARM:
            self.emg_forearm = reading
        elif location == IMULocation.EMG_UPPERARM:
            self.emg_upperarm = reading
    
    def is_complete(self) -> bool:
        """Check if all 4 IMU readings are available."""
        return all([
            self.robot_forearm is not None,
            self.robot_upperarm is not None,
            self.emg_forearm is not None,
            self.emg_upperarm is not None
        ])
    
    def get_all_readings(self) -> list[ProcessedIMUReading]:
        """Get all available readings as a list."""
        readings = []
        for location in IMULocation:
            reading = self.get_reading(location)
            if reading is not None:
                readings.append(reading)
        return readings
