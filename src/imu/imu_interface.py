"""
IMU interface for reading data from 4 IMUs in the robotic arm system.

This module provides interfaces for communicating with IMUs via different protocols
(serial, I2C, simulation) and managing the 4 IMU sensors.
"""

import time
import threading
from abc import ABC, abstractmethod
from typing import Dict, Optional, Callable, List
import numpy as np
import serial
import logging

from .data_structures import IMUReading, IMULocation, ProcessedIMUReading, IMUDataCollection, IMUCalibration


class IMUProtocol(ABC):
    """Abstract base class for IMU communication protocols."""
    
    @abstractmethod
    def read_imu_data(self) -> Optional[IMUReading]:
        """Read raw IMU data from the sensor."""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if the IMU is connected and ready."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the connection to the IMU."""
        pass


class SerialIMUProtocol(IMUProtocol):
    """Serial communication protocol for IMU sensors."""
    
    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 1.0):
        """
        Initialize serial IMU protocol.
        
        Args:
            port: Serial port (e.g., '/dev/ttyUSB0' or 'COM3')
            baudrate: Baud rate for serial communication
            timeout: Timeout for serial operations
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_conn: Optional[serial.Serial] = None
        self._connect()
    
    def _connect(self) -> None:
        """Establish serial connection."""
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            logging.info(f"Connected to IMU on {self.port}")
        except Exception as e:
            logging.error(f"Failed to connect to IMU on {self.port}: {e}")
            self.serial_conn = None
    
    def read_imu_data(self) -> Optional[IMUReading]:
        """Read IMU data from serial connection."""
        if not self.is_connected():
            return None
        
        try:
            # Read line from serial port
            line = self.serial_conn.readline().decode('utf-8').strip()
            if not line:
                return None
            
            # Parse CSV format: accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,mag_x,mag_y,mag_z,timestamp
            parts = line.split(',')
            if len(parts) != 10:
                return None
            
            data = [float(x) for x in parts[:9]]
            timestamp = float(parts[9])
            
            return IMUReading(
                accel_x=data[0], accel_y=data[1], accel_z=data[2],
                gyro_x=data[3], gyro_y=data[4], gyro_z=data[5],
                mag_x=data[6], mag_y=data[7], mag_z=data[8],
                timestamp=timestamp,
                location=IMULocation.ROBOT_FOREARM  # Will be set by IMUManager
            )
        except Exception as e:
            logging.error(f"Error reading IMU data: {e}")
            return None
    
    def is_connected(self) -> bool:
        """Check if serial connection is active."""
        return self.serial_conn is not None and self.serial_conn.is_open
    
    def close(self) -> None:
        """Close serial connection."""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            logging.info(f"Closed IMU connection on {self.port}")


class SimulatedIMUProtocol(IMUProtocol):
    """Simulated IMU protocol for testing and development."""
    
    def __init__(self, location: IMULocation, noise_level: float = 0.1):
        """
        Initialize simulated IMU protocol.
        
        Args:
            location: IMU location for simulation
            noise_level: Noise level for simulated data
        """
        self.location = location
        self.noise_level = noise_level
        self.start_time = time.time()
        
        # Simulate different motion patterns for different IMU locations
        if location in [IMULocation.ROBOT_FOREARM, IMULocation.EMG_FOREARM]:
            # Forearm motion: more dynamic, smaller movements
            self.motion_amplitude = 0.5
            self.motion_frequency = 2.0
        else:
            # Upper arm motion: larger, slower movements
            self.motion_amplitude = 1.0
            self.motion_frequency = 1.0
    
    def read_imu_data(self) -> Optional[IMUReading]:
        """Generate simulated IMU data."""
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Generate sinusoidal motion patterns
        accel_x = self.motion_amplitude * np.sin(self.motion_frequency * elapsed) + np.random.normal(0, self.noise_level)
        accel_y = self.motion_amplitude * np.cos(self.motion_frequency * elapsed * 0.7) + np.random.normal(0, self.noise_level)
        accel_z = 9.81 + self.motion_amplitude * 0.3 * np.sin(self.motion_frequency * elapsed * 0.5) + np.random.normal(0, self.noise_level)
        
        gyro_x = self.motion_amplitude * 0.1 * np.cos(self.motion_frequency * elapsed) + np.random.normal(0, self.noise_level * 0.1)
        gyro_y = self.motion_amplitude * 0.1 * np.sin(self.motion_frequency * elapsed * 0.8) + np.random.normal(0, self.noise_level * 0.1)
        gyro_z = self.motion_amplitude * 0.1 * np.cos(self.motion_frequency * elapsed * 1.2) + np.random.normal(0, self.noise_level * 0.1)
        
        # Magnetometer pointing roughly north
        mag_x = 20.0 + np.random.normal(0, self.noise_level * 2)
        mag_y = 5.0 + np.random.normal(0, self.noise_level * 2)
        mag_z = 40.0 + np.random.normal(0, self.noise_level * 2)
        
        return IMUReading(
            accel_x=accel_x, accel_y=accel_y, accel_z=accel_z,
            gyro_x=gyro_x, gyro_y=gyro_y, gyro_z=gyro_z,
            mag_x=mag_x, mag_y=mag_y, mag_z=mag_z,
            timestamp=current_time,
            location=self.location,
            temperature=25.0 + np.random.normal(0, 1.0)
        )
    
    def is_connected(self) -> bool:
        """Simulated IMU is always connected."""
        return True
    
    def close(self) -> None:
        """No-op for simulated IMU."""
        pass


class IMUManager:
    """Manager for all 4 IMU sensors in the system."""
    
    def __init__(self, use_simulation: bool = True, serial_ports: Optional[Dict[IMULocation, str]] = None):
        """
        Initialize IMU manager.
        
        Args:
            use_simulation: If True, use simulated IMUs for testing
            serial_ports: Dictionary mapping IMU locations to serial ports
        """
        self.use_simulation = use_simulation
        self.serial_ports = serial_ports or {}
        self.protocols: Dict[IMULocation, IMUProtocol] = {}
        self.calibrations: Dict[IMULocation, IMUCalibration] = {}
        self.running = False
        self.data_callback: Optional[Callable[[IMUDataCollection], None]] = None
        self.reading_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
        # Initialize protocols for all 4 IMUs
        self._initialize_protocols()
        
        # Set up default calibrations
        self._setup_default_calibrations()
    
    def _initialize_protocols(self) -> None:
        """Initialize communication protocols for all IMUs."""
        for location in IMULocation:
            if self.use_simulation:
                self.protocols[location] = SimulatedIMUProtocol(location)
            else:
                port = self.serial_ports.get(location)
                if port:
                    self.protocols[location] = SerialIMUProtocol(port)
                else:
                    logging.warning(f"No serial port specified for {location}, using simulation")
                    self.protocols[location] = SimulatedIMUProtocol(location)
    
    def _setup_default_calibrations(self) -> None:
        """Set up default calibration parameters for all IMUs."""
        for location in IMULocation:
            self.calibrations[location] = IMUCalibration(
                accel_bias=np.zeros(3),
                gyro_bias=np.zeros(3),
                mag_bias=np.zeros(3),
                accel_scale=np.ones(3),
                gyro_scale=np.ones(3),
                mag_scale=np.ones(3),
                accel_noise_var=0.01,
                gyro_noise_var=0.001,
                mag_noise_var=0.1
            )
    
    def set_calibration(self, location: IMULocation, calibration: IMUCalibration) -> None:
        """Set calibration parameters for a specific IMU."""
        with self.lock:
            self.calibrations[location] = calibration
    
    def get_calibration(self, location: IMULocation) -> IMUCalibration:
        """Get calibration parameters for a specific IMU."""
        with self.lock:
            return self.calibrations[location]
    
    def process_reading(self, raw_reading: IMUReading) -> ProcessedIMUReading:
        """Process raw IMU reading with calibration and filtering."""
        calibration = self.get_calibration(raw_reading.location)
        
        # Apply calibration
        accel_raw = np.array([raw_reading.accel_x, raw_reading.accel_y, raw_reading.accel_z])
        gyro_raw = np.array([raw_reading.gyro_x, raw_reading.gyro_y, raw_reading.gyro_z])
        mag_raw = np.array([raw_reading.mag_x, raw_reading.mag_y, raw_reading.mag_z])
        
        accel_calibrated = calibration.accel_scale * (accel_raw - calibration.accel_bias)
        gyro_calibrated = calibration.gyro_scale * (gyro_raw - calibration.gyro_bias)
        mag_calibrated = calibration.mag_scale * (mag_raw - calibration.mag_bias)
        
        # Calculate derived quantities
        angular_velocity_magnitude = np.linalg.norm(gyro_calibrated)
        linear_acceleration_magnitude = np.linalg.norm(accel_calibrated)
        
        return ProcessedIMUReading(
            accel=accel_calibrated,
            gyro=gyro_calibrated,
            mag=mag_calibrated,
            timestamp=raw_reading.timestamp,
            location=raw_reading.location,
            temperature=raw_reading.temperature,
            angular_velocity_magnitude=angular_velocity_magnitude,
            linear_acceleration_magnitude=linear_acceleration_magnitude
        )
    
    def set_data_callback(self, callback: Callable[[IMUDataCollection], None]) -> None:
        """Set callback function for receiving IMU data collections."""
        self.data_callback = callback
    
    def start_reading(self, sample_rate: float = 100.0) -> None:
        """Start reading from all IMUs at specified sample rate."""
        if self.running:
            return
        
        self.running = True
        self.reading_thread = threading.Thread(
            target=self._reading_loop,
            args=(sample_rate,),
            daemon=True
        )
        self.reading_thread.start()
        logging.info("Started IMU reading thread")
    
    def stop_reading(self) -> None:
        """Stop reading from all IMUs."""
        self.running = False
        if self.reading_thread:
            self.reading_thread.join(timeout=2.0)
        logging.info("Stopped IMU reading thread")
    
    def _reading_loop(self, sample_rate: float) -> None:
        """Main reading loop for all IMUs."""
        sample_interval = 1.0 / sample_rate
        
        while self.running:
            start_time = time.time()
            
            # Collect readings from all IMUs
            data_collection = IMUDataCollection(timestamp=start_time)
            
            for location, protocol in self.protocols.items():
                if protocol.is_connected():
                    raw_reading = protocol.read_imu_data()
                    if raw_reading:
                        processed_reading = self.process_reading(raw_reading)
                        data_collection.set_reading(location, processed_reading)
            
            # Call callback if data is available
            if self.data_callback and data_collection.get_all_readings():
                self.data_callback(data_collection)
            
            # Maintain sample rate
            elapsed = time.time() - start_time
            sleep_time = max(0, sample_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def read_single_collection(self) -> IMUDataCollection:
        """Read a single data collection from all IMUs (synchronous)."""
        data_collection = IMUDataCollection(timestamp=time.time())
        
        for location, protocol in self.protocols.items():
            if protocol.is_connected():
                raw_reading = protocol.read_imu_data()
                if raw_reading:
                    processed_reading = self.process_reading(raw_reading)
                    data_collection.set_reading(location, processed_reading)
        
        return data_collection
    
    def is_connected(self, location: IMULocation) -> bool:
        """Check if a specific IMU is connected."""
        protocol = self.protocols.get(location)
        return protocol is not None and protocol.is_connected()
    
    def get_connection_status(self) -> Dict[IMULocation, bool]:
        """Get connection status for all IMUs."""
        return {location: self.is_connected(location) for location in IMULocation}
    
    def close(self) -> None:
        """Close all IMU connections."""
        self.stop_reading()
        for protocol in self.protocols.values():
            protocol.close()
        logging.info("Closed all IMU connections")
