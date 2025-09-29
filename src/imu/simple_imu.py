"""
Simplified IMU system for LSM6DSO32 (6 DOF) IMU.

This module provides a simplified interface for reading accelerometer and gyroscope
data from the LSM6DSO32 IMU, focusing on the core functionality needed for VIO.
"""

import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable
import logging


@dataclass
class IMUData:
    """Simple IMU data structure for LSM6DSO32 (6 DOF)."""
    
    # Accelerometer data (m/s²)
    accel: np.ndarray  # [x, y, z]
    
    # Gyroscope data (rad/s)
    gyro: np.ndarray   # [x, y, z]
    
    # Timestamp (seconds since epoch)
    timestamp: float
    
    def __post_init__(self):
        """Ensure arrays are numpy arrays."""
        self.accel = np.asarray(self.accel, dtype=np.float64)
        self.gyro = np.asarray(self.gyro, dtype=np.float64)


class SimpleIMUReader:
    """Simple IMU reader for LSM6DSO32 (6 DOF) IMU."""
    
    def __init__(self, use_simulation: bool = True, serial_port: Optional[str] = None):
        """
        Initialize simple IMU reader.
        
        Args:
            use_simulation: If True, use simulated data for testing
            serial_port: Serial port for real hardware (e.g., '/dev/ttyUSB0')
        """
        self.use_simulation = use_simulation
        self.serial_port = serial_port
        self.serial_conn = None
        self.logger = logging.getLogger(__name__)
        
        if not use_simulation and serial_port:
            self._init_serial_connection()
    
    def _init_serial_connection(self):
        """Initialize serial connection for real hardware."""
        try:
            import serial
            self.serial_conn = serial.Serial(
                port=self.serial_port,
                baudrate=115200,
                timeout=1.0
            )
            self.logger.info(f"Connected to IMU on {self.serial_port}")
        except Exception as e:
            self.logger.error(f"Failed to connect to IMU: {e}")
            self.serial_conn = None
    
    def read_data(self) -> Optional[IMUData]:
        """
        Read IMU data.
        
        Returns:
            IMUData object or None if reading failed
        """
        if self.use_simulation:
            return self._read_simulated_data()
        else:
            return self._read_serial_data()
    
    def _read_simulated_data(self) -> IMUData:
        """Generate simulated IMU data for testing."""
        current_time = time.time()
        
        # Simulate realistic IMU data with some motion
        t = current_time * 0.5  # Slow down time for simulation
        
        # Simulate arm movement patterns
        accel_x = 0.1 * np.sin(t) + np.random.normal(0, 0.01)
        accel_y = 0.1 * np.cos(t * 0.7) + np.random.normal(0, 0.01)
        accel_z = 9.81 + 0.05 * np.sin(t * 0.3) + np.random.normal(0, 0.01)  # Gravity + small variations
        
        gyro_x = 0.05 * np.cos(t) + np.random.normal(0, 0.001)
        gyro_y = 0.05 * np.sin(t * 0.8) + np.random.normal(0, 0.001)
        gyro_z = 0.05 * np.cos(t * 1.2) + np.random.normal(0, 0.001)
        
        return IMUData(
            accel=np.array([accel_x, accel_y, accel_z]),
            gyro=np.array([gyro_x, gyro_y, gyro_z]),
            timestamp=current_time
        )
    
    def _read_serial_data(self) -> Optional[IMUData]:
        """Read data from serial connection."""
        if not self.serial_conn or not self.serial_conn.is_open:
            return None
        
        try:
            # Read line from serial port
            line = self.serial_conn.readline().decode('utf-8').strip()
            if not line:
                return None
            
            # Parse CSV format: accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,timestamp
            parts = line.split(',')
            if len(parts) != 7:
                return None
            
            accel = np.array([float(parts[0]), float(parts[1]), float(parts[2])])
            gyro = np.array([float(parts[3]), float(parts[4]), float(parts[5])])
            timestamp = float(parts[6])
            
            return IMUData(accel=accel, gyro=gyro, timestamp=timestamp)
            
        except Exception as e:
            self.logger.error(f"Error reading IMU data: {e}")
            return None
    
    def is_connected(self) -> bool:
        """Check if IMU is connected."""
        if self.use_simulation:
            return True
        return self.serial_conn is not None and self.serial_conn.is_open
    
    def close(self):
        """Close IMU connection."""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            self.logger.info("Closed IMU connection")


class IMUDataBuffer:
    """Buffer for storing IMU data with time-based access."""
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize IMU data buffer.
        
        Args:
            max_size: Maximum number of data points to store
        """
        self.max_size = max_size
        self.data: list[IMUData] = []
        self.lock = threading.Lock()
    
    def add_data(self, imu_data: IMUData):
        """Add new IMU data to buffer."""
        with self.lock:
            self.data.append(imu_data)
            if len(self.data) > self.max_size:
                self.data.pop(0)  # Remove oldest data
    
    def get_data_in_range(self, start_time: float, end_time: float) -> list[IMUData]:
        """Get IMU data within time range."""
        with self.lock:
            return [data for data in self.data if start_time <= data.timestamp <= end_time]
    
    def get_latest_data(self) -> Optional[IMUData]:
        """Get the most recent IMU data."""
        with self.lock:
            return self.data[-1] if self.data else None
    
    def clear(self):
        """Clear all data from buffer."""
        with self.lock:
            self.data.clear()


class IMUManager:
    """Manager for IMU data collection and processing."""
    
    def __init__(self, use_simulation: bool = True, serial_port: Optional[str] = None):
        """
        Initialize IMU manager.
        
        Args:
            use_simulation: If True, use simulated data
            serial_port: Serial port for real hardware
        """
        self.imu_reader = SimpleIMUReader(use_simulation, serial_port)
        self.data_buffer = IMUDataBuffer()
        self.running = False
        self.data_callback: Optional[Callable[[IMUData], None]] = None
        self.reading_thread: Optional[threading.Thread] = None
        self.logger = logging.getLogger(__name__)
    
    def set_data_callback(self, callback: Callable[[IMUData], None]):
        """Set callback function for receiving IMU data."""
        self.data_callback = callback
    
    def start_reading(self, sample_rate: float = 100.0):
        """Start reading IMU data at specified sample rate."""
        if self.running:
            return
        
        self.running = True
        self.reading_thread = threading.Thread(
            target=self._reading_loop,
            args=(sample_rate,),
            daemon=True
        )
        self.reading_thread.start()
        self.logger.info("Started IMU reading thread")
    
    def stop_reading(self):
        """Stop reading IMU data."""
        self.running = False
        if self.reading_thread:
            self.reading_thread.join(timeout=2.0)
        self.logger.info("Stopped IMU reading thread")
    
    def _reading_loop(self, sample_rate: float):
        """Main reading loop."""
        sample_interval = 1.0 / sample_rate
        
        while self.running:
            start_time = time.time()
            
            # Read IMU data
            imu_data = self.imu_reader.read_data()
            if imu_data:
                # Add to buffer
                self.data_buffer.add_data(imu_data)
                
                # Call callback
                if self.data_callback:
                    self.data_callback(imu_data)
            
            # Maintain sample rate
            elapsed = time.time() - start_time
            sleep_time = max(0, sample_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def read_single_data(self) -> Optional[IMUData]:
        """Read a single IMU data point (synchronous)."""
        return self.imu_reader.read_data()
    
    def get_latest_data(self) -> Optional[IMUData]:
        """Get the most recent IMU data."""
        return self.data_buffer.get_latest_data()
    
    def get_data_in_range(self, start_time: float, end_time: float) -> list[IMUData]:
        """Get IMU data within time range."""
        return self.data_buffer.get_data_in_range(start_time, end_time)
    
    def is_connected(self) -> bool:
        """Check if IMU is connected."""
        return self.imu_reader.is_connected()
    
    def close(self):
        """Close IMU connection and stop reading."""
        self.stop_reading()
        self.imu_reader.close()
        self.logger.info("IMU manager closed")


# Import threading at the top level
import threading
