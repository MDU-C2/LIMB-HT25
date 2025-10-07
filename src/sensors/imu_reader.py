"""
IMU Reader - Python interface to ESP32 IMU sensor

This module provides a clean Python interface to the ESP32 running the LSM6DSO32 IMU code.
It handles serial communication, data parsing, and provides data in the format expected
by the fusion system.
"""

import serial
import serial.tools.list_ports
import json
import numpy as np
import time
import threading
from typing import Optional, Dict, Any
from queue import Queue, Empty
from dataclasses import dataclass

@dataclass
class IMUData:
    """IMU data structure."""
    angular_velocity: np.ndarray  # [wx, wy, wz] in rad/s
    linear_acceleration: np.ndarray  # [ax, ay, az] in m/s²
    timestamp: float  # seconds

class IMUReader:
    """Python interface to ESP32 IMU sensor via serial communication."""
    
    def __init__(self, port: Optional[str] = None, baudrate: int = 115200):
        """
        Initialize IMU reader.
        
        Args:
            port: Serial port (auto-detect if None)
            baudrate: Serial communication baudrate
        """
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.running = False
        self.data_queue = Queue(maxsize=100)
        self.read_thread = None
        
        # Statistics
        self.read_count = 0
        self.error_count = 0
        self.last_read_time = None
        self.boot_time = None
        
        # IMUData is a class, not an instance
        self.IMUData = IMUData
       
        print(f"IMUReader initialized (port: {self.port or 'auto-detect'}, baudrate: {self.baudrate})")
    
    def activate(self) -> bool:
        """Activate IMU sensor (connect to ESP32)."""
        if self.running:
            return True
        
        # Auto-detect port if not specified
        if not self.port:
            self.port = self._auto_detect_esp32()
            if not self.port:
                print("Error: Could not auto-detect ESP32 device")
                return False
        
        try:
            # Open serial connection
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1.0,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS
            )
            
            # Start reading thread
            self.running = True
            self.boot_time = time.time()
            self.read_thread = threading.Thread(target=self._read_loop)
            self.read_thread.daemon = True
            self.read_thread.start()
            
            print(f"IMU activated on {self.port}")
            return True
            
        except serial.SerialException as e:
            print(f"Failed to activate IMU: {e}")
            return False
        except Exception as e:
            print(f"Error activating IMU: {e}")
            return False
    
    def deactivate(self):
        """Deactivate IMU sensor."""
        if not self.running:
            return
        
        self.running = False
        
        # Wait for read thread to finish
        if self.read_thread:
            self.read_thread.join(timeout=2.0)
        
        # Close serial connection
        if self.serial_conn:
            self.serial_conn.close()
            self.serial_conn = None
        
        print("IMU deactivated")
    
    def get_latest_data(self) -> Optional[Any]:
        """Get latest IMU data (non-blocking)."""
        try:
            return self.data_queue.get_nowait()
        except Empty:
            return None
    
    def get_latest_data_blocking(self, timeout: float = 1.0) -> Optional[Any]:
        """Get latest IMU data (blocking with timeout)."""
        try:
            return self.data_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def _auto_detect_esp32(self) -> Optional[str]:
        """Auto-detect ESP32 devices."""
        ports = serial.tools.list_ports.comports()
        
        for port in ports:
            # Look for common ESP32 identifiers
            if any(identifier in port.description.lower() for identifier in 
                   ['cp210', 'ch340', 'ftdi', 'usb serial', 'esp32']):
                print(f"Auto-detected ESP32 device: {port.device} ({port.description})")
                return port.device
            
            # Check by VID/PID if available
            if hasattr(port, 'vid') and hasattr(port, 'pid'):
                esp32_chips = [
                    (0x10C4, 0xEA60),  # Silicon Labs CP210x
                    (0x1A86, 0x7523),  # QinHeng Electronics CH340
                    (0x0403, 0x6001),  # FTDI FT232
                    (0x303A, 0x1001),  # Espressif ESP32-S2
                    (0x303A, 0x0001),  # Espressif ESP32
                ]
                
                if (port.vid, port.pid) in esp32_chips:
                    print(f"Auto-detected ESP32 device by VID/PID: {port.device}")
                    return port.device
        
        print("No ESP32 device auto-detected")
        return None
    
    def _read_loop(self):
        """Main reading loop (runs in separate thread)."""
        buffer = ""
        
        while self.running and self.serial_conn:
            try:
                # Read data from serial
                if self.serial_conn.in_waiting > 0:
                    data = self.serial_conn.read(self.serial_conn.in_waiting).decode('utf-8', errors='ignore')
                    buffer += data
                    
                    # Process complete lines
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        
                        if line:
                            self._process_line(line)
                
                time.sleep(0.001)  # Small delay to prevent busy waiting
                
            except serial.SerialException as e:
                print(f"Serial error: {e}")
                self.error_count += 1
                time.sleep(0.1)
            except Exception as e:
                print(f"Error in IMU read loop: {e}")
                self.error_count += 1
                time.sleep(0.1)
    
    def _process_line(self, line: str):
        """Process a single line of data."""
        try:
            # Skip ESP-IDF log messages (format: "I (timestamp) TAG: message")
            if line.startswith(('I (', 'W (', 'E (', 'D (')):
                return  # Skip log lines
            
            # Only parse lines that look like JSON
            if line.startswith('{'):
                data = json.loads(line)
                
                # Convert to IMUData format if available
                imu_data = self.IMUData(
                    angular_velocity=np.array([
                        data.get('gyro', {}).get('x', 0.0),
                        data.get('gyro', {}).get('y', 0.0),
                        data.get('gyro', {}).get('z', 0.0)
                    ], dtype=np.float64),
                    linear_acceleration=np.array([
                        data.get('accel', {}).get('x', 0.0),
                        data.get('accel', {}).get('y', 0.0),
                        data.get('accel', {}).get('z', 0.0)
                    ], dtype=np.float64),
                    timestamp=data.get('timestamp', time.time())
                )
                
                # Add to queue (non-blocking)
                try:
                    self.data_queue.put_nowait(imu_data)
                except:
                    pass  # Queue full, skip this sample
                
                # Update statistics
                self.read_count += 1
                self.last_read_time = time.time()
            
        except json.JSONDecodeError:
            # Silently skip lines that aren't valid JSON
            pass
        except Exception as e:
            print(f"Error processing IMU line: {e}")
            self.error_count += 1
    
    def get_status(self) -> Dict[str, Any]:
        """Get IMU reader status."""
        return {
            'running': self.running,
            'port': self.port,
            'baudrate': self.baudrate,
            'read_count': self.read_count,
            'error_count': self.error_count,
            'queue_size': self.data_queue.qsize(),
            'last_read_time': self.last_read_time,
            'uptime': time.time() - (self.boot_time or time.time()) if self.boot_time else 0,
            'data_rate': self.read_count / max(1, time.time() - (self.boot_time or time.time())) if self.boot_time else 0
        }
    
    def reset_statistics(self):
        """Reset reading statistics."""
        self.read_count = 0
        self.error_count = 0
        self.last_read_time = None
        self.boot_time = time.time()
    
    def cleanup(self):
        """Cleanup resources."""
        self.deactivate()
        print("IMU reader cleanup completed")
