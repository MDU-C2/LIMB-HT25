"""
EMG Reader - Python interface to EMG sensor

This module provides a clean Python interface to EMG sensors.
It handles serial communication, data parsing, and provides data in the format expected
by the fusion system and LSTM models.
"""

import serial
import serial.tools.list_ports
import json
import numpy as np
import time
import threading
from typing import Optional, Dict, Any, List
from queue import Queue, Empty
from dataclasses import dataclass


@dataclass
class EMGData:
    """EMG data structure."""
    channels: np.ndarray  # Shape: (n_channels,) - EMG signal from multiple channels
    timestamp: float  # seconds
    sampling_rate: Optional[float] = None  # Hz


class EMGReader:
    """Python interface to EMG sensor via serial communication."""
    
    def __init__(
        self,
        port: Optional[str] = None,
        baudrate: int = 115200,
        n_channels: int = 8
    ):
        """
        Initialize EMG reader.
        
        Args:
            port: Serial port (auto-detect if None)
            baudrate: Serial communication baudrate
            n_channels: Number of EMG channels to read
        """
        self.port = port
        self.baudrate = baudrate
        self.n_channels = n_channels
        self.serial_conn = None
        self.running = False
        self.data_queue = Queue(maxsize=100)
        self.read_thread = None
        
        # Statistics
        self.read_count = 0
        self.error_count = 0
        self.last_read_time = None
        self.boot_time = None
        
        self.EMGData = EMGData
       
        print(f"EMGReader initialized (port: {self.port or 'auto-detect'}, "
              f"baudrate: {self.baudrate}, channels: {self.n_channels})")
    
    def activate(self) -> bool:
        """Activate EMG sensor (connect to device)."""
        if self.running:
            return True
        
        # Auto-detect port if not specified
        if not self.port:
            self.port = self._auto_detect_emg()
            if not self.port:
                print("Error: Could not auto-detect EMG device")
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
            
            print(f"EMG activated on {self.port}")
            return True
            
        except serial.SerialException as e:
            print(f"Failed to activate EMG: {e}")
            return False
        except Exception as e:
            print(f"Error activating EMG: {e}")
            return False
    
    def deactivate(self):
        """Deactivate EMG sensor."""
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
        
        print("EMG deactivated")
    
    def get_latest_data(self) -> Optional[EMGData]:
        """Get latest EMG data (non-blocking)."""
        try:
            return self.data_queue.get_nowait()
        except Empty:
            return None
    
    def get_latest_data_blocking(self, timeout: float = 1.0) -> Optional[EMGData]:
        """Get latest EMG data (blocking with timeout)."""
        try:
            return self.data_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def _auto_detect_emg(self) -> Optional[str]:
        """Auto-detect EMG devices."""
        ports = serial.tools.list_ports.comports()
        
        for port in ports:
            # Look for common EMG device identifiers
            # Modify this based on your specific EMG hardware
            if any(identifier in port.description.lower() for identifier in 
                   ['emg', 'myo', 'arduino', 'usb serial']):
                print(f"Auto-detected EMG device: {port.device} ({port.description})")
                return port.device
            
            # Check by VID/PID if available (add your EMG device IDs here)
            if hasattr(port, 'vid') and hasattr(port, 'pid'):
                emg_devices = [
                    # Add your EMG device VID/PID here
                    # Example: (0x2341, 0x0043),  # Arduino Uno
                ]
                
                if (port.vid, port.pid) in emg_devices:
                    print(f"Auto-detected EMG device by VID/PID: {port.device}")
                    return port.device
        
        print("No EMG device auto-detected")
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
                print(f"Error in EMG read loop: {e}")
                self.error_count += 1
                time.sleep(0.1)
    
    def _process_line(self, line: str):
        """Process a single line of data."""
        try:
            # Skip log messages
            if line.startswith(('I (', 'W (', 'E (', 'D (')):
                return
            
            # Parse JSON format: {"channels": [ch1, ch2, ..., ch8], "timestamp": 123.456}
            if line.startswith('{'):
                data = json.loads(line)
                
                # Extract channel data
                channels = data.get('channels', [])
                if len(channels) != self.n_channels:
                    print(f"Warning: Expected {self.n_channels} channels, got {len(channels)}")
                    # Pad or truncate to match expected channels
                    if len(channels) < self.n_channels:
                        channels = channels + [0.0] * (self.n_channels - len(channels))
                    else:
                        channels = channels[:self.n_channels]
                
                # Convert to EMGData format
                emg_data = self.EMGData(
                    channels=np.array(channels, dtype=np.float64),
                    timestamp=data.get('timestamp', time.time()),
                    sampling_rate=data.get('sampling_rate', None)
                )
                
                # Add to queue (non-blocking)
                try:
                    self.data_queue.put_nowait(emg_data)
                except:
                    pass  # Queue full, skip this sample
                
                # Update statistics
                self.read_count += 1
                self.last_read_time = time.time()
            
            # Alternative format: comma-separated values (CSV)
            elif ',' in line:
                values = [float(v.strip()) for v in line.split(',')]
                
                if len(values) != self.n_channels:
                    # Pad or truncate
                    if len(values) < self.n_channels:
                        values = values + [0.0] * (self.n_channels - len(values))
                    else:
                        values = values[:self.n_channels]
                
                emg_data = self.EMGData(
                    channels=np.array(values, dtype=np.float64),
                    timestamp=time.time(),
                    sampling_rate=None
                )
                
                try:
                    self.data_queue.put_nowait(emg_data)
                except:
                    pass
                
                self.read_count += 1
                self.last_read_time = time.time()
            
        except json.JSONDecodeError:
            # Silently skip lines that aren't valid JSON
            pass
        except ValueError as e:
            # Skip lines that can't be parsed as numbers
            pass
        except Exception as e:
            print(f"Error processing EMG line: {e}")
            self.error_count += 1
    
    def get_status(self) -> Dict[str, Any]:
        """Get EMG reader status."""
        return {
            'running': self.running,
            'port': self.port,
            'baudrate': self.baudrate,
            'n_channels': self.n_channels,
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
        print("EMG reader cleanup completed")


# Simulated EMG reader for testing without hardware
class SimulatedEMGReader(EMGReader):
    """Simulated EMG reader for testing without hardware."""
    
    def __init__(self, n_channels: int = 8, sampling_rate: float = 1000.0):
        """
        Initialize simulated EMG reader.
        
        Args:
            n_channels: Number of EMG channels to simulate
            sampling_rate: Simulated sampling rate in Hz
        """
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.running = False
        self.data_queue = Queue(maxsize=100)
        self.read_thread = None
        
        self.read_count = 0
        self.error_count = 0
        self.last_read_time = None
        self.boot_time = None
        
        self.EMGData = EMGData
        
        print(f"SimulatedEMGReader initialized (channels: {n_channels}, rate: {sampling_rate} Hz)")
    
    def activate(self) -> bool:
        """Activate simulated EMG sensor."""
        if self.running:
            return True
        
        self.running = True
        self.boot_time = time.time()
        self.read_thread = threading.Thread(target=self._simulate_loop)
        self.read_thread.daemon = True
        self.read_thread.start()
        
        print("Simulated EMG activated")
        return True
    
    def _simulate_loop(self):
        """Simulate EMG data generation."""
        sample_period = 1.0 / self.sampling_rate
        
        while self.running:
            try:
                # Generate simulated EMG data (random noise + some patterns)
                t = time.time() - self.boot_time
                
                # Simulate muscle activation patterns
                channels = []
                for i in range(self.n_channels):
                    # Base noise
                    noise = np.random.randn() * 0.1
                    
                    # Add some sinusoidal patterns (simulating muscle contractions)
                    pattern = 0.3 * np.sin(2 * np.pi * 0.5 * t + i * np.pi / 4)
                    
                    channels.append(noise + pattern)
                
                emg_data = self.EMGData(
                    channels=np.array(channels, dtype=np.float64),
                    timestamp=time.time(),
                    sampling_rate=self.sampling_rate
                )
                
                try:
                    self.data_queue.put_nowait(emg_data)
                except:
                    pass  # Queue full
                
                self.read_count += 1
                self.last_read_time = time.time()
                
                time.sleep(sample_period)
                
            except Exception as e:
                print(f"Error in simulated EMG loop: {e}")
                self.error_count += 1
                time.sleep(0.1)


if __name__ == "__main__":
    print("Testing SimulatedEMGReader...")
    
    # Create simulated EMG reader
    emg_reader = SimulatedEMGReader(n_channels=8, sampling_rate=100.0)
    
    if emg_reader.activate():
        print("Reading simulated EMG data for 3 seconds...")
        
        start_time = time.time()
        sample_count = 0
        
        while time.time() - start_time < 3.0:
            emg_data = emg_reader.get_latest_data()
            
            if emg_data:
                sample_count += 1
                if sample_count % 10 == 0:  # Print every 10th sample
                    print(f"EMG Sample {sample_count}: channels={emg_data.channels[:3]}... "
                          f"(showing first 3 of {len(emg_data.channels)})")
            
            time.sleep(0.01)
        
        print(f"\nTotal samples read: {sample_count}")
        print(f"Status: {emg_reader.get_status()}")
        
        emg_reader.deactivate()
        print("Test completed")
    else:
        print("Failed to activate EMG reader")
