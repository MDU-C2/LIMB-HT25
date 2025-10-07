from glob import iglob
import serial
import json
import numpy as np
from typing import Optional
import time

class IMUSerialReader:
    def __init__(self, port: str = None, baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.running = False

    def connect(self):
        """Connect to the serial port."""
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1.0)
            self.running = True
            print(f"Connected to {self.port} at {self.baudrate} baud")
            return True
        except Exception as e:
            print(f"Error connecting to {self.port}: {e}")
            return False

    def disconnect(self):
        """Disconnect from the serial port."""
        self.running = False
        if self.serial_conn:
            self.serial_conn.close()

    def read_imu_data(self):
        """Read IMU data from the serial port."""
        #print(self.running, self.serial_conn)
        if not self.running or not self.serial_conn:
            return None

        try:
            line = self.serial_conn.readline().decode('utf-8').strip()
            #print("line:", line)
            if line:
                data = json.loads(line)
                return data
        except Exception as e:
            print(f"Error reading IMU data: {e}")
        
        return None
            