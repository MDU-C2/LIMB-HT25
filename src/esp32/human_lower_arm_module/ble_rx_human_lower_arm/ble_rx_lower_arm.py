import asyncio
import struct
import numpy as np
from collections import deque
from bleak import BleakScanner, BleakClient

# --- DEVICE CONFIGURATION ---
TARGET_NAME = "LIMBServer"

# BLE Characteristic UUIDs (Must match sensors_service.c in firmware)
EMG_CHAR_UUID   = "24011525-1212-efde-1523-785feabcd122"
PIEZO_CHAR_UUID = "26011525-1212-efde-1523-785feabcd122" 
IMU_CHAR_UUID   = "25011525-1212-efde-1523-785feabcd122"

class DataAssembler:
    """
    Handles the synchronization and windowing of disparate sensor streams.
    Ensures that EMG (4kHz), Piezo (1kHz), and IMU (100Hz) data align correctly
    into discrete 100ms processing windows.
    """
    def __init__(self):
        # Temporary buffers to assemble a 100ms window
        self.reset_temp_buffers()
        # History of the last 5 complete windows (rolling 500ms context for AI)
        self.history = deque(maxlen=5)
        self.window_count = 0

    def reset_temp_buffers(self):
        self.temp_emg1 = []
        self.temp_emg2 = []
        self.temp_piezo = []
        self.temp_imu1 = []
        self.temp_imu2 = []

    def add_data(self, sensor_type, data1, data2=None):
        """Adds incoming micro-packet data to the corresponding temporary buffer."""
        if sensor_type == "EMG":
            self.temp_emg1.extend(data1)
            self.temp_emg2.extend(data2)
        elif sensor_type == "PIEZO":
            self.temp_piezo.extend(data1)
        elif sensor_type == "IMU":
            # Scale conversion: Firmware sends int16 (val * 1000)
            # We divide by 1000.0 to recover real physical units (G's and deg/s)
            imu1_real = [val / 1000.0 for val in data1]
            imu2_real = [val / 1000.0 for val in data2]
            
            self.temp_imu1.append(imu1_real) 
            self.temp_imu2.append(imu2_real)

        self.check_and_build()

    def check_and_build(self):
        """
        Validates if enough samples have arrived to complete a 100ms window.
        Expected counts: 
        - EMG: 40 samples * 10 packets = 400
        - PIEZO: 10 samples * 10 packets = 100
        - IMU: 1 sample * 10 packets = 10
        """
        if (len(self.temp_emg1) >= 400 and 
            len(self.temp_piezo) >= 100 and 
            len(self.temp_imu1) >= 10):
            
            # Create a structured window object with NumPy arrays 
            window = {
                "id": self.window_count,
                "emg1": np.array(self.temp_emg1[:400]),
                "emg2": np.array(self.temp_emg2[:400]),
                "piezo": np.array(self.temp_piezo[:100]),
                "imu1": np.array(self.temp_imu1[:10]),
                "imu2": np.array(self.temp_imu2[:10])
            }
            
            # Store window in rolling history
            self.history.append(window)
            self.window_count += 1
            self.reset_temp_buffers()
            
            # --- Diagnostic Output ---
            acc_z_real1 = window['imu1'][-1][2] # Last Z-axis accel from IMU1
            acc_z_real2 = window['imu2'][-1][2] # Last Z-axis accel from IMU2

            print(f"[ASSEMBLER] Window #{window['id']} assembled.")
            print(f"    -> AccZ Real1: {acc_z_real1:.3f} G | AccZ Real2: {acc_z_real2:.3f} G ")
            print(f"    -> EMG1 Mean: {np.mean(window['emg1']):.1f} | EMG2 Mean: {np.mean(window['emg2']):.1f} | Piezo Mean: {np.mean(window['piezo']):.1f}")

class LimbSensorClient:
    """
    BLE Client that manages connection and data reception from the LIMB Server.
    It unpacks raw binary packets and forwards them to the DataAssembler.
    """
    def __init__(self):
        self.client = None
        self.assembler = DataAssembler()

    # --- RAW DATA HANDLERS ---
    
    async def _handle_emg_raw(self, sender, data):
        # Format: < (Little Endian), H (Header), I (Seq), Q (Timestamp), 80H (EMG samples)
        fmt = '<H I Q 80H'
        unpacked = struct.unpack(fmt, data)
        samples = unpacked[3:]
        # Split interleaved EMG1 and EMG2 data
        self.assembler.add_data("EMG", samples[:40], samples[40:])

    async def _handle_piezo_raw(self, sender, data):
        # Format: <, H, I, Q, 10H (Piezo samples)
        fmt = '<H I Q 10H'
        unpacked = struct.unpack(fmt, data)
        samples = unpacked[3:]
        self.assembler.add_data("PIEZO", samples)

    async def _handle_imu_raw(self, sender, data):
        # Format: <, H, I, Q, 12h (signed int16 axes for IMU1 and IMU2)
        fmt = '<H I Q 12h'
        unpacked = struct.unpack(fmt, data)
        imu1_axes = unpacked[3:9]  # (ax, ay, az, gx, gy, gz)
        imu2_axes = unpacked[9:15]
        self.assembler.add_data("IMU", imu1_axes, imu2_axes)

    # --- LIFECYCLE METHODS ---

    async def start(self):
        """Scans, connects, and subscribes to sensor characteristics."""
        print(f"Scanning for '{TARGET_NAME}'...")
        device = await BleakScanner.find_device_by_name(TARGET_NAME)
        if not device:
            print("Device not found.")
            return

        async with BleakClient(device) as client:
            print(f"Connected to {device.address}")
            # Subscribe to notifications for all data streams
            await client.start_notify(EMG_CHAR_UUID, self._handle_emg_raw)
            await client.start_notify(PIEZO_CHAR_UUID, self._handle_piezo_raw)
            await client.start_notify(IMU_CHAR_UUID, self._handle_imu_raw)

            try:
                # Keep the connection alive
                while True:
                    await asyncio.sleep(1.0)
            except KeyboardInterrupt:
                print("Disconnecting...")

if __name__ == "__main__":
    receiver = LimbSensorClient()
    try:
        asyncio.run(receiver.start())
    except KeyboardInterrupt:
        print("\nReceiver stopped by user.")