# BLE Receiver Implementation for NVIDIA Jetson AGX Orin

## Overview

The Lower ARM ESP32 sends fused sensor data via Bluetooth Low Energy (BLE) to the AGX Orin. The data packet is **64 bytes** and contains sensor readings from both Upper and Lower arms.

---

## BLE Data Structure (64 bytes)

```c
typedef struct {
    uint32_t timestamp;        // Milliseconds (4 bytes)

    // Upper ARM data (7 floats = 28 bytes)
    float upper_emg;
    float upper_imu_ax, upper_imu_ay, upper_imu_az;
    float upper_imu_gx, upper_imu_gy, upper_imu_gz;

    // Lower ARM data (8 floats = 32 bytes)
    float lower_emg;
    float lower_piezo;
    float lower_imu_ax, lower_imu_ay, lower_imu_az;
    float lower_imu_gx, lower_imu_gy, lower_imu_gz;
} BLE_FUSED_PACKET_t;  // Total: 64 bytes
```

---

## Python Implementation (AGX)

### 1. **Install Required Packages**

```bash
pip install bluepy pybluez
```

### 2. **Python BLE Receiver Code**

```python
#!/usr/bin/env python3

import struct
import threading
import time
from bluepy.btle import Peripheral, DefaultDelegate, UUID

class NotificationDelegate(DefaultDelegate):
    def __init__(self, on_data_callback):
        DefaultDelegate.__init__(self)
        self.on_data_callback = on_data_callback

    def handleNotification(self, cHandle, data):
        """Called when BLE notification is received"""
        self.on_data_callback(data)

class ProsthesisReceiver:
    def __init__(self, mac_address):
        """
        Initialize BLE receiver
        mac_address: MAC address of Lower ARM ESP32 (e.g., "AA:BB:CC:DD:EE:FF")
        """
        self.mac_address = mac_address
        self.device = None
        self.running = False

    def connect(self):
        """Connect to Lower ARM ESP32"""
        try:
            print(f"[BLE] Connecting to {self.mac_address}...")
            self.device = Peripheral(self.mac_address, "public")
            self.device.setDelegate(NotificationDelegate(self.on_notification))
            print("[BLE] ✓ Connected successfully!")
            return True
        except Exception as e:
            print(f"[BLE] ✗ Connection failed: {e}")
            return False

    def discover_services(self):
        """Discover available GATT services"""
        if not self.device:
            print("[BLE] Not connected")
            return

        print("[BLE] Discovering services...")
        services = self.device.getServices()
        for service in services:
            print(f"  Service: {service.uuid}")
            for char in service.getCharacteristics():
                print(f"    Characteristic: {char.uuid} (Handle: {char.valHandle})")

    def on_notification(self, data):
        """Parse incoming BLE notification"""
        if len(data) < 64:
            print(f"[BLE] Invalid packet size: {len(data)} bytes (expected 64)")
            return

        try:
            # Unpack the 64-byte packet: 1 uint32_t + 15 floats
            packet_data = struct.unpack('<I 7f 8f', data)

            timestamp = packet_data[0]

            # Upper ARM data (indices 1-7)
            upper_emg = packet_data[1]
            upper_imu_ax = packet_data[2]
            upper_imu_ay = packet_data[3]
            upper_imu_az = packet_data[4]
            upper_imu_gx = packet_data[5]
            upper_imu_gy = packet_data[6]
            upper_imu_gz = packet_data[7]

            # Lower ARM data (indices 8-15)
            lower_emg = packet_data[8]
            lower_piezo = packet_data[9]
            lower_imu_ax = packet_data[10]
            lower_imu_ay = packet_data[11]
            lower_imu_az = packet_data[12]
            lower_imu_gx = packet_data[13]
            lower_imu_gy = packet_data[14]
            lower_imu_gz = packet_data[15]

            # Print parsed data
            print(f"\n[BLE RX] ts={timestamp}ms")
            print(f"  Upper: EMG={upper_emg:.2f}, IMU=({upper_imu_ax:.2f}, {upper_imu_ay:.2f}, {upper_imu_az:.2f})")
            print(f"  Lower: EMG={lower_emg:.2f}, Piezo={lower_piezo:.2f}, IMU=({lower_imu_ax:.2f}, {lower_imu_ay:.2f}, {lower_imu_az:.2f})")

            # Use the data
            self.process_data(
                timestamp,
                upper_emg, (upper_imu_ax, upper_imu_ay, upper_imu_az), (upper_imu_gx, upper_imu_gy, upper_imu_gz),
                lower_emg, lower_piezo, (lower_imu_ax, lower_imu_ay, lower_imu_az), (lower_imu_gx, lower_imu_gy, lower_imu_gz)
            )

        except struct.error as e:
            print(f"[BLE] Unpack error: {e}")

    def process_data(self, timestamp, upper_emg, upper_accel, upper_gyro, lower_emg, lower_piezo, lower_accel, lower_gyro):
        """Process received sensor data"""
        # TODO: Implement your prosthesis control logic here
        pass

    def start_listening(self):
        """Start listening for BLE notifications"""
        if not self.device:
            print("[BLE] Not connected")
            return

        self.running = True
        print("[BLE] Listening for notifications...")

        try:
            while self.running:
                # Wait for notifications (blocking call with timeout)
                if self.device.waitForNotifications(0.5):
                    # Notification received and handled in callback
                    pass
        except KeyboardInterrupt:
            print("\n[BLE] Stopped by user")
        except Exception as e:
            print(f"[BLE] Error: {e}")

    def disconnect(self):
        """Disconnect from Lower ARM"""
        self.running = False
        if self.device:
            self.device.disconnect()
            print("[BLE] Disconnected")

# Main execution
if __name__ == "__main__":
    # Replace with your Lower ARM MAC address
    LOWER_ARM_MAC = "AA:BB:CC:DD:EE:FF"

    receiver = ProsthesisReceiver(LOWER_ARM_MAC)

    if receiver.connect():
        receiver.discover_services()
        receiver.start_listening()

    receiver.disconnect()
```

---

## Finding Lower ARM MAC Address

On AGX Orin:

```bash
# Scan for BLE devices
bluetoothctl scan on

# Output will show:
# [NEW] Device AA:BB:CC:DD:EE:FF Lower_ARM
```

---

## Testing

### 1. **Verify BLE Connection**

```bash
# On AGX
bluetoothctl
> scan on
> pair AA:BB:CC:DD:EE:FF
> connect AA:BB:CC:DD:EE:FF
```

### 2. **Run Python Receiver**

```bash
python3 ble_receiver.py
```

### 3. **Monitor Serial Output**

On Lower ARM ESP32:

```bash
idf.py -p /dev/ttyUSB0 monitor
```

You should see:

```
[LOWER_ARM] BLE Client connected (conn_id=1)
[LOWER_ARM] BLE TX: U_EMG=50.23 L_EMG=45.12 L_PIEZO=120.45 ts=12345
```

---

## Notes

- **Packet Rate**: ~100 Hz (every 10ms from acquisition_task)
- **Byte Order**: Little-endian (standard for Intel/ARM)
- **Timestamp**: Milliseconds since ESP32 startup
- **No ACK needed**: Uses BLE notifications (fire-and-forget)
- **Data Freshness**: Always receives latest data within 100ms

---

## Troubleshooting

| Issue             | Solution                                                      |
| ----------------- | ------------------------------------------------------------- |
| Connection fails  | Check MAC address, ensure ESP32 is powered on and advertising |
| No notifications  | Verify characteristic handle, check BLE permissions           |
| Packet corruption | Check cable quality, reduce distance, check MTU size          |
| Performance lag   | Increase task priority, reduce other BLE traffic              |
