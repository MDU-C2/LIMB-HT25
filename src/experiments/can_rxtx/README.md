# CAN RX/TX with IMU

ESP32-C3 project for transmitting IMU data over CAN bus and receiving it on a second ESP32.

## Overview

This project consists of two ESP32-C3 applications:

- **`can_tx/`** - Transmitter: Reads IMU (LSM6DSO32) data and transmits it on CAN bus
- **`can_rx/`** - Receiver: Receives CAN frames and logs IMU data

The transmitter reads accelerometer and gyroscope data from an LSM6DSO32 sensor via I2C and sends it over CAN at 20 Hz (50ms period). The receiver listens for these frames and displays the parsed data.

## Hardware Requirements

### Transmitter (can_tx)
- ESP32-C3 development board
- LSM6DSO32 IMU sensor
- CAN transceiver (e.g., MCP2551, TJA1050)
- Pull-up resistors (2.2kΩ - 4.7kΩ) for I2C bus

### Receiver (can_rx)
- ESP32-C3 development board
- CAN transceiver (same type as transmitter)

### CAN Bus
- CAN_H and CAN_L wires connecting both devices
- 120Ω termination resistors at both ends of the bus
- Common ground (GND) connection

## Pin Configuration

### Transmitter (can_tx)

#### CAN/TWAI
- **TX Pin**: GPIO 5
- **RX Pin**: GPIO 4
- **Baudrate**: 500 kbit/s

#### I2C (LSM6DSO32)
- **SDA**: GPIO 7
- **SCL**: GPIO 6
- **Frequency**: 400 kHz
- **Address**: 0x6A (SDO=0)

### Receiver (can_rx)

#### CAN/TWAI
- **TX Pin**: GPIO 5
- **RX Pin**: GPIO 4
- **Baudrate**: 500 kbit/s

## CAN Frame Format

### IMU Data Frame (ID: 0x100)

**Payload**: 8 bytes (4 × int16 values, little-endian)

| Byte | Content | Description |
|------|---------|-------------|
| 0-1  | accel_x | Raw accelerometer X (int16) |
| 2-3  | accel_y | Raw accelerometer Y (int16) |
| 4-5  | gyro_x  | Raw gyroscope X (int16) |
| 6-7  | gyro_y  | Raw gyroscope Y (int16) |

**Scaling**:
- Accelerometer: Divide by 8192.0 for ±4g range (g units)
- Gyroscope: Divide by 16384.0 for ±250 dps range (rad/s)

### Control Frame (ID: 0x200)

**Payload**: Variable
- Byte 0 = 1: Recalibrate/zero command

## Building and Flashing

### Prerequisites

1. Install ESP-IDF v5.x or later
2. Source ESP-IDF environment:
   ```bash
   source ~/esp/esp-idf/export.sh
   ```

### Build and Flash Transmitter

```bash
cd can_tx
idf.py set-target esp32c3
idf.py build flash monitor
```

Or specify port manually:
```bash
idf.py build flash monitor -p /dev/cu.usbmodem101
```

### Build and Flash Receiver

```bash
cd can_rx
idf.py set-target esp32c3
idf.py build flash monitor
```

Or specify port manually:
```bash
idf.py build flash monitor -p /dev/cu.usbmodem114101
```

## Usage

1. **Flash both devices** with their respective firmware
2. **Connect CAN bus**:
   - Connect CAN_H to CAN_H between devices
   - Connect CAN_L to CAN_L between devices
   - Connect GND to GND
   - Add 120Ω termination resistors at both ends
3. **Power on both devices**
4. **Monitor output**:
   - Transmitter will log: `TX: accel_x=..., accel_y=..., gyro_x=..., gyro_y=...`
   - Receiver will log: `RX: ID=0x100, DLC=8, data=[...]` and `IMU Data: accel_x=..., accel_y=..., gyro_x=..., gyro_y=...`

## Project Structure

```
can_rxtx/
├── can_tx/              # Transmitter project
│   ├── CMakeLists.txt
│   ├── main/
│   │   ├── CMakeLists.txt
│   │   ├── main.c       # Main application code
│   │   ├── can_driver.c # CAN driver implementation
│   │   └── can_driver.h # CAN driver header
│   └── sdkconfig
├── can_rx/              # Receiver project
│   ├── CMakeLists.txt
│   ├── main/
│   │   ├── CMakeLists.txt
│   │   ├── main.c       # Main application code
│   │   ├── can_driver.c # CAN driver implementation
│   │   └── can_driver.h # CAN driver header
│   └── sdkconfig
└── README.md            # This file
```

## Configuration

### Changing CAN Baudrate

Edit in both `can_tx/main/main.c` and `can_rx/main/main.c`:
```c
#define CAN_BAUDRATE         500000  // Change to desired baudrate
```

Supported baudrates: 25000, 50000, 100000, 125000, 250000, 500000, 800000, 1000000

### Changing I2C Pins

Edit in `can_tx/main/main.c`:
```c
#define I2C_MASTER_SDA_IO     7  // Change SDA pin
#define I2C_MASTER_SCL_IO     6  // Change SCL pin
```

### Changing CAN Pins

Edit in both projects:
```c
#define CAN_TX_PIN           5  // Change TX pin
#define CAN_RX_PIN           4  // Change RX pin
```

### Changing IMU Update Rate

Edit in `can_tx/main/main.c`:
```c
#define IMU_TX_PERIOD_MS      50  // Change period (ms), e.g., 100 = 10 Hz
```

## Troubleshooting

### I2C Read Failures

If you see `Failed to read IMU data` errors:

1. **Check wiring**: Verify SDA/SCL connections and pull-up resistors
2. **Check I2C address**: LSM6DSO32 can be 0x6A (SDO=0) or 0x6B (SDO=1)
3. **Check power**: Ensure IMU is powered (3.3V) and has common ground
4. **Reduce I2C frequency**: Try 100 kHz instead of 400 kHz if issues persist

### CAN Communication Issues

If no frames are received:

1. **Check CAN bus wiring**: 
   - CAN_H to CAN_H
   - CAN_L to CAN_L
   - GND to GND
2. **Check termination**: 120Ω resistors at both ends
3. **Verify baudrate**: Must match on both devices
4. **Check transceivers**: Ensure both devices have working CAN transceivers
5. **Check pins**: Verify TX/RX pins are correct

### Port Busy Error

If you see `port is busy or doesn't exist`:

```bash
# Check what's using the port
lsof /dev/cu.usbmodem101

# Kill the process if needed
kill -9 $(lsof -t /dev/cu.usbmodem101)
```

### Finding Serial Ports

```bash
# List all USB serial ports
ls /dev/cu.usb*

# List all serial ports
ls /dev/cu.*
```

## CAN Driver API

The `can_driver` provides a simple API:

```c
// Initialize CAN bus
esp_err_t can_init(int tx_pin, int rx_pin, int baudrate);

// Send a CAN message
esp_err_t can_send(uint32_t id, const uint8_t *data, uint8_t len);

// Receive a CAN message (blocking with timeout)
esp_err_t can_receive(uint32_t *id, uint8_t *data, uint8_t *len, int timeout_ms);

// Deinitialize CAN
void can_deinit(void);
```

## Notes

- CAN frames are limited to 8 bytes, so IMU data is sent as 4 int16 values (8 bytes total)
- The transmitter sends data at 20 Hz (every 50ms)
- I2C timeout is set to 2000ms for reliability
- Both projects use ESP-IDF v5.x TWAI driver (legacy API)

## License

This is a development/testing project, not for production use.

