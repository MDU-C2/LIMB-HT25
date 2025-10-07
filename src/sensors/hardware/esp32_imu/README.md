# ESP32 IMU Firmware

This directory contains the ESP32-C3 firmware for reading LSM6DSO32 IMU data and sending it over serial (UART) as JSON.

## Directory Structure

```
esp32_imu/
├── main/
│   ├── main.c              # Main IMU reading code
│   ├── CMakeLists.txt      # Component CMakeLists
│   └── Kconfig.projbuild   # Project configuration
├── CMakeLists.txt          # Project CMakeLists
├── sdkconfig              # ESP-IDF configuration
├── deploy.sh              # Build and flash script
└── build/                 # Build output (auto-generated)
```

## Hardware Configuration

- **Microcontroller**: ESP32-C3
- **IMU Sensor**: LSM6DSO32 (6-axis accelerometer + gyroscope)
- **Interface**: I2C
  - SCL: GPIO 5
  - SDA: GPIO 4
  - Address: 0x6A (SDO=0)
- **Serial Output**: UART0 (default console)
  - Baud rate: 115200
  - Format: JSON

## Build and Flash

### Quick Deploy
```bash
./deploy.sh
```

### Manual Build
```bash
# Set ESP-IDF environment
. $HOME/esp/esp-idf/export.sh

# Build
idf.py build

# Flash
idf.py -p /dev/cu.usbmodem1101 flash

# Monitor
idf.py -p /dev/cu.usbmodem1101 monitor
```

## Output Format

The firmware outputs JSON data over serial at ~100Hz:

```json
{
  "timestamp": 1234567890,
  "accelerometer": [0.12, -0.05, 9.81],
  "gyroscope": [0.01, -0.02, 0.00],
  "temperature": 25.5
}
```

## Integration with Python

The Python `IMUReader` class in `src/sensors/imu_reader.py` reads this serial data and converts it to `IMUData` objects for use in the robot control system.

## Notes

- All ESP-IDF logs are disabled to ensure clean JSON output
- The firmware uses the legacy I2C driver for compatibility
- Temperature readings are from the IMU's internal sensor

