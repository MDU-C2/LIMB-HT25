# Elbow Command Node

This is a simple CAN command node for testing the robot elbow module.

## Setup Instructions

### Hardware Requirements
- 2x ESP32 boards
- 2x CAN transceivers (e.g., MCP2551, SN65HVD230)
- CAN bus wiring (CANH, CANL, GND, 120Ω termination resistors)

### CAN Configuration
Both nodes must use the same configuration:
- **TX Pin**: 5
- **RX Pin**: 4
- **Baudrate**: 125000

### Building and Flashing

#### Node 1: Elbow Module (robot_elbow_module)
```bash
cd src/esp32/robot_elbow_module
idf.py set-target esp32c3  # or your ESP32 variant
idf.py build flash monitor
```

#### Node 2: Command Node (elbow_command_node)
```bash
cd src/esp32/elbow_command_node
idf.py set-target esp32c3  # or your ESP32 variant
idf.py build flash monitor
```

## How It Works

### Command Node (elbow_command_node)
- Sends target angle commands every 5 seconds
- Cycles through test angles: 0°, 30°, -30°, 45°, -45°, 60°, -60°, 0°
- Receives and displays status messages from the elbow module

### Elbow Module (robot_elbow_module)
- Receives CAN commands with target angles
- Moves stepper motor to target angle
- Sends status updates (current angle) every 100ms

## CAN Message Format

### Command Message (ID: 0x010)
- Byte 0: Target angle in degrees (signed 8-bit, -128 to +127)

### Status Message (ID: 0x030)
- Byte 0-1: Current angle in 0.1 degree resolution (signed 16-bit, little-endian)

## Customization

To change the test angles, edit `elbow_command_node/main/main.c`:
```c
int8_t test_angles[] = {0, 30, -30, 45, -45, 60, -60, 0};
```

To change the command interval, modify:
```c
vTaskDelay(pdMS_TO_TICKS(5000));  // 5 seconds between commands
```

