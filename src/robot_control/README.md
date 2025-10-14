# Robot Control Module

This module implements an intelligent robotic manipulation system for cup manipulation tasks using LSTM-based intention recognition and sensor-guided execution.

## Architecture Overview

The system uses a **three-layer architecture**:

1. **LSTM Gatekeeper** - Determines WHAT to do (high-level intentions)
2. **Sensor Execution Guidance** - Determines HOW to move (IMU direction + Vision target)
3. **Sensor Feedback Control** - Ensures adequate grip (Pressure + Slip sensors)

## Key Features

- **LSTM Intention Classification**: Recognizes user intentions (`"rest"`, `"grip"`, `"move"`)
- **Direction-Only Movement**: IMU provides direction, speed is hard-coded for consistency
- **Sensor-Guided Execution**: Vision provides targets, IMU provides movement direction
- **Adaptive Grip Control**: Pressure and slip sensors ensure adequate gripping
- **Modular State Machine**: Easy to extend and modify manipulation sequences

## System Components

### Core Components

| Component | File | Description |
|-----------|------|-------------|
| **State Machine** | `state_machine.py` | Main orchestrator with LSTM gatekeeper |
| **Main Entry** | `main.py` | System initialization and execution |
| **Robot Arm** | `hardware/robot_arm.py` | Robot arm control interface |
| **Action States** | `states/` | Individual manipulation actions |

### Sensor Integration

| Sensor | Role | Data Provided |
|--------|------|---------------|
| **LSTM** | Gatekeeper | Intention classification (`"rest"`, `"grip"`, `"move"`) |
| **IMU** | Direction | Movement direction vector (speed hard-coded) |
| **Vision** | Target | Target position and object detection |
| **Pressure** | Feedback | Grip force monitoring |
| **Slip** | Feedback | Slip detection for grip adequacy |

## File Structure

```
robot_control/
├── main.py                 # Main entry point
├── state_machine.py        # Core state machine with LSTM integration
├── hardware/              # Robot hardware interfaces
│   ├── robot_arm.py       # Robot arm control
│   ├── gripper.py         # Gripper control
│   └── actuators.py       # Actuator interfaces
└── states/                # Action state implementations
    ├── base_state.py      # Abstract base class
    ├── move_to_cup.py     # Move to cup action
    ├── grab_cup.py        # Grab cup with grip feedback
    ├── lift_cup.py        # Lift cup action
    ├── move_cup_ab.py     # Move cup to position A/B
    ├── place_down_cup.py  # Place cup down
    ├── release_cup.py     # Release cup
    └── move_back_hand.py  # Move hand back to neutral
```

## Usage

### Basic Usage

```python
from robot_control.state_machine import StateMachine
from robot_control.hardware.robot_arm import RobotArm
from sensors import SensorManager

# Initialize components
sensor_manager = SensorManager()
robot_arm = RobotArm()
state_machine = StateMachine(sensor_manager, robot_arm)

# Run the manipulation system
state_machine.run()
```

### Command Line Usage

```bash
# Run with default settings
python main.py

# Run with specific configuration
python main.py --config config.json

# Test sensors only
python main.py --test-sensors

# Run in simulation mode
python main.py --simulate

# Specify IMU port
python main.py --imu-port /dev/ttyUSB0
```

## State Machine Architecture

### LSTM Gatekeeper

The LSTM classifier acts as a gatekeeper, determining high-level intentions:

```python
# LSTM classifies intention from sensor data
intention = lstm_classifier.classify_intention(sensor_data)
# Returns: "rest", "grip", or "move"

# State machine maps intention to action
next_state = intention_to_state[intention]
```

### Intention-to-State Mapping

| LSTM Intention | Robot State | Description |
|----------------|-------------|-------------|
| `"rest"` | `MOVE_back_hand` | Move hand to neutral position |
| `"grip"` | `GRAB_cup` | Grasp the cup object |
| `"move"` | `MOVE_to_cup` | Move towards the cup |

### Execution Guidance

For movement actions, the system uses:

- **IMU Direction**: Movement direction vector from user
- **Hard-coded Speed**: Consistent movement speed (0.1 units/second)
- **Vision Target**: Target position from camera system

```python
# Get execution guidance
guidance = state_machine.get_execution_guidance()
# Returns: imu_direction, hardcoded_speed, vision_target

# Execute movement
robot_arm.move_with_direction_guidance(
    target=guidance['vision_target'],
    direction=guidance['imu_direction'],
    speed=guidance['hardcoded_speed']
)
```

### Grip Feedback Control

For gripping actions, the system uses:

- **Pressure Sensor**: Monitors grip force
- **Slip Sensor**: Detects object slipping
- **Adaptive Control**: Adjusts grip force based on feedback

```python
# Get grip feedback
feedback = state_machine.get_grip_feedback()
# Returns: pressure_force, slip_detected, grip_adequate

# Adjust grip based on feedback
if feedback['slip_detected']:
    robot_arm.increase_grip_force()
elif feedback['pressure_force'] > max_force:
    robot_arm.decrease_grip_force()
```

## Action States

### Move to Cup State
- **Purpose**: Move robot hand to cup position
- **Sensors**: IMU (direction), Vision (target), Piezo (contact)
- **Execution**: Uses IMU direction + Vision target with hard-coded speed

### Grab Cup State
- **Purpose**: Grasp the cup with adequate grip
- **Sensors**: Pressure (force), Slip (slipping detection)
- **Execution**: Continuous monitoring with force adjustment

### Other States
- **Lift Cup**: Lift the grasped cup
- **Move Cup AB**: Move cup between positions
- **Place Down Cup**: Place cup at target location
- **Release Cup**: Release grip on cup
- **Move Back Hand**: Return hand to neutral position

## Configuration

### Robot Arm Configuration

```python
# In main.py or config file
robot_config = {
    'connection_type': 'ethernet',
    'ip_address': '192.168.1.100',
    'port': 502
}
```

### Sensor Configuration

```python
# IMU settings
imu_config = {
    'port': '/dev/ttyUSB0',
    'baudrate': 115200
}

# Camera settings
camera_config = {
    'device_id': 0,
    'resolution': (640, 480)
}
```

## Hardware Integration

### Robot Arm Interface

The `RobotArm` class provides a hardware-agnostic interface:

```python
class RobotArm:
    def connect(self) -> bool
    def move_to_position(self, position: List[float]) -> bool
    def move_with_direction_guidance(self, target, direction, speed) -> bool
    def start_gripping(self)
    def increase_grip_force(self)
    def decrease_grip_force(self)
    def get_status(self) -> Dict[str, Any]
```

### Sensor Integration

The system integrates with various sensors through the `SensorManager`:

```python
# Get sensor data
imu_data = sensor_manager.get_sensor_data('imu')
vision_data = sensor_manager.get_sensor_data('vision')
pressure_data = sensor_manager.get_sensor_data('pressure')
slip_data = sensor_manager.get_sensor_data('slip')
```

## Error Handling

The system includes comprehensive error handling:

- **Sensor Failures**: Graceful degradation when sensors fail
- **Robot Communication**: Automatic fallback to simulation mode
- **State Machine Errors**: Retry logic for failed actions
- **Timeout Handling**: Prevents infinite loops in action states

## Debugging

### State Machine Status

```python
# Get current status
status = state_machine.get_status()
print(f"Current state: {status['current_state']}")
print(f"LSTM available: {status['lstm_available']}")
print(f"Last intention: {status['last_intention']}")
```

### Sensor Testing

```bash
# Test all sensors
python main.py --test-sensors

# Test specific sensor
sensor_manager.test_sensor('imu')
```

### State History

```python
# Get transition history
history = state_machine.get_state_history()
for transition in history:
    print(f"{transition['from_state']} → {transition['to_state']}")
    print(f"LSTM intention: {transition['lstm_intention']}")
```

## Performance Considerations

- **LSTM Classification**: Runs continuously for real-time intention recognition
- **Sensor Data**: Optimized for low-latency sensor data processing
- **Movement Control**: Hard-coded speed ensures consistent performance
- **Grip Control**: Adaptive feedback prevents over/under gripping

## Future Enhancements

- **Real LSTM Models**: Integration with trained LSTM models
- **Advanced Sensor Fusion**: Multi-modal sensor data fusion
- **Learning Capabilities**: Adaptation based on user behavior
- **Multi-Object Support**: Extension to multiple object types
- **Safety Features**: Enhanced safety monitoring and control

## Dependencies

- **NumPy**: Numerical computations
- **Sensor Modules**: IMU, Vision, Pressure, Slip sensors
- **Robot Hardware**: Specific robot arm control libraries
- **LSTM Classifier**: Intention recognition system

## Contributing

When adding new action states:

1. Inherit from `BaseState`
2. Implement `execute()` method
3. Define `get_required_sensors()` method
4. Add state to state machine configuration
5. Update intention-to-state mapping if needed

## License

This module is part of the LIMB-HT25 robotic manipulation system.
