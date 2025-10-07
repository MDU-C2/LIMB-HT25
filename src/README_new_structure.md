# Robotic Manipulation System - New Structure

## 📁 Directory Structure

```
src/
├── sensors/                    # Sensor Management System
│   ├── __init__.py
│   ├── sensor_manager.py       # Central sensor coordinator
│   ├── imu_reader.py          # ESP32 IMU interface
│   ├── vision_system.py       # Vision system wrapper
│   ├── pressure_sensor.py     # Pressure sensor interface
│   ├── slip_sensor.py         # Slip sensor interface
│   ├── piezo_sensor.py        # Piezo sensor interface
│   └── hardware/
│       └── esp32_imu/         # Your C code here!
│           ├── main.c         # Your existing IMU code
│           ├── CMakeLists.txt
│           └── deploy.sh      # Build and flash script
├── data_fusion/               # Data Fusion System
│   ├── __init__.py
│   ├── fusion_system.py       # Main fusion system
│   ├── smoothing.py           # IMU smoothing and validation
│   ├── frames.py              # Coordinate frame definitions
│   ├── hand_pose.py           # Hand pose estimation
│   ├── cup_3d.py              # 3D cup position estimation
│   ├── relative_pose.py       # Relative pose calculation
│   └── calibration.py         # Sensor calibration
├── robot_control/             # Robot Control System
│   ├── __init__.py
│   ├── main.py               # Main entry point
│   ├── state_machine.py      # State machine logic
│   ├── states/               # Action state implementations
│   │   ├── __init__.py
│   │   ├── base_state.py     # Abstract base class
│   │   ├── move_to_cup.py    # MOVE to cup action
│   │   ├── grab_cup.py       # GRAB cup action
│   │   ├── lift_cup.py       # LIFT cup action
│   │   ├── move_cup_ab.py    # MOVE cup A->B action
│   │   ├── place_down_cup.py # PLACE DOWN cup action
│   │   ├── release_cup.py    # RELEASE cup action
│   │   └── move_back_hand.py # MOVE back hand action
│   └── hardware/             # Robot hardware interfaces
│       ├── __init__.py
│       └── robot_arm.py      # Robot arm control
└── config/                   # Configuration files
    └── sensor_config.json    # Sensor configuration
```

## 🚀 Quick Start

### 1. Deploy ESP32 IMU Code

The ESP32-C3 firmware is located in `src/sensors/hardware/esp32_imu/`:

```bash
cd src/sensors/hardware/esp32_imu/
./deploy.sh
```

See `src/sensors/hardware/esp32_imu/README.md` for detailed hardware and firmware documentation.

### 2. Test Sensors
```bash
cd src/robot_control/
python3 main.py --test-sensors
```

### 3. Run Full System
```bash
cd src/robot_control/
python3 main.py
```

## 🔧 Key Features

- **Modular Design**: Each sensor and action state is independent
- **Dynamic Sensor Activation**: Only activates required sensors per state
- **Data Fusion**: Advanced sensor fusion for pose estimation and tracking
- **Clean Integration**: Your existing IMU and vision code integrates seamlessly
- **State Machine**: Implements the flowchart exactly as specified
- **Configurable**: Easy to modify sensor requirements and parameters

## 📊 Sensor Usage Per State

Based on your flowchart:
- **MOVE to cup**: vision + imu + piezo
- **GRAB cup**: vision + pressure + slip
- **LIFT cup**: vision + pressure + slip + imu
- **MOVE cup A->B**: vision + pressure + slip + imu
- **PLACE DOWN cup**: vision + pressure + slip + imu
- **RELEASE cup**: vision + pressure + slip
- **MOVE back hand**: vision + imu + piezo

This structure perfectly implements your flowchart requirements!
