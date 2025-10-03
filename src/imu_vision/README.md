# IMU-Vision Fusion Package

This package implements a **fiducial + depth approach** for robot control. It provides a simple, robust system for hand pose estimation and cup detection using ArUco markers and YOLO.

## Overview

The system uses three main coordinate frames:
- **W**: Robot base/shoulder frame (world frame)
- **C**: Camera frame (rigidly fixed to W)
- **H**: Hand/tool frame (ArUco board on the hand)

## Key Components

### 1. Frame Management (`frames.py`)
- `FrameType`: Enum for frame identifiers (WORLD, CAMERA, HAND)
- `TransformManager`: Manages 4x4 transforms between frames

### 2. Calibration (`calibration.py`)
- `CalibrationManager`: Handles calibration data
- **T_WC**: Camera→world transform (constant, calibrated once)
- **T_CH_fixed**: Camera→hand tag transform (known from tag size/layout)

### 3. Hand Pose Estimation (`hand_pose.py`)
- `HandPoseEstimator`: Converts ArUco detection to hand pose
- **T_CH_meas** (camera to hand) to **T_WH** (world→hand)

### 4. Cup 3D Estimation (`cup_3d.py`)
- `Cup3DEstimator`: Estimates cup 3D position from YOLO + depth
- **bbox** → **depth ROI** → **robust depth pooling** → **p_C^cup** → **p_W^cup**

### 5. Relative Pose Calculation (`relative_pose.py`)
- `RelativePoseCalculator`: Calculates control information
- **T_HW** = (T_WH)⁻¹
- **p_H^cup** = T_HW · p_W^cup

### 6. IMU Smoothing (`smoothing.py`)
- `IMUSmoother`: Optional smoothing when ArUco is temporarily lost
- Methods: complementary filter, EKF, position hold
- Propagates hand pose with IMU at 200-400 Hz

### 7. Main Fusion System (`fusion_system.py`)
- `FiducialDepthSystem`: Orchestrates all components
- Main interface for the complete system

## Usage

### Basic Usage

```python
import numpy as np
from imu_vision import FiducialDepthSystem

# Initialize system
camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
system = FiducialDepthSystem(
    camera_matrix=camera_matrix,
    enable_imu_smoothing=True
)

# Process frame
result = system.process_frame(
    tag_detection_result=tag_result,
    cup_detection_result=cup_result,
    imu_data=imu_data
)

# Get control command
control_command = system.get_control_command()
```

### Integration with Existing Vision System

```python
from vision.system import VisionSystem
from imu_vision import FiducialDepthSystem

# Initialize both systems
vision_system = VisionSystem(camera_matrix, dist_coeffs, marker_length_m=0.03)
fiducial_system = FiducialDepthSystem(camera_matrix, enable_imu_smoothing=True)

# Process frame
vision_result = vision_system.process_frame(frame, mode="combined")
fiducial_result = fiducial_system.process_frame(
    tag_detection_result=vision_result.get("tag_result"),
    cup_detection_result=vision_result.get("cup_result"),
    imu_data=imu_data
)
```

## Calibration

### Setting Calibration

```python
# Set camera→world transform
T_WC = np.eye(4)
T_WC[2, 3] = 0.05  # Camera 5cm above world origin

# Set camera→hand tag transform
T_CH_fixed = np.eye(4)
T_CH_fixed[2, 3] = 0.15  # Hand tag 15cm in front of camera

system.set_calibration(T_WC, T_CH_fixed)
```

### Calibration File

The system automatically saves/loads calibration from `calibration.json`:

```json
{
  "T_WC": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.05], [0, 0, 0, 1]],
  "T_CH_fixed": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.15], [0, 0, 0, 1]]
}
```

## IMU Smoothing

When ArUco detection is temporarily lost, the system can use IMU data for pose prediction:

```python
from imu_vision.smoothing import IMUData

# Create IMU data
imu_data = IMUData(
    angular_velocity=np.array([0.01, 0.02, -0.01]),  # rad/s
    linear_acceleration=np.array([0.1, 0.05, 9.8]),  # m/s²
    timestamp=time.time()
)

# System will automatically use IMU prediction when vision is lost
result = system.process_frame(
    tag_detection_result=None,  # No ArUco detection
    cup_detection_result=cup_result,
    imu_data=imu_data
)
```

## Control Information

The system provides comprehensive control information:

```python
control_command = system.get_control_command()

# Access control data
hand_position = control_command['hand_pose']['position']
cup_position_hand_frame = control_command['cup_position']['hand_frame']
distance_to_cup = control_command['distance_to_cup']
approach_vector = control_command['approach_vector']
target_position = control_command['target_position']
```

## Examples

**All examples should be run from the `src` folder:**

### Run Basic Demo
```bash
cd src
python -m imu_vision.example_usage
```

### Run Integration Example
```bash
cd src
python -m imu_vision.integration_example --show --enable-imu
```

### Run Quick Start
```bash
cd src
python -m imu_vision.quick_start
```

### Run Vision System
```bash
cd src
python -m vision.main --mode combined --show
```

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   ArUco Tags    │    │   YOLO + Depth   │    │   IMU Data      │
│   (Hand Pose)   │    │   (Cup 3D)       │    │   (Smoothing)   │
└─────────┬───────┘    └─────────┬────────┘    └─────────┬───────┘
          │                      │                       │
          ▼                      ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Hand Pose       │    │ Cup 3D           │    │ IMU Smoother    │
│ Estimator       │    │ Estimator        │    │ (Optional)      │
└─────────┬───────┘    └─────────┬────────┘    └─────────┬───────┘
          │                      │                       │
          └──────────────────────┼───────────────────────┘
                                 ▼
                    ┌──────────────────────┐
                    │ Relative Pose        │
                    │ Calculator           │
                    └─────────┬────────────┘
                              ▼
                    ┌──────────────────────┐
                    │ Control Commands     │
                    │ (T_HW, p_H^cup)      │
                    └──────────────────────┘
```

## Advantages

1. **Simple**: No complex VIO algorithms
2. **Robust**: ArUco markers provide reliable pose estimation
3. **Flexible**: Optional IMU smoothing for temporary vision loss
4. **Extensible**: Easy to add new components or modify existing ones
5. **Well-tested**: Comprehensive examples and integration scripts

## Future Extensions

- Full EKF implementation for IMU smoothing
- Multiple ArUco marker support
- Advanced depth processing algorithms
- Real-time performance optimization
- Integration with robot control systems
