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
- **Enhanced Integration**: Works with `TagDetectionResult` structure
- **Backward Compatibility**: Supports legacy dict formats

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

### 7. IMU Validation (`smoothing.py`)
- `IMUValidator`: Validates vision poses using IMU data
- **Motion Validation**: Checks if acceleration/angular velocity are physically reasonable
- **Orientation Validation**: Validates rotation matrix properties
- **Temporal Consistency**: Detects large position/orientation jumps
- **Independent Operation**: Can be used without IMU smoothing

### 8. Main Fusion System (`fusion_system.py`)
- `FiducialDepthSystem`: Orchestrates all components
- Main interface for the complete system
- Supports both IMU smoothing and validation independently
- **Updated Integration**: Seamlessly handles `TagDetectionResult` structure

## Tag Detection Integration

The system now seamlessly integrates with the updated tag detection system that provides structured `TagDetectionResult` objects:

### TagDetectionResult Structure
```python
@dataclass
class TagDetectionResult:
    tag_ids: np.ndarray               # shape (N,) - detected tag IDs
    rvecs: Optional[np.ndarray]       # shape (N, 1, 3) - rotation vectors
    tvecs: Optional[np.ndarray]       # shape (N, 1, 3) - translation vectors
    corners: Optional[List[np.ndarray]] # corner coordinates for visualization
    transforms: Optional[List[np.ndarray]]  # shape (N, 4, 4) - pre-computed transforms
    reproj_errors: Optional[np.ndarray]     # shape (N,) - reprojection errors
    timestamps: Optional[List[float]]       # detection timestamps
```

### Integration Benefits
- **Pre-computed Transforms**: Direct access to 4x4 transformation matrices
- **Quality Metrics**: Reprojection errors for pose quality assessment
- **Enhanced Data**: Structured access to all detection information
- **Backward Compatibility**: Legacy dict formats still supported
- **Performance**: Reduced computation overhead in fusion system

### Usage with New Structure
```python
from vision.tags.tag_detector import TagDetector, TagDetectionResult

# Initialize tag detector
tag_detector = TagDetector(camera_matrix, dist_coeffs, marker_length_m=0.03)

# Detect tags (returns TagDetectionResult object)
tag_result = tag_detector.detect_and_estimate(frame)

# Use directly with fusion system
fusion_result = system.process_frame(
    tag_detection_result=tag_result,  # Direct TagDetectionResult object
    cup_detection_result=cup_result,
    imu_data=imu_data
)
```

## Usage

### Basic Usage

```python
import numpy as np
from imu_vision import FiducialDepthSystem

# Initialize system
camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
system = FiducialDepthSystem(
    camera_matrix=camera_matrix,
    enable_imu_smoothing=True,
    enable_imu_validation=True  # NEW: Validate vision with IMU
)

# Process frame
result = system.process_frame(
    tag_detection_result=tag_result,
    cup_detection_result=cup_result,
    imu_data=imu_data
)

# Check validation results
if result['hand_pose']['validation']:
    validation = result['hand_pose']['validation']
    if validation['is_valid']:
        print(f"✅ Pose validated (confidence: {validation['confidence']:.2f})")
    else:
        print(f"❌ Pose rejected: {validation['issues']}")

# Get control command
control_command = system.get_control_command()
```

### Integration with Existing Vision System

```python
from vision.system import VisionSystem
from imu_vision import FiducialDepthSystem

# Initialize both systems
vision_system = VisionSystem(camera_matrix, dist_coeffs, marker_length_m=0.03)
fiducial_system = FiducialDepthSystem(
    camera_matrix, 
    enable_imu_smoothing=True,
    enable_imu_validation=True
)

# Process frame
vision_result = vision_system.process_frame(frame, mode="combined")
fiducial_result = fiducial_system.process_frame(
    tag_detection_result=vision_result.get("tag_result"),
    cup_detection_result=vision_result.get("cup_result"),
    imu_data=imu_data
)
```

### IMU Validation Only (No Smoothing)

```python
# Use IMU validation without smoothing
system = FiducialDepthSystem(
    camera_matrix=camera_matrix,
    enable_imu_smoothing=False,  # No smoothing
    enable_imu_validation=True   # But validate with IMU
)

# Process frame - validation will check vision poses
result = system.process_frame(
    tag_detection_result=tag_result,
    imu_data=imu_data
)

# Validation results are always available for vision poses
validation = result['hand_pose']['validation']
if validation and not validation['is_valid']:
    print(f"⚠️  Vision pose rejected: {validation['issues']}")
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

## IMU Validation

The system can validate vision poses using IMU data to detect physically unreasonable results:

```python
# Create IMU data for validation
imu_data = IMUData(
    angular_velocity=np.array([0.1, 0.05, -0.02]),  # rad/s
    linear_acceleration=np.array([0.2, -0.1, 9.8]),  # m/s²
    timestamp=time.time()
)

# Process frame with validation
result = system.process_frame(
    tag_detection_result=tag_result,
    imu_data=imu_data
)

# Check validation results
validation = result['hand_pose']['validation']
if validation:
    print(f"Validation: Valid={validation['is_valid']}")
    print(f"Confidence: {validation['confidence']:.3f}")
    if validation['issues']:
        print(f"Issues: {validation['issues']}")
    if validation['warnings']:
        print(f"Warnings: {validation['warnings']}")
```

### Custom Validation Thresholds

```python
# Custom validation thresholds
custom_thresholds = {
    'max_acceleration': 100.0,      # Allow higher acceleration
    'max_angular_velocity': 20.0,   # Allow faster rotation
    'max_position_jump': 1.0,       # Allow larger position jumps
    'max_orientation_jump': 0.8,    # Allow larger orientation changes
}

system = FiducialDepthSystem(
    camera_matrix=camera_matrix,
    enable_imu_validation=True,
    validation_thresholds=custom_thresholds
)
```

### Validation Features

- **Motion Validation**: Checks if IMU acceleration and angular velocity are physically reasonable
- **Orientation Validation**: Validates rotation matrix properties (orthogonality, determinant)
- **Temporal Consistency**: Detects large position/orientation jumps between frames
- **Confidence Scoring**: Provides confidence scores for validation results
- **Independent Operation**: Can be used without IMU smoothing

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

## Package Structure

```
imu_vision/
├── __init__.py                 # Main package
├── fusion_system.py           # Main fusion system
├── smoothing.py               # IMU smoothing and validation
├── calibration.py             # Calibration management
├── hand_pose.py               # Hand pose estimation
├── cup_3d.py                  # Cup 3D estimation
├── relative_pose.py           # Relative pose calculation
├── frames.py                  # Frame management
├── examples/                  # Example scripts
│   ├── __init__.py
│   ├── example_usage.py       # Basic usage examples
│   ├── integration_example.py # Integration with vision system
│   ├── quick_start.py         # Quick start guide
│   └── validation_example.py  # IMU validation examples
├── tests/                     # Test scripts
│   ├── __init__.py
│   └── test_validation.py     # IMU validation tests
└── README.md                  # This file
```

## Examples

**All examples should be run from the `src` folder:**

### Run Basic Demo
```bash
cd src
python -m imu_vision.examples.example_usage
```

### Run Integration Example
```bash
cd src
python -m imu_vision.examples.integration_example --show --enable-imu
```

### Run Quick Start
```bash
cd src
python -m imu_vision.examples.quick_start
```

### Test IMU Validation
```bash
cd src
python -m imu_vision.tests.test_validation
```

### Run Validation Examples
```bash
cd src
python -m imu_vision.examples.validation_example
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
│   (Hand Pose)   │    │   (Cup 3D)       │    │ (Smoothing +    │
│                 │    │                  │    │  Validation)    │
└─────────┬───────┘    └─────────┬────────┘    └─────────┬───────┘
          │                      │                       │
          ▼                      ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Hand Pose       │    │ Cup 3D           │    │ IMU Smoother    │
│ Estimator       │    │ Estimator        │    │ (Optional)      │
└─────────┬───────┘    └─────────┬────────┘    └─────────┬───────┘
          │                      │                       │
          ▼                      │                       │
┌─────────────────┐              │                       │
│ IMU Validator   │              │                       │
│ (Optional)      │              │                       │
└─────────┬───────┘              │                       │
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
4. **Quality Control**: IMU validation detects bad vision results
5. **Independent Features**: Use validation without smoothing
6. **Extensible**: Easy to add new components or modify existing ones
7. **Well-tested**: Comprehensive examples and integration scripts

## Validation Benefits

- **Safety**: Prevent dangerous robot movements from bad vision data
- **Debugging**: Get detailed information about pose quality issues
- **Robustness**: Flag suspicious poses even without IMU smoothing
- **Quality Assurance**: Validate vision results with physical constraints
- **Flexible Thresholds**: Customize validation criteria for your application

## Future Extensions

- Full EKF implementation for IMU smoothing
- Multiple ArUco marker support
- Advanced depth processing algorithms
- Real-time performance optimization
- Integration with robot control systems
- Enhanced validation algorithms (Kalman filter-based validation)
- Machine learning-based anomaly detection
