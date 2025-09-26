# IMU System for Robotic Arm

This module provides a comprehensive IMU (Inertial Measurement Unit) system for the robotic arm project, supporting 4 IMUs for enhanced pose estimation and sensor fusion with the vision system.

## Overview

The IMU system consists of:
- **2 IMUs on the robotic arm**: One on the forearm, one on the upper arm
- **2 IMUs on the EMG band/cuff**: One on the user's forearm, one on the user's upper arm

These IMUs work together with the vision system to provide accurate pose estimation and relative distance calculations between the cup and the arm/hand.

## Features

- **Multi-IMU Support**: Manages 4 IMUs simultaneously
- **Multiple Communication Protocols**: Serial and simulated IMU protocols
- **Calibration System**: Comprehensive calibration for accelerometer, gyroscope, and magnetometer
- **Data Processing**: Filtering, noise reduction, and orientation estimation
- **Vision Integration**: Seamless integration with the existing vision system
- **Real-time Processing**: High-frequency data collection and processing
- **Pose Estimation**: Advanced sensor fusion for accurate pose estimation

## Architecture

### Core Components

1. **Data Structures** (`data_structures.py`)
   - `IMUReading`: Raw sensor data
   - `ProcessedIMUReading`: Calibrated and filtered data
   - `IMUDataCollection`: Collection of readings from all IMUs
   - `IMUCalibration`: Calibration parameters

2. **Interface Layer** (`imu_interface.py`)
   - `IMUManager`: Manages all 4 IMUs
   - `IMUProtocol`: Abstract base for communication protocols
   - `SerialIMUProtocol`: Serial communication implementation
   - `SimulatedIMUProtocol`: Simulation for testing

3. **Calibration** (`calibration.py`)
   - `IMUCalibrator`: Calibration algorithms
   - `IMUDataProcessor`: Data processing and filtering
   - `IMUCalibrationCollector`: Data collection for calibration

4. **Vision Integration** (`vision_integration.py`)
   - `IMUPoseEstimator`: Pose estimation from IMU data
   - `MultiIMUPoseEstimator`: Multi-IMU sensor fusion
   - `VisionIMUIntegrator`: Integration with vision system

## Usage

### Basic IMU Reading

```python
from imu import IMUManager, IMULocation

# Create IMU manager (simulation mode)
imu_manager = IMUManager(use_simulation=True)

# Read single data collection
data_collection = imu_manager.read_single_collection()

# Print data from all IMUs
for location in IMULocation:
    reading = data_collection.get_reading(location)
    if reading:
        print(f"{location.value}: {reading.accel} m/s²")
```

### Real-time Data Collection

```python
def data_callback(data_collection):
    print(f"Received data at {data_collection.timestamp}")

# Set up callback
imu_manager.set_data_callback(data_callback)

# Start reading at 100 Hz
imu_manager.start_reading(sample_rate=100.0)

# Stop reading
imu_manager.stop_reading()
```

### Enhanced Vision System

```python
from imu import create_enhanced_vision_system
import numpy as np

# Camera parameters
camera_matrix = np.array([[800, 0, 640], [0, 800, 360], [0, 0, 1]])
dist_coeffs = np.zeros(5)

# Create enhanced vision system
enhanced_system = create_enhanced_vision_system(
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs,
    marker_length_m=0.03,
    use_simulation=True
)

# Start IMU reading
enhanced_system.start_imu_reading(sample_rate=100.0)

# Process frame with IMU integration
result = enhanced_system.process_frame_with_imu(frame)
print(f"Enhanced pose: {result['enhanced_arm_pose']}")
```

### Command Line Interface

The main script provides a command-line interface for testing and data collection:

```bash
# Single reading mode (default)
python -m imu.main

# Data collection mode
python -m imu.main --mode collect --duration 60 --sample-rate 100

# Calibration mode
python -m imu.main --mode calibrate --location robot_forearm

# Real hardware mode
python -m imu.main --real-hardware --serial-ports /dev/ttyUSB0 /dev/ttyUSB1 /dev/ttyUSB2 /dev/ttyUSB3
```

## Hardware Setup

### Serial Communication

For real hardware, the IMUs should output data in CSV format:
```
accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,mag_x,mag_y,mag_z,timestamp
```

Example:
```
0.1,0.2,9.8,0.01,0.02,0.03,20.5,5.2,40.1,1234567890.123
```

### IMU Locations

- **Robot Forearm**: IMU mounted on robotic arm forearm
- **Robot Upper Arm**: IMU mounted on robotic arm upper arm  
- **EMG Forearm**: IMU in EMG band on user's forearm
- **EMG Upper Arm**: IMU in EMG band on user's upper arm

## Calibration

The system supports comprehensive calibration for all sensors:

1. **Accelerometer Calibration**: Keep IMU stationary in multiple orientations
2. **Gyroscope Calibration**: Keep IMU stationary to estimate bias
3. **Magnetometer Calibration**: Rotate IMU slowly in various orientations

### Calibration Process

```python
from imu import IMUCalibrator, IMUCalibrationCollector

# Collect calibration data
collector = IMUCalibrationCollector()

# Add readings for each sensor type
collector.add_reading(reading, 'accel')  # Accelerometer data
collector.add_reading(reading, 'gyro')   # Gyroscope data
collector.add_reading(reading, 'mag')    # Magnetometer data

# Generate calibration
calibration = collector.get_calibration(IMULocation.ROBOT_FOREARM)
imu_manager.set_calibration(IMULocation.ROBOT_FOREARM, calibration)
```

## Data Processing

The system includes several data processing features:

- **Low-pass Filtering**: Reduces high-frequency noise
- **Complementary Filtering**: Combines accelerometer and gyroscope data
- **Orientation Estimation**: Quaternion-based orientation from sensor fusion
- **Pose Integration**: Integration of angular velocity to estimate pose changes

## Integration with Vision System

The IMU system integrates seamlessly with the existing vision system:

1. **IMU Delta Transforms**: Provides motion deltas between frames
2. **Sensor Fusion**: Combines IMU and vision data for robust pose estimation
3. **Enhanced Accuracy**: Improves pose estimation accuracy and robustness
4. **Real-time Processing**: Maintains real-time performance

## Configuration

### Sample Rates
- Default: 100 Hz
- Configurable: 10-1000 Hz (depending on hardware)

### Filter Parameters
- Accelerometer cutoff: 5 Hz
- Gyroscope cutoff: 10 Hz
- Complementary filter alpha: 0.02

### Weights for Multi-IMU Fusion
- Robot Forearm: 0.3
- Robot Upper Arm: 0.3
- EMG Forearm: 0.2
- EMG Upper Arm: 0.2

## Dependencies

- `numpy`: Numerical computations
- `scipy`: Scientific computing and optimization
- `pyserial`: Serial communication
- `opencv-python`: Computer vision (for integration)

## Testing

The system includes comprehensive testing capabilities:

```bash
# Run tests
python -m pytest tests/

# Test with simulation
python -m imu.main --mode single

# Test data collection
python -m imu.main --mode collect --duration 10
```

## Troubleshooting

### Common Issues

1. **Serial Connection Failed**
   - Check port permissions
   - Verify baud rate settings
   - Ensure IMU is powered and connected

2. **No Data Received**
   - Check data format (CSV with 10 fields)
   - Verify timestamp format
   - Check for data corruption

3. **Poor Calibration**
   - Ensure sufficient data collection time
   - Follow calibration procedure exactly
   - Check for magnetic interference

### Debug Mode

Enable debug logging for detailed information:

```bash
python -m imu.main --log-level DEBUG
```

## Future Enhancements

- **Advanced Sensor Fusion**: Kalman filtering and particle filters
- **Machine Learning**: Learned calibration and noise models
- **Wireless Communication**: Bluetooth and WiFi support
- **Real-time Visualization**: Live data plotting and monitoring
- **Advanced Filtering**: Adaptive filtering based on motion detection

## Contributing

When contributing to the IMU system:

1. Follow the existing code structure
2. Add comprehensive tests
3. Update documentation
4. Ensure backward compatibility
5. Test with both simulation and real hardware
