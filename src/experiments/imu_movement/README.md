# IMU Movement Detection Experiment

Simple experiment that reads IMU data and detects high-level movements.

## Setup

1. Build and flash the ESP32 firmware:
```bash
cd src/experiments/imu_movement
idf.py build flash
```

2. Run the Python script:
```bash
python process_movement.py /dev/ttyUSB0
```
(Replace `/dev/ttyUSB0` with your serial port)

## How it works

- **ESP32 (main.c)**: Reads IMU data at ~200Hz and sends JSON via serial
- **Python (process_movement.py)**: Receives data, maintains a window of 100 samples, and detects movements using the same algorithm as the processing layer

## Output

The Python script prints detected movements:
```
Movement: forward  | Confidence: 85.00% | Magnitude: 0.450 m/s²
Movement: right   | Confidence: 72.00% | Magnitude: 0.380 m/s²
Movement: none     | Confidence: 0.00%  | Magnitude: 0.120 m/s²
```

