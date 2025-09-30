# Vision System

## Overview
This vision system provides:
- **Tag detection** with ArUco markers using OpenCV
- **Cup detection** using YOLO models (YOLOv11) with optimized inference
- **Camera calibration** with ChArUco boards
- **Real-time performance optimization** with multi-threading and frame skipping
- **Comprehensive performance monitoring** and benchmarking
- **Structured JSON output** for integration with robotic systems

## Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

### Basic Usage
```bash
# Cup detection mode (default)
python -m vision.main --mode cup --show

# Tag detection mode
python -m vision.main --mode tag --show

# Combined detection mode
python -m vision.main --mode combined --show
```

### Performance Optimization
```bash
# High performance mode (60 FPS, low latency)
python -m vision.main --mode cup --target-fps 60 --skip-frames 0 --low-latency --show

# Balanced mode (30 FPS, good quality)
python -m vision.main --mode cup --target-fps 30 --skip-frames 1 --show

# High quality mode (15 FPS, best quality)
python -m vision.main --mode cup --target-fps 15 --skip-frames 2 --show
```

### Camera Calibration
1. **Generate ChArUco board:**
   ```bash
   python -m vision.tags.camera_calibration gen-board --squares-x 8 --squares-y 11 --square-length 200 --marker-length 120 --dict DICT_4X4_50 --out src/vision/tags/charuco/board.png
   ```

2. **Print the generated board** and calibrate camera:
   ```bash
   python -m vision.tags.camera_calibration calibrate --device 0 --samples 20 --squares-x 8 --squares-y 11 --square-length 0.025 --marker-length 0.018 --dict DICT_4X4_50 --output src/vision/tags/camera_calibration.json
   ```

3. **Use calibrated camera:**
   ```bash
   python -m vision.main --mode tag --calib src/vision/tags/camera_calibration.json --show
   ```

### Generate ArUco Tags
```bash
# Generate individual ArUco tags
python -m vision.tags.generate_aruco_tags --dict DICT_4X4_50 --ids 0-9 --size 600 --outdir src/vision/tags/generated_tags
```

## Advanced Usage

### Performance Monitoring
```bash
# Run with real-time performance monitoring
python -m vision.main --mode cup --show

# Interactive controls during runtime:
# Press 'p' - Print detailed performance statistics
# Press 's' - Save performance report to file
# Press '+' - Increase target FPS
# Press '-' - Decrease target FPS
# Press 'q' - Quit
```

### YOLO Model Options
```bash
# Use different YOLO models
python -m vision.main --mode cup --yolo-model yolo11s.pt --show

# Adjust detection sensitivity
python -m vision.main --mode cup --yolo-conf 0.5 --yolo-iou 0.3 --show

# GPU acceleration (Jetson/CUDA)
python -m vision.main --mode cup --yolo-device cuda:0 --show
```

### Performance Benchmarking
```bash
# Run comprehensive benchmark suite
python -m vision.utils.benchmark --duration 10 --model yolo11s.pt

# Quick system benchmark
python -m vision.main --benchmark

# Test optimization components
python -m vision.test_optimizations.py

# Run usage examples
python -m vision.example_optimized_usage.py
```

### Performance Utilities
The system includes several performance utilities:

- **`utils/performance_monitor.py`** - Real-time performance monitoring and statistics
- **`utils/camera_optimizer.py`** - Optimized camera capture with low latency
- **`utils/benchmark.py`** - Comprehensive benchmarking suite
- **`test_optimizations.py`** - Test script for optimization components


### Available YOLO Models
- `yolo11n.pt` - Nano (fastest, least accurate)
- `yolo11s.pt` - Small
- `yolo11m.pt` - Medium
- `yolo11l.pt` - Large
- `yolo11x.pt` - Extra Large (slowest, most accurate)

## Output Structure

### Cup Detection Mode
```json
{
  "cup_detected": true,
  "cup_relative_position": {"x": 0.12, "y": -0.03, "z": 0.45},
  "arm_pose_corrected": {"matrix": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]},
  "tag_detection": {"tag_detected": false, "tag_ids": [], "tag_positions": []}
}
```

### Tag Detection Mode
```json
{
  "cup_detected": false,
  "cup_relative_position": null,
  "arm_pose_corrected": {"matrix": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]},
  "tag_detection": {
    "tag_detected": true,
    "tag_ids": [0, 1, 2],
    "tag_positions": [
      {"id": 0, "position": {"x": 0.1, "y": 0.2, "z": 0.5}},
      {"id": 1, "position": {"x": -0.1, "y": 0.3, "z": 0.4}}
    ]
  }
}
```

## Command Line Options

### Main Application
- `--mode {cup,tag,combined}` - Select detection pipeline
- `--show` - Display camera feed with visualizations
- `--calib PATH` - Path to camera calibration JSON
- `--yolo-model PATH` - Path to YOLO model file
- `--yolo-device DEVICE` - Device for YOLO (cuda:0, cpu)
- `--yolo-conf FLOAT` - YOLO confidence threshold (0.0-1.0)
- `--yolo-iou FLOAT` - YOLO IoU threshold (0.0-1.0)

### Performance Options
- `--target-fps FLOAT` - Target processing FPS (default: 30.0)
- `--skip-frames INT` - Skip N frames between processing (default: 0)
- `--low-latency` - Enable low latency optimizations
- `--benchmark` - Run performance benchmark instead of normal operation
- `--camera-device INT` - Camera device ID (default: 0)

### Calibration
- `--device INDEX` - Camera device index
- `--samples COUNT` - Number of calibration frames
- `--squares-x COUNT` - Number of squares horizontally
- `--squares-y COUNT` - Number of squares vertically
- `--square-length METERS` - Physical square size
- `--marker-length METERS` - Physical marker size
- `--dict DICT_NAME` - ArUco dictionary (DICT_4X4_50, DICT_6X6_250, etc.)

## Hardware Requirements
- **Camera**: USB webcam or built-in camera
- **CPU**: Modern multi-core processor (optimized for multi-threading)
- **GPU** (optional): NVIDIA GPU with CUDA support for faster YOLO inference
- **Memory**: 4GB+ RAM recommended (8GB+ for high performance modes)
- **Storage**: SSD recommended for optimal performance

## Jetson Orin Setup
1. Install JetPack following NVIDIA instructions
2. Install PyTorch for Jetson
3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
4. Run with GPU acceleration and performance optimization:
   ```bash
   # High performance mode for Jetson Orin
   python -m vision.main --mode cup --yolo-device cuda:0 --target-fps 60 --low-latency --show
   
   # Balanced mode for Jetson Orin
   python -m vision.main --mode cup --yolo-device cuda:0 --target-fps 30 --skip-frames 1 --show
   ```

## Performance Features

### Real-time Optimization
- **Multi-threaded YOLO inference** prevents blocking and improves throughput
- **Frame skipping** allows processing every N+1 frames for better performance
- **Optimized camera capture** with minimal latency and buffer management
- **Dynamic performance tuning** with real-time FPS adjustment

### Performance Monitoring
- **Real-time statistics** including FPS, inference times, and drop rates
- **Comprehensive benchmarking** suite for performance analysis
- **Interactive controls** for runtime performance adjustment
- **Performance reports** with detailed timing and throughput metrics

### Expected Performance (Jetson Orin)
| Configuration | FPS | Inference Time | CPU Usage |
|---------------|-----|----------------|-----------|
| YOLO11n + No Skip | 45-60 | 15-20ms | 60-80% |
| YOLO11s + Skip 1 | 30-40 | 25-35ms | 40-60% |
| YOLO11s + Skip 2 | 20-30 | 25-35ms | 30-50% |

## Notes
- Camera calibration is essential for accurate pose estimation
- Use ChArUco boards for robust calibration
- YOLO models can be downloaded automatically on first use
- IMU fusion uses placeholder implementation - replace with proper EKF for production
- Performance optimizations are automatically enabled by default
- Use benchmarking tools to find optimal settings for your hardware

