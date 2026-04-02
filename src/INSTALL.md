# Installation Guide

This guide explains how to install all dependencies for the LIMB-HT25 system.

## Quick Install

```bash
cd src
pip install -r requirements.txt
```

## System Dependencies

### Linux (Ubuntu/Debian)

```bash
# CAN bus utilities
sudo apt-get update
sudo apt-get install can-utils python3-dev

# OpenCV dependencies
sudo apt-get install libgl1-mesa-glx libglib2.0-0

# For OAK-D camera (if using vision system)
# See: https://docs.luxonis.com/en/latest/pages/tutorials/first_steps/
```

### macOS

```bash
# CAN bus utilities (if available)
brew install can-utils

# Note: Some hardware interfaces may not be available on macOS
```

## Package Details

### Core Dependencies
- **numpy**: Numerical computing
- **scipy**: Signal processing (filters, FFT, etc.)
- **PyYAML**: Configuration file parsing

### Machine Learning
- **torch**: PyTorch for LSTM models
- **scikit-learn**: Data preprocessing (StandardScaler, etc.)

### Hardware Interfaces
- **python-can**: CAN bus communication (SocketCAN)
- **bleak**: Bluetooth Low Energy (BLE) for ESP32 communication

### Vision System
- **depthai**: OAK-D camera SDK
- **opencv-python**: Computer vision and image processing

### Optional
- **pandas**: Data processing (used in EMG scripts)
- **matplotlib**: Plotting and visualization

## Installation Order

For best results, install in this order:

1. System dependencies (CAN utils, etc.)
2. Core Python packages: `pip install numpy scipy PyYAML`
3. ML packages: `pip install torch scikit-learn`
4. Hardware interfaces: `pip install python-can bleak`
5. Vision: `pip install depthai opencv-python`
6. Optional: `pip install pandas matplotlib`

## Troubleshooting

### CAN Interface Issues
- Ensure `can-utils` is installed
- Check CAN interface permissions: `sudo chmod 666 /dev/can0`
- Verify interface exists: `ip link show can0`

### BLE Issues
- On Linux, may need: `sudo apt-get install libbluetooth-dev`
- Check BLE permissions

### Vision System Issues
- OAK-D camera requires USB3 connection
- May need to install DepthAI system dependencies
- See: https://docs.luxonis.com/

### PyTorch Installation
- For CUDA support, install from PyTorch website
- CPU-only: `pip install torch` (default)

## Verification

Test installation:

```bash
python3 -c "import numpy, scipy, torch, yaml, can, bleak; print('Core packages OK')"
python3 -c "import depthai, cv2; print('Vision packages OK')"
```

