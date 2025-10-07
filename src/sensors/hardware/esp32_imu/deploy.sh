#!/bin/bash
# Build and Deploy Script for ESP32 IMU

set -e

echo "Building and flashing ESP32 IMU..."

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Navigate to the project root (three levels up from esp32_imu)
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
PROJECT_DIR="$SCRIPT_DIR"

# Auto-detect ESP32 port
PORT=""
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    PORT=$(ls /dev/cu.usbmodem* 2>/dev/null | head -1)
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    PORT=$(ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null | head -1)
fi

if [ -z "$PORT" ]; then
    echo "Error: Could not auto-detect ESP32 port"
    echo "Please specify the port manually:"
    echo "  ./deploy.sh /dev/cu.usbmodem1101"
    exit 1
fi

echo "Using ESP32 port: $PORT"

# Check if ESP-IDF environment is set up
if [ -z "$IDF_PATH" ]; then
    echo "Setting up ESP-IDF environment..."
    source ~/esp/esp-idf/export.sh
fi

# Change to the project directory
cd "$PROJECT_DIR"

# Verify we're in the right place
if [ ! -f "CMakeLists.txt" ]; then
    echo "Error: CMakeLists.txt not found in $PROJECT_DIR"
    exit 1
fi

# Clean previous build
echo "Cleaning previous build..."
idf.py fullclean

# Build and flash in one command
echo "Building and flashing to $PORT..."
idf.py -p "$PORT" build flash

echo "✓ Build and flash completed!"
echo "ESP32 IMU is now running and sending data via serial."
echo ""
echo "To monitor the output:"
echo "  screen $PORT 115200"
echo ""
echo "To exit screen: Ctrl+A, then K, then Y"
