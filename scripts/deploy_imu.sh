#!/bin/bash

# ESP-IDF IMU Project Deploy Script

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Navigate to the project root (one level up from scripts)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$PROJECT_ROOT/src/imu/imu_test"
PORT="/dev/cu.usbmodem1101"

echo "Building and flashing IMU project..."

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

# Build and flash in one command
idf.py -p "$PORT" build flash

echo "Done! IMU is now running on the ESP32."
echo "To monitor: screen $PORT 115200"