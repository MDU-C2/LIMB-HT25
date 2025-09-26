"""
Main IMU system script for the robotic arm project.

This script provides functionality for:
- Reading data from 4 IMUs (2 robotic arm, 2 EMG band)
- Calibrating IMU sensors
- Real-time data collection and processing
- Integration with the vision system
"""

import argparse
import time
import json
import logging
from typing import Dict, Optional
import numpy as np

from .imu_interface import IMUManager
from .data_structures import IMULocation, IMUDataCollection
from .calibration import IMUCalibrator, IMUCalibrationCollector, IMUDataProcessor


def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def print_imu_status(imu_manager: IMUManager) -> None:
    """Print connection status of all IMUs."""
    print("\n=== IMU Connection Status ===")
    status = imu_manager.get_connection_status()
    for location, connected in status.items():
        status_str = "✓ Connected" if connected else "✗ Disconnected"
        print(f"{location.value}: {status_str}")
    print()


def print_imu_data(data_collection: IMUDataCollection) -> None:
    """Print IMU data collection in a formatted way."""
    print(f"\n=== IMU Data Collection (t={data_collection.timestamp:.3f}) ===")
    
    for location in IMULocation:
        reading = data_collection.get_reading(location)
        if reading:
            print(f"\n{location.value}:")
            print(f"  Accel: [{reading.accel[0]:6.3f}, {reading.accel[1]:6.3f}, {reading.accel[2]:6.3f}] m/s²")
            print(f"  Gyro:  [{reading.gyro[0]:6.3f}, {reading.gyro[1]:6.3f}, {reading.gyro[2]:6.3f}] rad/s")
            print(f"  Mag:   [{reading.mag[0]:6.1f}, {reading.mag[1]:6.1f}, {reading.mag[2]:6.1f}] μT")
            if reading.orientation is not None:
                print(f"  Quat:  [{reading.orientation[0]:6.3f}, {reading.orientation[1]:6.3f}, {reading.orientation[2]:6.3f}, {reading.orientation[3]:6.3f}]")
            if reading.angular_velocity_magnitude is not None:
                print(f"  |ω|:   {reading.angular_velocity_magnitude:.3f} rad/s")
            if reading.linear_acceleration_magnitude is not None:
                print(f"  |a|:   {reading.linear_acceleration_magnitude:.3f} m/s²")
        else:
            print(f"\n{location.value}: No data")


def data_callback(data_collection: IMUDataCollection) -> None:
    """Callback function for receiving IMU data collections."""
    # This function is called whenever new IMU data is available
    # You can add custom processing here, such as:
    # - Logging to file
    # - Sending to vision system
    # - Real-time analysis
    
    # For now, just print the data
    print_imu_data(data_collection)


def run_calibration_mode(imu_manager: IMUManager, location: IMULocation) -> None:
    """Run calibration mode for a specific IMU."""
    print(f"\n=== Calibrating {location.value} ===")
    print("Follow these steps:")
    print("1. Keep the IMU stationary for accelerometer calibration (10 seconds)")
    print("2. Rotate the IMU slowly for magnetometer calibration (20 seconds)")
    print("3. Keep the IMU stationary for gyroscope calibration (5 seconds)")
    print("\nPress Enter to start calibration...")
    input()
    
    collector = IMUCalibrationCollector()
    calibrator = IMUCalibrator()
    
    # Collect accelerometer data (stationary)
    print("\nCollecting accelerometer data... Keep IMU stationary!")
    start_time = time.time()
    while time.time() - start_time < 10.0:
        data_collection = imu_manager.read_single_collection()
        reading = data_collection.get_reading(location)
        if reading:
            # Convert back to raw reading for calibration
            raw_reading = reading  # This is a simplified approach
            collector.add_reading(raw_reading, 'accel')
        time.sleep(0.1)
    
    # Collect magnetometer data (rotation)
    print("\nCollecting magnetometer data... Rotate IMU slowly!")
    start_time = time.time()
    while time.time() - start_time < 20.0:
        data_collection = imu_manager.read_single_collection()
        reading = data_collection.get_reading(location)
        if reading:
            raw_reading = reading
            collector.add_reading(raw_reading, 'mag')
        time.sleep(0.1)
    
    # Collect gyroscope data (stationary)
    print("\nCollecting gyroscope data... Keep IMU stationary!")
    start_time = time.time()
    while time.time() - start_time < 5.0:
        data_collection = imu_manager.read_single_collection()
        reading = data_collection.get_reading(location)
        if reading:
            raw_reading = reading
            collector.add_reading(raw_reading, 'gyro')
        time.sleep(0.1)
    
    # Generate calibration
    if collector.has_sufficient_data():
        calibration = collector.get_calibration(location)
        imu_manager.set_calibration(location, calibration)
        print(f"\n✓ Calibration completed for {location.value}")
        print(f"Accel bias: {calibration.accel_bias}")
        print(f"Gyro bias: {calibration.gyro_bias}")
        print(f"Mag bias: {calibration.mag_bias}")
    else:
        print(f"\n✗ Insufficient data for calibration of {location.value}")


def run_data_collection_mode(imu_manager: IMUManager, duration: float = 60.0, sample_rate: float = 100.0) -> None:
    """Run data collection mode."""
    print(f"\n=== Data Collection Mode ===")
    print(f"Duration: {duration} seconds")
    print(f"Sample rate: {sample_rate} Hz")
    print("Press Ctrl+C to stop early")
    
    # Set up data callback
    imu_manager.set_data_callback(data_callback)
    
    # Start reading
    imu_manager.start_reading(sample_rate)
    
    try:
        time.sleep(duration)
    except KeyboardInterrupt:
        print("\nData collection stopped by user")
    finally:
        imu_manager.stop_reading()


def run_single_reading_mode(imu_manager: IMUManager) -> None:
    """Run single reading mode for testing."""
    print("\n=== Single Reading Mode ===")
    print("Press Enter to take a reading, 'q' to quit")
    
    while True:
        user_input = input("\nPress Enter for reading (q to quit): ").strip().lower()
        if user_input == 'q':
            break
        
        data_collection = imu_manager.read_single_collection()
        print_imu_data(data_collection)


def main() -> None:
    """Main function for IMU system."""
    parser = argparse.ArgumentParser(description="IMU System for Robotic Arm")
    parser.add_argument("--mode", choices=["single", "collect", "calibrate"], default="single",
                       help="Operation mode")
    parser.add_argument("--location", type=str, choices=[loc.value for loc in IMULocation],
                       help="IMU location for calibration mode")
    parser.add_argument("--duration", type=float, default=60.0,
                       help="Duration for data collection mode (seconds)")
    parser.add_argument("--sample-rate", type=float, default=100.0,
                       help="Sample rate for data collection (Hz)")
    parser.add_argument("--serial-ports", type=str, nargs=4,
                       help="Serial ports for IMUs in order: robot_forearm, robot_upperarm, emg_forearm, emg_upperarm")
    parser.add_argument("--simulation", action="store_true", default=True,
                       help="Use simulated IMUs (default)")
    parser.add_argument("--real-hardware", action="store_true",
                       help="Use real hardware IMUs")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    
    # Determine if using simulation
    use_simulation = args.simulation and not args.real_hardware
    
    # Set up serial ports if provided
    serial_ports = None
    if args.serial_ports and not use_simulation:
        serial_ports = {
            IMULocation.ROBOT_FOREARM: args.serial_ports[0],
            IMULocation.ROBOT_UPPERARM: args.serial_ports[1],
            IMULocation.EMG_FOREARM: args.serial_ports[2],
            IMULocation.EMG_UPPERARM: args.serial_ports[3],
        }
    
    # Initialize IMU manager
    print("Initializing IMU system...")
    imu_manager = IMUManager(use_simulation=use_simulation, serial_ports=serial_ports)
    
    # Print connection status
    print_imu_status(imu_manager)
    
    try:
        if args.mode == "single":
            run_single_reading_mode(imu_manager)
        elif args.mode == "collect":
            run_data_collection_mode(imu_manager, args.duration, args.sample_rate)
        elif args.mode == "calibrate":
            if not args.location:
                print("Error: --location is required for calibration mode")
                return
            location = IMULocation(args.location)
            run_calibration_mode(imu_manager, location)
    
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        imu_manager.close()
        print("IMU system closed.")


if __name__ == "__main__":
    main()
