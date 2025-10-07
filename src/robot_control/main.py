"""
Main Entry Point for Robotic Manipulation System

This is the main orchestrator that initializes all components and starts the
state machine for robotic cup manipulation.
"""

import sys
import os
import time
import argparse
import numpy as np
from typing import Optional

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sensors import SensorManager
from robot_control.state_machine import StateMachine
from robot_control.hardware.robot_arm import RobotArm

def load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from file."""
    # TODO: Implement actual config loading
    default_config = {
        'camera_matrix': None,  # Will be auto-detected
        'dist_coeffs': None,    # Will be auto-detected
        'imu_port': None,       # Will be auto-detected
        'imu_baudrate': 115200,
        'robot_arm_config': {
            'connection_type': 'ethernet',
            'ip_address': '192.168.1.100',
            'port': 502
        }
    }
    
    if config_path and os.path.exists(config_path):
        # TODO: Load from JSON/YAML file
        pass
    
    return default_config

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Robotic Manipulation System")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--imu-port", type=str, help="IMU serial port")
    parser.add_argument("--imu-baudrate", type=int, default=115200, help="IMU baudrate")
    parser.add_argument("--camera-device", type=int, default=0, help="Camera device ID")
    parser.add_argument("--test-sensors", action="store_true", help="Test sensors and exit")
    parser.add_argument("--simulate", action="store_true", help="Run in simulation mode")
    
    args = parser.parse_args()
    
    print("Robotic Manipulation System")
    print("=" * 50)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.imu_port:
        config['imu_port'] = args.imu_port
    if args.imu_baudrate:
        config['imu_baudrate'] = args.imu_baudrate
    
    try:
        # Initialize sensor manager
        print("Initializing sensor manager...")
        sensor_manager = SensorManager()
        
        # Initialize robot arm
        print("Initializing robot arm...")
        robot_arm = RobotArm()
        
        if not args.simulate:
            # Connect to robot arm hardware
            if not robot_arm.connect():
                print("Warning: Failed to connect to robot arm, continuing in simulation mode")
                args.simulate = True
        
        # Test sensors if requested
        if args.test_sensors:
            print("\nTesting sensors...")
            test_sensors(sensor_manager)
            return
        
        # Initialize state machine
        print("Initializing state machine...")
        state_machine = StateMachine(sensor_manager, robot_arm)
        
        # Print system status
        print_system_status(sensor_manager, robot_arm, state_machine)
        
        # Start the manipulation task
        print("\nStarting robotic manipulation task...")
        print("Press Ctrl+C to stop")
        
        state_machine.run()
        
    except KeyboardInterrupt:
        print("\nSystem interrupted by user")
    except Exception as e:
        print(f"System error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("\nCleaning up system...")
        try:
            if 'sensor_manager' in locals():
                sensor_manager.cleanup()
            if 'robot_arm' in locals():
                robot_arm.cleanup()
        except Exception as e:
            print(f"Error during cleanup: {e}")
        
        print("System shutdown completed")

def test_sensors(sensor_manager: SensorManager):
    """Test all available sensors."""
    available_sensors = ['imu', 'vision', 'pressure', 'slip', 'piezo']
    
    print("Testing available sensors...")
    for sensor_name in available_sensors:
        try:
            success = sensor_manager.test_sensor(sensor_name)
            if success:
                print(f"✓ {sensor_name} test passed")
            else:
                print(f"✗ {sensor_name} test failed")
        except Exception as e:
            print(f"✗ {sensor_name} test error: {e}")
    
    print("\nSensor testing completed")

def print_system_status(sensor_manager: SensorManager, robot_arm: RobotArm, state_machine: StateMachine):
    """Print system status information."""
    print("\nSystem Status:")
    print("-" * 30)
    
    # Sensor status
    print("Sensors:")
    sensor_status = sensor_manager.get_status()
    for sensor_name in sensor_status['available_sensors']:
        status = "active" if sensor_name in sensor_status['active_sensors'] else "inactive"
        print(f"  - {sensor_name}: {status}")
    
    # Robot arm status
    print("Robot Arm:")
    robot_status = robot_arm.get_status()
    print(f"  - Connected: {robot_status['connected']}")
    print(f"  - Position: {robot_status['current_position']}")
    print(f"  - Gripper: {'open' if robot_status['gripper_open'] else 'closed'}")
    
    # State machine status
    print("State Machine:")
    sm_status = state_machine.get_status()
    print(f"  - Current state: {sm_status['current_state']}")
    print(f"  - Running: {sm_status['running']}")
    print(f"  - Available states: {len(sm_status['available_states']['waiting']) + len(sm_status['available_states']['action'])}")

if __name__ == "__main__":
    main()
