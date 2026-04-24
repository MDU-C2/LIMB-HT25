"""
IMU Movement Intention to Elbow Control

This script reads IMU movement intention and sends elbow actuation commands via CAN.
It maps movement directions to elbow angles and sends commands to the elbow node.

Usage:
    python imu_elbow_control.py [--interface <interface>] [--bitrate <bitrate>]
"""

import sys
import time
import argparse
import os
import yaml
from typing import Optional

# Add parent directory to path
test_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(test_dir, '../..'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from hardware.can.can_socketcan import SocketCANInterface
from hardware.can.can_message_parser import CANMessageParser
from hardware.ble.ble_bleak import BleakBLEInterface
from layers.input.input_layer import InputLayer
from layers.processing.processing_layer import ProcessingLayer
from shared.queues import DataQueue


class IMUElbowController:
    """Controller that maps IMU movement intention to elbow actuation."""
    
    # Elbow node CAN IDs
    ELBOW_ACTUATION_ID = 0x240
    ELBOW_STOP_ID = 0x140
    
    def __init__(self, interface: str = "can0", bitrate: int = 1000000):
        self.can_interface = SocketCANInterface(interface=interface, bitrate=bitrate)
        self.can_parser = CANMessageParser()
        self.running = False
        
        # Elbow angle state (all in degrees)
        self.current_elbow_angle = 0.0  # Current angle in degrees
        self.target_elbow_angle = 0.0   # Target angle in degrees
        
        # Configuration (all in degrees)
        self.angle_step_size = 5.7      # Degrees per movement detection
        self.min_angle = 0.0            # Minimum angle in degrees
        self.max_angle = 90.0            # Maximum angle in degrees
        self.confidence_threshold = 0.3  # Minimum confidence to act
        
        # Movement direction to angle mapping
        # Only "up" and "down" are used for elbow control
        # "up" = bend elbow (positive angle), "down" = extend elbow (negative angle)
        self.direction_to_angle_delta = {
            "up": self.angle_step_size,      # Bend elbow
            "down": -self.angle_step_size,   # Extend elbow
            "none": 0.0                       # No movement
        }
    
    def start(self) -> bool:
        """Start the CAN interface."""
        if self.can_interface.start():
            self.running = True
            print("✓ CAN interface started")
            return True
        else:
            print("✗ Failed to start CAN interface")
            return False
    
    def stop(self):
        """Stop the CAN interface."""
        self.can_interface.stop()
        self.running = False
        print("✓ CAN interface stopped")
    
    def map_direction_to_angle(self, movement_intention: dict) -> Optional[float]:
        """
        Map IMU movement intention to elbow angle change.
        
        Args:
            movement_intention: Dict with 'direction', 'confidence', 'is_still'
        
        Returns:
            Target elbow angle in degrees, or None if no movement should occur
        """
        if not movement_intention:
            return None
        
        direction = movement_intention.get("direction", "none")
        confidence = movement_intention.get("confidence", 0.0)
        is_still = movement_intention.get("is_still", False)
        
        # Don't act if confidence is too low or device is still
        if confidence < self.confidence_threshold or is_still:
            return None
        
        # Only process "up" and "down" directions
        if direction not in ["up", "down"]:
            return None
        
        # Get angle delta for this direction
        angle_delta = self.direction_to_angle_delta.get(direction, 0.0)
        
        if angle_delta == 0.0:
            # No elbow movement for this direction
            return None
        
        # Calculate new target angle
        new_target = self.current_elbow_angle + angle_delta
        
        # Clamp to joint limits
        new_target = max(self.min_angle, min(self.max_angle, new_target))
        
        return new_target
    
    def send_elbow_command(self, angle_deg: float) -> bool:
        """
        Send elbow actuation command via CAN.
        
        Args:
            angle_deg: Target angle in degrees
        """
        if not self.running:
            return False
        
        # Encode using parser (angle in degrees, velocity in degrees/s)
        encoded = self.can_parser.encode(
            "robot_elbow_up_down_actuation",
            {"angle": angle_deg, "velocity": 5.0},
        )
        if not encoded:
            print(f"✗ Failed to encode elbow command")
            return False
        
        can_id, data = encoded
        
        # Verify CAN ID
        if can_id != self.ELBOW_ACTUATION_ID:
            print(f"⚠ Warning: CAN ID mismatch! Expected {self.ELBOW_ACTUATION_ID:03X}, got {can_id:03X}")
        
        success = self.can_interface.send(can_id, data)
        
        if success:
            self.target_elbow_angle = angle_deg
            print(f"✓ Sent elbow command: {angle_deg:.1f}°")
        else:
            print(f"✗ Failed to send elbow command")
        
        return success
    
    def send_stop_command(self) -> bool:
        """Send stop command to elbow node."""
        if not self.running:
            return False
        
        success = self.can_interface.send(self.ELBOW_STOP_ID, b"")
        
        if success:
            print("✓ Sent stop command")
        else:
            print("✗ Failed to send stop command")
        
        return success
    
    def update_from_movement_intention(self, movement_intention: dict) -> bool:
        """
        Update elbow position based on movement intention.
        
        Args:
            movement_intention: Movement intention dict from processing layer
        
        Returns:
            True if command was sent, False otherwise
        """
        target_angle = self.map_direction_to_angle(movement_intention)
        
        if target_angle is None:
            return False
        
        # Only send if angle changed significantly (0.5 degrees threshold)
        if abs(target_angle - self.current_elbow_angle) < 0.5:
            return False
        
        success = self.send_elbow_command(target_angle)
        
        if success:
            self.current_elbow_angle = target_angle
        
        return success
    
    def run_with_processing_layer(self, processing_queue):
        """
        Run controller by reading from processing layer queue.
        
        This integrates with the existing system pipeline.
        """
        if not self.start():
            return
        
        print("\n" + "="*60)
        print("IMU Elbow Controller Running")
        print("="*60)
        print("Waiting for movement intention data...")
        print("Press Ctrl+C to stop\n")
        
        try:
            while self.running:
                try:
                    # Get processed packet from queue
                    packet = processing_queue.get(timeout=0.1)
                    
                    # Extract movement intention
                    movement_intention = packet.metadata.get("movement_intention")
                    
                    if movement_intention:
                        direction = movement_intention.get("direction", "none")
                        confidence = movement_intention.get("confidence", 0.0)
                        
                        # Only process "up" and "down" movements
                        if direction in ["up", "down"]:
                            print(f"Movement: {direction} (confidence: {confidence:.2f})")
                            
                            # Update elbow based on movement
                            self.update_from_movement_intention(movement_intention)
                    
                except Exception as e:
                    # Queue timeout or other error - continue
                    time.sleep(0.01)
                    continue
        
        except KeyboardInterrupt:
            print("\n\nStopping controller...")
        finally:
            self.send_stop_command()
            self.stop()
    
    def run_standalone_test(self, movement_intentions: list):
        """
        Run controller with test movement intentions.
        
        Useful for testing without full pipeline.
        """
        if not self.start():
            return
        
        print("\n" + "="*60)
        print("IMU Elbow Controller - Standalone Test")
        print("="*60)
        
        for i, intention in enumerate(movement_intentions):
            print(f"\nTest {i+1}: {intention}")
            self.update_from_movement_intention(intention)
            time.sleep(0.5)
        
        print("\n" + "="*60)
        print("Test completed")
        print("="*60)
        
        self.send_stop_command()
        self.stop()
    
    def run_live_mode(self, ble_device_name: str = "LIMBServer", ble_scan_timeout: float = 10.0):
        """
        Run controller with real IMU hardware via BLE.
        
        This sets up the full pipeline:
        BLE IMU → Input Layer → Processing Layer → Movement Intention → Elbow Controller
        
        Args:
            ble_device_name: BLE device name to connect to
            ble_scan_timeout: Timeout for BLE device scan
        """
        print("\n" + "="*60)
        print("IMU Elbow Controller - Live Mode")
        print("="*60)
        print("Setting up hardware pipeline...")
        
        # Load config for processing layer parameters
        config_path = os.path.join(os.path.dirname(__file__), "../../config/system_config.yaml")
        config = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            except Exception as e:
                print(f"⚠ Warning: Could not load config: {e}")
                print("Using default parameters")
        
        proc_config = config.get("processing_layer", {})
        imu_config = proc_config.get("imu_intention", {})
        
        # Initialize hardware interfaces
        print("\n1. Initializing CAN interface...")
        can_interface = SocketCANInterface(interface=self.can_interface.interface, bitrate=self.can_interface.bitrate)
        
        print("2. Initializing BLE interface...")
        print(f"   Looking for device: {ble_device_name}")
        ble_interface = BleakBLEInterface(
            device_name=ble_device_name,
            scan_timeout=ble_scan_timeout
        )
        
        # Create queues
        print("3. Creating data queues...")
        input_to_processing = DataQueue(max_size=5)
        processing_to_control = DataQueue(max_size=5)
        
        # Initialize layers
        print("4. Initializing Input Layer...")
        input_layer = InputLayer(
            can_interface=can_interface,
            ble_interface=ble_interface,
            output_queue=input_to_processing,
            window_size=100,
            sample_rate=100.0
        )
        
        print("5. Initializing Processing Layer...")
        processing_layer = ProcessingLayer(
            input_queue=input_to_processing,
            output_queue=processing_to_control,
            # IMU intention parameters (only parameters that ProcessingLayer accepts)
            imu_accel_threshold=imu_config.get("accel_threshold", 0.3),
            imu_gravity_removal=imu_config.get("gravity_removal", True),
        )
        
        # Set IMU intention parameters that are instance variables
        processing_layer.imu_velocity_threshold = imu_config.get("velocity_threshold", 0.2)
        processing_layer.imu_direction_timeout = imu_config.get("direction_timeout", 4.0)
        processing_layer.ACCEL_STILL_THRESH = imu_config.get("accel_still_thresh", 0.5)
        processing_layer.GYRO_STILL_THRESH = imu_config.get("gyro_still_thresh", 0.1)
        processing_layer.DEADBAND_THRESH = imu_config.get("deadband_thresh", 0.15)
        processing_layer.VELOCITY_DECAY = imu_config.get("velocity_decay", 0.95)
        processing_layer.GRAVITY_EMA_ALPHA = imu_config.get("gravity_ema_alpha", 0.05)
        processing_layer.imu_madgwick_beta = imu_config.get("madgwick_beta", 0.05)
        processing_layer.imu_bias_update_duration = imu_config.get("bias_update_duration", 0.5)
        
        # Start layers
        print("\n6. Starting layers...")
        try:
            input_layer.start()
            time.sleep(0.5)
            print("   ✓ Input Layer started")
            
            processing_layer.start()
            time.sleep(0.5)
            print("   ✓ Processing Layer started")
            
            print("\n" + "="*60)
            print("System Ready!")
            print("="*60)
            print("Move the IMU up/down to control the elbow")
            print("Press Ctrl+C to stop\n")
            
            # Run controller with processing queue
            self.run_with_processing_layer(processing_to_control)
            
        except KeyboardInterrupt:
            print("\n\nStopping system...")
        except Exception as e:
            print(f"\n✗ Error during execution: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("\n7. Shutting down layers...")
            try:
                processing_layer.stop()
                processing_layer.join(timeout=2.0)
                print("   ✓ Processing Layer stopped")
            except Exception as e:
                print(f"   ✗ Error stopping Processing Layer: {e}")
            
            try:
                input_layer.stop()
                input_layer.join(timeout=2.0)
                print("   ✓ Input Layer stopped")
            except Exception as e:
                print(f"   ✗ Error stopping Input Layer: {e}")
            
            try:
                can_interface.stop()
                print("   ✓ CAN interface stopped")
            except Exception as e:
                print(f"   ✗ Error stopping CAN interface: {e}")
            
            try:
                ble_interface.stop()
                print("   ✓ BLE interface stopped")
            except Exception as e:
                print(f"   ✗ Error stopping BLE interface: {e}")
            
            print("\nShutdown complete!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="IMU movement intention to elbow control")
    parser.add_argument("--interface", default="can0", help="CAN interface (default: can0)")
    parser.add_argument("--bitrate", type=int, default=1000000, help="CAN bitrate (default: 1000000)")
    parser.add_argument("--test", action="store_true", help="Run standalone test with mock data")
    parser.add_argument("--live", action="store_true", help="Run with real IMU hardware via BLE")
    parser.add_argument("--ble-device", default="LIMBServer", help="BLE device name (default: LIMBServer)")
    parser.add_argument("--ble-timeout", type=float, default=10.0, help="BLE scan timeout in seconds (default: 10.0)")
    parser.add_argument("--angle-step", type=float, default=5.7, help="Angle step size in degrees (default: 5.7°)")
    parser.add_argument("--confidence-threshold", type=float, default=0.3, help="Minimum confidence threshold (default: 0.3)")
    
    args = parser.parse_args()
    
    controller = IMUElbowController(interface=args.interface, bitrate=args.bitrate)
    controller.angle_step_size = args.angle_step
    controller.confidence_threshold = args.confidence_threshold
    
    if args.test:
        # Run standalone test with mock data
        test_intentions = [
            {"direction": "up", "confidence": 0.8, "is_still": False},
            {"direction": "up", "confidence": 0.7, "is_still": False},
            {"direction": "none", "confidence": 0.1, "is_still": True},
            {"direction": "down", "confidence": 0.6, "is_still": False},
            {"direction": "down", "confidence": 0.5, "is_still": False},
        ]
        controller.run_standalone_test(test_intentions)
    elif args.live:
        # Run with real IMU hardware
        controller.run_live_mode(
            ble_device_name=args.ble_device,
            ble_scan_timeout=args.ble_timeout
        )
    else:
        print("No mode specified. Use one of:")
        print("  --test    : Run with mock movement intentions")
        print("  --live    : Run with real IMU hardware via BLE")
        print("\nExample:")
        print("  python test_imu_elbow_control.py --live --ble-device LIMBServer")


if __name__ == "__main__":
    main()
