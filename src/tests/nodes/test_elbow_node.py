"""

This script tests the elbow node hardware.
It sends actuation commands, stop commands, and reads feedback messages.
It also tests the interactive control mode.
It can be run with different options to test different aspects of the elbow node.

Usage:
    python test_elbow_node.py [--interface <interface>] [--bitrate <bitrate>] [--interactive] [--feedback-only]

Options:
    --interface <interface>  CAN interface (default: can0)
    --bitrate <bitrate>      CAN bitrate (default: 1000000)
    --interactive            Run interactive test
    --feedback-only          Only test feedback reading

Ensure CAN is set up on AGX:
    # Run the setup script (as root)
    sudo scripts/agx_setup_can.sh

    # Verify CAN interface is up
    ip link show can0

    # Check CAN statistics
    ip -s -s link show can0
"""



import sys
import time
import struct
from typing import Optional
import os

# Add parent directory to path
test_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(test_dir, "../../"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from hardware.can.can_socketcan import SocketCANInterface
from hardware.can.can_message_parser import CANMessageParser

class ElbowNodeTester:
    """Test harness for the elbow node."""

    # Elbow node CAN IDs
    ELBOW_ACTUATION_ID = 0x240
    ELBOW_STOP_ID = 0x140
    ELBOW_POTENTIOMETER_ID = 0x4A0
    ELBOW_IMU_GYRO_ID = 0x5A2
    ELBOW_IMU_ACCEL_ID = 0x5A3

    def __init__(self, interface: str = "can0", bitrate: int = 1000000):
        self.can_interface = SocketCANInterface(interface="can0", bitrate=1000000)
        self.can_parser = CANMessageParser()
        self.running = False

    def start(self):
        """Start the CAN interface."""
        if self.can_interface.start():
            self.running = True
            print("CAN started.")
            return True
        else:
            print("Failed to start CAN interface.")
            return False

    def stop(self):
        """Stop the CAN interface."""
        self.can_interface.stop()
        self.running = False
        print("CAN stopped.")

    def send_actuation_command(self, angle: float, velocity: float = 5.0) -> bool:
        """
        Send an actuation command to the elbow node.
        """
        if not self.running:
            print("CAN interface not running.")
            return False

        encoded = self.can_parser.encode(
            "robot_elbow_up_down_actuation",
            {"angle": angle, "velocity": velocity},
        )
        if not encoded:
            print("Failed to encode actuation command.")
            return False
        
        can_id, data = encoded
        success = self.can_interface.send(can_id, data)

        if success:
            print(f"Sent actuation command: angle={angle}, velocity={velocity}")
        else:
            print(f"Failed to send actuation command: angle={angle}, velocity={velocity}")

        return success
    
    def send_stop_command(self) -> bool:
        """Send a stop command to the elbow node."""
        if not self.running:
            print("CAN interface not running.")
            return False
        
        success = self.can_interface.send(self.ELBOW_STOP_ID, b"")

        if success:
            print("Sent stop command.")
        else:
            print("Failed to send stop command.")
        
        return success

    def read_feedback(self, timeout: float = 1.0) -> dict:
        """
        Read feedback messages from the elbow node.
        """
        if not self.running:
            return {}

        feedback = {
            "potentiometer": None,
            "imu_gyro": None,
            "imu_accel": None,
        }

        messages = self.can_interface.read(timeout=timeout)

        for msg in messages:
            msg_type = getattr(msg, "message_type", None)
            parsed_data = getattr(msg, "parsed_data", None)

            if msg_type == "robot_elbow_up_down_potentiometer":
                feedback["potentiometer"] = parsed_data.get("value")
            elif msg_type == "robot_upper_arm_imu_gyro":
                feedback["imu_gyro"] = parsed_data.get("data")
            elif msg_type == "robot_upper_arm_imu_accel":
                feedback["imu_accel"] = parsed_data.get("data")

        return feedback

    def test_actuation(self):
        """Test the actuation command with different values."""
        print("\n" + "="*60)
        print("TEST 1: Actuation Commands")
        print("="*60)
        
        test_values = [0.0, 10.0, 20.0, 10.0, 0.0]
        
        for value in test_values:
            print(f"\nSending actuation: {value}")
            self.send_actuation_command(value)
            time.sleep(0.5)  # Wait for motor to respond
            
            # Read potentiometer feedback
            feedback = self.read_feedback(timeout=0.2)
            if feedback['potentiometer'] is not None:
                print(f"  Potentiometer reading: {feedback['potentiometer']}")

    def test_stop_command(self):
        """Test stop command."""
        print("\n" + "="*60)
        print("TEST 2: Stop Command")
        print("="*60)

        # Send a movement command first
        print("\n1. Sending movement command...")
        self.send_actuation_command(0.5)
        time.sleep(1.0)

        # Send stop command
        print("\n2. Sending stop command...")
        self.send_stop_command()
        time.sleep(0.5)
        
        # Read potentiometer feedback and check if movement stopped
        feedback = self.read_feedback(timeout=0.5)
        print(f"\n3. Potentiometer feedback: {feedback['potentiometer']}")

    def test_feedback_reading(self, duration: float = 5.0):
        """Test reading feedback messages continuously."""
        print("\n" + "="*60)
        print(f"TEST 3: Feedback Reading (for {duration}s)")
        print("="*60)

        print("\nReading feedback continuously...")
        print("Press Ctrl+C to stop early\n")

        start_time = time.time()
        message_counts = {
            'potentiometer': 0,
            'imu_gyro': 0,
            'imu_accel': 0
        }

        try:
            while time.time() - start_time < duration:
                feedback = self.read_feedback(timeout=0.1)
                
                if feedback['potentiometer'] is not None:
                    message_counts['potentiometer'] += 1
                    print(f"  Potentiometer: {feedback['potentiometer']:.4f}")
                
                if feedback['imu_gyro'] is not None:
                    message_counts['imu_gyro'] += 1
                    gx, gy, gz = feedback['imu_gyro']
                    print(f"  IMU Gyro: [{gx:.4f}, {gy:.4f}, {gz:.4f}]")
                
                if feedback['imu_accel'] is not None:
                    message_counts['imu_accel'] += 1
                    ax, ay, az = feedback['imu_accel']
                    print(f"  IMU Accel: [{ax:.4f}, {ay:.4f}, {az:.4f}]")
                
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            print("\n\nStopped by user")
        
        print(f"\nMessage counts:")
        print(f"  Potentiometer: {message_counts['potentiometer']}")
        print(f"  IMU Gyro: {message_counts['imu_gyro']}")
        print(f"  IMU Accel: {message_counts['imu_accel']}")


    def test_interactive_control(self):
        """Interactive test - user can control the elbow."""
        print("\n" + "="*60)
        print("TEST 4: Interactive Control")
        print("="*60)
        print("\nCommands:")
        print("  <value>  - Send actuation command (e.g., 0.5, -0.3)")
        print("  stop     - Send stop command")
        print("  read     - Read current feedback")
        print("  quit     - Exit")
        print()
        
        while True:
            try:
                cmd = input("Elbow> ").strip().lower()
                
                if cmd == "quit":
                    break
                elif cmd == "stop":
                    self.send_stop_command()
                elif cmd == "read":
                    feedback = self.read_feedback(timeout=0.5)
                    print(f"  Potentiometer: {feedback.get('potentiometer', 'N/A')}")
                    if feedback['imu_gyro']:
                        print(f"  IMU Gyro: {feedback['imu_gyro']}")
                    if feedback['imu_accel']:
                        print(f"  IMU Accel: {feedback['imu_accel']}")
                else:
                    try:
                        value = float(cmd)
                        self.send_actuation_command(value)
                    except ValueError:
                        print("  Invalid command")
            
            except KeyboardInterrupt:
                print("\nExiting...")
                break

    def run_all_tests(self):
        """Run all automated tests."""
        if not self.start():
            return False
        
        try:
            self.test_feedback_reading(duration=3.0)
            self.test_actuation()
            self.test_stop_command()
            
            print("\n" + "="*60)
            print("All tests completed!")
            print("="*60)
            
            stats = self.can_interface.get_statistics()
            print(f"\nCAN Statistics:")
            print(f"  TX count: {stats['tx_count']}")
            print(f"  RX count: {stats['rx_count']}")
            print(f"  Errors: {stats['error_count']}")
        
        finally:
            self.stop()

    
def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test elbow node hardware")
    parser.add_argument("--interface", default="can0", help="CAN interface (default: can0)")
    parser.add_argument("--bitrate", type=int, default=1000000, help="CAN bitrate (default: 1000000)")
    parser.add_argument("--interactive", action="store_true", help="Run interactive test")
    parser.add_argument("--feedback-only", action="store_true", help="Only test feedback reading")
    
    args = parser.parse_args()
    
    tester = ElbowNodeTester(interface=args.interface, bitrate=args.bitrate)
    
    if not tester.start():
        print("\nFailed to start CAN interface.")
        print("Make sure:")
        print("  1. CAN is set up: sudo scripts/agx_setup_can.sh")
        print("  2. Interface is up: ip link show can0")
        print("  3. Elbow node is connected and powered")
        sys.exit(1)
    
    try:
        if args.interactive:
            tester.test_interactive_control()
        elif args.feedback_only:
            tester.test_feedback_reading(duration=10.0)
        else:
            tester.run_all_tests()
    finally:
        tester.stop()


if __name__ == "__main__":
    main()
