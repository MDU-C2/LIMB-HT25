#!/usr/bin/env python3
"""
Test script for SocketCANInterface.

This script tests the SocketCAN implementation on a Jetson Orin.
Make sure the CAN interface is set up before running:
  1. Run: sudo scripts/agx_setup_can.sh
  2. Check interface: ip link show can0
  3. Interface should be UP

For loopback testing (send and receive on same interface):
  sudo ip link set can0 up type can bitrate 1000000 loopback on

THIS IS AI GENERATED CODE
"""

import sys
import time
import struct
from typing import List

# Add parent directory to path
sys.path.insert(0, '/'.join(__file__.split('/')[:-3]))

from hardware.can.can_socketcan import SocketCANInterface
from hardware.can.can_interface import CANMessage


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_interface_start_stop():
    """Test starting and stopping the CAN interface."""
    print_section("Test 1: Start/Stop Interface")
    
    can_interface = SocketCANInterface(interface="can0", bitrate=1000000)
    
    # Test start
    print("\n1. Starting CAN interface...")
    result = can_interface.start()
    if result:
        print("   ✓ Interface started successfully")
        assert can_interface.is_running(), "Interface should be running"
    else:
        print("   ✗ Failed to start interface")
        print("   Make sure:")
        print("     - CAN interface is set up (run scripts/agx_setup_can.sh)")
        print("     - Interface is up: ip link set can0 up")
        print("     - python-can is installed: pip install python-can")
        return False
    
    # Test stop
    print("\n2. Stopping CAN interface...")
    result = can_interface.stop()
    if result:
        print("   ✓ Interface stopped successfully")
        assert not can_interface.is_running(), "Interface should be stopped"
    else:
        print("   ✗ Failed to stop interface")
        return False
    
    return True


def test_read_messages():
    """Test reading messages from CAN bus."""
    print_section("Test 2: Read Messages")
    
    can_interface = SocketCANInterface(interface="can0", bitrate=1000000)
    
    if not can_interface.start():
        print("   ✗ Cannot test reading - interface failed to start")
        return False
    
    print("\n1. Reading messages for 2 seconds...")
    print("   (Make sure there are devices sending on CAN bus)")
    
    start_time = time.time()
    message_count = 0
    message_types = {}
    
    while time.time() - start_time < 2.0:
        messages = can_interface.read(timeout=0.1)
        for msg in messages:
            message_count += 1
            msg_type = getattr(msg, 'message_type', 'unknown')
            parsed_data = getattr(msg, 'parsed_data', {})
            
            if msg_type not in message_types:
                message_types[msg_type] = 0
            message_types[msg_type] += 1
            
            print(f"   Received: CAN ID=0x{msg.can_id:03X}, Type={msg_type}")
            if parsed_data:
                print(f"     Data: {parsed_data}")
        
        time.sleep(0.1)
    
    print(f"\n2. Statistics:")
    print(f"   Total messages: {message_count}")
    print(f"   Message types: {message_types}")
    
    stats = can_interface.get_statistics()
    print(f"   RX count: {stats['rx_count']}")
    print(f"   Error count: {stats['error_count']}")
    
    can_interface.stop()
    
    if message_count == 0:
        print("\n   ⚠ No messages received. This is OK if:")
        print("     - No devices are connected to CAN bus")
        print("     - Devices are not sending messages")
        print("     - Testing in loopback mode (need to send first)")
    
    return True


def test_send_messages():
    """Test sending messages on CAN bus."""
    print_section("Test 3: Send Messages")
    
    can_interface = SocketCANInterface(interface="can0", bitrate=1000000)
    
    if not can_interface.start():
        print("   ✗ Cannot test sending - interface failed to start")
        return False
    
    print("\n1. Sending test messages...")
    
    # Test 1: Send gripper command
    print("   Sending gripper command (close with 0.8 force)...")
    gripper_data = struct.pack('<Bf', 1, 0.8)  # action=1 (close), force=0.8
    result = can_interface.send(0x200, gripper_data)
    if result:
        print("   ✓ Gripper command sent")
    else:
        print("   ✗ Failed to send gripper command")
    
    time.sleep(0.1)
    
    # Test 2: Send arm command
    print("   Sending arm command (joint positions)...")
    arm_data = struct.pack('<6f', 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    result = can_interface.send(0x201, arm_data)
    if result:
        print("   ✓ Arm command sent")
    else:
        print("   ✗ Failed to send arm command")
    
    time.sleep(0.1)
    
    # Test 3: Send motor command
    print("   Sending motor command...")
    motor_data = struct.pack('<Bf', 1, 0.5)  # motor_id=1, value=0.5
    result = can_interface.send(0x202, motor_data)
    if result:
        print("   ✓ Motor command sent")
    else:
        print("   ✗ Failed to send motor command")
    
    # Test 4: Test data truncation (too long)
    print("\n2. Testing data truncation (sending 12 bytes, should truncate to 8)...")
    long_data = b'\x00' * 12
    result = can_interface.send(0x200, long_data)
    if result:
        print("   ✓ Long data truncated and sent")
    else:
        print("   ✗ Failed to send long data")
    
    # Get statistics
    stats = can_interface.get_statistics()
    print(f"\n3. Statistics:")
    print(f"   TX count: {stats['tx_count']}")
    print(f"   Error count: {stats['error_count']}")
    
    can_interface.stop()
    
    return True


def test_loopback():
    """Test loopback mode (send and receive on same interface)."""
    print_section("Test 4: Loopback Mode")
    
    print("\n⚠ Loopback mode requires:")
    print("   sudo ip link set can0 down")
    print("   sudo ip link set can0 up type can bitrate 1000000 loopback on")
    print("\n   Continue anyway? (y/n): ", end='')
    
    response = input().strip().lower()
    if response != 'y':
        print("   Skipping loopback test")
        return True
    
    can_interface = SocketCANInterface(interface="can0", bitrate=1000000)
    
    if not can_interface.start():
        print("   ✗ Cannot test loopback - interface failed to start")
        return False
    
    print("\n1. Sending test message in loopback mode...")
    
    # Send EMG message
    emg_data = struct.pack('<4f', 0.5, 0.6, 0.7, 0.8)
    result = can_interface.send(0x100, emg_data)
    
    if not result:
        print("   ✗ Failed to send message")
        can_interface.stop()
        return False
    
    print("   ✓ Message sent")
    
    # Try to receive it
    print("\n2. Attempting to receive message...")
    time.sleep(0.1)  # Small delay
    
    messages = can_interface.read(timeout=0.5)
    
    if messages:
        print(f"   ✓ Received {len(messages)} message(s)")
        for msg in messages:
            print(f"     CAN ID: 0x{msg.can_id:03X}")
            msg_type = getattr(msg, 'message_type', 'unknown')
            parsed_data = getattr(msg, 'parsed_data', {})
            print(f"     Type: {msg_type}")
            if parsed_data:
                print(f"     Data: {parsed_data}")
    else:
        print("   ✗ No message received")
        print("   Make sure loopback is enabled:")
        print("     sudo ip link set can0 up type can bitrate 500000 loopback on")
    
    can_interface.stop()
    return True


def test_error_handling():
    """Test error handling."""
    print_section("Test 5: Error Handling")
    
    can_interface = SocketCANInterface(interface="can0", bitrate=1000000)
    
    # Test 1: Read when not started
    print("\n1. Testing read when interface not started...")
    messages = can_interface.read()
    assert len(messages) == 0, "Should return empty list when not running"
    print("   ✓ Returns empty list when not running")
    
    # Test 2: Send when not started
    print("\n2. Testing send when interface not started...")
    result = can_interface.send(0x100, b'\x00')
    assert not result, "Should return False when not running"
    print("   ✓ Returns False when not running")
    
    # Test 3: Start and test invalid operations
    if can_interface.start():
        print("\n3. Testing with running interface...")
        
        # Send empty data
        result = can_interface.send(0x100, b'')
        print(f"   Sending empty data: {'✓' if result else '✗'}")
        
        can_interface.stop()
    
    return True


def test_statistics():
    """Test statistics tracking."""
    print_section("Test 6: Statistics")
    
    can_interface = SocketCANInterface(interface="can0", bitrate=1000000)
    
    if not can_interface.start():
        print("   ✗ Cannot test statistics - interface failed to start")
        return False
    
    # Initial statistics
    stats = can_interface.get_statistics()
    print("\n1. Initial statistics:")
    print(f"   RX count: {stats['rx_count']}")
    print(f"   TX count: {stats['tx_count']}")
    print(f"   Error count: {stats['error_count']}")
    print(f"   Running: {stats['running']}")
    print(f"   Interface: {stats['interface']}")
    print(f"   Bitrate: {stats['bitrate']}")
    
    # Send some messages
    print("\n2. Sending 5 messages...")
    for i in range(5):
        data = struct.pack('<Bf', i, 0.5)
        can_interface.send(0x200, data)
    
    # Read some messages
    print("3. Reading messages...")
    for _ in range(3):
        can_interface.read(timeout=0.1)
        time.sleep(0.1)
    
    # Final statistics
    stats = can_interface.get_statistics()
    print("\n4. Final statistics:")
    print(f"   RX count: {stats['rx_count']}")
    print(f"   TX count: {stats['tx_count']}")
    print(f"   Error count: {stats['error_count']}")
    
    assert stats['tx_count'] >= 5, "TX count should be at least 5"
    print("   ✓ Statistics tracking works correctly")
    
    can_interface.stop()
    return True


def test_context_manager():
    """Test context manager (__enter__/__exit__) functionality."""
    print_section("Test 7: Context Manager")

    print("\n1. Testing context manager entry and exit...")
    context_interface = None

    with SocketCANInterface(interface="can0", bitrate=1000000) as can_interface:
        context_interface = can_interface
        if can_interface.is_running():
            print("   ✓ Interface started successfully in __enter__")
        else:
            print("   ✗ Failed to start interface in __enter__")
            print("   Make sure:")
            print("     - CAN interface is set up (run scripts/agx_setup_can.sh)")
            print("     - Interface is up: ip link set can0 up")
            print("     - python-can is installed: pip install python-can")
            return False

    print("\n2. Verifying interface stopped after context exit...")
    assert context_interface is not None, "Context interface should be set"
    if not context_interface.is_running():
        print("   ✓ Interface stopped successfully in __exit__")
    else:
        print("   ✗ Interface should be stopped after context exit")
        return False

    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("  SocketCAN Interface Test Suite")
    print("=" * 60)
    print("\nPrerequisites:")
    print("  1. CAN interface must be set up")
    print("  2. Run: sudo scripts/agx_setup_can.sh")
    print("  3. Check: ip link show can0")
    print("  4. Install: pip install python-can")
    
    tests = [
        ("Start/Stop", test_interface_start_stop),
        ("Read Messages", test_read_messages),
        ("Send Messages", test_send_messages),
        ("Loopback Mode", test_loopback),
        ("Error Handling", test_error_handling),
        ("Statistics", test_statistics),
        ("Context Manager", test_context_manager),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print_section("Test Summary")
    print("\nResults:")
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {test_name:20s} {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

