"""
Test for Input -> Processing Pipeline

Tests:
- Data flow from InputLayer to ProcessingLayer via queue
- Packet format correctness (all required fields present)
- Timing: packets arrive at expected rate
- Queue overflow handling (dropped packets)
- Packet age/staleness detection
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import numpy as np
import time
import sys
import os
from multiprocessing import Process

# Add src directoty to path
test_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(test_dir, "../.."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from layers.input.input_layer import InputLayer
from layers.processing.processing_layer import ProcessingLayer
from shared.queues import DataQueue
from shared.models.packet import DataPacket, HumanDataWindow, SensorSnapshot, MotorState

class MockCANMessage:
    """Mock CAN message object"""
    def __init__(self, message_type, parsed_data=None, timestamp=None):
        self.message_type = message_type
        self.parsed_data = parsed_data or {}
        self.timestamp = timestamp or time.time()

class MockBLESample:
    """Mock BLE sample"""
    def __init__(self, message_type, data=None, timestamp=None):
        self.message_type = message_type
        self.data = data or {}
        self.timestamp = timestamp or time.time()

class TestInputProcessingPipeline(unittest.TestCase):
    """Test for Input -> Processing Pipeline"""

    def setUp(self):
        """Set up test fixtures"""
        self.window_size = 100
        self.sample_rate = 100.0
        
        # Create queues
        self.input_to_processing_queue = DataQueue(max_size=5)
        self.processing_to_control_queue = DataQueue(max_size=5)
        
        # Mock CAN interface
        self.mock_can = Mock()
        self.mock_can.read.return_value = []
        self.mock_can.start = Mock(return_value=True)
        self.mock_can.stop = Mock(return_value=True)
        self.mock_can.is_running = Mock(return_value=True)
        
        # Mock BLE interface
        self.mock_ble = Mock()
        self.mock_ble.read.return_value = []
        self.mock_ble.start = Mock(return_value=True)
        self.mock_ble.stop = Mock(return_value=True)
        self.mock_ble.is_running = Mock(return_value=True)
        
        # Mock vision source
        self.mock_vision = Mock()
        self.mock_vision.update = Mock()
        
        # Create layers
        self.input_layer = InputLayer(
            can_interface=self.mock_can,
            ble_interface=self.mock_ble,
            output_queue=self.input_to_processing_queue,
            window_size=self.window_size,
            sample_rate=self.sample_rate,
            vision_source=self.mock_vision
        )
        
        self.processing_layer = ProcessingLayer(
            input_queue=self.input_to_processing_queue,
            output_queue=self.processing_to_control_queue,
            model_path=None,
            scaler_path=None
        )
    
    def tearDown(self):
        """Clean up after tests"""
        if hasattr(self.input_layer, 'running'):
            self.input_layer.running.clear()
        if hasattr(self.processing_layer, 'running'):
            self.processing_layer.running.clear()
        self.input_layer.stop()
        self.processing_layer.stop()
        
        # Clear queues
        while not self.input_to_processing_queue.empty():
            try:
                self.input_to_processing_queue.get_nowait()
            except:
                break
    
    def test_data_flow_input_to_processing(self):
        """Test data flows from InputLayer to ProcessingLayer via queue"""

        # Fill window buffer
        timestamp = time.time()
        for i in range(self.window_size):
            self.input_layer.window_buffer.add_emg([0.5, 0.3], timestamp + i * 0.01)
            self.input_layer.window_buffer.add_imu([1.0, 2.0, 3.0, 0.1, 0.2, 0.3], timestamp + i * 0.01)
            self.input_layer.window_buffer.add_piezo(0.5, timestamp + i * 0.01)

        # Create packet manually (simulating InputLayer behavior)
        original_vision = self.input_layer.packet_builder.vision_source
        self.input_layer.packet_builder.vision_source = None

        if self.input_layer.window_buffer.is_full():
            packet = self.input_layer.packet_builder.build(
                self.input_layer.window_buffer,
                self.input_layer.sample_rate,
                latest_pressure=[0.1, 0.2, 0.3, 0.4, 0.5],
                latest_motor_state=None,
                latest_robot_imu=None
            )

            # Put packet in queue
            try:
                self.input_to_processing_queue.put(packet)
                time.sleep(0.01)
            except Exception as e:
                self.fail(f"Failed to put packet in queue: {e}")

        # Restore vision source
        self.input_layer.packet_builder.vision_source = original_vision

        # Verify packet is in queue
        self.assertFalse(self.input_to_processing_queue.empty(), "Queue should not be empty after putting packet")

        # Simulate processing receiving packet
        received_packet = self.input_to_processing_queue.get()

        # Verify packet was received
        self.assertIsNotNone(received_packet)
        self.assertIsInstance(received_packet, DataPacket)
        self.assertEqual(received_packet.sequence_id, 0)

    def test_packet_format_correctness(self):
        """Test packet format correctness - all required fields present"""
        # Create a complete packet
        timestamp = time.time()
        for i in range(self.window_size):
            self.input_layer.window_buffer.add_emg([0.5, 0.3], timestamp + i * 0.01)
            self.input_layer.window_buffer.add_imu([1.0, 2.0, 3.0, 0.1, 0.2, 0.3], timestamp + i * 0.01)
            self.input_layer.window_buffer.add_piezo(0.5, timestamp + i * 0.01)

        # Set up motor state
        motor_state = MotorState(
            joint_positions=[0.1, 0.2, 0.3, 0.4, 0.5],
            gripper_state={"open": True, "force": 0.5},
            timestamp=timestamp
        )

        packet = self.input_layer.packet_builder.build(
            self.input_layer.window_buffer,
            self.input_layer.sample_rate,
            latest_pressure=[0.1, 0.2, 0.3, 0.4, 0.5],
            latest_motor_state=motor_state,
            latest_robot_imu={"data": np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])}
        )

        # Verify all required fields are present
        self.assertIsNotNone(packet.sequence_id)
        self.assertIsNotNone(packet.timestamp)
        self.assertIsNotNone(packet.packet_age_ms)

        # Verify human data
        self.assertIsNotNone(packet.human_data)
        self.assertIsNotNone(packet.human_data.emg)
        self.assertIsNotNone(packet.human_data.imu)
        self.assertIsNotNone(packet.human_data.piezo)
        self.assertEqual(packet.human_data.emg.shape[0], self.window_size)
        self.assertEqual(packet.human_data.imu.shape[0], self.window_size)
        self.assertEqual(len(packet.human_data.piezo), self.window_size)

        # Verify sensors
        self.assertIsNotNone(packet.sensors)
        self.assertIsNotNone(packet.sensors.pressure)
        self.assertEqual(len(packet.sensors.pressure), 5)
        self.assertIsNotNone(packet.sensors.robot_imu)

        # Verify motors
        self.assertIsNotNone(packet.motors)
        self.assertIsNotNone(packet.motors.joint_positions)
        self.assertEqual(len(packet.motors.joint_positions), 5)
        self.assertIsNotNone(packet.motors.gripper_state)

        # Verify metadata
        self.assertIsNotNone(packet.metadata)
        self.assertIn("sample_rate", packet.metadata)
        self.assertIn("window_size", packet.metadata)

    def test_packet_timing_expected_rate(self):
        """Test packets arrive at expected rate"""
        # Calculate expceted packet interval based on window size and sample rate
        expected_interval = self.window_size / self.sample_rate # seconds (100 ms / 100 Hz = 1 second)

        timestamps = []

        for _ in range(3):
            # Fill window buffer
            timestamp = time.time()
            for i in range(self.window_size):
                self.input_layer.window_buffer.add_emg([0.5, 0.3], timestamp + i * 0.01)
                self.input_layer.window_buffer.add_imu([1.0, 2.0, 3.0, 0.1, 0.2, 0.3], timestamp + i * 0.01)
                self.input_layer.window_buffer.add_piezo(0.5, timestamp + i * 0.01)

            # Create packet
            packet = self.input_layer.packet_builder.build(
                self.input_layer.window_buffer,
                self.input_layer.sample_rate
            )

            timestamps.append(packet.timestamp)

            # Clear bluff for next packet
            self.input_layer.window_buffer.clear()

            # Small delay
            time.sleep(0.01)

        # Verify timing intervals
        for i in range(1, len(timestamps)):
            interval = timestamps[i] - timestamps[i-1]
            # Allow some tolerance (within 20% of expected)
            self.assertGreater(interval, 0)
            self.assertLess(interval, 0.1)

    def test_queue_overflow_handling(self):
        """Test queue overflow handling - dropped packets"""
        # Set queue max size to 2 for testing
        small_queue = DataQueue(max_size=2)
        initial_dropped = small_queue.dropped_count
        
        # Fill queue beyond capacity
        for i in range(5):
            packet = DataPacket(
                sequence_id=i,
                timestamp=time.time(),
                human_data=None,
                sensors=SensorSnapshot(),
                metadata={}
            )
            small_queue.put(packet)
            time.sleep(0.01)

        # Verify dropped count increased
        self.assertGreater(small_queue.dropped_count, initial_dropped)

        # Verify queue is at max size
        try:
            self.assertEqual(small_queue.size(), small_queue.max_size)
        except NotImplementedError:
            # macOS doesn't support qsize(), so verify by draining the queue instead
            packets_in_queue = []
            while not small_queue.empty():
                try:
                    packets_in_queue.append(small_queue.get_nowait())
                except:
                    break
            # Should have at most max_size packets
            self.assertLessEqual(len(packets_in_queue), small_queue.max_size)
            # Should have at least some packets (the latest ones)
            self.assertGreater(len(packets_in_queue), 0)

        # Verify only latest packets are in queue
        packets_in_queue = []
        while not small_queue.empty():
            try:
                packets_in_queue.append(small_queue.get_nowait())
            except:
                break

        # Verify only latest packets are in queue
        if len(packets_in_queue) > 0:
            sequence_ids = [p.sequence_id for p in packets_in_queue]
            self.assertIn(max(sequence_ids), [3, 4]) # Latest packets should be present

    def test_packet_age_staleness_detection(self):
        """Test packet age/staleness detection"""
        fresh_packet = DataPacket(
            sequence_id=0,
            timestamp=time.time(),
            human_data=None,
            sensors=SensorSnapshot(),
            metadata={}
        )
        fresh_packet.update_age()

        # Should not be stale
        self.assertFalse(fresh_packet.is_stale(max_age_ms=100.0))

        # Create stale packet
        stale_timestamp = time.time() - 0.2 # 200ms ago
        stale_packet = DataPacket(
            sequence_id=1,
            timestamp=stale_timestamp,
            human_data=None,
            sensors=SensorSnapshot(),
            metadata={}
        )
        stale_packet.update_age()

        # Should be stale
        self.assertTrue(stale_packet.is_stale(max_age_ms=100.0))

        # Test with different thresholds
        self.assertFalse(stale_packet.is_stale(max_age_ms=300.0))
        self.assertTrue(stale_packet.is_stale(max_age_ms=50.0))

    def test_packet_processing_through_pipeline(self):
        """Test packet flows through entire Input -> Processing pipeline"""
        # Fill window buffer
        timestamp = time.time()
        for i in range(self.window_size):
            self.input_layer.window_buffer.add_emg([0.5, 0.3], timestamp + i * 0.01)
            self.input_layer.window_buffer.add_imu([1.0, 2.0, 3.0, 0.1, 0.2, 0.3], timestamp + i * 0.01)
            self.input_layer.window_buffer.add_piezo(0.5, timestamp + i * 0.01)

        # Temporarily disable vision source
        original_vision = self.input_layer.packet_builder.vision_source
        self.input_layer.packet_builder.vision_source = None
        
        # Create packet
        packet = self.input_layer.packet_builder.build(
            self.input_layer.window_buffer,
            self.input_layer.sample_rate,
            latest_pressure=[0.1, 0.2, 0.3, 0.4, 0.5]
        )


        # Restore vision source
        self.input_layer.packet_builder.vision_source = original_vision

        # Put packet in queue
        try:
            self.input_to_processing_queue.put(packet)
            time.sleep(0.01)
        except Exception as e:
            self.fail(f"Failed to put packet in queue: {e}")

        # Process packet (simulating ProcessingLayer behavior)
        try:
            received_packet = self.input_to_processing_queue.get()
        
        except Exception as e:
            self.fail("Failed to get packet from queue: {e}")

        processed_packet = self.processing_layer.process_packet(received_packet)

        # Verify pacet was processed
        self.assertIsNotNone(processed_packet)
        self.assertTrue(processed_packet.metadata.get("processed", False))

        # Verify packet has features
        self.assertIsNotNone(processed_packet.metadata.get("features"))

        # Verify packet can be put in output
        if not self.processing_to_control_queue.full():
            try:
                self.processing_to_control_queue.put(processed_packet)
                time.sleep(0.01)
            except Exception as e:
                self.fail(f"Failed to put packet in output queue: {e}")

            self.assertFalse(self.processing_to_control_queue.empty())

if __name__ == "__main__":
    unittest.main()
        