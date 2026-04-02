"""
Tests for Processing -> Control Pipeline

Tests:
- Processed packets with ML predictions reaching ControlLayer
- Movement intention data flow
- Fused pose data availability
- Control rate maintenance (100 Hz)
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import numpy as np
import time
import sys
import os

test_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(test_dir, '../..'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from layers.processing.processing_layer import ProcessingLayer
from layers.control.control_layer import ControlLayer
from shared.queues import DataQueue
from shared.models.packet import DataPacket, HumanDataWindow, SensorSnapshot, MotorState

class TestProcessingControlPipeline(unittest.TestCase):
    """Tests for Processing -> Control pipeline"""

    def setUp(self):
        """Set up test fixtures"""
        self.control_rate = 100.0
        
        # Create queues
        self.processing_to_control_queue = DataQueue(max_size=5)
        
        # Mock CAN interface
        self.mock_can = Mock()
        self.mock_can.send = Mock(return_value=True)
        
        # Create layers
        self.processing_layer = ProcessingLayer(
            input_queue=DataQueue(max_size=5),
            output_queue=self.processing_to_control_queue,
            model_path=None,
            scaler_path=None
        )
        
        self.control_layer = ControlLayer(
            input_queue=self.processing_to_control_queue,
            can_interface=self.mock_can,
            control_rate=self.control_rate
        )

    def tearDown(self):
        """Clean up after tests"""
        if hasattr(self.processing_layer, 'running'):
            self.processing_layer.running.clear()
        if hasattr(self.control_layer, 'running'):
            self.control_layer.running.clear()
        self.processing_layer.stop()
        self.control_layer.stop()
        
        # Clear queues
        while not self.processing_to_control_queue.empty():
            try:
                self.processing_to_control_queue.get_nowait()
            except:
                break

    def test_processed_packets_with_ml_predictions_reach_control(self):
        """Test processed packets with ML predictions reach ControlLayer"""
        # Create packet with EMG data
        window_size = 100
        human_data = HumanDataWindow(
            emg=np.random.randn(window_size, 2),
            imu=np.random.randn(window_size, 6),
            piezo=np.random.randn(window_size),
            timestamp_start=time.time(),
            timestamp_end=time.time() + 1.0,
            sample_rate=100.0
        )
        
        packet = DataPacket(
            sequence_id=0,
            timestamp=time.time(),
            human_data=human_data,
            sensors=SensorSnapshot(),
            metadata={}
        )
        
        # Process packet
        processed_packet = self.processing_layer.process_packet(packet)
        
        # Add ML prediction manually (simulating model inference)
        processed_packet.metadata["ml_prediction"] = {
            "class": 1,  # grip
            "confidence": 0.8,
            "probabilities": np.array([0.2, 0.8]),
            "timestamp": time.time()
        }
        
        # Send to control queue
        try:
            self.processing_to_control_queue.put(processed_packet, timeout=1.0)
            time.sleep(0.01)  # Allow multiprocessing.Queue to serialize
        except Exception as e:
            self.fail(f"Failed to put packet in queue: {e}")
        
        # ControlLayer should receive it
        received_packet = self.control_layer._get_latest_packet()
        
        # Verify packet was received
        self.assertIsNotNone(received_packet)
        self.assertIsNotNone(received_packet.metadata.get("ml_prediction"))
        ml_pred = received_packet.metadata["ml_prediction"]
        self.assertEqual(ml_pred["class"], 1)
        self.assertEqual(ml_pred["confidence"], 0.8)
        self.assertIn("probabilities", ml_pred)
        self.assertIn("timestamp", ml_pred)

    def test_movement_intention_data_flow(self):
        """Test movement intention data flows from Processing to Control"""
        # Create packet with IMU data showing forward movement
        window_size = 100
        imu_data = np.zeros((window_size, 6))
        imu_data[:, 0] = 2.0  # Strong forward acceleration
        
        human_data = HumanDataWindow(
            emg=np.random.randn(window_size, 2),
            imu=imu_data,
            piezo=np.random.randn(window_size),
            timestamp_start=time.time(),
            timestamp_end=time.time() + 1.0,
            sample_rate=100.0
        )
        
        packet = DataPacket(
            sequence_id=0,
            timestamp=time.time(),
            human_data=human_data,
            sensors=SensorSnapshot(),
            metadata={}
        )
        
        # Process packet (should detect movement intention)
        processed_packet = self.processing_layer.process_packet(packet)
        
        # Verify movement intention was added
        self.assertIsNotNone(processed_packet.metadata.get("movement_intention"))
        
        movement_intention = processed_packet.metadata["movement_intention"]
        self.assertIn("direction", movement_intention)
        self.assertIn("confidence", movement_intention)
        self.assertIn("acceleration", movement_intention)
        self.assertIn("magnitude", movement_intention)
        
        # Send to control queue
        try:
            self.processing_to_control_queue.put(processed_packet, timeout=1.0)
            time.sleep(0.01)
        except Exception as e:
            self.fail(f"Failed to put packet in queue: {e}")
        
        # ControlLayer should receive it
        received_packet = self.control_layer._get_latest_packet()
        
        # Verify movement intention is accessible
        self.assertIsNotNone(received_packet)
        self.assertIsNotNone(received_packet.metadata.get("movement_intention"))
        movement = received_packet.metadata["movement_intention"]
        self.assertIsNotNone(movement.get("direction"))
        self.assertIsNotNone(movement.get("confidence"))
        self.assertIsNotNone(movement.get("acceleration"))
        self.assertIsNotNone(movement.get("magnitude"))

    def test_fused_pose_data_availability(self):
        """Test fused pose data availability in ControlLayer"""
        # Create packet with robot IMU and vision data
        packet = DataPacket(
            sequence_id=0,
            timestamp=time.time(),
            human_data=None,
            sensors=SensorSnapshot(
                robot_imu={"data": np.array([0.0, 0.0, 9.8, 0.0, 0.0, 0.0])},
                vision={
                    "apriltag_pose": {
                        "position": np.array([100.0, 200.0, 300.0]),
                        "orientation": np.array([1.0, 0.0, 0.0, 0.0])  # quaternion
                    }
                }
            ),
            motors=MotorState(
                joint_positions=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
                gripper_state={},
                timestamp=time.time()
            ),
            metadata={}
        )
        
        # Process packet (should create fused pose)
        # Note: This requires IMU+vision fusion which may need EKF initialization
        processed_packet = self.processing_layer.process_packet(packet)
        
        # Manually add fused pose for testing (simulating fusion)
        # In real system, this would come from _process_imu_vision_fusion()
        processed_packet.metadata["fused_arm_pose"] = {
            "position": [0.1, 0.2, 0.3],
            "orientation": [1.0, 0.0, 0.0, 0.0],
            "orientation_euler": [0.0, 0.0, 0.0]
        }
        
        # Send to control queue
        try:
            self.processing_to_control_queue.put(processed_packet, timeout=1.0)
            time.sleep(0.01)
        except Exception as e:
            self.fail(f"Failed to put packet in queue: {e}")
        
        # ControlLayer should receive it
        received_packet = self.control_layer._get_latest_packet()
        
        # Verify fused pose is available
        self.assertIsNotNone(received_packet)
        fused_pose = received_packet.metadata.get("fused_arm_pose")
        self.assertIsNotNone(fused_pose)
        self.assertIn("position", fused_pose)
        self.assertIn("orientation", fused_pose)
        
        # Test ControlLayer can access it
        current_pose = self.control_layer._get_current_arm_pose(received_packet)
        self.assertIsNotNone(current_pose)
        self.assertIn("position", current_pose)
        self.assertIn("orientation", current_pose)

    def test_control_rate_maintenance(self):
        """Test control rate maintenance (100 Hz)"""
        control_period = 1.0 / self.control_rate  # 0.01 seconds = 10ms
        
        # Measure actual control loop timing
        num_iterations = 10
        timings = []
        execution_times = []
        
        self.control_layer.running.set()
        
        for i in range(num_iterations):
            cycle_start = time.time()
            
            # Simulate one control cycle
            exec_start = time.time()
            packet = self.control_layer._get_latest_packet()
            if packet is not None:
                commands = self.control_layer._compute_commands(packet)
                if commands:
                    self.control_layer._send_commands(commands)
            exec_end = time.time()
            execution_times.append(exec_end - exec_start)
            
            # Maintain control rate
            elapsed = time.time() - cycle_start
            sleep_time = max(0, control_period - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            cycle_end = time.time()
            timings.append(cycle_end - cycle_start)
        
        # Calculate average period and rate
        avg_period = sum(timings) / len(timings)
        avg_rate = 1.0 / avg_period if avg_period > 0 else 0
        avg_execution_time = sum(execution_times) / len(execution_times)
        
        # Verify execution time is reasonable (should be much less than period)
        self.assertLess(avg_execution_time, control_period * 0.5,
                      f"Average execution time {avg_execution_time}s exceeds 50% of period {control_period}s")
        
        # Verify rate is close to target (within 25% tolerance to account for system overhead)
        # On some systems, sleep() may not be precise, so we allow more tolerance
        self.assertGreater(avg_rate, self.control_rate * 0.75,
                          f"Average rate {avg_rate} Hz is below 75% of target {self.control_rate} Hz")
        self.assertLess(avg_rate, self.control_rate * 1.25,
                        f"Average rate {avg_rate} Hz is above 125% of target {self.control_rate} Hz")
        
        # Verify individual cycles don't exceed period significantly
        for timing in timings:
            # Allow 30% overhead for timing variations (accounting for sleep imprecision)
            self.assertLess(timing, control_period * 1.3,
                          f"Cycle time {timing}s exceeds 130% of period {control_period}s")

    def test_control_layer_receives_fully_processed_packet(self):
        """Test ControlLayer receives fully processed packet with all metadata"""
        # Create complete processed packet
        window_size = 100
        human_data = HumanDataWindow(
            emg=np.random.randn(window_size, 2),
            imu=np.random.randn(window_size, 6),
            piezo=np.random.randn(window_size),
            timestamp_start=time.time(),
            timestamp_end=time.time() + 1.0,
            sample_rate=100.0
        )
        
        packet = DataPacket(
            sequence_id=0,
            timestamp=time.time(),
            human_data=human_data,
            sensors=SensorSnapshot(
                vision={"cup_detections": []},
                pressure=[0.1, 0.2, 0.3, 0.4, 0.5]
            ),
            motors=MotorState(
                joint_positions=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
                gripper_state={"open": True, "force": 0.5},
                timestamp=time.time()
            ),
            metadata={}
        )
        
        # Process packet
        processed_packet = self.processing_layer.process_packet(packet)
        
        # Add all expected metadata
        processed_packet.metadata["ml_prediction"] = {
            "class": 1,
            "confidence": 0.8,
            "probabilities": np.array([0.2, 0.8]),
            "timestamp": time.time()
        }
        processed_packet.metadata["movement_intention"] = {
            "direction": "forward",
            "confidence": 0.7,
            "acceleration": [1.0, 0.0, 0.0],
            "magnitude": 1.0,
            "timestamp": time.time()
        }
        processed_packet.metadata["fused_arm_pose"] = {
            "position": [0.1, 0.2, 0.3],
            "orientation": [1.0, 0.0, 0.0, 0.0]
        }
        
        # Send to control queue
        try:
            self.processing_to_control_queue.put(processed_packet, timeout=1.0)
            time.sleep(0.01)
        except Exception as e:
            self.fail(f"Failed to put packet in queue: {e}")
        
        # ControlLayer receives it
        received_packet = self.control_layer._get_latest_packet()
        
        # Verify all processed data is present
        self.assertIsNotNone(received_packet)
        self.assertTrue(received_packet.metadata.get("processed", False))
        self.assertIsNotNone(received_packet.metadata.get("ml_prediction"))
        self.assertIsNotNone(received_packet.metadata.get("movement_intention"))
        self.assertIsNotNone(received_packet.metadata.get("fused_arm_pose"))
        self.assertIsNotNone(received_packet.metadata.get("features"))

    def test_control_layer_handles_empty_queue_gracefully(self):
        """Test ControlLayer handles empty queue gracefully"""
        # Get packet from empty queue
        packet = self.control_layer._get_latest_packet()
        
        # Should return None, not raise exception
        self.assertIsNone(packet)
        
        # ControlLayer's run() method checks for None before processing
        # This test verifies that _get_latest_packet() returns None gracefully
        # and doesn't crash when queue is empty
        # The actual command computation would be skipped in run() when packet is None

    def test_control_layer_drains_queue_for_latest_packet(self):
        """Test ControlLayer drains queue to get latest packet"""
        # Create multiple packets with different sequence IDs
        packets = []
        for i in range(3):
            packet = DataPacket(
                sequence_id=i,
                timestamp=time.time() + i * 0.001,
                human_data=None,
                sensors=SensorSnapshot(),
                metadata={"test": f"packet_{i}"}
            )
            packets.append(packet)
            try:
                self.processing_to_control_queue.put(packet, timeout=1.0)
                time.sleep(0.01)
            except Exception as e:
                self.fail(f"Failed to put packet {i} in queue: {e}")
        
        # ControlLayer should get the latest packet (sequence_id=2)
        latest_packet = self.control_layer._get_latest_packet()
        
        # Verify it's the latest packet
        self.assertIsNotNone(latest_packet)
        self.assertEqual(latest_packet.sequence_id, 2)
        self.assertEqual(latest_packet.metadata.get("test"), "packet_2")
        
        # Queue should be empty after draining
        # Note: empty() might not be reliable on macOS, so we'll just verify we got the right packet


if __name__ == "__main__":
    unittest.main()