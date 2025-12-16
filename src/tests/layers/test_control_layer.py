
import unittest
from unittest.mock import Mock, MagicMock, patch
import numpy as np
import time
import sys
import os

# Add src directory to path for imports (using absolute path)
test_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(test_dir, '../..'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from layers.control.control_layer import ControlLayer
from shared.queues import DataQueue
from shared.models.packet import DataPacket, HumanDataWindow, SensorSnapshot, MotorState


class TestControlLayer(unittest.TestCase):
    """Unit tests for ControlLayer"""

    def setUp(self):
        """Set up test fixtures"""
        self.input_queue = DataQueue(max_size=5)
        self.control_rate = 100.0
        
        # Mock CAN interface
        self.mock_can = Mock()
        self.mock_can.send = Mock(return_value=True)
        
        # Create ControlLayer instance
        self.control_layer = ControlLayer(
            input_queue=self.input_queue,
            can_interface=self.mock_can,
            control_rate=self.control_rate
        )

    def tearDown(self):
        """Clean up after tests"""
        if hasattr(self.control_layer, 'running'):
            self.control_layer.running.clear()
        self.control_layer.stop()

    def test_initialization(self):
        """Test that ControlLayer initializes correctly"""
        self.assertEqual(self.control_layer.control_rate, self.control_rate)
        self.assertEqual(self.control_layer.control_period, 1.0 / self.control_rate)
        self.assertEqual(self.control_layer.current_state, "Waiting")
        self.assertFalse(self.control_layer.grip_command_sent)
        self.assertEqual(self.control_layer.conf_threshold, 0.5)

    def test_state_machine_initial_state(self):
        """Test that state machine starts in Waiting state"""
        self.assertEqual(self.control_layer.current_state, "Waiting")
        self.assertEqual(len(self.control_layer.state_history), 0)

    def test_get_latest_packet(self):
        """Test getting latest packet from queue"""
        # Create multiple packets
        packet1 = DataPacket(sequence_id=1, timestamp=time.time() - 0.1)
        packet2 = DataPacket(sequence_id=2, timestamp=time.time() - 0.05)
        packet3 = DataPacket(sequence_id=3, timestamp=time.time())
        
        # Add packets to queue
        self.input_queue.put(packet1)
        self.input_queue.put(packet2)
        self.input_queue.put(packet3)
        
        # Get latest packet (should drain queue and return newest)
        latest = self.control_layer._get_latest_packet()
        
        # Verify latest packet is returned
        self.assertIsNotNone(latest)
        self.assertEqual(latest.sequence_id, 3)
        
        # Verify queue is empty (drained)
        self.assertTrue(self.input_queue.empty())

    def test_get_latest_packet_stale_packets(self):
        """Test that stale packets are dropped"""
        # Create stale packet
        stale_packet = DataPacket(sequence_id=1, timestamp=time.time() - 1.0)
        stale_packet.update_age()
        
        # Create fresh packet
        fresh_packet = DataPacket(sequence_id=2, timestamp=time.time())
        
        self.input_queue.put(stale_packet)
        self.input_queue.put(fresh_packet)
        
        latest = self.control_layer._get_latest_packet()
        
        # Should return fresh packet
        self.assertIsNotNone(latest)
        self.assertEqual(latest.sequence_id, 2)

    def test_interpret_intention_grip(self):
        """Test interpreting ML prediction as grip intention"""
        ml_prediction = {
            "class": 1,  # grip
            "confidence": 0.8
        }
        
        intention = self.control_layer._interpret_intention(ml_prediction)
        self.assertEqual(intention, "grip")

    def test_interpret_intention_rest(self):
        """Test interpreting ML prediction as rest intention"""
        ml_prediction = {
            "class": 0,  # rest
            "confidence": 0.8
        }
        
        intention = self.control_layer._interpret_intention(ml_prediction)
        self.assertEqual(intention, "rest")

    def test_interpret_intention_low_confidence(self):
        """Test that low confidence predictions return None"""
        ml_prediction = {
            "class": 1,
            "confidence": 0.3  # Below threshold
        }
        
        intention = self.control_layer._interpret_intention(ml_prediction)
        self.assertIsNone(intention)

    def test_check_hand_cup_distance(self):
        """Test hand-cup distance validation"""
        # Create vision data with cup detection
        vision_data = {
            "cup_detections": [{
                "position": np.array([0.1, 0.1, 0.1])  # Cup at 10cm
            }]
        }
        
        hand_position = np.array([0.15, 0.1, 0.1])  # Hand at 15cm (5cm away)
        
        is_close = self.control_layer._check_hand_cup_distance(vision_data, hand_position)
        self.assertTrue(is_close)  # Within 20cm threshold

    def test_check_hand_cup_distance_too_far(self):
        """Test hand-cup distance when too far"""
        vision_data = {
            "cup_detections": [{
                "position": np.array([0.0, 0.0, 0.0])
            }]
        }
        
        hand_position = np.array([0.5, 0.0, 0.0])  # 50cm away
        
        is_close = self.control_layer._check_hand_cup_distance(vision_data, hand_position)
        self.assertFalse(is_close)  # Beyond 20cm threshold

    def test_check_stable_grip(self):
        """Test stable grip detection"""
        gripper_state = {
            "open": False,
            "force": 0.5,
            "stable": True
        }
        
        is_stable = self.control_layer._check_stable_grip(gripper_state)
        self.assertTrue(is_stable)

    def test_check_stable_grip_unstable(self):
        """Test unstable grip detection"""
        gripper_state = {
            "open": False,
            "force": 0.5,
            "stable": False
        }
        
        is_stable = self.control_layer._check_stable_grip(gripper_state)
        self.assertFalse(is_stable)

    def test_validate_joint_positions(self):
        """Test joint position validation"""
        valid_positions = [0.0, 0.5, -0.5, 1.0, -1.0]  # Within limits
        self.assertTrue(self.control_layer._validate_joint_positions(valid_positions))
        
        invalid_positions = [4.0, 0.5, -0.5, 1.0, -1.0]  # First joint out of range
        self.assertFalse(self.control_layer._validate_joint_positions(invalid_positions))

    def test_validate_target_pose(self):
        """Test target pose validation"""
        valid_pose = {
            "position": np.array([0.2, 0.2, 0.3])  # Within workspace
        }
        self.assertTrue(self.control_layer._validate_target_pose(valid_pose))
        
        invalid_pose = {
            "position": np.array([1.0, 0.0, 0.0])  # X out of workspace
        }
        self.assertFalse(self.control_layer._validate_target_pose(invalid_pose))

    def test_encode_arm_command(self):
        """Test encoding arm command"""
        joint_positions = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        encoded = self.control_layer._encode_arm_command(joint_positions)
        
        # Should return (can_id, data) tuple
        self.assertIsNotNone(encoded)
        self.assertEqual(len(encoded), 2)
        self.assertIsInstance(encoded[0], int)  # CAN ID
        self.assertIsInstance(encoded[1], bytes)  # Data

    def test_encode_arm_command_invalid(self):
        """Test encoding invalid arm command"""
        invalid_positions = [0.1, 0.2]  # Wrong length
        encoded = self.control_layer._encode_arm_command(invalid_positions)
        self.assertIsNone(encoded)

    def test_encode_gripper_command(self):
        """Test encoding gripper command"""
        encoded = self.control_layer._encode_gripper_command("close", 0.5)
        
        self.assertIsNotNone(encoded)
        self.assertEqual(len(encoded), 2)
        self.assertIsInstance(encoded[0], int)  # CAN ID
        self.assertIsInstance(encoded[1], bytes)  # Data

    def test_transition_to_state(self):
        """Test state transition"""
        initial_state = self.control_layer.current_state
        self.control_layer._transition_to_state("Gripping", "Test transition")
        
        self.assertEqual(self.control_layer.current_state, "Gripping")
        self.assertEqual(len(self.control_layer.state_history), 1)
        self.assertEqual(self.control_layer.state_history[0]["from_state"], initial_state)
        self.assertEqual(self.control_layer.state_history[0]["to_state"], "Gripping")

    def test_compute_waiting_state_command(self):
        """Test command computation in Waiting state"""
        # Create packet with vision data
        vision_data = {
            "cup_detections": [{
                "position": np.array([0.2, 0.2, 0.2])
            }]
        }
        
        sensors = SensorSnapshot(vision=vision_data)
        
        # Mock current arm pose
        packet = DataPacket(
            sequence_id=0,
            timestamp=time.time(),
            sensors=sensors,
            metadata={
                "fused_arm_pose": {
                    "position": [0.1, 0.1, 0.1],
                    "orientation": [0, 0, 0]
                }
            }
        )
        
        # Mock inverse kinematics
        with patch.object(self.control_layer, '_inverse_kinematics') as mock_ik, \
             patch.object(self.control_layer, '_get_current_arm_pose') as mock_pose:
            mock_pose.return_value = {
                "position": np.array([0.1, 0.1, 0.1]),
                "orientation": [0, 0, 0]
            }
            mock_ik.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
            
            command = self.control_layer._compute_waiting_state_command(packet)
            
            # Should compute approach command
            self.assertIsNotNone(command)
            self.assertIn("joint_positions", command)

    def test_compute_gripping_state_command(self):
        """Test command computation in Gripping state"""
        # Create packet with motor state
        motor_state = MotorState(
            joint_positions=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            gripper_state={"open": False, "force": 0.5},
            timestamp=time.time()
        )
        
        packet = DataPacket(
            sequence_id=0,
            timestamp=time.time(),
            motors=motor_state,
            metadata={}
        )
        
        command = self.control_layer._compute_gripping_state_command(packet)
        
        # Should maintain current position
        self.assertIsNotNone(command)
        self.assertIn("joint_positions", command)
        self.assertTrue(command.get("maintain_position", False))

    def test_apply_safety_limits(self):
        """Test safety limit application"""
        commands = {
            "arm": {
                "joint_positions": [0.1, 0.2, 0.3, 0.4, 0.5]
            }
        }
        
        motor_state = MotorState(
            joint_positions=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            gripper_state={},
            timestamp=time.time()
        )
        
        validated = self.control_layer._apply_safety_limits(commands, motor_state)
        
        # Should pass validation
        self.assertIsNotNone(validated)
        self.assertIn("arm", validated)

    def test_apply_safety_limits_invalid(self):
        """Test safety limit application with invalid commands"""
        commands = {
            "arm": {
                "joint_positions": [10.0, 0.2, 0.3, 0.4, 0.5]  # Invalid
            }
        }
        
        motor_state = MotorState(
            joint_positions=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            gripper_state={},
            timestamp=time.time()
        )
        
        validated = self.control_layer._apply_safety_limits(commands, motor_state)
        
        # Should reject invalid commands
        self.assertIsNone(validated) or self.assertNotIn("arm", validated)

    def test_send_commands(self):
        """Test sending commands via CAN"""
        commands = {
            "arm": {
                "joint_positions": [0.1, 0.2, 0.3, 0.4, 0.5]
            }
        }
        
        self.control_layer._send_commands(commands)
        
        # Verify CAN send was called
        self.mock_can.send.assert_called()

    def test_stop_method(self):
        """Test that stop method works correctly"""
        self.control_layer.running.set()
        self.control_layer.stop()
        
        # Verify running flag is cleared
        self.assertFalse(self.control_layer.running.is_set())


if __name__ == "__main__":
    unittest.main()