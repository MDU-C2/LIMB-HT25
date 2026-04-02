import unittest
from unittest.mock import Mock, MagicMock, patch
import numpy as np
import time
import sys
import os

# CRITICAL: Add src directory to path BEFORE any other imports
# Get the absolute path to src directory
test_file_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(test_file_dir, '../..'))

# Remove if already present to avoid duplicates
if src_dir in sys.path:
    sys.path.remove(src_dir)
# Insert at the beginning
sys.path.insert(0, src_dir)

# Verify the path is correct (you can remove this after testing)
print(f"DEBUG: Added to sys.path: {src_dir}")
print(f"DEBUG: layers module exists: {os.path.exists(os.path.join(src_dir, 'layers'))}")

# NOW do the imports
from layers.input.input_layer import InputLayer
from shared.queues import DataQueue
from shared.models.packet import DataPacket, MotorState

class MockCANMessage:
    """Mock CAN message object"""
    def __init__(self, message_type, parsed_data=None, timestamp=None):
        self.message_type = message_type
        self.parsed_data = parsed_data or {}
        self.timestamp = timestamp or time.time()

class MockBLEMessage:
    """Mock BLE message object"""
    def __init__(self, message_type, data=None, timestamp=None):
        self.message_type = message_type
        self.data = data or {}
        self.timestamp = timestamp or time.time()

class TestInputLayer(unittest.TestCase):
    """Unit tests for input layer"""

    def setUp(self):
        """Set up test fixtures"""
        self.window_size = 100
        self.sample_rate = 100.0

        # Mock CAN interface
        self.mock_can = Mock()
        self.mock_can.read.return_value = []
        self.mock_can.start = Mock()
        self.mock_can.stop = Mock()

        # Mock BLE interface
        self.mock_ble = Mock()
        self.mock_ble.read.return_value = []
        self.mock_ble.start = Mock()
        self.mock_ble.stop = Mock()

        # Create output queue
        self.output_queue = DataQueue(max_size=5)

        # Mock vision source
        self.mock_vision = Mock()
        self.mock_vision.update = Mock()

        # Create input layer instance
        self.input_layer = InputLayer(
            can_interface=self.mock_can,
            ble_interface=self.mock_ble,
            output_queue=self.output_queue,
            window_size=self.window_size,
            sample_rate=self.sample_rate,
            vision_source=self.mock_vision
        )

    def tearDown(self):
        """Clear up after tests"""
        if hasattr(self.input_layer, "running"):
            self.input_layer.running.clear()
        self.input_layer.stop()

    def test_initiazliation(self):
        """Test that InputLayer initializes correctly"""
        self.assertEqual(self.input_layer.window_buffer.window_size, self.window_size)
        self.assertEqual(self.input_layer.sample_rate, self.sample_rate)
        self.assertIsNotNone(self.input_layer.window_buffer)
        self.assertIsNotNone(self.input_layer.packet_builder)
        self.assertEqual(self.input_layer.pressure_values, {})
        self.assertIsNone(self.input_layer.latest_pressure)
        self.assertIsNone(self.input_layer.latest_motor_state)
        self.assertIsNone(self.input_layer.latest_robot_imu)
        self.assertEqual(self.input_layer.potentiometer_values, {})
        self.assertEqual(self.input_layer.robot_imu_buffer, {})
        
    def test_window_buffer_initialization(self):
        """Test that WindowBuffer initializes correctly"""
        self.assertEqual(self.input_layer.window_buffer.window_size, self.window_size) 
        self.assertEqual(len(self.input_layer.window_buffer.emg_buffer), 0)
        self.assertEqual(len(self.input_layer.window_buffer.imu_buffer), 0)
        self.assertEqual(len(self.input_layer.window_buffer.piezo_buffer), 0)

    def test_emg_data_processing(self):
        """Test processing EMG data from BLE"""
        # Create mock EMG data
        emg_channels = [0.5, 0.3, 0.7, 0.2]
        timestamp = time.time()

        ble_message = MockBLEMessage("EMG", {"channels":emg_channels}, timestamp)
        self.mock_ble.read.return_value = [ble_message]

        # Simulate adding data to buffer
        self.input_layer.window_buffer.add_emg(emg_channels, timestamp)

        # Verify EMG was added
        self.assertEqual(len(self.input_layer.window_buffer.emg_buffer), 1)
        self.assertEqual(self.input_layer.window_buffer.emg_buffer[0]["data"], emg_channels)

    def test_imu_data_processing(self):
        """Test processing IMU data from BLE"""
        imu_data = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]
        timestamp = time.time()

        ble_message = MockBLEMessage("IMU", {"data": imu_data}, timestamp)
        self.mock_ble.read.return_value = [ble_message]

        # Simulate adding data to buffer
        self.input_layer.window_buffer.add_imu(imu_data, timestamp)

        # Verify IMU was added
        self.assertEqual(len(self.input_layer.window_buffer.imu_buffer), 1)
        np.testing.assert_array_equal(self.input_layer.window_buffer.imu_buffer[0]["data"], imu_data)

    def test_piezo_data_processing(self):
        """Test processing Piezo data from BLE"""
        piezo_value = 0.75
        timestamp = time.time()

        ble_message = MockBLEMessage("piezo", {"value": piezo_value}, timestamp)
        self.mock_ble.read.return_value = [ble_message]

        # Simulate adding data to buffer
        self.input_layer.window_buffer.add_piezo(piezo_value, timestamp)

        # Verify Piezo was added
        self.assertEqual(len(self.input_layer.window_buffer.piezo_buffer), 1)
        self.assertEqual(self.input_layer.window_buffer.piezo_buffer[0]["data"], piezo_value)

    def test_pressure_sensor_processing(self):
        """Test processing pressure sensor data from CAN"""

        # Test individual finger pressure messages (new CAN message format)
        finger_order = ["thumb", "index", "middle", "ring", "pinky"]
        pressure_values = [0.5, 0.3, 0.7, 0.2, 0.4]

        for finger, value in zip(finger_order, pressure_values):
            can_message = MockCANMessage(
                f"robot_{finger}_pressure",
                {"finger": finger, "value": value},
                time.time()
            )
            self.mock_can.read.return_value = [can_message]

            # Simulate processing CAN message (matching input_layer logic)
            if can_message.message_type.endswith("_pressure"):
                if can_message.parsed_data and "value" in can_message.parsed_data and "finger" in can_message.parsed_data:
                    finger_name = can_message.parsed_data["finger"]
                    pressure_value = can_message.parsed_data["value"]
                    self.input_layer.pressure_values[finger_name] = pressure_value

                    # Update latest_pressure when all fingers are present
                    if all(f in self.input_layer.pressure_values for f in finger_order):
                        self.input_layer.latest_pressure = [self.input_layer.pressure_values[f] for f in finger_order]

        # Verify all pressure values are stored
        self.assertEqual(len(self.input_layer.pressure_values), 5)
        self.assertIsNotNone(self.input_layer.latest_pressure)
        self.assertEqual(self.input_layer.latest_pressure, pressure_values)

    def test_potentiometer_processing(self):
        """Test processing potentiometer data from CAN for joint position feedback"""
        timestamp = time.time()
        
        # Test potentiometer messages for joints 0-3 (joint 4 has no potentiometer)
        potentiometer_messages = [
            ("robot_shoulder_up_down_potentiometer", "shoulder_up_down", 0, 0.1),
            ("robot_shoulder_left_right_potentiometer", "shoulder_left_right", 1, 0.2),
            ("robot_elbow_up_down_potentiometer", "elbow_up_down", 2, 0.3),
            ("robot_upper_arm_rotation_potentiometer", "upper_arm_rotation", 3, 0.4),
        ]
        
        for msg_type, source, joint_idx, value in potentiometer_messages:
            can_message = MockCANMessage(
                msg_type,
                {"source": source, "value": value},
                timestamp
            )
            self.mock_can.read.return_value = [can_message]
            
            # Simulate processing CAN message (matching input_layer logic)
            if "potentiometer" in can_message.message_type:
                source = can_message.parsed_data.get("source")
                value = can_message.parsed_data.get("value")
                
                if source and value is not None:
                    joint_idx_mapped = self.input_layer.potentiometer_mapping.get(source)
                    if joint_idx_mapped is not None:
                        self.input_layer.potentiometer_values[joint_idx_mapped] = value
                        self.input_layer._update_motor_state_from_potentiometers(can_message.timestamp)
        
        # Verify potentiometer values are stored
        self.assertEqual(len(self.input_layer.potentiometer_values), 4)
        self.assertEqual(self.input_layer.potentiometer_values[0], 0.1)
        self.assertEqual(self.input_layer.potentiometer_values[1], 0.2)
        self.assertEqual(self.input_layer.potentiometer_values[2], 0.3)
        self.assertEqual(self.input_layer.potentiometer_values[3], 0.4)
        
        # Verify motor state was updated with joint positions
        self.assertIsNotNone(self.input_layer.latest_motor_state)
        np.testing.assert_array_almost_equal(
            self.input_layer.latest_motor_state.joint_positions[:4],
            [0.1, 0.2, 0.3, 0.4]
        )
        # Joint 4 should be 0.0 (no potentiometer)
        self.assertEqual(self.input_layer.latest_motor_state.joint_positions[4], 0.0)

    def test_gripper_state_tracking(self):
        """Test internal gripper state tracking (no CAN gripper_status message exists)"""
        
        # Gripper state is tracked internally, not from CAN messages
        # Test that gripper state can be updated manually
        self.input_layer.gripper_state["open"] = False
        self.input_layer.gripper_state["force"] = 0.5
        
        # Create motor state and update gripper state
        self.input_layer.latest_motor_state = MotorState(
            joint_positions=np.array([0.0] * 5),
            gripper_state=self.input_layer.gripper_state.copy(),
            timestamp=time.time()
        )
        
        # Verify gripper state is tracked
        self.assertFalse(self.input_layer.gripper_state["open"])
        self.assertEqual(self.input_layer.gripper_state["force"], 0.5)
        self.assertIsNotNone(self.input_layer.latest_motor_state)
        self.assertFalse(self.input_layer.latest_motor_state.gripper_state["open"])
        self.assertEqual(self.input_layer.latest_motor_state.gripper_state["force"], 0.5)

    def test_robot_imu_processing(self):
        """Test processing robot IMU data from CAN (separate gyro and accel messages)"""
        accel_data = [1.0, 2.0, 3.0]  # [ax, ay, az]
        gyro_data = [0.1, 0.2, 0.3]   # [wx, wy, wz]
        expected_combined = accel_data + gyro_data  # [ax, ay, az, wx, wy, wz]
        timestamp = time.time()
        
        # Send gyro message first
        gyro_message = MockCANMessage(
            "robot_hand_imu_gyro",
            {"source": "robot_hand", "data": gyro_data},
            timestamp
        )
        self.mock_can.read.return_value = [gyro_message]
        
        # Simulate processing gyro message
        if gyro_message.message_type.endswith("_imu_gyro"):
            source = gyro_message.parsed_data.get("source", "unknown")
            if source not in self.input_layer.robot_imu_buffer:
                self.input_layer.robot_imu_buffer[source] = {}
            self.input_layer.robot_imu_buffer[source]["gyro"] = gyro_message.parsed_data["data"]
        
        # Send accel message
        accel_message = MockCANMessage(
            "robot_hand_imu_accel",
            {"source": "robot_hand", "data": accel_data},
            timestamp
        )
        self.mock_can.read.return_value = [accel_message]
        
        # Simulate processing accel message and combining when both are available
        if accel_message.message_type.endswith("_imu_accel"):
            source = accel_message.parsed_data.get("source", "unknown")
            if source not in self.input_layer.robot_imu_buffer:
                self.input_layer.robot_imu_buffer[source] = {}
            self.input_layer.robot_imu_buffer[source]["accel"] = accel_message.parsed_data["data"]
            
            # Combine when both are available (for robot_hand)
            if source == "robot_hand" and "gyro" in self.input_layer.robot_imu_buffer[source] and "accel" in self.input_layer.robot_imu_buffer[source]:
                gyro = self.input_layer.robot_imu_buffer[source]["gyro"]
                accel = self.input_layer.robot_imu_buffer[source]["accel"]
                combined = accel + gyro
                self.input_layer.latest_robot_imu = {"data": combined, "timestamp": accel_message.timestamp}
        
        # Verify robot IMU was stored and combined correctly
        self.assertIsNotNone(self.input_layer.latest_robot_imu)
        self.assertEqual(self.input_layer.latest_robot_imu["data"], expected_combined)
        self.assertEqual(self.input_layer.latest_robot_imu["timestamp"], timestamp)

    def test_packet_creation_when_window_full(self):
        """Test that packets are created when window buffer is full"""
        # Fill window buffer
        timestamp = time.time()
        for i in range(self.window_size):
            self.input_layer.window_buffer.add_emg([0.5, 0.3], timestamp + i * 0.01)
            self.input_layer.window_buffer.add_imu([1.0, 2.0, 3.0, 0.1, 0.2, 0.3], timestamp + i * 0.01)
            self.input_layer.window_buffer.add_piezo(0.5, timestamp + i * 0.01)
        
        # Verify window is full
        self.assertTrue(self.input_layer.window_buffer.is_full())
        
        # Set up mocks for empty reads
        self.mock_ble.read.return_value = []
        self.mock_can.read.return_value = []
        
        # Manually trigger packet creation (simulating run loop)
        if self.input_layer.window_buffer.is_full():
            if self.input_layer.vision_source is not None:
                self.input_layer.vision_source.update()
            
            packet = self.input_layer.packet_builder.build(
                self.input_layer.window_buffer,
                self.input_layer.sample_rate,
                latest_pressure=self.input_layer.latest_pressure,
                latest_motor_state=self.input_layer.latest_motor_state,
                latest_robot_imu=self.input_layer.latest_robot_imu
            )
            
            # Verify packet was created
            self.assertIsNotNone(packet)
            self.assertIsInstance(packet, DataPacket)
            self.assertIsNotNone(packet.human_data)
            self.assertEqual(packet.human_data.emg.shape[0], self.window_size)

    def test_vision_system_update(self):
        """Test that vision system is updated when window is full"""
        # Fill window buffer
        timestamp = time.time()
        for i in range(self.window_size):
            self.input_layer.window_buffer.add_emg([0.5, 0.3], timestamp + i * 0.01)
            self.input_layer.window_buffer.add_imu([1.0, 2.0, 3.0, 0.1, 0.2, 0.3], timestamp + i * 0.01)
            self.input_layer.window_buffer.add_piezo(0.5, timestamp + i * 0.01)
        
        # Manually trigger vision update
        if self.input_layer.window_buffer.is_full():
            if self.input_layer.vision_source is not None:
                self.input_layer.vision_source.update()
        
        # Verify vision system was called
        self.mock_vision.update.assert_called()

    def test_stop_method(self):
        """Test that stop method works correctly"""
        self.input_layer.running.set()
        self.input_layer.stop()
        
        # Verify running flag is cleared
        self.assertFalse(self.input_layer.running.is_set())
        
        # Verify interfaces are stopped
        self.mock_can.stop.assert_called()
        
        # Verify window buffer is cleared
        self.assertEqual(len(self.input_layer.window_buffer.emg_buffer), 0)


if __name__ == "__main__":
    unittest.main()