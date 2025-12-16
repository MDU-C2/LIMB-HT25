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

        # Test individual finger pressure messages
        finger_order = ["thumb", "index", "middle", "ring", "little"]
        pressure_values = [0.5, 0.3, 0.7, 0.2, 0.4]

        for finger, value in zip(finger_order, pressure_values):
            can_message = MockCANMessage(
                f"pressure_{finger}",
                {"finger": finger, "value": value},
                time.time()
            )
            self.mock_can.read.return_value = [can_message]

            # Simulate processing CAN message
            if can_message.message_type.startswith("pressure_"):
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

    def test_motor_status_processing(self):
        """Test processing motor status from CAN"""
        joint_positions = [0.1, 0.2, 0.3, 0.4, 0.5]
        timestamp = time.time()

        can_message = MockCANMessage(
            "motor_status",
            {"joint_positions": joint_positions},
            timestamp
        )
        self.mock_can.read.return_value = [can_message]

        # Simulate processing CAN message
        if can_message.message_type == "motor_status":
            if can_message.parsed_data:
                positions = can_message.parsed_data.get("joint_positions", [])
                if len(positions) == 5:
                    self.input_layer.latest_motor_state = MotorState(
                        joint_positions=np.array(positions),
                        gripper_state={},
                        timestamp=timestamp
                    )

        # Verify motor state was updated
        self.assertIsNotNone(self.input_layer.latest_motor_state)
        np.testing.assert_array_equal(self.input_layer.latest_motor_state.joint_positions, joint_positions)

    def test_gripper_status_processing(self):
        """Test processing gripper status from CAN"""
        
        # First create a motor state
        self.input_layer.latest_motor_state = MotorState(
            joint_positions=np.array([0.0] * 5),
            gripper_state={},
            timestamp=time.time()
        )

        # Then update gripper state
        gripper_state = 1
        gripper_force = 0.5
        timestamp = time.time()

        can_message = MockCANMessage(
            "gripper_status",
            {"state": gripper_state, "force": gripper_force},
            timestamp
        )

        # Simulate processing CAN message
        if can_message.message_type == "gripper_status":
            if self.input_layer.latest_motor_state and can_message.parsed_data:
                self.input_layer.latest_motor_state.gripper_state = {
                    "open": can_message.parsed_data.get("state", 0) == 1,
                    "force": can_message.parsed_data.get("force", 0.0)
                }

        # Verify gripper state was updated
        self.assertTrue(self.input_layer.latest_motor_state.gripper_state["open"])
        self.assertEqual(self.input_layer.latest_motor_state.gripper_state["force"], gripper_force)

    def test_robot_imu_processing(self):
        """Test processing robot IMU data from CAN"""
        imu_data = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]  # [ax, ay, az, wx, wy, wz]
        timestamp = time.time()
        
        can_message = MockCANMessage(
            "IMU",
            {"data": imu_data},
            timestamp
        )
        self.mock_can.read.return_value = [can_message]
        
        # Simulate processing CAN message
        if can_message.message_type == "IMU":
            if can_message.parsed_data and "data" in can_message.parsed_data:
                self.input_layer.latest_robot_imu = {
                    "data": can_message.parsed_data["data"],
                    "timestamp": can_message.timestamp
                }
        
        # Verify robot IMU was stored
        self.assertIsNotNone(self.input_layer.latest_robot_imu)
        self.assertEqual(self.input_layer.latest_robot_imu["data"], imu_data)
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