import unittest
from unittest.mock import Mock, MagicMock, patch
import numpy as np
import time
import sys
import os

import os
test_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(test_dir, '../..'))
print(f"Test dir: {test_dir}")
print(f"Src dir: {src_dir}")
print(f"Layers exists: {os.path.exists(os.path.join(src_dir, 'layers'))}")

# Add src directory to path for imports (using absolute path)
test_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(test_dir, '../..'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from layers.processing.processing_layer import ProcessingLayer
from shared.queues import DataQueue
from shared.models.packet import DataPacket, HumanDataWindow, SensorSnapshot


class TestProcessingLayer(unittest.TestCase):
    """Unit tests for ProcessingLayer"""

    def setUp(self):
        """Set up test fixtures"""
        self.input_queue = DataQueue(max_size=5)
        self.output_queue = DataQueue(max_size=5)
        
        # Create ProcessingLayer instance without model (for basic tests)
        self.processing_layer = ProcessingLayer(
            input_queue=self.input_queue,
            output_queue=self.output_queue,
            model_path=None,
            scaler_path=None
        )

    def tearDown(self):
        """Clean up after tests"""
        if hasattr(self.processing_layer, 'running'):
            self.processing_layer.running.clear()
        self.processing_layer.stop()

    def test_initialization(self):
        """Test that ProcessingLayer initializes correctly"""
        self.assertIsNotNone(self.processing_layer.input_queue)
        self.assertIsNotNone(self.processing_layer.output_queue)
        self.assertIsNone(self.processing_layer.model)  # No model loaded
        self.assertIsNone(self.processing_layer.scaler)  # No scaler loaded
        self.assertEqual(self.processing_layer.seq_length, 10)
        self.assertEqual(self.processing_layer.num_classes, 2)

    def test_emg_preprocessing(self):
        """Test EMG preprocessing functionality"""
        # Create synthetic EMG data
        window_size = 100
        num_channels = 2
        emg_data = np.random.randn(window_size, num_channels)
        
        # Create HumanDataWindow
        human_data = HumanDataWindow(
            emg=emg_data,
            imu=np.random.randn(window_size, 6),
            piezo=np.random.randn(window_size),
            timestamp_start=time.time(),
            timestamp_end=time.time() + 1.0,
            sample_rate=100.0
        )
        
        # Create packet
        packet = DataPacket(
            sequence_id=0,
            timestamp=time.time(),
            human_data=human_data,
            sensors=SensorSnapshot(),
            metadata={}
        )
        
        # Test EMG processing
        features = self.processing_layer._process_emg(human_data)
        
        # Verify features were extracted
        self.assertIsNotNone(features)
        self.assertIsInstance(features, np.ndarray)
        # Should have 6 features per channel: num_channels * 6
        expected_features = num_channels * 6
        self.assertEqual(len(features), expected_features)

    def test_emg_preprocessing_empty_data(self):
        """Test EMG preprocessing with empty data"""
        human_data = HumanDataWindow(
            emg=None,
            imu=np.random.randn(100, 6),
            piezo=np.random.randn(100),
            timestamp_start=time.time(),
            timestamp_end=time.time() + 1.0,
            sample_rate=100.0
        )
        
        features = self.processing_layer._process_emg(human_data)
        self.assertIsNone(features)

    '''
    SKIP THIS TEST FOR NOW
    def test_imu_movement_intention_detection(self):
        """Test IMU movement intention detection"""
        # Create synthetic IMU data with clear forward movement
        window_size = 100
        # Strong forward acceleration (positive X)
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
        
        # Test movement intention detection
        movement_intention = self.processing_layer._process_imu_intention(human_data)
        
        # Verify movement was detected
        self.assertIsNotNone(movement_intention)
        self.assertEqual(movement_intention["direction"], "forward")
        self.assertEqual(movement_intention["confidence"], 0.0)
        self.assertLessEqual(movement_intention["confidence"], 1.0)
    '''
    
    def test_imu_movement_intention_no_movement(self):
        """Test IMU movement intention with no significant movement"""
        window_size = 100
        # Very small acceleration (below threshold)
        imu_data = np.zeros((window_size, 6))
        imu_data[:, 0] = 0.1  # Below threshold
        
        human_data = HumanDataWindow(
            emg=np.random.randn(window_size, 2),
            imu=imu_data,
            piezo=np.random.randn(window_size),
            timestamp_start=time.time(),
            timestamp_end=time.time() + 1.0,
            sample_rate=100.0
        )
        
        movement_intention = self.processing_layer._process_imu_intention(human_data)
        
        # Should detect no movement
        self.assertIsNotNone(movement_intention)
        self.assertEqual(movement_intention["direction"], "none")
        self.assertEqual(movement_intention["confidence"], 0.0)

    def test_packet_processing_without_model(self):
        """Test packet processing without ML model"""
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
        
        # Verify packet was processed
        self.assertIsNotNone(processed_packet)
        self.assertTrue(processed_packet.metadata.get("processed", False))
        # ML prediction should be None (no model)
        self.assertIsNone(processed_packet.metadata.get("ml_prediction"))
        # Features should be extracted
        self.assertIsNotNone(processed_packet.metadata.get("features"))

    def test_packet_processing_with_model(self):
        """Test packet processing with ML model (mocked)"""
        # Mock model
        mock_model = MagicMock()
        mock_logits = MagicMock()
        mock_logits.shape = (1, 2)
        
        # Mock softmax output
        mock_probs = MagicMock()
        mock_probs.__getitem__.return_value = MagicMock()
        mock_probs.__getitem__.return_value.cpu.return_value = MagicMock()
        mock_probs.__getitem__.return_value.cpu.return_value.numpy.return_value = [0.3, 0.7]
        
        mock_model.return_value = mock_logits
        
        # Replace model
        self.processing_layer.model = mock_model
        self.processing_layer.device = "cpu"
        
        # Create packet
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
        
        # Fill feature buffer to enable inference
        for _ in range(self.processing_layer.seq_length):
            features = self.processing_layer._process_emg(human_data)
            if features is not None:
                self.processing_layer.feature_buffer.append(features)
        
        # Process packet
        with patch('torch.softmax') as mock_softmax, \
             patch('torch.argmax') as mock_argmax:
            mock_softmax.return_value = mock_probs
            mock_argmax.return_value = MagicMock()
            mock_argmax.return_value.item.return_value = 1
            
            processed_packet = self.processing_layer.process_packet(packet)
        
        # Verify ML prediction was added
        self.assertIsNotNone(processed_packet.metadata.get("ml_prediction"))

    def test_stale_packet_handling(self):
        """Test that stale packets are skipped"""
        # Create stale packet
        old_timestamp = time.time() - 1.0  # 1 second ago
        packet = DataPacket(
            sequence_id=0,
            timestamp=old_timestamp,
            human_data=None,
            sensors=SensorSnapshot(),
            metadata={}
        )
        packet.update_age()  # Update age
        
        # Add to queue
        self.input_queue.put(packet)
        
        # Verify packet is stale
        self.assertTrue(packet.is_stale(max_age_ms=100.0))

    def test_stop_method(self):
        """Test that stop method works correctly"""
        self.processing_layer.running.set()
        self.processing_layer.stop()
        
        # Verify running flag is cleared
        self.assertFalse(self.processing_layer.running.is_set())


if __name__ == "__main__":
    unittest.main()