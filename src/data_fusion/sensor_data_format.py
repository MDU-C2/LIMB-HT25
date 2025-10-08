"""
Unified Sensor Data Format for LSTM Input

This module provides data structures and utilities for combining IMU and EMG data
into a format suitable for LSTM neural networks. The format is designed to be:
- Time-series compatible (sequential data)
- Normalized and standardized
- Easy to batch and feed into PyTorch/TensorFlow LSTMs
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np
import time


@dataclass
class EMGData:
    """EMG (Electromyography) data structure."""
    channels: np.ndarray  # Shape: (n_channels,) - EMG signal from multiple channels
    timestamp: float  # seconds
    sampling_rate: Optional[float] = None  # Hz
    
    def __post_init__(self):
        """Validate EMG data."""
        if self.channels.ndim != 1:
            raise ValueError(f"EMG channels must be 1D array, got shape {self.channels.shape}")


@dataclass
class IMUData:
    """IMU (Inertial Measurement Unit) data structure."""
    angular_velocity: np.ndarray  # [wx, wy, wz] in rad/s
    linear_acceleration: np.ndarray  # [ax, ay, az] in m/s²
    timestamp: float  # seconds
    
    def __post_init__(self):
        """Validate IMU data."""
        if self.angular_velocity.shape != (3,):
            raise ValueError(f"Angular velocity must be shape (3,), got {self.angular_velocity.shape}")
        if self.linear_acceleration.shape != (3,):
            raise ValueError(f"Linear acceleration must be shape (3,), got {self.linear_acceleration.shape}")


@dataclass
class SensorFusionData:
    """
    Combined IMU + EMG data for LSTM input.
    
    This structure represents a single timestep of sensor data.
    For LSTM input, multiple timesteps will be stacked into sequences.
    """
    imu: Optional[IMUData] = None
    emg: Optional[EMGData] = None
    timestamp: float = field(default_factory=time.time)
    
    def to_feature_vector(self, normalize: bool = True) -> np.ndarray:
        """
        Convert sensor data to a flat feature vector for LSTM input.
        
        Args:
            normalize: Whether to normalize the features
        
        Returns:
            Feature vector: [gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z, emg_ch1, emg_ch2, ...]
        """
        features = []
        
        # Add IMU features (6 values: 3 gyro + 3 accel)
        if self.imu is not None:
            features.extend(self.imu.angular_velocity.tolist())
            features.extend(self.imu.linear_acceleration.tolist())
        else:
            # Fill with zeros if IMU data is missing
            features.extend([0.0] * 6)
        
        # Add EMG features (n_channels values)
        if self.emg is not None:
            features.extend(self.emg.channels.tolist())
        else:
            # Fill with zeros if EMG data is missing (need to know expected channels)
            pass  # Will be handled by the sequence builder
        
        feature_vector = np.array(features, dtype=np.float32)
        
        if normalize:
            # Simple normalization (can be replaced with learned statistics)
            feature_vector = self._normalize_features(feature_vector)
        
        return feature_vector
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features using typical sensor ranges.
        
        IMU ranges (approximate):
        - Gyroscope: ±2000 deg/s = ±34.9 rad/s
        - Accelerometer: ±16g = ±156.8 m/s²
        
        EMG ranges:
        - Typically 0-1 or normalized to unit variance
        """
        normalized = features.copy()
        
        # Normalize gyroscope (first 3 values)
        if len(normalized) >= 3:
            normalized[0:3] = normalized[0:3] / 35.0  # Normalize to ~[-1, 1]
        
        # Normalize accelerometer (next 3 values)
        if len(normalized) >= 6:
            normalized[3:6] = normalized[3:6] / 160.0  # Normalize to ~[-1, 1]
        
        # EMG is typically already normalized or in mV range
        # Keep as-is or apply custom normalization
        
        return normalized
    
    def get_feature_names(self, emg_channel_count: int = 0) -> List[str]:
        """Get names of all features in the feature vector."""
        names = [
            'gyro_x', 'gyro_y', 'gyro_z',
            'accel_x', 'accel_y', 'accel_z'
        ]
        
        # Add EMG channel names
        for i in range(emg_channel_count):
            names.append(f'emg_ch{i+1}')
        
        return names


class SensorSequenceBuilder:
    """
    Builds sequences of sensor data for LSTM training/inference.
    
    LSTMs require input shape: (batch_size, sequence_length, feature_dim)
    This class manages the temporal windowing and batching.
    """
    
    def __init__(
        self,
        sequence_length: int = 50,
        overlap: int = 25,
        emg_channel_count: int = 8,
        normalize: bool = True
    ):
        """
        Initialize sequence builder.
        
        Args:
            sequence_length: Number of timesteps in each sequence
            overlap: Number of overlapping timesteps between sequences
            emg_channel_count: Number of EMG channels expected
            normalize: Whether to normalize features
        """
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.emg_channel_count = emg_channel_count
        self.normalize = normalize
        
        # Buffer for building sequences
        self.data_buffer: List[SensorFusionData] = []
        
        # Statistics for normalization (can be learned from data)
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None
    
    def add_sample(self, sensor_data: SensorFusionData):
        """Add a new sensor sample to the buffer."""
        self.data_buffer.append(sensor_data)
        
        # Keep buffer size manageable
        max_buffer_size = self.sequence_length * 10
        if len(self.data_buffer) > max_buffer_size:
            self.data_buffer = self.data_buffer[-max_buffer_size:]
    
    def get_latest_sequence(self) -> Optional[np.ndarray]:
        """
        Get the most recent sequence of sensor data.
        
        Returns:
            Sequence array of shape (sequence_length, feature_dim) or None if not enough data
        """
        if len(self.data_buffer) < self.sequence_length:
            return None
        
        # Get the last sequence_length samples
        recent_samples = self.data_buffer[-self.sequence_length:]
        
        # Convert to feature vectors
        sequence = []
        for sample in recent_samples:
            features = sample.to_feature_vector(normalize=self.normalize)
            
            # Ensure consistent feature dimension (pad EMG if needed)
            expected_dim = 6 + self.emg_channel_count
            if len(features) < expected_dim:
                # Pad with zeros for missing EMG channels
                features = np.pad(features, (0, expected_dim - len(features)), mode='constant')
            
            sequence.append(features)
        
        sequence_array = np.array(sequence, dtype=np.float32)
        
        # Apply learned normalization if available
        if self.feature_mean is not None and self.feature_std is not None:
            sequence_array = (sequence_array - self.feature_mean) / (self.feature_std + 1e-8)
        
        return sequence_array
    
    def get_all_sequences(self, stride: Optional[int] = None) -> List[np.ndarray]:
        """
        Get all possible sequences from the buffer with specified stride.
        
        Args:
            stride: Step size between sequences (default: sequence_length - overlap)
        
        Returns:
            List of sequence arrays, each of shape (sequence_length, feature_dim)
        """
        if stride is None:
            stride = self.sequence_length - self.overlap
        
        sequences = []
        
        for i in range(0, len(self.data_buffer) - self.sequence_length + 1, stride):
            samples = self.data_buffer[i:i + self.sequence_length]
            
            # Convert to feature vectors
            sequence = []
            for sample in samples:
                features = sample.to_feature_vector(normalize=self.normalize)
                
                # Ensure consistent feature dimension
                expected_dim = 6 + self.emg_channel_count
                if len(features) < expected_dim:
                    features = np.pad(features, (0, expected_dim - len(features)), mode='constant')
                
                sequence.append(features)
            
            sequence_array = np.array(sequence, dtype=np.float32)
            
            # Apply learned normalization if available
            if self.feature_mean is not None and self.feature_std is not None:
                sequence_array = (sequence_array - self.feature_mean) / (self.feature_std + 1e-8)
            
            sequences.append(sequence_array)
        
        return sequences
    
    def fit_normalization(self, data_samples: List[SensorFusionData]):
        """
        Compute normalization statistics from a dataset.
        
        Args:
            data_samples: List of sensor fusion data samples
        """
        all_features = []
        
        for sample in data_samples:
            features = sample.to_feature_vector(normalize=False)
            
            # Ensure consistent feature dimension
            expected_dim = 6 + self.emg_channel_count
            if len(features) < expected_dim:
                features = np.pad(features, (0, expected_dim - len(features)), mode='constant')
            
            all_features.append(features)
        
        if all_features:
            feature_matrix = np.array(all_features, dtype=np.float32)
            self.feature_mean = np.mean(feature_matrix, axis=0)
            self.feature_std = np.std(feature_matrix, axis=0)
            
            print(f"Normalization statistics computed from {len(data_samples)} samples")
            print(f"Feature mean: {self.feature_mean}")
            print(f"Feature std: {self.feature_std}")
    
    def save_normalization_stats(self, filepath: str):
        """Save normalization statistics to file."""
        if self.feature_mean is not None and self.feature_std is not None:
            np.savez(filepath, mean=self.feature_mean, std=self.feature_std)
            print(f"Normalization stats saved to {filepath}")
    
    def load_normalization_stats(self, filepath: str):
        """Load normalization statistics from file."""
        data = np.load(filepath)
        self.feature_mean = data['mean']
        self.feature_std = data['std']
        print(f"Normalization stats loaded from {filepath}")
    
    def clear_buffer(self):
        """Clear the data buffer."""
        self.data_buffer = []
    
    def get_buffer_size(self) -> int:
        """Get current buffer size."""
        return len(self.data_buffer)
    
    def get_feature_dimension(self) -> int:
        """Get the feature dimension for LSTM input."""
        return 6 + self.emg_channel_count


def create_batch_for_lstm(
    sequences: List[np.ndarray],
    labels: Optional[List[Any]] = None
) -> Dict[str, np.ndarray]:
    """
    Create a batch of sequences for LSTM training/inference.
    
    Args:
        sequences: List of sequence arrays, each of shape (sequence_length, feature_dim)
        labels: Optional list of labels for supervised learning
    
    Returns:
        Dictionary with 'sequences' and optionally 'labels' as numpy arrays
    """
    if not sequences:
        raise ValueError("Cannot create batch from empty sequence list")
    
    # Stack sequences into batch
    batch_sequences = np.stack(sequences, axis=0)  # Shape: (batch_size, seq_len, feature_dim)
    
    result = {'sequences': batch_sequences}
    
    if labels is not None:
        result['labels'] = np.array(labels)
    
    return result


# Example usage and testing
if __name__ == "__main__":
    print("Testing SensorFusionData format...")
    
    # Create sample IMU data
    imu_data = IMUData(
        angular_velocity=np.array([0.1, 0.2, 0.3]),
        linear_acceleration=np.array([9.8, 0.1, 0.2]),
        timestamp=time.time()
    )
    
    # Create sample EMG data (8 channels)
    emg_data = EMGData(
        channels=np.array([0.5, 0.6, 0.4, 0.7, 0.3, 0.8, 0.2, 0.5]),
        timestamp=time.time(),
        sampling_rate=1000.0
    )
    
    # Create fused data
    fused_data = SensorFusionData(imu=imu_data, emg=emg_data)
    
    # Convert to feature vector
    features = fused_data.to_feature_vector(normalize=True)
    print(f"Feature vector shape: {features.shape}")
    print(f"Feature vector: {features}")
    print(f"Feature names: {fused_data.get_feature_names(emg_channel_count=8)}")
    
    # Test sequence builder
    print("\nTesting SensorSequenceBuilder...")
    builder = SensorSequenceBuilder(
        sequence_length=10,
        overlap=5,
        emg_channel_count=8,
        normalize=True
    )
    
    # Add multiple samples
    for i in range(20):
        sample_imu = IMUData(
            angular_velocity=np.random.randn(3),
            linear_acceleration=np.random.randn(3) + np.array([9.8, 0, 0]),
            timestamp=time.time() + i * 0.01
        )
        sample_emg = EMGData(
            channels=np.random.randn(8) * 0.5,
            timestamp=time.time() + i * 0.01
        )
        sample_fused = SensorFusionData(imu=sample_imu, emg=sample_emg)
        builder.add_sample(sample_fused)
    
    # Get latest sequence
    sequence = builder.get_latest_sequence()
    if sequence is not None:
        print(f"Latest sequence shape: {sequence.shape}")
        print(f"Expected shape: (sequence_length={builder.sequence_length}, feature_dim={builder.get_feature_dimension()})")
    
    # Get all sequences
    all_sequences = builder.get_all_sequences()
    print(f"Total sequences generated: {len(all_sequences)}")
    
    # Create batch for LSTM
    if all_sequences:
        batch = create_batch_for_lstm(all_sequences)
        print(f"Batch shape: {batch['sequences'].shape}")
        print(f"Expected: (batch_size, sequence_length, feature_dim)")
    
    print("\nAll tests passed!")
