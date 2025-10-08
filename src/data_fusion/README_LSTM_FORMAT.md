# IMU + EMG Data Format for LSTM

This document describes the unified data format for combining IMU and EMG sensor data for LSTM neural network input.

## Overview

The data format is designed to:
- **Combine multiple sensor modalities** (IMU + EMG)
- **Support time-series processing** with LSTM networks
- **Provide normalized features** for better training
- **Handle missing data** gracefully
- **Be framework-agnostic** (works with PyTorch, TensorFlow, etc.)

## Data Structures

### 1. `IMUData`
Represents a single IMU measurement:
```python
@dataclass
class IMUData:
    angular_velocity: np.ndarray  # [wx, wy, wz] in rad/s (shape: 3)
    linear_acceleration: np.ndarray  # [ax, ay, az] in m/s² (shape: 3)
    timestamp: float  # seconds
```

### 2. `EMGData`
Represents a single EMG measurement:
```python
@dataclass
class EMGData:
    channels: np.ndarray  # EMG signal from multiple channels (shape: n_channels)
    timestamp: float  # seconds
    sampling_rate: Optional[float]  # Hz
```

### 3. `SensorFusionData`
Combines IMU and EMG data at a single timestep:
```python
@dataclass
class SensorFusionData:
    imu: Optional[IMUData]
    emg: Optional[EMGData]
    timestamp: float
```

**Key Method:**
```python
feature_vector = sensor_fusion_data.to_feature_vector(normalize=True)
# Returns: [gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z, emg_ch1, emg_ch2, ...]
```

## Feature Vector Format

The feature vector combines all sensor data into a single 1D array:

| Index | Feature | Range | Description |
|-------|---------|-------|-------------|
| 0 | `gyro_x` | ±35 rad/s | Angular velocity around X-axis |
| 1 | `gyro_y` | ±35 rad/s | Angular velocity around Y-axis |
| 2 | `gyro_z` | ±35 rad/s | Angular velocity around Z-axis |
| 3 | `accel_x` | ±160 m/s² | Linear acceleration along X-axis |
| 4 | `accel_y` | ±160 m/s² | Linear acceleration along Y-axis |
| 5 | `accel_z` | ±160 m/s² | Linear acceleration along Z-axis |
| 6+ | `emg_ch1...N` | Normalized | EMG channels (typically 8 channels) |

**Total feature dimension:** 6 (IMU) + N (EMG channels)

For 8 EMG channels: **14 features per timestep**

## LSTM Input Format

### Sequence Building

The `SensorSequenceBuilder` class creates sequences for LSTM input:

```python
builder = SensorSequenceBuilder(
    sequence_length=50,      # Number of timesteps per sequence
    overlap=25,              # Overlapping timesteps between sequences
    emg_channel_count=8,     # Number of EMG channels
    normalize=True           # Apply normalization
)

# Add samples
for sensor_data in data_stream:
    builder.add_sample(sensor_data)

# Get latest sequence
sequence = builder.get_latest_sequence()
# Shape: (sequence_length, feature_dim) = (50, 14)
```

### Batch Format

For LSTM training, sequences are batched:

```python
sequences = builder.get_all_sequences()
batch = create_batch_for_lstm(sequences, labels)

# batch['sequences'] shape: (batch_size, sequence_length, feature_dim)
# batch['labels'] shape: (batch_size,) or (batch_size, n_classes)
```

**Example dimensions:**
- Batch size: 32
- Sequence length: 50 timesteps
- Feature dimension: 14 (6 IMU + 8 EMG)
- **Final shape: (32, 50, 14)**

## Usage Examples

### 1. Basic Data Collection

```python
from data_fusion.sensor_data_format import (
    SensorFusionData, IMUData, EMGData, SensorSequenceBuilder
)
from sensors.imu_reader import IMUReader
from sensors.emg_reader import EMGReader

# Initialize sensors
imu_reader = IMUReader()
emg_reader = EMGReader(n_channels=8)

imu_reader.activate()
emg_reader.activate()

# Initialize sequence builder
builder = SensorSequenceBuilder(
    sequence_length=50,
    emg_channel_count=8,
    normalize=True
)

# Collect data
while collecting:
    imu_data = imu_reader.get_latest_data()
    emg_data = emg_reader.get_latest_data()
    
    fused = SensorFusionData(imu=imu_data, emg=emg_data)
    builder.add_sample(fused)
    
    # Get sequence when ready
    if builder.get_buffer_size() >= 50:
        sequence = builder.get_latest_sequence()
        # Use sequence for inference or training
```

### 2. Training Data Collection

```bash
# Collect 60 seconds of data
python src/data_fusion/imu_emg_lstm_example.py \
    --mode collect \
    --duration 60 \
    --output training_data.npz \
    --sequence-length 50 \
    --emg-channels 8
```

### 3. PyTorch LSTM Training

```python
import torch
import torch.nn as nn
import numpy as np

# Load collected data
data = np.load('training_data.npz')
sequences = data['sequences']  # Shape: (n_sequences, seq_len, feature_dim)

# Convert to PyTorch tensors
X = torch.from_numpy(sequences).float()
y = torch.from_numpy(labels).long()

# Define LSTM model
class SensorLSTM(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=128, num_layers=2, num_classes=7):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # Take last timestep output
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)

# Initialize model
model = SensorLSTM(input_dim=14, hidden_dim=128, num_classes=7)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    outputs = model(X)
    loss = criterion(outputs, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 4. TensorFlow/Keras LSTM Training

```python
import tensorflow as tf
from tensorflow import keras

# Load data
data = np.load('training_data.npz')
X = data['sequences']  # Shape: (n_sequences, seq_len, feature_dim)
y = data['labels']

# Define model
model = keras.Sequential([
    keras.layers.LSTM(128, return_sequences=True, input_shape=(50, 14)),
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(64),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(7, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)
```

## Normalization

### Built-in Normalization
Features are normalized using typical sensor ranges:
- **Gyroscope:** Divided by 35 rad/s (≈2000 deg/s)
- **Accelerometer:** Divided by 160 m/s² (≈16g)
- **EMG:** Typically pre-normalized

### Learned Normalization
For better performance, compute statistics from your dataset:

```python
# Fit normalization to your data
builder.fit_normalization(all_training_samples)

# Save for later use
builder.save_normalization_stats('normalization.npz')

# Load in production
builder.load_normalization_stats('normalization.npz')
```

## Data Collection Best Practices

1. **Synchronization:** Ensure IMU and EMG data are time-synchronized
2. **Sampling Rate:** Match sampling rates or use interpolation
3. **Missing Data:** The format handles missing sensors gracefully (fills with zeros)
4. **Sequence Length:** Choose based on your task:
   - Short sequences (20-30): Fast reactions
   - Medium sequences (50-100): General purpose
   - Long sequences (100-200): Complex patterns
5. **Overlap:** Use 50% overlap for more training samples
6. **Normalization:** Always normalize for better LSTM training

## File Format

Data is saved in NumPy `.npz` format:

```python
data = np.load('data.npz')

# Available fields:
sequences = data['sequences']      # (n_sequences, seq_len, feature_dim)
metadata = data['metadata'].item() # Dictionary with collection info
feature_mean = data['feature_mean'] # Normalization statistics
feature_std = data['feature_std']   # Normalization statistics
```

## Integration with Robot Control States

For reinforcement learning with robot states:

```python
from robot_control.states import MoveToCupState

# Collect data during state execution
while state.execute():
    sensor_data = collect_sensor_data()
    builder.add_sample(sensor_data)
    
    # Get current sequence for RL agent
    sequence = builder.get_latest_sequence()
    action = rl_agent.predict(sequence)
    robot.execute_action(action)
```

## Troubleshooting

### Issue: Inconsistent feature dimensions
**Solution:** Ensure `emg_channel_count` is set correctly in `SensorSequenceBuilder`

### Issue: Missing data
**Solution:** The format fills missing sensors with zeros. Check sensor connectivity.

### Issue: Poor LSTM performance
**Solution:** 
- Increase sequence length
- Apply learned normalization
- Collect more training data
- Try different LSTM architectures

### Issue: Memory issues with large datasets
**Solution:** Use data generators or streaming for training instead of loading all data at once

## References

- IMU data format: `src/data_fusion/smoothing.py`
- EMG reader: `src/sensors/emg_reader.py`
- Example script: `src/data_fusion/imu_emg_lstm_example.py`
- Sensor fusion: `src/data_fusion/sensor_data_format.py`
