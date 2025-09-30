# EMG Signal Processing and Intent Classification

This package provides a comprehensive solution for EMG (Electromyography) signal processing and intent classification using LSTM neural networks.

## Overview

The EMG package implements:
- **Feature Extraction**: Time-domain, frequency-domain, and time-frequency features
- **LSTM Model**: Temporal modeling for gesture/intent classification
- **Data Processing**: Dataset management and preprocessing
- **Real-time Processing**: Live EMG signal processing and prediction

## Key Components

### 1. LSTM Model (`lstm_model.py`)
- `EMGLSTMModel`: PyTorch LSTM model with attention mechanism
- `EMGClassifier`: High-level interface for training and inference
- Supports bidirectional LSTM, dropout, and multi-head attention

### 2. Feature Extraction (`feature_extractor.py`)
- `EMGFeatureExtractor`: Comprehensive feature extraction
- **Time-domain**: MAV, RMS, VAR, WL, ZC, SSC, WAMP, skewness, kurtosis
- **Frequency-domain**: MNF, MDF, peak frequency, spectral features
- **Time-frequency**: Wavelet transform features

### 3. Data Processing (`data_processor.py`)
- `EMGDataset`: PyTorch dataset for EMG sequences
- `EMGDataProcessor`: Complete data pipeline
- Handles loading, preprocessing, feature extraction, and dataset creation

### 4. Real-time Processing (`real_time_processor.py`)
- `RealTimeEMGProcessor`: Live EMG processing
- `EMGDataCollector`: Data collection for training
- Supports real-time prediction with smoothing and callbacks

## Usage

### Basic Usage

```python
from emg import EMGClassifier, EMGFeatureExtractor, EMGDataProcessor

# 1. Process EMG data
processor = EMGDataProcessor(sampling_rate=1000, window_size=200, seq_len=10)
processor.load_emg_data("path/to/emg_data.npy")
processor.extract_features()
sequences, labels = processor.create_sequences()
processor.normalize_features()

# 2. Train LSTM model
classifier = EMGClassifier(
    feature_dim=processor.feature_dim,
    seq_len=processor.seq_len,
    num_classes=len(processor.get_class_names())
)

train_loader, val_loader, test_loader = processor.split_data()
for epoch in range(10):
    metrics = classifier.train_epoch(train_loader, val_loader)
    print(f"Epoch {epoch}: Loss={metrics['loss']:.4f}, Acc={metrics['accuracy']:.2f}%")

# 3. Make predictions
predictions = classifier.predict_class(test_sequences)
```

### Real-time Processing

```python
from emg import RealTimeEMGProcessor

# Initialize real-time processor
processor = RealTimeEMGProcessor(
    model=classifier.model,
    feature_extractor=processor.feature_extractor,
    window_size=200,
    seq_len=10
)

# Set up callbacks
def on_gesture(class_id, confidence):
    print(f"Gesture detected: {class_id} (confidence: {confidence:.3f})")

processor.set_gesture_callback(on_gesture)
processor.start_processing()

# Feed EMG data
for emg_sample in emg_data_stream:
    processor.add_sample(emg_sample)
```

### Feature Extraction

```python
from emg import EMGFeatureExtractor

# Initialize feature extractor
extractor = EMGFeatureExtractor(sampling_rate=1000)

# Extract features from EMG data
features = extractor.extract_all_features(emg_data)

# Extract features from sliding windows
feature_matrix, feature_names = extractor.extract_features_from_windows(
    emg_data=emg_data,
    window_size=200,
    overlap=0.5
)
```

## Model Architecture

### LSTM Model Structure
```
Input: (batch_size, seq_len, feature_dim)
    ↓
LSTM Layers (bidirectional, dropout)
    ↓
Multi-head Attention
    ↓
Global Average Pooling
    ↓
Dense Layers (ReLU, Dropout)
    ↓
Output: (batch_size, num_classes)
```

### Feature Types

**Time-domain Features:**
- Mean Absolute Value (MAV)
- Root Mean Square (RMS)
- Variance (VAR)
- Waveform Length (WL)
- Zero Crossings (ZC)
- Slope Sign Changes (SSC)
- Willison Amplitude (WAMP)
- Skewness and Kurtosis

**Frequency-domain Features:**
- Mean Frequency (MNF)
- Median Frequency (MDF)
- Peak Frequency
- Spectral Centroid
- Spectral Rolloff
- Band Power (multiple frequency bands)

**Time-frequency Features:**
- Wavelet coefficients
- Wavelet energy
- Wavelet statistics

## Data Format

### Input Data Structure
```python
# Raw EMG data: (num_channels, num_samples)
emg_data = np.array([
    [sample1_ch1, sample2_ch1, ...],  # Channel 1
    [sample1_ch2, sample2_ch2, ...],  # Channel 2
    ...
])

# Features: (num_windows, num_features)
features = extractor.extract_features_from_windows(emg_data)

# Sequences: (num_sequences, seq_len, num_features)
sequences = processor.create_sequences()
```

### Supported File Formats
- **NumPy**: `.npy` files with dictionary structure
- **CSV**: Comma-separated values
- **JSON**: JSON format with arrays

## Configuration

### Model Configuration
```python
model_config = {
    "hidden_dim": 128,        # LSTM hidden dimension
    "num_layers": 2,          # Number of LSTM layers
    "dropout_rate": 0.3,      # Dropout rate
    "bidirectional": True     # Bidirectional LSTM
}
```

### Processing Configuration
```python
processor_config = {
    "sampling_rate": 1000,    # EMG sampling rate (Hz)
    "window_size": 200,       # Feature window size (samples)
    "overlap": 0.5,           # Window overlap ratio
    "seq_len": 10             # Sequence length for LSTM
}
```

## Examples

### Run Complete Demo
```bash
cd src
python -m emg.example_usage
```

### Train Custom Model
```python
# Load your EMG data
processor = EMGDataProcessor()
processor.load_emg_data("your_emg_data.npy")

# Extract features and create sequences
processor.extract_features()
sequences, labels = processor.create_sequences()
processor.normalize_features()

# Train model
classifier = EMGClassifier(
    feature_dim=processor.feature_dim,
    seq_len=processor.seq_len,
    num_classes=len(processor.get_class_names())
)

# Training loop
train_loader, val_loader, test_loader = processor.split_data()
for epoch in range(50):
    metrics = classifier.train_epoch(train_loader, val_loader)
    print(f"Epoch {epoch}: {metrics}")

# Save model
classifier.save_model("emg_model.pth")
```

## Performance Considerations

### Optimization Tips
1. **Feature Selection**: Use only relevant features to reduce dimensionality
2. **Window Size**: Balance between temporal resolution and feature stability
3. **Sequence Length**: Longer sequences capture more temporal patterns
4. **Model Size**: Adjust hidden dimensions based on data complexity
5. **Real-time Processing**: Use smaller models for real-time applications

### Hardware Requirements
- **Training**: GPU recommended for large datasets
- **Inference**: CPU sufficient for real-time processing
- **Memory**: Depends on sequence length and feature dimensions

## Installation

### Quick Install

```bash
cd src/emg
pip install -r requirements.txt
```

### Development Install

```bash
cd src/emg
pip install -e .[dev]
```

For detailed installation instructions, see [INSTALL.md](INSTALL.md).

## Dependencies

### Core Dependencies
- `torch>=1.9.0` - Deep learning framework
- `numpy>=1.21.0` - Numerical computing
- `scipy>=1.7.0` - Scientific computing
- `scikit-learn>=1.0.0` - Machine learning utilities
- `pywavelets>=1.1.0` - Wavelet transforms

### Optional Dependencies
- `matplotlib>=3.4.0` - Visualization
- `seaborn>=0.11.0` - Statistical visualization
- `pandas>=1.3.0` - Data manipulation
- `jupyter>=1.0.0` - Jupyter notebook support
- `librosa>=0.8.0` - Advanced signal processing
- `sounddevice>=0.4.0` - Real-time audio processing

## Future Enhancements

- [ ] Support for more EMG sensors and channels
- [ ] Advanced feature selection algorithms
- [ ] Ensemble methods for improved accuracy
- [ ] Online learning capabilities
- [ ] Integration with robotic control systems
- [ ] Real-time visualization tools
