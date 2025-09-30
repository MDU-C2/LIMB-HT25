#!/usr/bin/env python3
"""Test script to verify the EMG package works correctly."""

import numpy as np
import sys
import os

# Add parent directory to path if running directly
if __name__ == "__main__":
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

def test_emg_imports():
    """Test that EMG package imports work correctly."""
    print("Testing EMG package imports...")
    
    try:
        from emg.lstm_model import EMGLSTMModel, EMGClassifier
        print("✓ LSTM model imports successful")
        
        from emg.feature_extractor import EMGFeatureExtractor
        print("✓ Feature extractor import successful")
        
        from emg.data_processor import EMGDataProcessor, EMGDataset
        print("✓ Data processor imports successful")
        
        from emg.real_time_processor import RealTimeEMGProcessor, EMGDataCollector
        print("✓ Real-time processor imports successful")
        
        # Test package-level imports
        from emg import EMGLSTMModel, EMGClassifier, EMGFeatureExtractor, EMGDataProcessor, RealTimeEMGProcessor
        print("✓ Package-level imports successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_feature_extraction():
    """Test EMG feature extraction."""
    print("\nTesting EMG feature extraction...")
    
    try:
        from emg.feature_extractor import EMGFeatureExtractor
        
        # Create synthetic EMG data
        sampling_rate = 1000
        duration = 1.0  # 1 second
        num_samples = int(sampling_rate * duration)
        num_channels = 4
        
        # Generate synthetic EMG signal
        t = np.linspace(0, duration, num_samples)
        emg_data = np.zeros((num_channels, num_samples))
        
        for ch in range(num_channels):
            # Create EMG-like signal with different frequencies
            freq = 50 + ch * 20
            signal = np.sin(2 * np.pi * freq * t) + 0.1 * np.random.randn(num_samples)
            emg_data[ch, :] = signal
        
        # Initialize feature extractor
        extractor = EMGFeatureExtractor(sampling_rate=sampling_rate)
        
        # Extract features
        features = extractor.extract_all_features(emg_data, include_time_freq=False)
        feature_names = extractor.get_feature_names()
        
        print(f"✓ Extracted {len(features)} features")
        print(f"✓ Feature names: {len(feature_names)} features")
        print(f"✓ Sample features: {features[:5]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature extraction error: {e}")
        return False

def test_lstm_model():
    """Test LSTM model creation and basic functionality."""
    print("\nTesting LSTM model...")
    
    try:
        from emg.lstm_model import EMGLSTMModel, EMGClassifier
        
        # Model parameters
        feature_dim = 50
        seq_len = 10
        num_classes = 5
        
        # Test model creation
        model = EMGLSTMModel(
            feature_dim=feature_dim,
            seq_len=seq_len,
            num_classes=num_classes,
            hidden_dim=64,
            num_layers=2,
            dropout_rate=0.3,
            bidirectional=True
        )
        print("✓ LSTM model created successfully")
        
        # Test forward pass
        batch_size = 4
        x = np.random.randn(batch_size, seq_len, feature_dim)
        x_tensor = torch.FloatTensor(x)
        
        with torch.no_grad():
            output = model(x_tensor)
        
        print(f"✓ Forward pass successful: input {x.shape} -> output {output.shape}")
        
        # Test classifier
        classifier = EMGClassifier(
            feature_dim=feature_dim,
            seq_len=seq_len,
            num_classes=num_classes
        )
        print("✓ EMG classifier created successfully")
        
        # Test prediction
        predictions = classifier.predict(x)
        print(f"✓ Prediction successful: {predictions.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ LSTM model error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_processing():
    """Test data processing functionality."""
    print("\nTesting data processing...")
    
    try:
        from emg.data_processor import EMGDataProcessor
        
        # Create synthetic EMG data
        num_gestures = 3
        num_channels = 4
        samples_per_gesture = 500
        
        emg_data = {}
        for i in range(num_gestures):
            gesture_name = f"gesture_{i}"
            # Create synthetic data with different characteristics
            t = np.linspace(0, 1, samples_per_gesture)
            signal = np.sin(2 * np.pi * (50 + i * 20) * t) + 0.1 * np.random.randn(samples_per_gesture)
            emg_data[gesture_name] = np.tile(signal, (num_channels, 1))
        
        # Initialize processor
        processor = EMGDataProcessor(
            sampling_rate=1000,
            window_size=100,
            overlap=0.5,
            seq_len=5
        )
        
        # Load data
        processor.raw_data = emg_data
        print("✓ Data loaded successfully")
        
        # Extract features
        features = processor.extract_features(include_time_freq=False)
        print("✓ Features extracted successfully")
        
        # Create sequences
        sequences, labels = processor.create_sequences()
        print(f"✓ Sequences created: {sequences.shape}")
        
        # Normalize features
        sequences_normalized = processor.normalize_features()
        print("✓ Features normalized successfully")
        
        # Get data info
        info = processor.get_data_info()
        print(f"✓ Data info: {info['num_sequences']} sequences, {info['num_classes']} classes")
        
        return True
        
    except Exception as e:
        print(f"❌ Data processing error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("EMG Package Test")
    print("=" * 60)
    
    # Test imports
    if test_emg_imports():
        print("\n" + "=" * 60)
        print("✅ Import tests passed!")
        
        # Test feature extraction
        if test_feature_extraction():
            print("\n" + "=" * 60)
            print("✅ Feature extraction tests passed!")
            
            # Test LSTM model
            if test_lstm_model():
                print("\n" + "=" * 60)
                print("✅ LSTM model tests passed!")
                
                # Test data processing
                if test_data_processing():
                    print("\n" + "=" * 60)
                    print("✅ All tests passed!")
                    print("The EMG package is working correctly.")
                    print("=" * 60)
                else:
                    print("\n" + "=" * 60)
                    print("❌ Data processing tests failed!")
                    print("=" * 60)
            else:
                print("\n" + "=" * 60)
                print("❌ LSTM model tests failed!")
                print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("❌ Feature extraction tests failed!")
            print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ Import tests failed!")
        print("=" * 60)

if __name__ == "__main__":
    main()
