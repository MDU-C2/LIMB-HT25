#!/usr/bin/env python3
"""Example usage of the EMG LSTM model for intent classification.

This script demonstrates how to:
1. Load and preprocess EMG data
2. Extract features from EMG signals
3. Train an LSTM model for gesture classification
4. Evaluate the model performance
5. Use the model for real-time prediction
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import os
import sys
import time

# Add parent directory to path if running directly
if __name__ == "__main__":
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

from emg.lstm_model import EMGLSTMModel, EMGClassifier
from emg.feature_extractor import EMGFeatureExtractor
from emg.data_processor import EMGDataProcessor
from emg.real_time_processor import RealTimeEMGProcessor


def create_synthetic_emg_data(
    num_gestures: int = 5,
    num_channels: int = 8,
    samples_per_gesture: int = 1000,
    sampling_rate: int = 1000,
    num_trials_per_gesture: int = 10
) -> dict:
    """
    Create synthetic EMG data for demonstration.
    
    Args:
        num_gestures: Number of different gestures
        num_channels: Number of EMG channels
        samples_per_gesture: Number of samples per gesture trial
        sampling_rate: Sampling rate in Hz
        num_trials_per_gesture: Number of trials per gesture
        
    Returns:
        Dictionary mapping gesture names to list of EMG data trials
    """
    print("Creating synthetic EMG data...")
    
    gestures = [f"gesture_{i}" for i in range(num_gestures)]
    emg_data = {}
    
    for i, gesture in enumerate(gestures):
        gesture_trials = []
        
        for trial in range(num_trials_per_gesture):
            # Create synthetic EMG signal with different characteristics for each gesture
            t = np.linspace(0, samples_per_gesture / sampling_rate, samples_per_gesture)
            
            # Different frequency components for each gesture
            base_freq = 50 + i * 20  # 50, 70, 90, 110, 130 Hz
            harmonic_freq = base_freq * 2
            
            # Generate multi-channel EMG data
            gesture_data = np.zeros((num_channels, samples_per_gesture))
            
            for ch in range(num_channels):
                # Add different phase and amplitude for each channel
                phase = ch * np.pi / 4
                amplitude = 0.5 + 0.3 * np.sin(ch * np.pi / 2)
                
                # Add trial variation
                trial_variation = 1 + 0.1 * np.sin(trial * np.pi / 3)
                amplitude *= trial_variation
                
                # Base signal with harmonics
                signal = (amplitude * np.sin(2 * np.pi * base_freq * t + phase) +
                         0.3 * amplitude * np.sin(2 * np.pi * harmonic_freq * t + phase))
                
                # Add noise with trial variation
                noise_level = 0.1 + 0.02 * trial
                noise = noise_level * np.random.randn(samples_per_gesture)
                signal += noise
                
                # Add some non-stationarity
                envelope = 1 + 0.2 * np.sin(2 * np.pi * 0.5 * t)
                signal *= envelope
                
                gesture_data[ch, :] = signal
            
            gesture_trials.append(gesture_data)
        
        emg_data[gesture] = gesture_trials
        print(f"  Created {gesture}: {len(gesture_trials)} trials of shape {gesture_trials[0].shape}")
    
    return emg_data


def demonstrate_feature_extraction():
    """Demonstrate EMG feature extraction."""
    print("\n" + "="*60)
    print("EMG Feature Extraction Demo")
    print("="*60)
    
    # Create synthetic data
    emg_data = create_synthetic_emg_data(num_gestures=3, samples_per_gesture=500, num_trials_per_gesture=1)
    
    # Initialize feature extractor
    feature_extractor = EMGFeatureExtractor(sampling_rate=1000)
    
    # Extract features for each gesture
    for gesture_name, trials in emg_data.items():
        print(f"\nExtracting features for {gesture_name}...")
        
        # Use the first trial for demonstration
        data = trials[0]
        
        # Extract all features
        features = feature_extractor.extract_all_features(data)
        feature_names = feature_extractor.get_feature_names()
        
        print(f"  Extracted {len(features)} features")
        print(f"  Feature names: {feature_names[:5]}...")  # Show first 5
        print(f"  Feature values: {features[:5]}")  # Show first 5 values


def demonstrate_data_processing():
    """Demonstrate EMG data processing and sequence creation."""
    print("\n" + "="*60)
    print("EMG Data Processing Demo")
    print("="*60)
    
    # Create synthetic data with multiple trials per gesture
    emg_data = create_synthetic_emg_data(num_gestures=4, samples_per_gesture=1000, num_trials_per_gesture=8)
    
    # Initialize data processor
    processor = EMGDataProcessor(
        sampling_rate=1000,
        window_size=200,
        overlap=0.5,
        seq_len=10
    )
    
    # Load data - flatten the trials into a single list per gesture
    flattened_data = {}
    for gesture_name, trials in emg_data.items():
        # Concatenate all trials for this gesture
        concatenated_data = np.concatenate(trials, axis=1)
        flattened_data[gesture_name] = concatenated_data
    
    processor.raw_data = flattened_data
    
    # Extract features
    print("Extracting features...")
    features = processor.extract_features(include_time_freq=True)
    
    # Create sequences
    print("Creating sequences...")
    sequences, labels = processor.create_sequences()
    
    # Normalize features
    print("Normalizing features...")
    sequences_normalized = processor.normalize_features()
    
    # Get data info
    info = processor.get_data_info()
    print(f"\nData Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    return processor, sequences_normalized, labels


def demonstrate_model_training():
    """Demonstrate LSTM model training."""
    print("\n" + "="*60)
    print("LSTM Model Training Demo")
    print("="*60)
    
    # Process data
    processor, sequences, labels = demonstrate_data_processing()
    
    # Split data
    train_loader, val_loader, test_loader = processor.split_data(
        test_size=0.2,
        val_size=0.2
    )
    
    # Initialize classifier
    classifier = EMGClassifier(
        feature_dim=processor.feature_dim,
        seq_len=processor.seq_len,
        num_classes=len(processor.get_class_names()),
        model_config={
            "hidden_dim": 64,
            "num_layers": 2,
            "dropout_rate": 0.3,
            "bidirectional": True
        }
    )
    
    print(classifier.get_model_summary())
    
    # Train model
    print("\nTraining model...")
    num_epochs = 10
    
    for epoch in range(num_epochs):
        metrics = classifier.train_epoch(train_loader, val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {metrics['loss']:.4f}, Train Acc: {metrics['accuracy']:.2f}%")
        if 'val_loss' in metrics:
            print(f"  Val Loss: {metrics['val_loss']:.4f}, Val Acc: {metrics['val_accuracy']:.2f}%")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = classifier.evaluate(test_loader)
    print(f"Test Loss: {test_metrics['loss']:.4f}, Test Acc: {test_metrics['accuracy']:.2f}%")
    
    # Make predictions on test set
    print("\nMaking predictions...")
    all_predictions = []
    all_labels = []
    
    for data, labels_batch in test_loader:
        predictions = classifier.predict_class(data.numpy())
        all_predictions.extend(predictions)
        all_labels.extend(labels_batch.numpy())
    
    # Classification report
    class_names = processor.get_class_names()
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    return classifier, processor


def demonstrate_real_time_processing():
    """Demonstrate real-time EMG processing."""
    print("\n" + "="*60)
    print("Real-time EMG Processing Demo")
    print("="*60)
    
    # Train a model first
    classifier, processor = demonstrate_model_training()
    
    # Initialize real-time processor
    real_time_processor = RealTimeEMGProcessor(
        model=classifier.model,
        feature_extractor=processor.feature_extractor,
        window_size=200,
        overlap=0.5,
        seq_len=10,
        sampling_rate=1000,
        prediction_threshold=0.7,
        smoothing_window=3,
        num_channels=8
    )
    
    # Set up callbacks
    def prediction_callback(class_id, confidence, probabilities):
        class_name = processor.get_class_names()[class_id]
        print(f"Prediction: {class_name} (confidence: {confidence:.3f})")
    
    def gesture_callback(class_id, confidence):
        class_name = processor.get_class_names()[class_id]
        print(f"GESTURE DETECTED: {class_name} (confidence: {confidence:.3f})")
    
    real_time_processor.set_prediction_callback(prediction_callback)
    real_time_processor.set_gesture_callback(gesture_callback)
    
    # Simulate real-time data
    print("\nSimulating real-time EMG data...")
    real_time_processor.start_processing()
    
    # Generate some test data
    test_gesture_data = create_synthetic_emg_data(num_gestures=1, samples_per_gesture=2000, num_trials_per_gesture=1)
    test_data = test_gesture_data["gesture_0"][0]  # Get the first (and only) trial
    
    # Feed data sample by sample
    for i in range(0, test_data.shape[1], 10):  # Process every 10th sample
        # Get a sample from all channels at time i
        sample = test_data[:, i:i+10].flatten()
        real_time_processor.add_sample(sample)
        time.sleep(0.01)  # Simulate real-time delay
    
    # Get statistics
    stats = real_time_processor.get_statistics()
    print(f"\nProcessing Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    real_time_processor.stop_processing()


def main():
    """Run all demonstrations."""
    print("EMG LSTM Model for Intent Classification")
    print("="*60)
    
    try:
        # Run demonstrations
        demonstrate_feature_extraction()
        demonstrate_model_training()
        demonstrate_real_time_processing()
        
        print("\n" + "="*60)
        print("All demonstrations completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
