"""Real-time EMG processing for live gesture recognition.

This module provides real-time processing capabilities for EMG signals,
including feature extraction, sequence management, and live prediction.
"""

import numpy as np
import torch
from collections import deque
from typing import List, Dict, Optional, Callable, Tuple
import threading
import time
from queue import Queue, Empty


class RealTimeEMGProcessor:
    """
    Real-time EMG signal processor for live gesture recognition.
    
    Handles continuous EMG data processing, feature extraction,
    sequence management, and real-time predictions.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        feature_extractor,
        window_size: int = 200,
        overlap: float = 0.5,
        seq_len: int = 10,
        sampling_rate: int = 1000,
        prediction_threshold: float = 0.7,
        smoothing_window: int = 5,
        num_channels: int = 8
    ):
        """
        Initialize the real-time EMG processor.
        
        Args:
            model: Trained LSTM model for prediction
            feature_extractor: EMG feature extractor
            window_size: Size of feature extraction window in samples
            overlap: Overlap ratio between windows
            seq_len: Number of windows in sequence for LSTM
            sampling_rate: Sampling rate of EMG signals in Hz
            prediction_threshold: Minimum confidence for predictions
            smoothing_window: Number of predictions to average for smoothing
            num_channels: Number of EMG channels
        """
        self.model = model
        self.feature_extractor = feature_extractor
        self.window_size = window_size
        self.overlap = overlap
        self.seq_len = seq_len
        self.sampling_rate = sampling_rate
        self.prediction_threshold = prediction_threshold
        self.smoothing_window = smoothing_window
        self.num_channels = num_channels
        
        # Calculate step size for sliding windows
        self.step_size = int(window_size * (1 - overlap))
        
        # Data buffers
        self.raw_buffer = deque(maxlen=window_size * 2)  # Keep extra samples
        self.feature_buffer = deque(maxlen=seq_len)
        self.prediction_buffer = deque(maxlen=smoothing_window)
        
        # Processing state
        self.is_processing = False
        self.is_running = False
        self.last_prediction_time = 0
        self.prediction_interval = 1.0 / 10  # 10 Hz prediction rate
        
        # Callbacks
        self.prediction_callback: Optional[Callable] = None
        self.gesture_callback: Optional[Callable] = None
        
        # Statistics
        self.stats = {
            "total_samples": 0,
            "total_predictions": 0,
            "avg_processing_time": 0.0,
            "last_gesture": None,
            "last_confidence": 0.0
        }
    
    def add_sample(self, sample: np.ndarray):
        """
        Add a new EMG sample to the processing buffer.
        
        Args:
            sample: EMG sample of shape (num_channels,) or (num_channels, 1)
        """
        if sample.ndim == 2 and sample.shape[1] == 1:
            sample = sample.flatten()
        
        self.raw_buffer.extend(sample)
        self.stats["total_samples"] += len(sample)
        
        # Process if we have enough data
        if len(self.raw_buffer) >= self.window_size:
            self._process_window()
    
    def add_samples(self, samples: np.ndarray):
        """
        Add multiple EMG samples to the processing buffer.
        
        Args:
            samples: EMG samples of shape (num_samples, num_channels)
        """
        for sample in samples:
            self.add_sample(sample)
    
    def _process_window(self):
        """Process a window of EMG data."""
        if len(self.raw_buffer) < self.window_size:
            return
        
        # Extract window
        window_data = np.array(list(self.raw_buffer)[-self.window_size:])
        
        # The raw_buffer contains flattened samples from multiple channels
        # We need to reshape it back to (channels, samples) format
        samples_per_channel = self.window_size // self.num_channels
        
        if samples_per_channel * self.num_channels != self.window_size:
            # If the window size doesn't divide evenly, pad or truncate
            window_data = window_data[:samples_per_channel * self.num_channels]
        
        # Reshape to (channels, samples)
        window_reshaped = window_data.reshape(self.num_channels, samples_per_channel)
        
        # Extract features
        features = self.feature_extractor.extract_all_features(
            window_reshaped,
            include_time_freq=True
        )
        
        # Add to feature buffer
        self.feature_buffer.append(features)
        
        # Make prediction if we have enough features
        if len(self.feature_buffer) >= self.seq_len:
            self._make_prediction()
    
    def _make_prediction(self):
        """Make a prediction using the current feature sequence."""
        current_time = time.time()
        
        # Throttle predictions
        if current_time - self.last_prediction_time < self.prediction_interval:
            return
        
        self.last_prediction_time = current_time
        
        # Prepare sequence
        sequence = np.array(list(self.feature_buffer))
        
        # Normalize if scaler is available
        if hasattr(self.feature_extractor, 'scaler') and self.feature_extractor.scaler is not None:
            sequence = self.feature_extractor.scaler.transform(
                sequence.reshape(-1, sequence.shape[-1])
            ).reshape(sequence.shape)
        
        # Convert to tensor
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)  # Add batch dimension
        
        # Make prediction
        start_time = time.time()
        
        with torch.no_grad():
            self.model.eval()
            output = self.model(sequence_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
            confidence = confidence.item()
            predicted_class = predicted_class.item()
        
        processing_time = time.time() - start_time
        
        # Update statistics
        self.stats["total_predictions"] += 1
        self.stats["avg_processing_time"] = (
            (self.stats["avg_processing_time"] * (self.stats["total_predictions"] - 1) + processing_time) /
            self.stats["total_predictions"]
        )
        
        # Add to prediction buffer for smoothing
        self.prediction_buffer.append((predicted_class, confidence))
        
        # Apply smoothing
        smoothed_class, smoothed_confidence = self._smooth_predictions()
        
        # Update stats
        self.stats["last_gesture"] = smoothed_class
        self.stats["last_confidence"] = smoothed_confidence
        
        # Call callbacks
        if self.prediction_callback:
            self.prediction_callback(smoothed_class, smoothed_confidence, probabilities.numpy())
        
        if smoothed_confidence >= self.prediction_threshold and self.gesture_callback:
            self.gesture_callback(smoothed_class, smoothed_confidence)
    
    def _smooth_predictions(self) -> Tuple[int, float]:
        """Apply smoothing to predictions using a moving average."""
        if not self.prediction_buffer:
            return 0, 0.0
        
        # Get most recent predictions
        recent_predictions = list(self.prediction_buffer)
        
        # Weight recent predictions more heavily
        weights = np.linspace(0.5, 1.0, len(recent_predictions))
        weights = weights / np.sum(weights)
        
        # Calculate weighted average of class probabilities
        class_probs = {}
        for (pred_class, confidence), weight in zip(recent_predictions, weights):
            if pred_class not in class_probs:
                class_probs[pred_class] = 0.0
            class_probs[pred_class] += confidence * weight
        
        # Get the class with highest probability
        if class_probs:
            best_class = max(class_probs, key=class_probs.get)
            best_confidence = class_probs[best_class]
        else:
            best_class = recent_predictions[-1][0]
            best_confidence = recent_predictions[-1][1]
        
        return best_class, best_confidence
    
    def set_prediction_callback(self, callback: Callable[[int, float, np.ndarray], None]):
        """
        Set callback for prediction events.
        
        Args:
            callback: Function that takes (class_id, confidence, probabilities)
        """
        self.prediction_callback = callback
    
    def set_gesture_callback(self, callback: Callable[[int, float], None]):
        """
        Set callback for high-confidence gesture events.
        
        Args:
            callback: Function that takes (class_id, confidence)
        """
        self.gesture_callback = callback
    
    def start_processing(self):
        """Start real-time processing."""
        self.is_running = True
        self.is_processing = True
        print("Real-time EMG processing started")
    
    def stop_processing(self):
        """Stop real-time processing."""
        self.is_running = False
        self.is_processing = False
        print("Real-time EMG processing stopped")
    
    def reset_buffers(self):
        """Reset all data buffers."""
        self.raw_buffer.clear()
        self.feature_buffer.clear()
        self.prediction_buffer.clear()
        print("Buffers reset")
    
    def get_current_gesture(self) -> Tuple[Optional[int], float]:
        """
        Get the current predicted gesture and confidence.
        
        Returns:
            Tuple of (gesture_class, confidence)
        """
        return self.stats["last_gesture"], self.stats["last_confidence"]
    
    def get_statistics(self) -> Dict:
        """Get processing statistics."""
        return self.stats.copy()
    
    def update_model(self, new_model: torch.nn.Module):
        """Update the prediction model."""
        self.model = new_model
        print("Model updated")
    
    def set_prediction_threshold(self, threshold: float):
        """Set the prediction confidence threshold."""
        self.prediction_threshold = threshold
        print(f"Prediction threshold set to {threshold}")
    
    def set_smoothing_window(self, window_size: int):
        """Set the smoothing window size."""
        self.smoothing_window = window_size
        self.prediction_buffer = deque(maxlen=window_size)
        print(f"Smoothing window set to {window_size}")


class EMGDataCollector:
    """
    Utility class for collecting EMG data for training.
    
    Provides functionality to record EMG data with gesture labels
    for creating training datasets.
    """
    
    def __init__(
        self,
        sampling_rate: int = 1000,
        buffer_size: int = 10000
    ):
        """
        Initialize the data collector.
        
        Args:
            sampling_rate: Sampling rate of EMG signals
            buffer_size: Size of data buffer
        """
        self.sampling_rate = sampling_rate
        self.buffer_size = buffer_size
        
        self.is_recording = False
        self.current_gesture = None
        self.data_buffer = []
        self.recorded_data = {}
        
    def start_recording(self, gesture_name: str):
        """
        Start recording data for a specific gesture.
        
        Args:
            gesture_name: Name of the gesture being recorded
        """
        self.is_recording = True
        self.current_gesture = gesture_name
        self.data_buffer = []
        print(f"Started recording gesture: {gesture_name}")
    
    def stop_recording(self):
        """Stop recording and save the data."""
        if not self.is_recording:
            return
        
        if self.current_gesture and self.data_buffer:
            # Convert buffer to numpy array
            data = np.array(self.data_buffer)
            
            # Store data
            if self.current_gesture not in self.recorded_data:
                self.recorded_data[self.current_gesture] = []
            
            self.recorded_data[self.current_gesture].append(data)
            print(f"Stopped recording. Collected {len(data)} samples for {self.current_gesture}")
        
        self.is_recording = False
        self.current_gesture = None
        self.data_buffer = []
    
    def add_sample(self, sample: np.ndarray):
        """Add a sample to the current recording."""
        if self.is_recording:
            self.data_buffer.append(sample.copy())
    
    def get_recorded_data(self) -> Dict[str, List[np.ndarray]]:
        """Get all recorded data."""
        return self.recorded_data.copy()
    
    def save_data(self, filepath: str):
        """Save recorded data to file."""
        np.save(filepath, self.recorded_data)
        print(f"Data saved to {filepath}")
    
    def load_data(self, filepath: str):
        """Load recorded data from file."""
        self.recorded_data = np.load(filepath, allow_pickle=True).item()
        print(f"Data loaded from {filepath}")
        print(f"Loaded data for gestures: {list(self.recorded_data.keys())}")
