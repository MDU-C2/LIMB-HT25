"""EMG signal processing and intent classification package.

This package provides:
- EMG data preprocessing and feature extraction
- LSTM-based temporal models for intent classification
- Real-time EMG processing capabilities
- Gesture recognition and control interfaces
"""

from .lstm_model import EMGLSTMModel, EMGClassifier
from .feature_extractor import EMGFeatureExtractor
from .data_processor import EMGDataProcessor
from .real_time_processor import RealTimeEMGProcessor

__all__ = [
    "EMGLSTMModel",
    "EMGClassifier", 
    "EMGFeatureExtractor",
    "EMGDataProcessor",
    "RealTimeEMGProcessor"
]
