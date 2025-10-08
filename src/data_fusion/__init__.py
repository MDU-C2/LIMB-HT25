"""
Data Fusion System

This module provides data fusion capabilities for combining sensor data from multiple sources.
It includes IMU smoothing, vision fusion, sensor validation, and LSTM-ready data formatting.
"""

from .fusion_system import FiducialDepthSystem
from .smoothing import IMUData, IMUSmoother, IMUValidator
from .sensor_data_format import (
    SensorFusionData,
    EMGData,
    SensorSequenceBuilder,
    create_batch_for_lstm
)

__all__ = [
    'FiducialDepthSystem',
    'IMUData',
    'IMUSmoother', 
    'IMUValidator',
    'SensorFusionData',
    'EMGData',
    'SensorSequenceBuilder',
    'create_batch_for_lstm'
]