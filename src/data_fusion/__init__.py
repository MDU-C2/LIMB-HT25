"""
Data Fusion System

This module provides data fusion capabilities for combining sensor data from multiple sources.
It includes IMU smoothing, vision fusion, and sensor validation components.
"""

from .fusion_system import FiducialDepthSystem
from .smoothing import IMUData, IMUSmoother, IMUValidator

__all__ = [
    'FiducialDepthSystem',
    'IMUData',
    'IMUSmoother', 
    'IMUValidator'
]