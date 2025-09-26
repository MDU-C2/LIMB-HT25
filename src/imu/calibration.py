"""
IMU calibration and data preprocessing utilities.

This module provides tools for calibrating IMU sensors and preprocessing
raw sensor data for use in pose estimation.
"""

import numpy as np
from typing import List, Tuple, Optional
import logging
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .data_structures import IMUReading, IMUCalibration, ProcessedIMUReading, IMULocation


class IMUCalibrator:
    """Calibrator for IMU sensors using various calibration methods."""
    
    def __init__(self):
        """Initialize IMU calibrator."""
        self.logger = logging.getLogger(__name__)
    
    def calibrate_accelerometer(self, readings: List[IMUReading], gravity_magnitude: float = 9.81) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calibrate accelerometer using ellipsoid fitting.
        
        Args:
            readings: List of accelerometer readings from static positions
            gravity_magnitude: Expected magnitude of gravity vector
            
        Returns:
            Tuple of (bias, scale) arrays
        """
        if len(readings) < 6:
            raise ValueError("At least 6 accelerometer readings are required for calibration")
        
        # Extract accelerometer data
        accel_data = np.array([[r.accel_x, r.accel_y, r.accel_z] for r in readings])
        
        # Ellipsoid fitting for accelerometer calibration
        bias, scale = self._fit_ellipsoid(accel_data, gravity_magnitude)
        
        self.logger.info(f"Accelerometer calibration completed. Bias: {bias}, Scale: {scale}")
        return bias, scale
    
    def calibrate_gyroscope(self, readings: List[IMUReading]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calibrate gyroscope using static bias estimation.
        
        Args:
            readings: List of gyroscope readings from static positions
            
        Returns:
            Tuple of (bias, scale) arrays
        """
        if len(readings) < 10:
            raise ValueError("At least 10 gyroscope readings are required for calibration")
        
        # Extract gyroscope data
        gyro_data = np.array([[r.gyro_x, r.gyro_y, r.gyro_z] for r in readings])
        
        # Estimate bias as mean of static readings
        bias = np.mean(gyro_data, axis=0)
        
        # Estimate scale factors (assuming unit scale for now)
        scale = np.ones(3)
        
        self.logger.info(f"Gyroscope calibration completed. Bias: {bias}, Scale: {scale}")
        return bias, scale
    
    def calibrate_magnetometer(self, readings: List[IMUReading], field_magnitude: float = 50.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calibrate magnetometer using ellipsoid fitting.
        
        Args:
            readings: List of magnetometer readings from various orientations
            field_magnitude: Expected magnitude of magnetic field
            
        Returns:
            Tuple of (bias, scale) arrays
        """
        if len(readings) < 6:
            raise ValueError("At least 6 magnetometer readings are required for calibration")
        
        # Extract magnetometer data
        mag_data = np.array([[r.mag_x, r.mag_y, r.mag_z] for r in readings])
        
        # Ellipsoid fitting for magnetometer calibration
        bias, scale = self._fit_ellipsoid(mag_data, field_magnitude)
        
        self.logger.info(f"Magnetometer calibration completed. Bias: {bias}, Scale: {scale}")
        return bias, scale
    
    def _fit_ellipsoid(self, data: np.ndarray, expected_magnitude: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit ellipsoid to sensor data to estimate bias and scale factors.
        
        Args:
            data: Nx3 array of sensor readings
            expected_magnitude: Expected magnitude of the sensor readings
            
        Returns:
            Tuple of (bias, scale) arrays
        """
        def objective(params):
            bias = params[:3]
            scale = params[3:6]
            
            # Apply calibration
            calibrated = scale * (data - bias)
            
            # Calculate magnitude error
            magnitudes = np.linalg.norm(calibrated, axis=1)
            error = np.sum((magnitudes - expected_magnitude) ** 2)
            
            return error
        
        # Initial guess
        initial_bias = np.mean(data, axis=0)
        initial_scale = np.ones(3)
        initial_params = np.concatenate([initial_bias, initial_scale])
        
        # Optimize
        result = minimize(objective, initial_params, method='BFGS')
        
        if not result.success:
            self.logger.warning("Ellipsoid fitting optimization failed, using initial estimates")
            return initial_bias, initial_scale
        
        bias = result.x[:3]
        scale = result.x[3:6]
        
        return bias, scale
    
    def create_calibration(self, location: IMULocation, accel_readings: List[IMUReading], 
                          gyro_readings: List[IMUReading], mag_readings: List[IMUReading]) -> IMUCalibration:
        """
        Create complete calibration for an IMU.
        
        Args:
            location: IMU location
            accel_readings: Accelerometer calibration readings
            gyro_readings: Gyroscope calibration readings
            mag_readings: Magnetometer calibration readings
            
        Returns:
            Complete IMU calibration
        """
        # Calibrate each sensor
        accel_bias, accel_scale = self.calibrate_accelerometer(accel_readings)
        gyro_bias, gyro_scale = self.calibrate_gyroscope(gyro_readings)
        mag_bias, mag_scale = self.calibrate_magnetometer(mag_readings)
        
        # Estimate noise variances
        accel_noise_var = self._estimate_noise_variance(accel_readings, accel_bias, accel_scale)
        gyro_noise_var = self._estimate_noise_variance(gyro_readings, gyro_bias, gyro_scale)
        mag_noise_var = self._estimate_noise_variance(mag_readings, mag_bias, mag_scale)
        
        return IMUCalibration(
            accel_bias=accel_bias,
            gyro_bias=gyro_bias,
            mag_bias=mag_bias,
            accel_scale=accel_scale,
            gyro_scale=gyro_scale,
            mag_scale=mag_scale,
            accel_noise_var=accel_noise_var,
            gyro_noise_var=gyro_noise_var,
            mag_noise_var=mag_noise_var
        )
    
    def _estimate_noise_variance(self, readings: List[IMUReading], bias: np.ndarray, scale: np.ndarray) -> float:
        """Estimate noise variance from calibrated readings."""
        data = np.array([[r.accel_x, r.accel_y, r.accel_z] for r in readings])
        calibrated = scale * (data - bias)
        
        # Calculate variance of magnitude
        magnitudes = np.linalg.norm(calibrated, axis=1)
        return np.var(magnitudes)


class IMUDataProcessor:
    """Data processor for IMU readings with filtering and orientation estimation."""
    
    def __init__(self, sample_rate: float = 100.0):
        """
        Initialize IMU data processor.
        
        Args:
            sample_rate: Sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
        
        # Complementary filter parameters
        self.alpha = 0.02  # Trust factor for accelerometer/magnetometer
        
        # Low-pass filter parameters
        self.accel_cutoff = 5.0  # Hz
        self.gyro_cutoff = 10.0  # Hz
        
        # Initialize filters
        self._init_filters()
    
    def _init_filters(self):
        """Initialize low-pass filters for sensor data."""
        # Simple first-order low-pass filter coefficients
        self.accel_alpha = 2 * np.pi * self.accel_cutoff * self.dt / (1 + 2 * np.pi * self.accel_cutoff * self.dt)
        self.gyro_alpha = 2 * np.pi * self.gyro_cutoff * self.dt / (1 + 2 * np.pi * self.gyro_cutoff * self.dt)
        
        # Filter state
        self.accel_filtered = np.zeros(3)
        self.gyro_filtered = np.zeros(3)
        self.mag_filtered = np.zeros(3)
    
    def process_reading(self, reading: IMUReading, calibration: IMUCalibration) -> ProcessedIMUReading:
        """
        Process raw IMU reading with calibration and filtering.
        
        Args:
            reading: Raw IMU reading
            calibration: IMU calibration parameters
            
        Returns:
            Processed IMU reading
        """
        # Apply calibration
        accel_raw = np.array([reading.accel_x, reading.accel_y, reading.accel_z])
        gyro_raw = np.array([reading.gyro_x, reading.gyro_y, reading.gyro_z])
        mag_raw = np.array([reading.mag_x, reading.mag_y, reading.mag_z])
        
        accel_calibrated = calibration.accel_scale * (accel_raw - calibration.accel_bias)
        gyro_calibrated = calibration.gyro_scale * (gyro_raw - calibration.gyro_bias)
        mag_calibrated = calibration.mag_scale * (mag_raw - calibration.mag_bias)
        
        # Apply low-pass filtering
        self.accel_filtered = self._low_pass_filter(accel_calibrated, self.accel_filtered, self.accel_alpha)
        self.gyro_filtered = self._low_pass_filter(gyro_calibrated, self.gyro_filtered, self.gyro_alpha)
        self.mag_filtered = self._low_pass_filter(mag_calibrated, self.mag_filtered, self.accel_alpha)
        
        # Estimate orientation using complementary filter
        orientation = self._estimate_orientation(self.accel_filtered, self.gyro_filtered, self.mag_filtered)
        
        # Calculate derived quantities
        angular_velocity_magnitude = np.linalg.norm(self.gyro_filtered)
        linear_acceleration_magnitude = np.linalg.norm(self.accel_filtered)
        
        return ProcessedIMUReading(
            accel=self.accel_filtered,
            gyro=self.gyro_filtered,
            mag=self.mag_filtered,
            timestamp=reading.timestamp,
            location=reading.location,
            temperature=reading.temperature,
            orientation=orientation,
            angular_velocity_magnitude=angular_velocity_magnitude,
            linear_acceleration_magnitude=linear_acceleration_magnitude
        )
    
    def _low_pass_filter(self, new_value: np.ndarray, filtered_value: np.ndarray, alpha: float) -> np.ndarray:
        """Apply first-order low-pass filter."""
        return alpha * new_value + (1 - alpha) * filtered_value
    
    def _estimate_orientation(self, accel: np.ndarray, gyro: np.ndarray, mag: np.ndarray) -> np.ndarray:
        """
        Estimate orientation using complementary filter.
        
        Args:
            accel: Filtered accelerometer data
            gyro: Filtered gyroscope data
            mag: Filtered magnetometer data
            
        Returns:
            Orientation quaternion [w, x, y, z]
        """
        # Normalize accelerometer and magnetometer
        accel_norm = accel / np.linalg.norm(accel)
        mag_norm = mag / np.linalg.norm(mag)
        
        # Calculate roll and pitch from accelerometer
        roll = np.arctan2(accel_norm[1], accel_norm[2])
        pitch = np.arcsin(-accel_norm[0])
        
        # Calculate yaw from magnetometer
        mag_x = mag_norm[0] * np.cos(pitch) + mag_norm[1] * np.sin(roll) * np.sin(pitch) + mag_norm[2] * np.cos(roll) * np.sin(pitch)
        mag_y = mag_norm[1] * np.cos(roll) - mag_norm[2] * np.sin(roll)
        yaw = np.arctan2(-mag_y, mag_x)
        
        # Convert to quaternion
        orientation = Rotation.from_euler('xyz', [roll, pitch, yaw]).as_quat()
        
        return orientation  # [x, y, z, w] format from scipy


class IMUCalibrationCollector:
    """Collector for gathering calibration data from IMUs."""
    
    def __init__(self):
        """Initialize calibration collector."""
        self.logger = logging.getLogger(__name__)
        self.accel_readings: List[IMUReading] = []
        self.gyro_readings: List[IMUReading] = []
        self.mag_readings: List[IMUReading] = []
    
    def add_reading(self, reading: IMUReading, reading_type: str) -> None:
        """
        Add a reading for calibration.
        
        Args:
            reading: IMU reading
            reading_type: Type of reading ('accel', 'gyro', 'mag')
        """
        if reading_type == 'accel':
            self.accel_readings.append(reading)
        elif reading_type == 'gyro':
            self.gyro_readings.append(reading)
        elif reading_type == 'mag':
            self.mag_readings.append(reading)
        else:
            raise ValueError(f"Invalid reading type: {reading_type}")
    
    def clear_readings(self) -> None:
        """Clear all collected readings."""
        self.accel_readings.clear()
        self.gyro_readings.clear()
        self.mag_readings.clear()
    
    def get_calibration(self, location: IMULocation) -> IMUCalibration:
        """
        Generate calibration from collected readings.
        
        Args:
            location: IMU location
            
        Returns:
            IMU calibration
        """
        calibrator = IMUCalibrator()
        return calibrator.create_calibration(location, self.accel_readings, self.gyro_readings, self.mag_readings)
    
    def has_sufficient_data(self) -> bool:
        """Check if sufficient data has been collected for calibration."""
        return (len(self.accel_readings) >= 6 and 
                len(self.gyro_readings) >= 10 and 
                len(self.mag_readings) >= 6)
