"""
Shared EMG processing utilities for CSV data processing.
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

# Feature calculation functions
def calculate_mav(window):
    """Mean Absolut Value"""
    return np.mean(np.abs(window))

def calculate_rms(window):
    """Root Mean Square"""
    return np.sqrt(np.mean(window**2))

def calculate_wl(window):
    """Waveform Length"""
    return np.sum(np.abs(np.diff(window)))

def calculate_zc(window, threshold=1e-5):
    """Zero Crossings."""
    return np.sum(np.diff(np.sign(window - threshold)) != 0)

def calculate_ssc(window):
    """Slope Sign Change"""
    return np.sum(np.diff(np.sign(np.diff(window))) != 0)

def calculate_var(window):
    """Variance"""
    return np.var(window)

def extract_time_domain_features(window_matrix, num_channels):
    """
    Extract 6 time-domain features for each window per channel.

    This function iterates through each flattened window, reshapes it back to its
    multi-channel form, and calculates a feature vector for each channel.

    Args:
        window_matrix (np.ndarray): The matrix from create_windows (windows, flattened_samples).
        num_channels (int): The number of original EMG channels 

    Returns:
        np.ndarray: The final feature matrix with shape (num_windows, num_channels * 6).
    """
    # TODO: Implement this function, that from generate_ch_dataset.py
    pass

def create_sequences(features, labels, seq_length=10):
    """
    Convert feature matrix into overlapping sequences for LSTM.

    Args:
        features (np.ndarray): The feature matrix from extract_time_domain_features.
        labels (np.ndarray): The corresponding labels for each window.
        seq_length (int): The length of the sequences to create.

    Returns:
        np.ndarray: The sequences matrix with shape (num_sequences, seq_length, num_features).
        np.ndarray: The sequence labels with shape (num_sequences,).
    """
    # TODO: Implement this function, that from generate_ch_dataset.py
    pass