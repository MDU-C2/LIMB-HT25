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
    num_windows = window_matrix.shape[0]
    samples_per_row = window_matrix.shape[1]
    samples_per_channel_window = samples_per_row // num_channels
    
    features_dataset = []
    
    for i in range(num_windows):
        flat_window = window_matrix[i, :]
        reshaped_window = flat_window.reshape(num_channels, samples_per_channel_window)
        
        features_for_current_window = []
        for j in range(num_channels):
            current_channel = reshaped_window[j, :]
            
            mav = calculate_mav(current_channel)
            rms = calculate_rms(current_channel)
            wl = calculate_wl(current_channel)
            zc = calculate_zc(current_channel)
            ssc = calculate_ssc(current_channel)
            var = calculate_var(current_channel)
            
            features_for_current_window.extend([mav, rms, wl, zc, ssc, var])
        
        features_dataset.append(features_for_current_window)
    
    return np.array(features_dataset)

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
    sequences = []
    sequence_labels = []
    
    for i in range(len(features) - seq_length + 1):
        sequence = features[i:i + seq_length]
        sequences.append(sequence)
        label = labels[i + seq_length - 1]
        sequence_labels.append(label)
    
    if len(sequences) > 0:
        return np.array(sequences), np.array(sequence_labels)
    else:
        return np.array([]), np.array([])