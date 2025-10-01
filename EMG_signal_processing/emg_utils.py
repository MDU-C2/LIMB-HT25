import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, welch
import matplotlib.pyplot as plt

# preprocessing

def preprocess_emg_signal(signal, fs, lowcut, highcut, notch_freq, order=4):
    """
    Preprocesses a raw, multi-channel EMG signal.
    Order: 1. Remove DC offset -> 2. Filter (Band-pass + Notch).
    """
    # --- 1. Remove DC offset from each channel ---
    # Centers the signal around zero for accurate filtering.
    signal_mean_removed = signal - np.mean(signal, axis=1, keepdims=True)

    # --- 2. Define filter parameters ---
    # lowcut = 20.0      # Lower cutoff frequency (Hz) for band-pass
    # highcut = 450.0    # Upper cutoff frequency (Hz) for band-pass
    # notch_freq = 50.0  # Powerline frequency to eliminate (Hz)
    Q = 30             # Quality factor for the Notch filter
    # order = 4          # Order for the Butterworth band-pass filter

    # --- 3. Design filters ---
    # Design Butterworth band-pass filter
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b_band, a_band = butter(order, [low, high], btype='band')

    # Design Notch filter
    b_notch, a_notch = iirnotch(notch_freq, Q, fs)

    # --- 4. Apply filters to each channel ---
    # Use filtfilt for zero-phase filtering (no time delay).
    signal_filtered = np.zeros_like(signal_mean_removed)
    for i in range(signal_mean_removed.shape[0]):
        # Apply band-pass filter first
        ch_band_filtered = filtfilt(b_band, a_band, signal_mean_removed[i, :])
        # Then apply notch filter to the result
        ch_notch_filtered = filtfilt(b_notch, a_notch, ch_band_filtered)
        signal_filtered[i, :] = ch_notch_filtered

    return signal_filtered

# visualize

def plot_emg_channels(signal, fs, channels_to_plot=3, title="Señal EMG"):
    """
    Plots a specified number of channels from an EMG signal over time.

    Args:
        signal (np.ndarray): The multi-channel EMG signal (channels, samples).
        fs (int): The sampling frequency in Hz.
        channels_to_plot (int): The number of channels to display from the top.
        title (str): The main title for the plot.
    """
    
    num_channels = signal.shape[0]
    if channels_to_plot > num_channels:
        channels_to_plot = num_channels

    # Create the time vector in seconds
    num_samples = signal.shape[1]
    t = np.arange(num_samples) / fs

    # Create subplots
    fig, axes = plt.subplots(channels_to_plot, 1, figsize=(14, 2 * channels_to_plot), sharex=True)
    
    # Handle the case of a single channel to plot
    if channels_to_plot == 1:
        axes = [axes]

    # Plot each channel
    for i in range(channels_to_plot):
        axes[i].plot(t, signal[i, :])
        axes[i].set_ylabel(f"Ch {i+1}")
        axes[i].grid(True)

    # Set labels and title
    axes[-1].set_xlabel("Tiempo (s)")
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_psd_comparison(raw_signal, preprocessed_signal, fs, channel_idx=0):
    """
    Plots the Power Spectral Density (PSD) of a signal before and after preprocessing
    to visually verify the effect of the filters.

    Args:
        raw_signal (np.ndarray): The original, unfiltered signal.
        preprocessed_signal (np.ndarray): The signal after filtering.
        fs (int): The sampling frequency in Hz.
        channel_idx (int): The index of the channel to visualize.
    """
    # --- 1. Calculate PSD for the raw signal ---
    raw_freqs, raw_psd = welch(raw_signal[channel_idx, :], fs=fs, nperseg=1024)

    # --- 2. Calculate PSD for the preprocessed signal ---
    pre_freqs, pre_psd = welch(preprocessed_signal[channel_idx, :], fs=fs, nperseg=1024)

    # --- 3. Plot both spectra for comparison ---
    plt.figure(figsize=(14, 7))
    # Use a logarithmic scale 
    plt.semilogy(raw_freqs, raw_psd, label='Raw Signal', color='blue', alpha=0.7)
    plt.semilogy(pre_freqs, pre_psd, label='Preprocessed Signal', color='red', alpha=0.9)

    # --- 4. Add visual guides for filter frequencies ---
    plt.axvline(x=20, color='gray', linestyle='--', label='Band-pass Cutoff (20 Hz)')
    plt.axvline(x=450, color='gray', linestyle='--', label='Band-pass Cutoff (450 Hz)')
    plt.axvline(x=50, color='green', linestyle=':', label='Notch Filter (50 Hz)')

    # --- 5. Set labels and finalize plot ---
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (V^2/Hz)')
    plt.title(f'Preprocessing Effect on Frequency Spectrum (Channel {channel_idx + 1})')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.xlim(0, fs / 2) # Show up to the Nyquist frequency
    plt.show()

def plot_emg_windows(window_matrix, window_indices, title="Sample EMG Windows"):
    """
    Plots specific windows from the window matrix by their indices.

    Args:
        window_matrix (np.ndarray): The matrix from create_windows (windows, flattened_samples).
        window_indices (list): A list of integer indices for the windows to plot.
        title (str): The main title for the plot.
    """
    num_to_plot = len(window_indices)

    # Create subplots 
    fig, axes = plt.subplots(num_to_plot, 1, figsize=(14, 2 * num_to_plot), sharex=True)

    # Handle the case of a single window index
    if num_to_plot == 1:
        axes = [axes]

    for i, window_idx in enumerate(window_indices):

        if window_idx >= window_matrix.shape[0]:
            continue
        
        axes[i].plot(window_matrix[window_idx, :])
        axes[i].set_ylabel(f"Window #{window_idx}")
        axes[i].grid(True)

    axes[-1].set_xlabel("Samples within the window (all channels concatenated)")
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_feature_snapshot(feature_matrix, num_channels, window_idx, feature_idx, feature_name):
    """
    Plots a bar chart of a single feature across all channels for one specific time window.

    This provides a "spatial snapshot" of muscle activation at a single moment.

    Args:
        feature_matrix (np.ndarray): The final feature matrix (windows, total_features).
        num_channels (int): The number of EMG channels.
        window_idx (int): The index of the window to visualize.
        feature_idx (int): The index of the feature to visualize (e.g., 1 for RMS).
        feature_name (str): The name of the feature for the plot title.
    """
    # --- 1. Calculate how many features exist per channel ---
    num_features_per_channel = feature_matrix.shape[1] // num_channels
    
    # --- 2. Extract the feature value for each channel in the specified window ---
    values = []
    for i in range(num_channels):
        # Calculate the correct column index for the desired feature and channel
        col_idx = i * num_features_per_channel + feature_idx
        values.append(feature_matrix[window_idx, col_idx])
        
    channel_labels = [f'Channel {i+1}' for i in range(num_channels)]
    
    # --- 3. Create the bar plot ---
    plt.figure(figsize=(12, 6))
    plt.bar(channel_labels, values, color='skyblue')
    plt.title(f'"{feature_name}" Snapshot at Window #{window_idx}')
    plt.xlabel('Sensor Channel')
    plt.ylabel('Feature Value')
    plt.grid(axis='y', linestyle='--')
    plt.show()

def plot_feature_evolution(feature_matrix, num_channels, feature_idx, feature_name):
    """
    Plots the evolution of a single feature over all time windows for all channels.

    This provides the "temporal pattern" of a feature for a full gesture.

    Args:
        feature_matrix (np.ndarray): The final feature matrix (windows, total_features).
        num_channels (int): The number of EMG channels.
        feature_idx (int): The index of the feature to visualize (e.g., 1 for RMS).
        feature_name (str): The name of the feature for the plot title.
    """
    # --- 1. Calculate how many features exist per channel ---
    num_features_per_channel = feature_matrix.shape[1] // num_channels

    # --- 2. Create the plot ---
    plt.figure(figsize=(14, 7))
    
    # --- 3. Iterate through each channel and plot its feature evolution ---
    for i in range(num_channels):
        # Calculate the correct column index for the desired feature and channel
        col_idx = i * num_features_per_channel + feature_idx
        # Plot the entire column, which represents the feature's value over time
        plt.plot(feature_matrix[:, col_idx], label=f'Channel {i + 1}')

    # --- 4. Set labels and finalize plot ---
    plt.title(f'Evolution of "{feature_name}" Over Time ({num_channels} Channels)')
    plt.xlabel('Window Number (Time)')
    plt.ylabel('Feature Value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # Move legend outside plot
    plt.grid(True)
    plt.tight_layout() # Adjust layout to prevent legend cutoff
    plt.show()

# Windowing

def create_windows(signal, label, fs, window_size_ms, overlap_ms):
    """
    Creates flattened, overlapping windows from a multi-channel signal.

    This function segments a signal, applies a windowing function, and reshapes
    the output so that each row contains all channel data for a single time window.

    Args:
        signal (np.ndarray): Input signal with shape (channels, samples).
        label (int): Integer label for the gesture.
        fs (int): Sampling frequency in Hz.
        window_size_ms (float): Window duration in milliseconds.
        overlap_ms (float): Overlap duration in milliseconds.
        window_type (int): Window function to apply: 1=Rectangular, 2=Hamming, etc.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the flattened data matrix
                                       and the corresponding labels vector.
    """
    # --- 1. Convert parameters from ms to samples ---
    window_size_samples = int(fs * window_size_ms / 1000)
    overlap_samples = int(fs * overlap_ms / 1000)
    step_size = window_size_samples - overlap_samples
    
    num_channels, total_samples = signal.shape
    
    window_func = np.ones(window_size_samples)

    # --- 3. Segment each channel ---
    all_channels_segmented = []
    for i in range(num_channels):
        channel_data = signal[i, :]
        
        # Calculate the number of windows and required padding
        num_windows = int(np.ceil((total_samples - window_size_samples) / step_size)) + 1
        required_length = (num_windows - 1) * step_size + window_size_samples
        padding_needed = max(0, required_length - total_samples)
        
        # Pad the signal with zeros at the end
        padded_channel = np.pad(channel_data, (0, padding_needed), mode='constant')
        
        channel_segments = np.zeros((num_windows, window_size_samples))
        
        # Extract and apply window function to each segment
        for j in range(num_windows):
            start_idx = j * step_size
            end_idx = start_idx + window_size_samples
            segment = padded_channel[start_idx:end_idx] * window_func
            channel_segments[j, :] = segment
            
        all_channels_segmented.append(channel_segments)

    # --- 4. Combine and flatten the windows ---
    # Stack channels into a 3D array (channels, windows, samples)
    stacked_array = np.array(all_channels_segmented)
    
    # Transpose to (windows, channels, samples) for easier reshaping
    transposed_array = stacked_array.transpose(1, 0, 2)
    
    # Reshape into the final 2D matrix (windows, channels * samples)
    num_windows = transposed_array.shape[0]
    data_matrix = transposed_array.reshape(num_windows, -1)

    # --- 5. Create the corresponding labels vector ---
    labels_vector = np.full(num_windows, label, dtype=int)
    
    return data_matrix, labels_vector

# Feature Extraction

# [mav, rms, wl, zc, ssc, var]

def calculate_mav(window):
    """1. Mean Absolute Value: Measures the average intensity of the contraction."""
    return np.mean(np.abs(window))

def calculate_rms(window):
    """2. Root Mean Square: Measures signal power, related to muscle force."""
    return np.sqrt(np.mean(window**2))

def calculate_wl(window):
    """3. Waveform Length: Measures signal complexity, related to total activity."""
    return np.sum(np.abs(np.diff(window)))

def calculate_zc(window, threshold=1e-5):
    """4. Zero Crossings: Measures signal frequency by counting zero crosses."""
    return np.sum(np.diff(np.sign(window - threshold)) != 0)

def calculate_ssc(window, threshold=1e-5):
    """5. Slope Sign Changes: Measures signal frequency by counting peaks and valleys."""
    diff_signal = np.diff(window)
    return np.sum(np.diff(np.sign(diff_signal)) != 0)

def calculate_var(window):
    """6. Variance: Measures signal power, similar to RMS."""
    # For a zero-mean signal, VAR is simply the mean of the squares.
    return np.var(window)

def extract_time_domain_features(window_matrix, num_channels):
    """
    Extracts a set of 6 time-domain features for each window.

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

    # --- 1. Iterate through each window (each row) ---
    for i in range(num_windows):
        flat_window = window_matrix[i, :]
        # Reshape the flat row back to its multi-channel form 
        reshaped_window = flat_window.reshape(num_channels, samples_per_channel_window)

        features_for_current_window = []
        # --- 2. Iterate through each channel within the window ---
        for j in range(num_channels):
            current_channel = reshaped_window[j, :]

            # --- 3. Calculate all selected features for the current channel ---
            mav = calculate_mav(current_channel)
            rms = calculate_rms(current_channel)
            wl = calculate_wl(current_channel)
            zc = calculate_zc(current_channel)
            ssc = calculate_ssc(current_channel)
            var = calculate_var(current_channel)

            # --- 4. Append features in a consistent order ---
            features_for_current_window.extend([mav, rms, wl, zc, ssc, var])

        # Append the full feature vector for the window to the main list
        features_dataset.append(features_for_current_window)

    # Convert the list of lists to a 2D NumPy array
    return np.array(features_dataset)






































