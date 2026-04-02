"""
Generate EMG sequence dataset for a single channel.

This script processes EMG data from the data_rest folder and creates a dataset
containing sequences from only one specified channel.
"""

import argparse
import os
import glob
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch
import sys

# Add path to import emg_utils functions if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../EMG_signal_processing'))


def preprocess_emg_signal(signal, fs, lowcut, highcut, notch_freq, order=4):
    """Preprocesses a raw, multi-channel EMG signal."""
    # Remove DC offset
    signal_mean_removed = signal - np.mean(signal, axis=1, keepdims=True)
    
    # Design filters
    Q = 30
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b_band, a_band = butter(order, [low, high], btype='band')
    b_notch, a_notch = iirnotch(notch_freq, Q, fs)
    
    # Apply filters
    signal_filtered = np.zeros_like(signal_mean_removed)
    for i in range(signal_mean_removed.shape[0]):
        ch_band_filtered = filtfilt(b_band, a_band, signal_mean_removed[i, :])
        ch_notch_filtered = filtfilt(b_notch, a_notch, ch_band_filtered)
        signal_filtered[i, :] = ch_notch_filtered
    
    return signal_filtered


def create_windows(signal, label, fs, window_size_ms, overlap_ms):
    """Creates flattened, overlapping windows from a signal."""
    window_size_samples = int(fs * window_size_ms / 1000)
    overlap_samples = int(fs * overlap_ms / 1000)
    step_size = window_size_samples - overlap_samples
    
    num_channels, total_samples = signal.shape
    window_func = np.ones(window_size_samples)
    
    all_channels_segmented = []
    for i in range(num_channels):
        channel_data = signal[i, :]
        
        num_windows = int(np.ceil((total_samples - window_size_samples) / step_size)) + 1
        required_length = (num_windows - 1) * step_size + window_size_samples
        padding_needed = max(0, required_length - total_samples)
        
        padded_channel = np.pad(channel_data, (0, padding_needed), mode='constant')
        channel_segments = np.zeros((num_windows, window_size_samples))
        
        for j in range(num_windows):
            start_idx = j * step_size
            end_idx = start_idx + window_size_samples
            segment = padded_channel[start_idx:end_idx] * window_func
            channel_segments[j, :] = segment
            
        all_channels_segmented.append(channel_segments)
    
    stacked_array = np.array(all_channels_segmented)
    transposed_array = stacked_array.transpose(1, 0, 2)
    num_windows = transposed_array.shape[0]
    data_matrix = transposed_array.reshape(num_windows, -1)
    
    labels_vector = np.full(num_windows, label, dtype=int)
    return data_matrix, labels_vector


def calculate_mav(window):
    """Mean Absolute Value."""
    return np.mean(np.abs(window))


def calculate_rms(window):
    """Root Mean Square."""
    return np.sqrt(np.mean(window**2))


def calculate_wl(window):
    """Waveform Length."""
    return np.sum(np.abs(np.diff(window)))


def calculate_zc(window, threshold=1e-5):
    """Zero Crossings."""
    return np.sum(np.diff(np.sign(window - threshold)) != 0)


def calculate_ssc(window, threshold=1e-5):
    """Slope Sign Changes."""
    diff_signal = np.diff(window)
    return np.sum(np.diff(np.sign(diff_signal)) != 0)


def calculate_var(window):
    """Variance."""
    return np.var(window)


def extract_time_domain_features(window_matrix, num_channels):
    """Extracts 6 time-domain features for each window per channel."""
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
    """Converts feature matrix into overlapping sequences for LSTM."""
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


def main():
    parser = argparse.ArgumentParser(
        description="Generate EMG sequence dataset for a single channel"
    )
    parser.add_argument(
        "--channel",
        type=int,
        required=True,
        help="Channel number to extract (1-based, e.g., 1 for first channel)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="../../EMG_signal_processing/data_rest",
        help="Path to data_rest folder (default: ../../EMG_signal_processing/data_rest)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename (default: emg_sequences_ch{channel}.npz)"
    )
    parser.add_argument(
        "--subjects",
        type=str,
        nargs="+",
        default=None,
        help="List of subjects to process (default: all S1-S19)"
    )
    parser.add_argument(
        "--gestures",
        type=str,
        nargs="+",
        default=["Hand_Close", "Hand_Open"],
        help="Gestures to process (default: Hand_Close Hand_Open)"
    )
    parser.add_argument(
        "--fs",
        type=int,
        default=985,
        help="Sampling frequency in Hz (default: 985)"
    )
    parser.add_argument(
        "--lowcut",
        type=float,
        default=20.0,
        help="Low cutoff frequency for band-pass filter (default: 20.0)"
    )
    parser.add_argument(
        "--highcut",
        type=float,
        default=450.0,
        help="High cutoff frequency for band-pass filter (default: 450.0)"
    )
    parser.add_argument(
        "--notch-freq",
        type=float,
        default=50.0,
        help="Notch filter frequency (default: 50.0)"
    )
    parser.add_argument(
        "--window-size-ms",
        type=float,
        default=200.0,
        help="Window size in milliseconds (default: 200.0)"
    )
    parser.add_argument(
        "--overlap-ms",
        type=float,
        default=50.0,
        help="Overlap size in milliseconds (default: 50.0)"
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=10,
        help="Sequence length for LSTM (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Validate channel index (1-based)
    if args.channel < 1:
        print(f"Error: Channel number must be >= 1, got {args.channel}")
        return
    
    # Convert to 0-based for array indexing
    channel_idx = args.channel - 1
    
    # Setup paths and parameters
    base_path = os.path.abspath(args.data_path)
    
    if args.subjects is None:
        subjects = [f"S{i}" for i in range(1, 20)]
    else:
        subjects = args.subjects
    
    labels_map = {"Hand_Close": 1, "Hand_Open": 0}
    
    if args.output is None:
        output_filename = f"emg_sequences_ch{args.channel}.npz"  # Uses 1-based channel number
    else:
        output_filename = args.output
    
    # Initialize storage
    all_sequences_X = []
    all_sequences_y = []
    
    print("=" * 70)
    print(f"Generating dataset for Channel {args.channel} (1-based indexing)")
    print(f"Output file: {output_filename}")
    print("=" * 70)
    print(f"Processing {len(subjects)} subjects")
    print(f"Gestures: {args.gestures}")
    print(f"Parameters: fs={args.fs}, window={args.window_size_ms}ms, "
          f"overlap={args.overlap_ms}ms, seq_len={args.seq_len}")
    print("=" * 70)
    
    # Process each subject and gesture
    for subject_id in subjects:
        subject_path = os.path.join(base_path, subject_id)
        if not os.path.exists(subject_path):
            print(f"[WARNING] Subject path not found: {subject_path}")
            continue
        
        print(f"\n[INFO] Processing Subject: {subject_id}")
        
        for gesture_name in args.gestures:
            search_pattern = os.path.join(subject_path, f"{gesture_name}-*.mat")
            files_found = sorted(glob.glob(search_pattern))
            print(f"  [INFO] Found {len(files_found)} files for gesture: {gesture_name}")
            
            for file_path in files_found:
                try:
                    # Load signal
                    raw_signal = loadmat(file_path)["value"]
                    
                    # Validate channel index (1-based input, 0-based array)
                    if channel_idx >= raw_signal.shape[0]:
                        print(f"    [WARNING] Channel {args.channel} not available in {os.path.basename(file_path)} "
                              f"(only {raw_signal.shape[0]} channels available). Skipping.")
                        continue
                    
                    # Extract single channel (keep 2D shape: (1, samples))
                    single_channel_signal = raw_signal[channel_idx:channel_idx+1, :]
                    
                    # Preprocess
                    preprocessed_signal = preprocess_emg_signal(
                        single_channel_signal, args.fs, args.lowcut, args.highcut, args.notch_freq
                    )
                    
                    # Create windows
                    label = labels_map[gesture_name]
                    windows, window_labels = create_windows(
                        preprocessed_signal, label, args.fs, args.window_size_ms, args.overlap_ms
                    )
                    
                    # Extract features (now only 1 channel, so 6 features total)
                    features = extract_time_domain_features(windows, num_channels=1)
                    
                    # Create sequences
                    if features.shape[0] >= args.seq_len:
                        X_seq_file, y_seq_file = create_sequences(features, window_labels, args.seq_len)
                        all_sequences_X.append(X_seq_file)
                        all_sequences_y.append(y_seq_file)
                    else:
                        print(f"    [WARNING] File {os.path.basename(file_path)} skipped "
                              f"(too short: {features.shape[0]} windows < {args.seq_len} required)")
                
                except Exception as e:
                    print(f"    [ERROR] Failed to process {os.path.basename(file_path)}: {e}")
                    continue
    
    # Save dataset
    if all_sequences_X:
        X_final = np.vstack(all_sequences_X)
        y_final = np.concatenate(all_sequences_y)
        
        print("\n" + "=" * 70)
        print("Dataset Creation Complete!")
        print("=" * 70)
        print(f"Final sequences matrix shape (X): {X_final.shape}")
        print("      -> (Total Sequences, Windows per Sequence, Features per Window)")
        print(f"Final labels vector shape (y): {y_final.shape}")
        print(f"Features per window: {X_final.shape[2]} (6 features for 1 channel)")
        print(f"Label distribution:")
        unique, counts = np.unique(y_final, return_counts=True)
        for label, count in zip(unique, counts):
            gesture_name = "Hand_Open" if label == 0 else "Hand_Close"
            print(f"  {gesture_name} (label {label}): {count} sequences")
        
        output_path = os.path.join(os.path.dirname(__file__), output_filename)
        np.savez(output_path, X=X_final, y=y_final)
        print(f"\nDataset saved to: {output_path}")
        print("=" * 70)
    else:
        print("\n[ERROR] No data was processed. Please check your parameters and data path.")


if __name__ == "__main__":
    main()

