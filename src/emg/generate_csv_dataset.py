"""
Generate EMG sequence dataset from CSV files.

Process CSV files where each row is pre-windowed a EMG sample.
Each file contains 80 windows: 20 rest + 40 gesture + 20 rest.
"""

import argparse
import pandas as pd
import numpy as np
import os
import glob
from .emg_processing_utils import extract_time_domain_features, create_sequences

# CONSTANTS from data collection code
WINDOWS_PER_CAPTURE = 80
REST_START_WINDOWS = 20
GESTURE_END_WINDOWS = 60
EMG_SAMPLES_PER_WINDOW = 400

def load_csv_file(file_path):
    """
    Load a CSV file and return a pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The DataFrame containing the CSV data.
    """
    df = pd.read_csv(file_path)
    emg_cols = [f"v{i}" for i in range(EMG_SAMPLES_PER_WINDOW)]
    windows = df[emg_cols].values # Shape (80, 400)
    return windows

def segment_windows(windows):
    """
    Segment the windows into overlapping sequences.

    Args:
        windows (np.ndarray): The windows to segment.

    Returns:
        np.ndarray: The segmented windows.
    """
    segmented = []

    # Rest windows (labeled as 0)
    for i in range(REST_START_WINDOWS):
        segmented.append((windows[i], 0))
    
    # Gesture windows (labeled as 1)
    for i in range(REST_START_WINDOWS, GESTURE_END_WINDOWS):
        segmented.append((windows[i], 1))
    
    # Rest windows (labeled as 0)
    for i in range(GESTURE_END_WINDOWS, WINDOWS_PER_CAPTURE):
        segmented.append((windows[i], 0))
    
    return segmented

def process_csv_file(csv_path, num_channels, seq_len):
    """
    Process a CSV file and extract sequences.

    Args:
        csv_path (str): The path to the CSV file.
        num_channels (int): The number of channels in the EMG signal.
        seq_len (int): The length of the sequences to extract.

    Returns:
        np.ndarray: The sequences matrix with shape (num_sequences, seq_length, num_features).
    """
    # Load windows
    windows = load_csv_file(csv_path)
    
    # Segment windows
    segmented = segment_windows(windows)
    window_data = np.array([w[0] for w in segmented]) # Shape (80, 400) = (num_windows, num_samples_per_window)
    labels = np.array([w[1] for w in segmented]) # Shape (80,) = (num_windows,)

    # Extract time domain features for each window
    features = extract_time_domain_features(window_data, num_channels)

    # Create sequences
    if features.shape[0] >= seq_len:
        sequence_X, sequence_y = create_sequences(features, labels, seq_len)
        return sequence_X, sequence_y
    else:
        print(f"Skipping {csv_path} (too short to create sequences of length {seq_len})")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Generate EMG sequence dataset from CSV files.")
    # Default path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_path = os.path.join(script_dir, "data", "raw_data")
    parser.add_argument(
        "--data-path",
        type=str,
        default=default_data_path,
        help="Path to raw_data folder containing subject folders (default: relative to script location)"
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=None,
        help="Channel number to extract (1-based). If None, uses all channels."
    )
    parser.add_argument(
        "--num-channels",
        type=int,
        default=1,
        help="Number of EMG channels (for reshaping 400 samples)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output NPZ filename"
    )
    parser.add_argument(
        "--subjects",
        type=str,
        nargs="+",
        default=["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"],
        help="List of subjects to process (default: all S0-S7)"
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=10,
        help="Sequence length for LSTM (default: 10)"
    )
    
    args = parser.parse_args()

    subjects = args.subjects
    
    # Resolve data path to absolute path for clarity
    data_path = os.path.abspath(args.data_path)
    print(f"Looking for data in: {data_path}")
    
    if args.output is None:
        if args.channel is None:
            output_filename = f"emg_sequences_ch{args.channel}.npz" if args.channel is not None else "emg_sequences_all.npz"
    else:
        output_filename = args.output

    # Process all CSV files
    all_sequences_X = []
    all_sequences_y = []

    for subject_id in subjects:
        subject_path = os.path.join(data_path, subject_id)
        csv_files = glob.glob(os.path.join(subject_path, "*.csv"))

        print(f"\nProcessing {subject_id}: {len(csv_files)} CSV files found in {subject_path}")

        for csv_file in csv_files:
            X, y = process_csv_file(csv_file, args.num_channels, args.seq_len)
            if X is not None:
                all_sequences_X.append(X)
                all_sequences_y.append(y)

    # Combine and save
    if all_sequences_X:
        X_final = np.vstack(all_sequences_X)
        y_final = np.concatenate(all_sequences_y)

        # Save to data/processed_data directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        processed_data_dir = os.path.join(script_dir, "data", "processed_data")
        os.makedirs(processed_data_dir, exist_ok=True)
        output_path = os.path.join(processed_data_dir, output_filename)
        np.savez(output_path, X=X_final, y=y_final)

        print(f"\n{'='*70}")
        print("Dataset Creation Complete!")
        print(f"{'='*70}")
        print(f"Output: {output_path}")
        print(f"Sequences shape (X): {X_final.shape}")
        print(f"Labels shape (y): {y_final.shape}")
        print(f"Label distribution:")
        unique, counts = np.unique(y_final, return_counts=True)
        for label, count in zip(unique, counts):
            label_name = "Hand_Open" if label == 0 else "Hand_Close"
            print(f"  {label_name} (label {label}): {count} sequences")
    else:
        print("\n[ERROR] No data was processed.")

if __name__ == "__main__":
    main()
