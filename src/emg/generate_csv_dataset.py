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
from emg_processing_utils import extract_time_domain_features, create_sequences

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

    # TODO: Reshape based on actual channel configuration
    # Option 1, if 8 channels: reshape to (80, 8, 50)
    # Option 2, if 2 channels: reshape to (80, 2, 200)
    # Option 3: if 1 channel: reshape to (80, 1, 400)

    # Segment windows
    segmented = segment_windows(windows)
    window_data = np.array([w[0] for w in segmented]) # Shape (80, 400)
    labels = np.array([w[1] for w in segmented]) # Shape (80,)

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
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw_data",
        help="Path to raw_data folder containing subject folders"
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
        default=8,
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
        default=None,
        help="List of subjects to process (default: all S0-S7)"
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=10,
        help="Sequence length for LSTM (default: 10)"
    )
    
    args = parser.parse_args()

    # Determine subjects to process
    if args.subjects is None:
        subjects = [f"S{i}" for i in range(0, 8)]
    else:
        subjects = args.subjects

    # Determine output filename
    if args.output is None:
        if args.channel is None:
            output_filename = f"emg_sequences_ch{args.channel}.npz"
        else:
            output_filename = f"emg_sequences_all.npz"
    else:
        output_filename = args.output

    # Process all CSV files
    all_sequences_X = []
    all_sequences_y = []

    for subject_id in subjects:
        subject_path = os.path.join(args.data_path, subject_id)
        csv_files = glob.glob(os.path.join(subject_path, "*.csv"))

        print(f"\nProcessing {subject_id}: {len(csv_files)} CSV files found")

        for csv_file in csv_files:
            X, y = process_csv_file(csv_file, args.num_channels, args.seq_len)
            if X is not None:
                all_sequences_X.append(X)
                all_sequences_y.append(y)

    # Combine and save
    if all_sequences_X:
        X_final = np.vstack(all_sequences_X)
        y_final = np.concatenate(all_sequences_y)

        output_path = os.path.join(os.path.dirname(__file__), output_filename)
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
