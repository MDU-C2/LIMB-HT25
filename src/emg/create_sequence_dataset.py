"""
Create sequence dataset from raw EMG .mat files.

This script processes raw EMG recordings, applies preprocessing, extracts features,
and creates sequences for LSTM training.
"""
import argparse
import os
import glob
import logging
import numpy as np
from scipy.io import loadmat
import sys

# Import emg_utils from EMG_signal_processing subdirectory
# Add the EMG_signal_processing directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
emg_processing_dir = os.path.join(current_dir, 'EMG_signal_processing')
if emg_processing_dir not in sys.path:
    sys.path.insert(0, emg_processing_dir)

import emg_utils as EMGU

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create EMG sequence dataset from raw .mat files"
    )
    
    # Data paths
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Path to data directory containing subject folders (default: EMG_signal_processing/data_rest)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output NPZ file path (default: EMG_signal_processing/emg_sequences_dataset.npz)'
    )
    
    # Subjects and gestures
    parser.add_argument(
        '--subjects',
        type=str,
        nargs='+',
        default=None,
        help='List of subjects to process (default: S1-S19)'
    )
    parser.add_argument(
        '--gestures',
        type=str,
        nargs='+',
        default=['Hand_Close', 'Hand_Open'],
        help='List of gestures to process (default: Hand_Close Hand_Open)'
    )
    
    # Processing parameters
    parser.add_argument(
        '--fs',
        type=int,
        default=985,
        help='Sampling frequency in Hz (default: 985)'
    )
    parser.add_argument(
        '--num-channels',
        type=int,
        default=8,
        help='Number of EMG channels (default: 8)'
    )
    parser.add_argument(
        '--lowcut',
        type=float,
        default=20.0,
        help='Low cutoff frequency for bandpass filter in Hz (default: 20.0)'
    )
    parser.add_argument(
        '--highcut',
        type=float,
        default=450.0,
        help='High cutoff frequency for bandpass filter in Hz (default: 450.0)'
    )
    parser.add_argument(
        '--notch-freq',
        type=float,
        default=50.0,
        help='Notch filter frequency in Hz (default: 50.0)'
    )
    parser.add_argument(
        '--window-size-ms',
        type=int,
        default=200,
        help='Window size in milliseconds (default: 200)'
    )
    parser.add_argument(
        '--overlap-ms',
        type=int,
        default=50,
        help='Window overlap in milliseconds (default: 50)'
    )
    parser.add_argument(
        '--seq-len',
        type=int,
        default=10,
        help='Sequence length (number of windows per sequence) (default: 10)'
    )
    
    # Labels mapping
    parser.add_argument(
        '--label-map',
        type=str,
        nargs='+',
        default=['Hand_Open:0', 'Hand_Close:1'],
        help='Label mapping as "Gesture:Label" pairs (default: Hand_Open:0 Hand_Close:1)'
    )
    
    return parser.parse_args()


def parse_label_map(label_map_args):
    """
    Parse label mapping from command-line arguments.
    
    Args:
        label_map_args: List of strings in format "Gesture:Label"
        
    Returns:
        Dictionary mapping gesture names to labels
    """
    label_map = {}
    for arg in label_map_args:
        try:
            gesture, label = arg.split(':')
            label_map[gesture] = int(label)
        except ValueError:
            logger.warning(f"Invalid label mapping format: {arg}. Expected 'Gesture:Label'")
    return label_map


def process_file(file_path, label, fs, num_channels, lowcut, highcut, notch_freq,
                 window_size_ms, overlap_ms, seq_len):
    """
    Process a single EMG recording file.
    
    Args:
        file_path: Path to the .mat file
        label: Label for this gesture
        fs: Sampling frequency
        num_channels: Number of EMG channels
        lowcut: Low cutoff frequency
        highcut: High cutoff frequency
        notch_freq: Notch filter frequency
        window_size_ms: Window size in milliseconds
        overlap_ms: Overlap in milliseconds
        seq_len: Sequence length
        
    Returns:
        Tuple of (X_sequences, y_sequences) or (None, None) if file is too short
    """
    try:
        # Load the signal
        raw_signal = loadmat(file_path)["value"]
        
        # Preprocess the signal
        preprocessed_signal = EMGU.preprocess_emg_signal(
            raw_signal, fs, lowcut, highcut, notch_freq
        )
        
        # Create windows and labels
        windows, labels = EMGU.create_windows(
            preprocessed_signal, label, fs, window_size_ms, overlap_ms
        )
        
        # Extract features from the windows
        features = EMGU.extract_time_domain_features(windows, num_channels)
        
        # Create sequences from this file's features
        if features.shape[0] >= seq_len:
            X_seq_file, y_seq_file = EMGU.create_sequences(features, labels, seq_len)
            return X_seq_file, y_seq_file
        else:
            logger.warning(
                f"File {os.path.basename(file_path)} skipped "
                f"(too short to create sequences of length {seq_len})"
            )
            return None, None
            
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return None, None


def create_sequence_dataset(data_dir, output_path, subjects, gestures, label_map,
                           fs, num_channels, lowcut, highcut, notch_freq,
                           window_size_ms, overlap_ms, seq_len):
    """
    Create sequence dataset from raw EMG files.
    
    Args:
        data_dir: Directory containing subject folders
        output_path: Path to save the output NPZ file
        subjects: List of subject IDs to process
        gestures: List of gesture names to process
        label_map: Dictionary mapping gesture names to labels
        fs: Sampling frequency
        num_channels: Number of EMG channels
        lowcut: Low cutoff frequency
        highcut: High cutoff frequency
        notch_freq: Notch filter frequency
        window_size_ms: Window size in milliseconds
        overlap_ms: Overlap in milliseconds
        seq_len: Sequence length
        
    Returns:
        Tuple of (X_final, y_final) or (None, None) if no data processed
    """
    all_sequences_X = []
    all_sequences_y = []
    
    logger.info("="*60)
    logger.info("Starting sequence dataset creation")
    logger.info("="*60)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Subjects: {subjects}")
    logger.info(f"Gestures: {gestures}")
    logger.info(f"Label map: {label_map}")
    logger.info(f"Processing parameters: fs={fs}, channels={num_channels}, "
                f"seq_len={seq_len}")
    logger.info("-"*60)
    
    total_files = 0
    processed_files = 0
    
    for subject_id in subjects:
        subject_path = os.path.join(data_dir, subject_id)
        
        if not os.path.exists(subject_path):
            logger.warning(f"Subject directory not found: {subject_path}")
            continue
            
        logger.info(f"Processing Subject: {subject_id}")
        
        for gesture_name in gestures:
            if gesture_name not in label_map:
                logger.warning(f"Gesture '{gesture_name}' not in label map, skipping")
                continue
                
            search_pattern = os.path.join(subject_path, f"{gesture_name}-*.mat")
            files_found = glob.glob(search_pattern)
            logger.info(f"  Found {len(files_found)} files for gesture: {gesture_name}")
            
            if len(files_found) == 0:
                logger.warning(f"  No files found for pattern: {search_pattern}")
            
            # Process each file
            for file_path in files_found:
                total_files += 1
                label = label_map[gesture_name]
                
                X_seq, y_seq = process_file(
                    file_path, label, fs, num_channels, lowcut, highcut, notch_freq,
                    window_size_ms, overlap_ms, seq_len
                )
                
                if X_seq is not None and y_seq is not None:
                    all_sequences_X.append(X_seq)
                    all_sequences_y.append(y_seq)
                    processed_files += 1
    
    logger.info("-"*60)
    logger.info(f"Processed {processed_files}/{total_files} files successfully")
    
    # Create final dataset
    if all_sequences_X:
        X_final = np.vstack(all_sequences_X)
        y_final = np.concatenate(all_sequences_y)
        
        logger.info("="*60)
        logger.info("Sequence Dataset Creation Complete!")
        logger.info(f"Final sequences matrix shape (X): {X_final.shape}")
        logger.info("      -> (Total Sequences, Windows per Sequence, Features per Window)")
        logger.info(f"Final labels vector shape (y): {y_final.shape}")
        logger.info(f"Label distribution: {np.bincount(y_final)}")
        logger.info("="*60)
        
        return X_final, y_final
    else:
        logger.error("No data was processed. Please check your data directory and parameters.")
        return None, None


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set default data directory
    if args.data_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        args.data_dir = os.path.join(current_dir, 'EMG_signal_processing', 'data_rest')
    
    # Set default output path
    if args.output is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        args.output = os.path.join(current_dir, 'EMG_signal_processing', 'emg_sequences_dataset.npz')
    
    # Set default subjects
    if args.subjects is None:
        args.subjects = [f"S{i}" for i in range(1, 20)]  # S1 to S19
    
    # Parse label map
    label_map = parse_label_map(args.label_map)
    
    # Validate paths
    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory not found: {args.data_dir}")
        return 1
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
    
    # Create dataset
    X_final, y_final = create_sequence_dataset(
        args.data_dir, args.output, args.subjects, args.gestures, label_map,
        args.fs, args.num_channels, args.lowcut, args.highcut, args.notch_freq,
        args.window_size_ms, args.overlap_ms, args.seq_len
    )
    
    if X_final is not None and y_final is not None:
        # Save dataset
        np.savez(args.output, X=X_final, y=y_final)
        logger.info(f"Dataset saved to: {args.output}")
        return 0
    else:
        logger.error("Failed to create dataset")
        return 1


if __name__ == "__main__":
    exit(main())

