"""
Visualize EMG data from .mat files in the data_rest folder.

This script allows you to:
- List available subjects and gestures
- Visualize raw EMG signals
- Optionally show preprocessed signals
- Compare frequency spectra (raw vs preprocessed)
"""

import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch, welch
import sys

# Add parent directory to path to import emg_utils if needed
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


def plot_emg_channels(signal, fs, channels_to_plot=None, title="EMG Signal", 
                       show_preprocessed=False, preprocessed_signal=None):
    """Plot EMG channels over time."""
    num_channels = signal.shape[0]
    if channels_to_plot is None:
        channels_to_plot = min(num_channels, 8)  # Default to 8 channels max
    channels_to_plot = min(channels_to_plot, num_channels)
    
    num_samples = signal.shape[1]
    t = np.arange(num_samples) / fs
    
    if show_preprocessed and preprocessed_signal is not None:
        fig, axes = plt.subplots(channels_to_plot, 2, figsize=(16, 2 * channels_to_plot), sharex=True)
        if channels_to_plot == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(channels_to_plot):
            axes[i, 0].plot(t, signal[i, :], 'b-', alpha=0.7)
            axes[i, 0].set_ylabel(f"Ch {i+1}")
            axes[i, 0].set_title("Raw Signal")
            axes[i, 0].grid(True)
            
            axes[i, 1].plot(t, preprocessed_signal[i, :], 'r-', alpha=0.7)
            axes[i, 1].set_ylabel(f"Ch {i+1}")
            axes[i, 1].set_title("Preprocessed Signal")
            axes[i, 1].grid(True)
        
        axes[-1, 0].set_xlabel("Time (s)")
        axes[-1, 1].set_xlabel("Time (s)")
        fig.suptitle(title, fontsize=16)
    else:
        fig, axes = plt.subplots(channels_to_plot, 1, figsize=(14, 2 * channels_to_plot), sharex=True)
        if channels_to_plot == 1:
            axes = [axes]
        
        for i in range(channels_to_plot):
            axes[i].plot(t, signal[i, :])
            axes[i].set_ylabel(f"Ch {i+1}")
            axes[i].grid(True)
        
        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_psd_comparison(raw_signal, preprocessed_signal, fs, channel_idx=0):
    """Plot Power Spectral Density comparison."""
    raw_freqs, raw_psd = welch(raw_signal[channel_idx, :], fs=fs, nperseg=1024)
    pre_freqs, pre_psd = welch(preprocessed_signal[channel_idx, :], fs=fs, nperseg=1024)
    
    plt.figure(figsize=(14, 7))
    plt.semilogy(raw_freqs, raw_psd, label='Raw Signal', color='blue', alpha=0.7)
    plt.semilogy(pre_freqs, pre_psd, label='Preprocessed Signal', color='red', alpha=0.9)
    
    plt.axvline(x=20, color='gray', linestyle='--', label='Band-pass Cutoff (20 Hz)')
    plt.axvline(x=450, color='gray', linestyle='--', label='Band-pass Cutoff (450 Hz)')
    plt.axvline(x=50, color='green', linestyle=':', label='Notch Filter (50 Hz)')
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (V^2/Hz)')
    plt.title(f'Preprocessing Effect on Frequency Spectrum (Channel {channel_idx + 1})')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.xlim(0, fs / 2)
    plt.tight_layout()
    plt.show()


def list_available_files(data_path):
    """List all available .mat files in the data_rest folder."""
    subjects = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    
    print("\nAvailable subjects and files:")
    print("=" * 60)
    
    for subject in subjects:
        subject_path = os.path.join(data_path, subject)
        mat_files = sorted(glob.glob(os.path.join(subject_path, "*.mat")))
        
        if mat_files:
            print(f"\n{subject}:")
            for mat_file in mat_files:
                filename = os.path.basename(mat_file)
                print(f"  - {filename}")
    
    print("\n" + "=" * 60)


def load_and_visualize(file_path, fs=985, channels_to_plot=None, 
                       show_preprocessed=False, show_psd=False,
                       lowcut=20.0, highcut=450.0, notch_freq=50.0):
    """Load a .mat file and visualize the EMG data."""
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return
    
    print(f"Loading: {file_path}")
    try:
        mat_data = loadmat(file_path)
        signal = mat_data["value"]
        print(f"Signal shape: {signal.shape} (channels, samples)")
        print(f"Duration: {signal.shape[1] / fs:.2f} seconds")
    except KeyError:
        print("Error: 'value' key not found in .mat file")
        return
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    # Plot raw signal
    title = f"EMG Signal: {os.path.basename(file_path)}"
    preprocessed_signal = None
    
    if show_preprocessed:
        print("Preprocessing signal...")
        preprocessed_signal = preprocess_emg_signal(signal, fs, lowcut, highcut, notch_freq)
        plot_emg_channels(signal, fs, channels_to_plot, title, 
                         show_preprocessed=True, preprocessed_signal=preprocessed_signal)
        
        if show_psd:
            plot_psd_comparison(signal, preprocessed_signal, fs, channel_idx=0)
    else:
        plot_emg_channels(signal, fs, channels_to_plot, title)
    
    print("Visualization complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize EMG data from .mat files in data_rest folder"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="../../EMG_signal_processing/data_rest",
        help="Path to data_rest folder (default: ../../EMG_signal_processing/data_rest)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available files"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to specific .mat file to visualize (e.g., S1/Hand_Open-1.mat)"
    )
    parser.add_argument(
        "--subject",
        type=str,
        help="Subject ID (e.g., S1) - will show first file for that subject"
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=None,
        help="Number of channels to plot (default: all, max 8)"
    )
    parser.add_argument(
        "--preprocessed",
        action="store_true",
        help="Show preprocessed signal alongside raw signal"
    )
    parser.add_argument(
        "--psd",
        action="store_true",
        help="Show Power Spectral Density comparison (requires --preprocessed)"
    )
    parser.add_argument(
        "--fs",
        type=int,
        default=985,
        help="Sampling frequency in Hz (default: 985)"
    )
    
    args = parser.parse_args()
    
    data_path = os.path.abspath(args.data_path)
    
    if args.list:
        list_available_files(data_path)
        return
    
    if args.file:
        # Full path or relative to data_path
        if os.path.isabs(args.file):
            file_path = args.file
        else:
            file_path = os.path.join(data_path, args.file)
        load_and_visualize(
            file_path, fs=args.fs, channels_to_plot=args.channels,
            show_preprocessed=args.preprocessed, show_psd=args.psd
        )
    elif args.subject:
        # Find first file for the subject
        subject_path = os.path.join(data_path, args.subject)
        mat_files = sorted(glob.glob(os.path.join(subject_path, "*.mat")))
        if mat_files:
            load_and_visualize(
                mat_files[0], fs=args.fs, channels_to_plot=args.channels,
                show_preprocessed=args.preprocessed, show_psd=args.psd
            )
        else:
            print(f"No .mat files found for subject {args.subject}")
    else:
        # Interactive mode: list files and let user choose
        list_available_files(data_path)
        print("\nUsage examples:")
        print("  python view_data_mat.py --file S1/Hand_Open-1.mat")
        print("  python view_data_mat.py --subject S1 --preprocessed")
        print("  python view_data_mat.py --file S1/Hand_Open-1.mat --preprocessed --psd")


if __name__ == "__main__":
    main()

