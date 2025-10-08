"""
IMU + EMG Data Collection for LSTM Training

This script demonstrates how to collect synchronized IMU and EMG data
and format it for LSTM neural network input.

Usage:
    python imu_emg_lstm_example.py --mode collect --duration 60 --output data.npz
    python imu_emg_lstm_example.py --mode visualize --input data.npz
"""

import sys
import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import sensor readers
from sensors.imu_reader import IMUReader
from sensors.emg_reader import SimulatedEMGReader  # Use real EMGReader when hardware is available

# Import data fusion format
from data_fusion.sensor_data_format import (
    SensorFusionData,
    IMUData,
    EMGData,
    SensorSequenceBuilder,
    create_batch_for_lstm
)


class IMUEMGCollector:
    """Collects synchronized IMU and EMG data for LSTM training."""
    
    def __init__(
        self,
        imu_port: Optional[str] = None,
        emg_port: Optional[str] = None,
        emg_channels: int = 8,
        sequence_length: int = 50,
        use_simulated_emg: bool = True
    ):
        """
        Initialize data collector.
        
        Args:
            imu_port: Serial port for IMU (None for auto-detect)
            emg_port: Serial port for EMG (None for auto-detect)
            emg_channels: Number of EMG channels
            sequence_length: Length of sequences for LSTM
            use_simulated_emg: Use simulated EMG data if True
        """
        self.imu_reader = IMUReader(port=imu_port)
        
        if use_simulated_emg:
            self.emg_reader = SimulatedEMGReader(n_channels=emg_channels, sampling_rate=100.0)
        else:
            from sensors.emg_reader import EMGReader
            self.emg_reader = EMGReader(port=emg_port, n_channels=emg_channels)
        
        self.sequence_builder = SensorSequenceBuilder(
            sequence_length=sequence_length,
            overlap=sequence_length // 2,
            emg_channel_count=emg_channels,
            normalize=True
        )
        
        self.collected_data: List[SensorFusionData] = []
        self.running = False
    
    def start(self) -> bool:
        """Start data collection."""
        print("Starting IMU and EMG sensors...")
        
        imu_ok = self.imu_reader.activate()
        emg_ok = self.emg_reader.activate()
        
        if not imu_ok:
            print("Warning: Failed to activate IMU")
        if not emg_ok:
            print("Warning: Failed to activate EMG")
        
        if not (imu_ok or emg_ok):
            print("Error: No sensors activated")
            return False
        
        self.running = True
        print("Data collection started")
        return True
    
    def collect(self, duration: float = 60.0, verbose: bool = True):
        """
        Collect data for specified duration.
        
        Args:
            duration: Collection duration in seconds
            verbose: Print progress if True
        """
        if not self.running:
            print("Error: Sensors not started. Call start() first.")
            return
        
        print(f"Collecting data for {duration} seconds...")
        print("Press Ctrl+C to stop early")
        
        start_time = time.time()
        last_print_time = start_time
        sample_count = 0
        
        try:
            while time.time() - start_time < duration:
                # Read latest sensor data
                imu_data = self.imu_reader.get_latest_data()
                emg_data = self.emg_reader.get_latest_data()
                
                # Create fused data sample
                if imu_data is not None or emg_data is not None:
                    # Convert IMU data format if needed
                    if imu_data is not None and not isinstance(imu_data, IMUData):
                        # If IMUData from imu_reader has different structure, convert it
                        from data_fusion.sensor_data_format import IMUData as FusionIMUData
                        imu_data = FusionIMUData(
                            angular_velocity=imu_data.angular_velocity,
                            linear_acceleration=imu_data.linear_acceleration,
                            timestamp=imu_data.timestamp
                        )
                    
                    fused_sample = SensorFusionData(
                        imu=imu_data,
                        emg=emg_data,
                        timestamp=time.time()
                    )
                    
                    self.collected_data.append(fused_sample)
                    self.sequence_builder.add_sample(fused_sample)
                    sample_count += 1
                
                # Print progress
                if verbose and time.time() - last_print_time >= 1.0:
                    elapsed = time.time() - start_time
                    remaining = duration - elapsed
                    imu_status = self.imu_reader.get_status()
                    emg_status = self.emg_reader.get_status()
                    
                    print(f"[{elapsed:.1f}s / {duration:.1f}s] "
                          f"Samples: {sample_count} | "
                          f"IMU: {imu_status['read_count']} | "
                          f"EMG: {emg_status['read_count']} | "
                          f"Remaining: {remaining:.1f}s")
                    
                    last_print_time = time.time()
                
                time.sleep(0.01)  # 100 Hz collection rate
        
        except KeyboardInterrupt:
            print("\nCollection interrupted by user")
        
        elapsed = time.time() - start_time
        print(f"\nCollection complete!")
        print(f"  Duration: {elapsed:.1f}s")
        print(f"  Total samples: {len(self.collected_data)}")
        print(f"  Sample rate: {len(self.collected_data) / elapsed:.1f} Hz")
    
    def stop(self):
        """Stop data collection."""
        print("Stopping sensors...")
        self.imu_reader.deactivate()
        self.emg_reader.deactivate()
        self.running = False
        print("Sensors stopped")
    
    def save_data(self, filepath: str):
        """Save collected data to file."""
        if not self.collected_data:
            print("Warning: No data to save")
            return
        
        print(f"Saving data to {filepath}...")
        
        # Convert to sequences
        sequences = self.sequence_builder.get_all_sequences()
        
        if not sequences:
            print("Warning: Not enough data to create sequences")
            return
        
        # Stack sequences into array
        sequences_array = np.stack(sequences, axis=0)
        
        # Save metadata
        metadata = {
            'n_samples': len(self.collected_data),
            'n_sequences': len(sequences),
            'sequence_length': self.sequence_builder.sequence_length,
            'feature_dim': self.sequence_builder.get_feature_dimension(),
            'emg_channels': self.sequence_builder.emg_channel_count,
            'collection_date': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save to file
        np.savez(
            filepath,
            sequences=sequences_array,
            metadata=metadata,
            feature_mean=self.sequence_builder.feature_mean,
            feature_std=self.sequence_builder.feature_std
        )
        
        print(f"Saved {len(sequences)} sequences to {filepath}")
        print(f"  Shape: {sequences_array.shape}")
        print(f"  Format: (n_sequences, sequence_length, feature_dim)")
    
    def load_data(self, filepath: str):
        """Load previously collected data."""
        print(f"Loading data from {filepath}...")
        
        data = np.load(filepath, allow_pickle=True)
        sequences = data['sequences']
        metadata = data['metadata'].item()
        
        print(f"Loaded data:")
        print(f"  Sequences: {sequences.shape}")
        print(f"  Samples: {metadata['n_samples']}")
        print(f"  Sequence length: {metadata['sequence_length']}")
        print(f"  Feature dimension: {metadata['feature_dim']}")
        print(f"  Collection date: {metadata['collection_date']}")
        
        return sequences, metadata
    
    def visualize_data(self, max_samples: int = 500):
        """Visualize collected data."""
        if not self.collected_data:
            print("Warning: No data to visualize")
            return
        
        print("Generating visualization...")
        
        # Extract data for plotting
        timestamps = []
        imu_gyro = []
        imu_accel = []
        emg_channels = []
        
        for sample in self.collected_data[:max_samples]:
            timestamps.append(sample.timestamp)
            
            if sample.imu:
                imu_gyro.append(sample.imu.angular_velocity)
                imu_accel.append(sample.imu.linear_acceleration)
            else:
                imu_gyro.append([0, 0, 0])
                imu_accel.append([0, 0, 0])
            
            if sample.emg:
                emg_channels.append(sample.emg.channels)
            else:
                emg_channels.append([0] * self.sequence_builder.emg_channel_count)
        
        timestamps = np.array(timestamps) - timestamps[0]  # Relative time
        imu_gyro = np.array(imu_gyro)
        imu_accel = np.array(imu_accel)
        emg_channels = np.array(emg_channels)
        
        # Create plots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot IMU gyroscope
        axes[0].plot(timestamps, imu_gyro[:, 0], label='Gyro X', alpha=0.7)
        axes[0].plot(timestamps, imu_gyro[:, 1], label='Gyro Y', alpha=0.7)
        axes[0].plot(timestamps, imu_gyro[:, 2], label='Gyro Z', alpha=0.7)
        axes[0].set_ylabel('Angular Velocity (rad/s)')
        axes[0].set_title('IMU Gyroscope Data')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot IMU accelerometer
        axes[1].plot(timestamps, imu_accel[:, 0], label='Accel X', alpha=0.7)
        axes[1].plot(timestamps, imu_accel[:, 1], label='Accel Y', alpha=0.7)
        axes[1].plot(timestamps, imu_accel[:, 2], label='Accel Z', alpha=0.7)
        axes[1].set_ylabel('Linear Acceleration (m/s²)')
        axes[1].set_title('IMU Accelerometer Data')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot EMG channels (first 4 channels for clarity)
        n_channels_to_plot = min(4, emg_channels.shape[1])
        for i in range(n_channels_to_plot):
            axes[2].plot(timestamps, emg_channels[:, i], label=f'EMG Ch{i+1}', alpha=0.7)
        axes[2].set_xlabel('Time (seconds)')
        axes[2].set_ylabel('EMG Amplitude')
        axes[2].set_title(f'EMG Data (showing {n_channels_to_plot} of {emg_channels.shape[1]} channels)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('imu_emg_data_visualization.png', dpi=150)
        print("Visualization saved to imu_emg_data_visualization.png")
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='IMU + EMG Data Collection for LSTM')
    parser.add_argument('--mode', type=str, default='collect',
                       choices=['collect', 'visualize'],
                       help='Operation mode')
    parser.add_argument('--duration', type=float, default=60.0,
                       help='Collection duration in seconds')
    parser.add_argument('--output', type=str, default='imu_emg_data.npz',
                       help='Output file path')
    parser.add_argument('--input', type=str, default='imu_emg_data.npz',
                       help='Input file path for visualization')
    parser.add_argument('--imu-port', type=str, default=None,
                       help='IMU serial port (auto-detect if not specified)')
    parser.add_argument('--emg-port', type=str, default=None,
                       help='EMG serial port (auto-detect if not specified)')
    parser.add_argument('--emg-channels', type=int, default=8,
                       help='Number of EMG channels')
    parser.add_argument('--sequence-length', type=int, default=50,
                       help='Sequence length for LSTM')
    parser.add_argument('--use-real-emg', action='store_true',
                       help='Use real EMG hardware (default: simulated)')
    
    args = parser.parse_args()
    
    if args.mode == 'collect':
        print("=" * 60)
        print("IMU + EMG Data Collection for LSTM Training")
        print("=" * 60)
        
        collector = IMUEMGCollector(
            imu_port=args.imu_port,
            emg_port=args.emg_port,
            emg_channels=args.emg_channels,
            sequence_length=args.sequence_length,
            use_simulated_emg=not args.use_real_emg
        )
        
        if collector.start():
            try:
                collector.collect(duration=args.duration, verbose=True)
                collector.save_data(args.output)
                collector.visualize_data()
            finally:
                collector.stop()
        
    elif args.mode == 'visualize':
        print("=" * 60)
        print("Visualizing Collected Data")
        print("=" * 60)
        
        collector = IMUEMGCollector(
            emg_channels=args.emg_channels,
            sequence_length=args.sequence_length
        )
        
        sequences, metadata = collector.load_data(args.input)
        
        print("\nSequence shape:", sequences.shape)
        print("Ready for LSTM input!")
        print("\nExample PyTorch usage:")
        print("  import torch")
        print("  sequences_tensor = torch.from_numpy(sequences)")
        print("  # sequences_tensor shape: (batch_size, seq_len, feature_dim)")
        print("  output = lstm_model(sequences_tensor)")


if __name__ == "__main__":
    main()
