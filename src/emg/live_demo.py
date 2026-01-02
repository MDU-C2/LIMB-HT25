"""
Live EMG Demo - Real-time grip prediction from EMG signals

Usage:
    python -m src.emg.live_demo --model checkpoints/model_XXX.pt --scaler checkpoints/scaler_XXX.pkl --config checkpoints/config_XXX.json
"""

import asyncio
import struct
import numpy as np
import torch
import torch.nn as nn
import json
import pickle
import argparse
import os
from collections import deque
from bleak import BleakScanner, BleakClient

# --- CONFIGURATION ---
TARGET_NAME = "LIMBServer"
EMG_CHAR_UUID = "24011525-1212-efde-1523-785feabcd122"

# --- ASSEMBLY PARAMETERS ---
CHUNKS_PER_WINDOW = 10
EMG_SAMPLES_PER_CHUNK = 40
EMG_WINDOW_SAMPLES = CHUNKS_PER_WINDOW * EMG_SAMPLES_PER_CHUNK
EMG_PACKET_FORMAT = f'<{EMG_SAMPLES_PER_CHUNK}H I'

# Label names
LABEL_NAMES = {0: 'Hand_Open', 1: 'Hand_Close'}


class SimpleLSTM(nn.Module):
    """LSTM model matching the training architecture."""
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h = self.dropout(h_n[-1])
        return self.fc(h)


class LiveEMGDemo:
    def __init__(self, model_path, scaler_path, config_path, num_channels=1, seq_length=10):
        """Initialize the live demo with trained model."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Initialize model
        self.model = SimpleLSTM(
            input_dim=self.config['input_dim'],
            hidden_dim=self.config['hidden_dim'],
            num_classes=self.config['num_classes'],
            dropout=0.5
        ).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.num_channels = num_channels
        self.seq_length = seq_length
        
        # Feature buffer for sequence creation
        self.feature_buffer = deque(maxlen=seq_length)
        
        # BLE connection state
        self.queue = asyncio.Queue()
        self.client = None
        self.running = False
        self.emg_buffer = []
        self.exp_emg_seq = None
        self.packet_count = 0
        self.window_count = 0
        
        # Statistics
        self.prediction_history = deque(maxlen=50)
        self.total_predictions = 0
        
        print(f"\nModel loaded successfully!")
        print(f"  Input dim: {self.config['input_dim']}")
        print(f"  Hidden dim: {self.config['hidden_dim']}")
        print(f"  Sequence length: {seq_length}")
        print(f"  Number of channels: {num_channels}")
    
    async def _handle_emg(self, data):
        """Handle incoming EMG data packets."""
        self.packet_count += 1
        try:
            *payload, seq = struct.unpack(EMG_PACKET_FORMAT, data)
        except Exception as e:
            self.exp_emg_seq = None
            self.emg_buffer.clear()
            return

        if self.exp_emg_seq is None:
            if seq % CHUNKS_PER_WINDOW == 0:
                self.emg_buffer = list(payload)
                self.exp_emg_seq = seq + 1
        elif seq == self.exp_emg_seq:
            self.emg_buffer.extend(payload)
            self.exp_emg_seq += 1
            
            if len(self.emg_buffer) == EMG_WINDOW_SAMPLES:
                self.window_count += 1
                await self.queue.put(('EMG', list(self.emg_buffer)))
                self.exp_emg_seq = None
        else:
            # Sequence mismatch - reset
            self.exp_emg_seq = None
            self.emg_buffer.clear()
    
    async def notification_handler(self, sender, data):
        """Handle BLE notifications."""
        if str(sender.uuid) == EMG_CHAR_UUID:
            await self._handle_emg(data)
    
    def extract_features(self, window_data):
        """Extract time-domain features from a window."""
        from .emg_processing_utils import (
            calculate_mav, calculate_rms, calculate_wl,
            calculate_zc, calculate_ssc, calculate_var
        )
        
        # Convert to numpy array
        window_array = np.array(window_data, dtype=np.float32)
        
        # Reshape for multi-channel processing
        samples_per_channel = len(window_array) // self.num_channels
        reshaped_window = window_array.reshape(self.num_channels, samples_per_channel)
        
        # Extract features for each channel
        features = []
        for channel_idx in range(self.num_channels):
            channel_data = reshaped_window[channel_idx, :]
            features.extend([
                calculate_mav(channel_data),
                calculate_rms(channel_data),
                calculate_wl(channel_data),
                calculate_zc(channel_data),
                calculate_ssc(channel_data),
                calculate_var(channel_data)
            ])
        
        return np.array(features)
    
    def predict(self, features):
        """Make prediction from feature sequence."""
        if len(self.feature_buffer) < self.seq_length:
            return None, None
        
        # Create sequence from buffer
        sequence = np.array(list(self.feature_buffer))
        
        # Standardize features
        sequence_flat = sequence.reshape(-1, sequence.shape[-1])
        sequence_std = self.scaler.transform(sequence_flat)
        sequence_std = sequence_std.reshape(sequence.shape)
        
        # Convert to tensor and add batch dimension
        sequence_tensor = torch.tensor(sequence_std, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            logits = self.model(sequence_tensor)
            probabilities = torch.softmax(logits, dim=1)
            pred_class = logits.argmax(1).item()
            confidence = probabilities[0, pred_class].item()
        
        return pred_class, confidence
    
    def display_prediction(self, pred_class, confidence):
        """Display prediction with visual feedback."""
        # Clear screen (works on most terminals)
        print("\033[2J\033[H", end='')
        
        # Header
        print("=" * 70)
        print("EMG Live Demo - Real-time Grip Prediction")
        print("=" * 70)
        print(f"Windows received: {self.window_count} | Predictions: {self.total_predictions}")
        print("=" * 70)
        
        if pred_class is None:
            print("\n⏳ Collecting data... (need {} windows for prediction)".format(self.seq_length))
            print(f"   Buffer: {len(self.feature_buffer)}/{self.seq_length} windows")
            return
        
        # Prediction display
        label = LABEL_NAMES.get(pred_class, f"Class_{pred_class}")
        confidence_pct = confidence * 100
        
        # Color coding
        if pred_class == 0:  # Hand_Open
            color = "\033[92m"  # Green
            symbol = "🟢"
            action = "REST"
        else:  # Hand_Close
            color = "\033[93m"  # Yellow/Orange
            symbol = "🟡"
            action = "GRIP"
        
        reset_color = "\033[0m"
        
        print(f"\n{color}{symbol} PREDICTION: {action} ({label}){reset_color}")
        print(f"   Confidence: {confidence_pct:.1f}%")
        
        # Visual bar
        bar_length = 50
        filled = int(confidence_pct / 100 * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"   [{bar}]")
        
        # Recent history
        if len(self.prediction_history) > 0:
            recent = list(self.prediction_history)[-10:]
            recent_str = "".join(["🟢" if p == 0 else "🟡" for p in recent])
            print(f"\n   Recent: {recent_str}")
        
        print("\n" + "=" * 70)
        print("Press Ctrl+C to stop")
    
    async def run(self):
        """Main demo loop."""
        print("=" * 70)
        print("EMG Live Demo")
        print("=" * 70)
        print(f"Target: {TARGET_NAME}")
        print("=" * 70)
        
        print(f"\nScanning for device '{TARGET_NAME}'...")
        device = await BleakScanner.find_device_by_name(TARGET_NAME)
        
        if not device:
            print(f"Error: Device '{TARGET_NAME}' not found.")
            return
        
        print(f"Connecting to {device.address}...")
        async with BleakClient(device) as client:
            self.client = client
            await client.start_notify(EMG_CHAR_UUID, self.notification_handler)
            
            self.running = True
            
            # Wait for connection to stabilize
            print("Waiting for data stream to stabilize...")
            await asyncio.sleep(2.0)
            
            # Clear initial data
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except:
                    break
            
            print("\nStarting live prediction...")
            print("Make hand gestures and watch the predictions!\n")
            await asyncio.sleep(1.0)
            
            try:
                while self.running:
                    try:
                        # Get next window with timeout
                        sensor_type, data = await asyncio.wait_for(
                            self.queue.get(),
                            timeout=1.0
                        )
                        
                        if sensor_type == 'EMG':
                            # Extract features
                            features = self.extract_features(data)
                            
                            # Add to buffer
                            self.feature_buffer.append(features)
                            
                            # Make prediction if buffer is full
                            if len(self.feature_buffer) >= self.seq_length:
                                pred_class, confidence = self.predict(features)
                                
                                if pred_class is not None:
                                    self.total_predictions += 1
                                    self.prediction_history.append(pred_class)
                                    self.display_prediction(pred_class, confidence)
                                else:
                                    self.display_prediction(None, None)
                            else:
                                self.display_prediction(None, None)
                    
                    except asyncio.TimeoutError:
                        # No data received, but continue running
                        continue
            
            except KeyboardInterrupt:
                print("\n\nDemo stopped by user.")
            finally:
                await client.stop_notify(EMG_CHAR_UUID)
                self.running = False
                
                # Final statistics
                print("\n" + "=" * 70)
                print("Session Statistics:")
                print(f"  Total windows received: {self.window_count}")
                print(f"  Total predictions made: {self.total_predictions}")
                if len(self.prediction_history) > 0:
                    open_count = sum(1 for p in self.prediction_history if p == 0)
                    close_count = sum(1 for p in self.prediction_history if p == 1)
                    print(f"  Recent predictions - Open: {open_count}, Close: {close_count}")
                print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Live EMG Demo - Real-time grip prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default checkpoint files (no arguments needed!)
  python -m src.emg.live_demo

  # Run with custom checkpoint files
  python -m src.emg.live_demo \\
      --model checkpoints/model_XXX.pt \\
      --scaler checkpoints/scaler_XXX.pkl \\
      --config checkpoints/config_XXX.json

  # Run with custom number of channels
  python -m src.emg.live_demo --num-channels 1
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='checkpoints/model_20251228_142314.pt',
        help='Path to trained model (.pt file) (default: checkpoints/model_20251228_142314.pt)'
    )
    parser.add_argument(
        '--scaler',
        type=str,
        default='checkpoints/scaler_20251228_142314.pkl',
        help='Path to StandardScaler (.pkl file) (default: checkpoints/scaler_20251228_142314.pkl)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='checkpoints/config_20251228_142314.json',
        help='Path to model config (.json file) (default: checkpoints/config_20251228_142314.json)'
    )
    parser.add_argument(
        '--num-channels',
        type=int,
        default=1,
        help='Number of EMG channels (default: 1)'
    )
    parser.add_argument(
        '--seq-length',
        type=int,
        default=None,
        help='Sequence length for LSTM (default: from config)'
    )
    
    args = parser.parse_args()
    
    # Resolve paths relative to script location if needed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    def resolve_path(path):
        if os.path.isabs(path):
            return path
        # Try relative to current directory first
        if os.path.exists(path):
            return path
        # Try relative to script directory
        script_path = os.path.join(script_dir, path)
        if os.path.exists(script_path):
            return script_path
        return path
    
    model_path = resolve_path(args.model)
    scaler_path = resolve_path(args.scaler)
    config_path = resolve_path(args.config)
    
    # Check if files exist
    for name, path in [("Model", model_path), ("Scaler", scaler_path), ("Config", config_path)]:
        if not os.path.exists(path):
            print(f"Error: {name} file not found: {path}")
            return
    
    # Get sequence length from config if not provided
    seq_length = args.seq_length
    if seq_length is None:
        with open(config_path, 'r') as f:
            config = json.load(f)
            seq_length = config.get('seq_length', 10)
    
    # Run demo
    demo = LiveEMGDemo(model_path, scaler_path, config_path, args.num_channels, seq_length)
    
    try:
        asyncio.run(demo.run())
    except KeyboardInterrupt:
        print("\nDemo stopped.")


if __name__ == "__main__":
    main()

