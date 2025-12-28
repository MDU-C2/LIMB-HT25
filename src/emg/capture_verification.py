"""
Simple script to capture 80 windows of EMG data (20 rest → 40 grip → 20 rest)
and display a plot for hardware verification.
"""

import asyncio
import struct
import numpy as np
import matplotlib.pyplot as plt
from bleak import BleakScanner, BleakClient
from datetime import datetime

# --- CONFIGURATION ---
TARGET_NAME = "LIMBServer"
EMG_CHAR_UUID = "24011525-1212-efde-1523-785feabcd122"

# --- ASSEMBLY PARAMETERS ---
CHUNKS_PER_WINDOW = 10
EMG_SAMPLES_PER_CHUNK = 40
EMG_WINDOW_SAMPLES = CHUNKS_PER_WINDOW * EMG_SAMPLES_PER_CHUNK
EMG_PACKET_FORMAT = f'<{EMG_SAMPLES_PER_CHUNK}H I'

# --- CAPTURE CONSTANTS ---
WINDOWS_PER_CAPTURE = 80
REST_START_WINDOWS = 20
GESTURE_END_WINDOWS = 60


def plot_emg_capture(captured_windows):
    """Plot the captured EMG data with phase zones highlighted."""
    if not captured_windows:
        print("Error: No data to plot.")
        return
    
    print("\n--- Creating verification plot ---")
    
    # Flatten all windows into a single array
    emg_data = []
    for window in captured_windows:
        emg_data.extend(window)
    
    y_values = np.array(emg_data, dtype=np.float32)
    x_values = np.arange(len(y_values))
    window_size = EMG_WINDOW_SAMPLES
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(x_values, y_values, label='EMG signal', color='#1f77b4', linewidth=0.8)
    ax.set_ylabel("ADC value")
    ax.set_title("EMG Capture Verification: 20 Rest → 40 Grip → 20 Rest")
    ax.set_xlabel("Samples")
    
    # Vertical lines for each window
    for x in range(window_size, len(x_values), window_size):
        ax.axvline(x=x, color='red', linestyle='--', linewidth=0.5, alpha=0.4)
    
    # Draw rest - gesture - rest zones
    # Initial rest zone (Windows 0-19)
    start_rest_end_x = REST_START_WINDOWS * window_size
    ax.axvspan(0, start_rest_end_x, alpha=0.2, color='green', label='Initial rest (0-19)')
    
    # Main gesture (Windows 20-59)
    gesture_end_x = GESTURE_END_WINDOWS * window_size
    ax.axvspan(start_rest_end_x, gesture_end_x, alpha=0.2, color='orange', label='Grip (20-59)')
    
    # Final rest zone (Windows 60-79)
    end_capture_x = WINDOWS_PER_CAPTURE * window_size
    ax.axvspan(gesture_end_x, end_capture_x, alpha=0.2, color='green', label='Final rest (60-79)')
    
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right')
    ax.set_xlim(0, len(x_values))
    plt.tight_layout()
    
    print("Plot displayed. Verify that:")
    print("  - Green zones show rest (low EMG activity)")
    print("  - Orange zone shows grip (higher EMG activity)")
    print("  - Signal looks reasonable (no obvious errors)")
    print("\nClose the plot window to exit.")
    plt.show()


class EMGCapture:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.client = None
        self.running = False
        self.emg_buffer = []
        self.exp_emg_seq = None
        self.packet_count = 0
        self.window_count = 0
    
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
    
    async def capture_windows(self, target_count=80):
        """Capture specified number of EMG windows."""
        print(f"\n--- CAPTURING {target_count} WINDOWS ---")
        print("Instructions:")
        print("  1. Keep your hand at REST for 2 seconds (windows 0-19)")
        print("  2. GRIP a mug/object for 4 seconds (windows 20-59)")
        print("  3. Release and keep hand at REST for 2 seconds (windows 60-79)")
        print("\nStarting capture in 2 seconds...")
        await asyncio.sleep(2)
        print("CAPTURING NOW!\n")
        
        # Clear any old data from queue
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except:
                break
        
        captured_windows = []
        count = 0
        
        while count < target_count:
            try:
                sensor_type, data = await asyncio.wait_for(
                    self.queue.get(), 
                    timeout=10.0  # Increased timeout
                )
                
                if sensor_type == 'EMG':
                    captured_windows.append(list(data))
                    count += 1
                    print(f"\rProgress: {count}/{target_count} windows captured", end='', flush=True)
            
            except asyncio.TimeoutError:
                print(f"\nWarning: Timeout waiting for window {count}. "
                      f"Captured {count}/{target_count} windows.")
                print(f"Packets received: {self.packet_count}, Windows assembled: {self.window_count}")
                break
        
        print(f"\n\nCapture complete! Captured {count} windows.")
        return captured_windows
    
    async def run(self):
        """Main capture routine."""
        print("="*70)
        print("EMG Hardware Verification Capture")
        print("="*70)
        print(f"Target: {WINDOWS_PER_CAPTURE} windows (20 rest → 40 grip → 20 rest)")
        print("="*70)
        
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
            
            # Wait for connection to stabilize and let data start flowing
            print("Waiting for data stream to stabilize...")
            await asyncio.sleep(2.0)
            
            # Check if we're receiving packets
            initial_packets = self.packet_count
            initial_windows = self.window_count
            await asyncio.sleep(1.0)
            packets_received = self.packet_count - initial_packets
            windows_received = self.window_count - initial_windows
            
            if packets_received > 0:
                print(f"✓ Receiving data: {packets_received} packets/sec, {windows_received} windows/sec")
            else:
                print("⚠ Warning: No data packets received. Check device connection.")
            
            # Clear any data that arrived during stabilization
            print("Clearing initial data...")
            cleared_count = 0
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                    cleared_count += 1
                except:
                    break
            if cleared_count > 0:
                print(f"Cleared {cleared_count} initial windows.")
            
            try:
                # Capture windows
                captured_windows = await self.capture_windows(WINDOWS_PER_CAPTURE)
                
                # Plot results
                if captured_windows:
                    plot_emg_capture(captured_windows)
                else:
                    print("No data captured. Cannot display plot.")
            
            except KeyboardInterrupt:
                print("\n\nCapture interrupted by user.")
            finally:
                await client.stop_notify(EMG_CHAR_UUID)
                self.running = False


async def main():
    """Main entry point."""
    capture = EMGCapture()
    try:
        await capture.run()
    except KeyboardInterrupt:
        print("\nProgram stopped by user.")


if __name__ == "__main__":
    asyncio.run(main())

