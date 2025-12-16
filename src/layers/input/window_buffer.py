import numpy as np
import time

class WindowBuffer:

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.emg_buffer = [] # List of EMG samples
        self.imu_buffer = [] # List of IMU samples
        self.piezo_buffer = [] # List of PIEZO samples

    def add_emg(self, samples, timestamp):
        """Add a new EMG sample to the buffer."""
        if timestamp is None:
            timestamp = time.time()

        self.emg_buffer.append({
            "data": samples,
            "timestamp": timestamp
        })

        # Keep only the last window_size samples (sliding window)
        if len(self.emg_buffer) > self.window_size:
            self.emg_buffer.pop(0) # Remove the oldest

    def add_imu(self, data, timestamp):
        """Add a new IMU sample to the buffer."""
        if timestamp is None:
            timestamp = time.time()

        self.imu_buffer.append({
            "data": data, # Make a copy to avoid modifying the original data (needed?)
            "timestamp": timestamp
        })

        # Keep only the last window_size samples (sliding window)
        if len(self.imu_buffer) > self.window_size:
            self.imu_buffer.pop(0) # Remove the oldest

    def add_piezo(self, data, timestamp):
        if timestamp is None:
            timestamp = time.time()

        self.piezo_buffer.append({
            "data": data,
            "timestamp": timestamp
        })

        if len(self.piezo_buffer) > self.window_size:
            self.piezo_buffer.pop(0) # Remove the oldest

    def is_full(self) -> bool:
        return (len(self.emg_buffer) >= self.window_size and 
                len(self.imu_buffer) >= self.window_size and
                len(self.piezo_buffer) >= self.window_size)

    def get_window(self):
        """Get complete window as nparray"""
        if not self.is_full():
            raise ValueError("Window buffer is not full")

        # Extract data arrays
        emg_data = np.array([sample["data"] for sample in self.emg_buffer])
        imu_data = np.array([sample["data"] for sample in self.imu_buffer])
        piezo_data = np.array([sample["data"] for sample in self.piezo_buffer])

        # Extract timestamps
        timestamps = np.array([sample["timestamp"] for sample in self.emg_buffer])
        t_start = timestamps[0] if len(timestamps) > 0 else time.time()
        t_end = timestamps[-1] if len(timestamps) > 0 else time.time()

        # Ensure the correct shapes
        # EMG
        if emg_data.ndim == 1:
            emg_data = emg_data.reshape(-1, 1)
        elif emg_data.ndim == 2 and emg_data.shape[0] != self.window_size:
            emg_data = emg_data[:self.window_size] # Take first window_size samples

        # IMU
        if imu_data.ndim == 1:
            imu_data = np.tile(imu_data, (self.window_size, 1))
        elif imu_data.ndim == 2:
            if imu_data.shape[0] != self.window_size:
                imu_data = imu_data[:self.window_size] # Take first window_size samples
            if imu_data.shape[1] != 6:
                # Pad or truncate to 6 values
                if imu_data.shape[1] < 6:
                    padding = np.zeros((imu_data.shape[0], 6 - imu_data.shape[1]))
                    imu_data = np.hstack([imu_data, padding])
                else:
                    imu_data = imu_data[:, :6] # Truncate to 6 values

        # Piezo
        if piezo_data.ndim == 0:
            # Scalar value, convert to 1D array
            piezo_data = np.array([piezo_data])
        elif piezo_data.ndim == 1:
            # Ensure correct length
            if len(piezo_data) != self.window_size:
                piezo_data = piezo_data[:self.window_size] if len(piezo_data) > self.window_size else np.pad(piezo_data, (0, self.window_size - len(piezo_data)), 'constant')
        elif piezo_data.ndim == 2:
            # If 2D, flatten or take first column
            if piezo_data.shape[0] != self.window_size:
                piezo_data = piezo_data[:self.window_size]
            if piezo_data.shape[1] == 1:
                piezo_data = piezo_data.flatten()
            else:
                # Take first column if multiple columns
                piezo_data = piezo_data[:, 0]

        return {
            "emg": emg_data, 
            "imu": imu_data, 
            "piezo": piezo_data,
            "timestamps": timestamps, 
            "timestamp_start": t_start, 
            "timestamp_end": t_end
        }

    def get_size(self):
        return {
            "emg": len(self.emg_buffer), 
            "imu": len(self.imu_buffer),
            "piezo": len(self.piezo_buffer)
        }
    
    def clear(self):
        self.emg_buffer.clear()
        self.imu_buffer.clear()
        self.piezo_buffer.clear()

