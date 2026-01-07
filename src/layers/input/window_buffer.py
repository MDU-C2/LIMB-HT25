import numpy as np
import time

class WindowBuffer:

    def __init__(self, window_size_ms: float = 200.0, overlap_ms: float = 0.0, 
                 emg_frequency: float = 4000.0, imu_frequency: float = 100.0, piezo_frequency: float = 100.0):
        """
        Initialize WindowBuffer with time-based windows and sensor-specific sample counts.
        
        Args:
            window_size_ms: Window duration in milliseconds
            overlap_ms: Overlap between windows in milliseconds (0 = no overlap)
            emg_frequency: EMG sampling frequency in Hz (default: 4000 Hz)
            imu_frequency: IMU sampling frequency in Hz (default: 100 Hz)
            piezo_frequency: Piezo sampling frequency in Hz (default: 100 Hz)
        """
        self.window_size_ms = window_size_ms
        self.overlap_ms = overlap_ms
        
        # Sensor-specific sampling frequencies
        self.emg_frequency = emg_frequency
        self.imu_frequency = imu_frequency
        self.piezo_frequency = piezo_frequency
        
        # Calculate sensor-specific sample counts for the time window
        self.emg_samples_per_window = int((window_size_ms / 1000.0) * emg_frequency)
        self.imu_samples_per_window = int((window_size_ms / 1000.0) * imu_frequency)
        self.piezo_samples_per_window = int((window_size_ms / 1000.0) * piezo_frequency)
        
        # Calculate step sizes (samples to advance for overlapping windows)
        step_size_ms = window_size_ms - overlap_ms
        self.emg_step_size = int((step_size_ms / 1000.0) * emg_frequency)
        self.imu_step_size = int((step_size_ms / 1000.0) * imu_frequency)
        self.piezo_step_size = int((step_size_ms / 1000.0) * piezo_frequency)
        
        # Track last window start positions for overlapping windows (in samples)
        self.last_window_start_emg = 0
        self.last_window_start_imu = 0
        self.last_window_start_piezo = 0
        
        # Calculate max buffer sizes needed for overlapping windows
        # Keep enough samples to support overlapping: window_size + step_size
        self.max_buffer_size_emg = self.emg_samples_per_window + self.emg_step_size
        self.max_buffer_size_imu = self.imu_samples_per_window + self.imu_step_size
        self.max_buffer_size_piezo = self.piezo_samples_per_window + self.piezo_step_size
        
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

        # Keep buffer size manageable: keep enough for overlapping windows
        # Remove oldest samples when buffer exceeds max_buffer_size
        if len(self.emg_buffer) > self.max_buffer_size_emg:
            # Remove samples that are no longer needed for the next window
            samples_to_remove = len(self.emg_buffer) - self.max_buffer_size_emg
            for _ in range(samples_to_remove):
                self.emg_buffer.pop(0)
                # Update last_window_start if we removed samples before it
                if self.last_window_start_emg > 0:
                    self.last_window_start_emg -= 1

    def add_imu(self, data, timestamp):
        """Add a new IMU sample to the buffer."""
        if timestamp is None:
            timestamp = time.time()

        self.imu_buffer.append({
            "data": data, # Make a copy to avoid modifying the original data (needed?)
            "timestamp": timestamp
        })

        # Keep buffer size manageable
        if len(self.imu_buffer) > self.max_buffer_size_imu:
            samples_to_remove = len(self.imu_buffer) - self.max_buffer_size_imu
            for _ in range(samples_to_remove):
                self.imu_buffer.pop(0)
                # Update last_window_start if we removed samples before it
                if self.last_window_start_imu > 0:
                    self.last_window_start_imu -= 1

    def add_piezo(self, data, timestamp):
        if timestamp is None:
            timestamp = time.time()

        self.piezo_buffer.append({
            "data": data,
            "timestamp": timestamp
        })

        if len(self.piezo_buffer) > self.max_buffer_size_piezo:
            samples_to_remove = len(self.piezo_buffer) - self.max_buffer_size_piezo
            for _ in range(samples_to_remove):
                self.piezo_buffer.pop(0)
                # Update last_window_start if we removed samples before it
                if self.last_window_start_piezo > 0:
                    self.last_window_start_piezo -= 1

    def is_full(self) -> bool:
        """
        Check if a new window can be created based on time-based requirements.
        Each sensor needs enough samples for the target time window.
        For overlapping windows, checks if we've collected enough new samples since the last window.
        """
        # Check if each buffer has enough samples for the time window
        if not (len(self.emg_buffer) >= self.emg_samples_per_window and 
                len(self.imu_buffer) >= self.imu_samples_per_window and
                len(self.piezo_buffer) >= self.piezo_samples_per_window):
            return False
        
        # For overlapping windows, check if we've collected step_size new samples for each sensor
        if self.overlap_ms > 0:
            # Check EMG samples since last window start
            emg_samples_since_last = len(self.emg_buffer) - self.last_window_start_emg
            if emg_samples_since_last < self.emg_step_size:
                return False
            
            # Check IMU samples since last window start
            imu_samples_since_last = len(self.imu_buffer) - self.last_window_start_imu
            if imu_samples_since_last < self.imu_step_size:
                return False
            
            # Check Piezo samples since last window start
            piezo_samples_since_last = len(self.piezo_buffer) - self.last_window_start_piezo
            if piezo_samples_since_last < self.piezo_step_size:
                return False
        
        return True

    def get_window(self):
        """
        Get complete window as nparray with sensor-specific sample counts.
        For overlapping windows, returns windows starting at sensor-specific start positions.
        """
        if not self.is_full():
            raise ValueError("Window buffer is not full")

        # Extract windows with sensor-specific sample counts
        if self.overlap_ms > 0:
            # Overlapping windows: extract starting at sensor-specific start positions
            emg_start = self.last_window_start_emg
            emg_end = emg_start + self.emg_samples_per_window
            imu_start = self.last_window_start_imu
            imu_end = imu_start + self.imu_samples_per_window
            piezo_start = self.last_window_start_piezo
            piezo_end = piezo_start + self.piezo_samples_per_window
            
            # Extract window data
            emg_window_samples = self.emg_buffer[emg_start:emg_end]
            imu_window_samples = self.imu_buffer[imu_start:imu_end]
            piezo_window_samples = self.piezo_buffer[piezo_start:piezo_end]
            
            # Update start positions for next window
            self.last_window_start_emg += self.emg_step_size
            self.last_window_start_imu += self.imu_step_size
            self.last_window_start_piezo += self.piezo_step_size
            
            # Extract data arrays
            emg_data = np.array([sample["data"] for sample in emg_window_samples])
            imu_data = np.array([sample["data"] for sample in imu_window_samples])
            piezo_data = np.array([sample["data"] for sample in piezo_window_samples])
            
            # Extract timestamps (use EMG timestamps as reference)
            timestamps = np.array([sample["timestamp"] for sample in emg_window_samples])
        else:
            # Non-overlapping windows: use last N samples for each sensor
            emg_data = np.array([sample["data"] for sample in self.emg_buffer[-self.emg_samples_per_window:]])
            imu_data = np.array([sample["data"] for sample in self.imu_buffer[-self.imu_samples_per_window:]])
            piezo_data = np.array([sample["data"] for sample in self.piezo_buffer[-self.piezo_samples_per_window:]])
            
            # Extract timestamps (use EMG timestamps as reference)
            timestamps = np.array([sample["timestamp"] for sample in self.emg_buffer[-self.emg_samples_per_window:]])

        t_start = timestamps[0] if len(timestamps) > 0 else time.time()
        t_end = timestamps[-1] if len(timestamps) > 0 else time.time()

        # Ensure the correct shapes
        # EMG: should be (emg_samples_per_window, num_channels)
        if emg_data.ndim == 1:
            emg_data = emg_data.reshape(-1, 1)
        elif emg_data.ndim == 2 and emg_data.shape[0] != self.emg_samples_per_window:
            emg_data = emg_data[:self.emg_samples_per_window]

        # IMU: should be (imu_samples_per_window, 6)
        if imu_data.ndim == 1:
            imu_data = np.tile(imu_data, (self.imu_samples_per_window, 1))
        elif imu_data.ndim == 2:
            if imu_data.shape[0] != self.imu_samples_per_window:
                imu_data = imu_data[:self.imu_samples_per_window]
            if imu_data.shape[1] != 6:
                # Pad or truncate to 6 values
                if imu_data.shape[1] < 6:
                    padding = np.zeros((imu_data.shape[0], 6 - imu_data.shape[1]))
                    imu_data = np.hstack([imu_data, padding])
                else:
                    imu_data = imu_data[:, :6]

        # Piezo: should be (piezo_samples_per_window,)
        if piezo_data.ndim == 0:
            # Scalar value, convert to 1D array
            piezo_data = np.array([piezo_data])
        elif piezo_data.ndim == 1:
            # Ensure correct length
            if len(piezo_data) != self.piezo_samples_per_window:
                if len(piezo_data) > self.piezo_samples_per_window:
                    piezo_data = piezo_data[:self.piezo_samples_per_window]
                else:
                    piezo_data = np.pad(piezo_data, (0, self.piezo_samples_per_window - len(piezo_data)), 'constant')
        elif piezo_data.ndim == 2:
            # If 2D, flatten or take first column
            if piezo_data.shape[0] != self.piezo_samples_per_window:
                piezo_data = piezo_data[:self.piezo_samples_per_window]
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
        self.last_window_start_emg = 0
        self.last_window_start_imu = 0
        self.last_window_start_piezo = 0

