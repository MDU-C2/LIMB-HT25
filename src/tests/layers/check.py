


import numpy as np


window_size = 5
channels = 2

print("--------------------------------")
emg_data = np.random.randn(window_size, channels)

print(f"EMG data: {emg_data}")
print(f"EMG data shape: {emg_data.shape}")

print("--------------------------------")
emg_transposed = emg_data.T

print(f"EMG transposed: {emg_transposed}")
print(f"EMG transposed shape: {emg_transposed.shape}")
print("--------------------------------")
emg_flatten = emg_transposed.flatten()

print(f"EMG flattened: {emg_flatten}")
print(f"EMG flattened shape: {emg_flatten.shape}")

print("--------------------------------")
emg_reshaped = emg_flatten.reshape(-1, 1)

print(f"EMG reshaped: {emg_reshaped}")
print(f"EMG reshaped shape: {emg_reshaped.shape}")

