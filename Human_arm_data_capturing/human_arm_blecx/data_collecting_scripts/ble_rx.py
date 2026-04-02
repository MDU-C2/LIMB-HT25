# pip install -r requirements.txt
import asyncio
import struct
import csv
import os
from bleak import BleakScanner, BleakClient
from datetime import datetime 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION ---
TARGET_NAME = "LIMBServer" 
EMG_CHAR_UUID = "24011525-1212-efde-1523-785feabcd122"
IMU_CHAR_UUID = "25011525-1212-efde-1523-785feabcd122"

# --- ASSEMBLY PARAMETERS ---
CHUNKS_PER_WINDOW = 10

# EMG Settings
EMG_SAMPLES_PER_CHUNK = 40
EMG_WINDOW_SAMPLES = CHUNKS_PER_WINDOW * EMG_SAMPLES_PER_CHUNK 
EMG_PACKET_FORMAT = f'<{EMG_SAMPLES_PER_CHUNK}H I' 

# IMU Settings
IMU_SAMPLES_PER_CHUNK = 1
IMU_WINDOW_SAMPLES = CHUNKS_PER_WINDOW * IMU_SAMPLES_PER_CHUNK 
IMU_PACKET_FORMAT = f'<9f I' 
IMU_HEADER = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z', 'temp', 'pitch', 'roll']

# --- DATASET SEGMENTATION CONSTANTS ---
WINDOWS_PER_CAPTURE = 80 
# starting window of the gesture
REST_START_WINDOWS = 20 
# ending window of the gesture
GESTURE_END_WINDOWS = 60 


def plot_emg_capture(captured_emg_list, raw_label, main_label):
    """
    Visualize all the windows of a single capture.

    captured_emg_list: list of windows
    raw_label:RAW identifier of the capture (ej: 1_20251210_091500).
    param main_label: main label of the capture (1: Holding o 2: Resting).
    """
    if not captured_emg_list:
        print("Error: empty capture")
        return

    print(f"\n--- Creating figure for EMG capture {raw_label}")
    
    # 1. create a dataframe with windows
    # Columns 0 y 1 are Raw_Label and Timestamp
    # Remaining columns are data
    
    data_columns = [f"v{i}" for i in range(EMG_WINDOW_SAMPLES)]
    header = ["Raw_Label", "Timestamp"] + data_columns
    
    df = pd.DataFrame(captured_emg_list, columns=header)
    
    # 2. flat the dataframe
    # Exclude non numerical columns
    numeric_data = df.drop(columns=df.columns[[0, 1]], errors='ignore') 
    numeric_data = numeric_data.select_dtypes(include=[np.number])
    
    y_values = numeric_data.values.flatten()
    x_values = np.arange(len(y_values))
    window_size = EMG_WINDOW_SAMPLES
    
    # 3. Create the figure
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(x_values, y_values, label='EMG signal', color='#1f77b4', linewidth=0.8)
    ax.set_ylabel("ADC value")
    
    title_text = f"RAW EMG Value: {raw_label}"
    ax.set_title(title_text)
    ax.set_xlabel("Samples")
    
    # 4. vertical lines for each window
    for x in range(window_size, len(x_values), window_size):
        ax.axvline(x=x, color='red', linestyle='--', linewidth=0.5, alpha=0.4)
    
    # 5. draw rest - gesture - rest zones
    
    # Initial rest zone (Label 2)
    start_rest_end_x = REST_START_WINDOWS * window_size
    ax.axvspan(0, start_rest_end_x, alpha=0.2, color='green', label='Initial rest (2)')
    
    # Main gesture (Label 1)
    gesture_end_x = GESTURE_END_WINDOWS * window_size
    ax.axvspan(start_rest_end_x, gesture_end_x, alpha=0.2, color='orange', label=f'Gesture ({main_label})')
    
    # Zona de Reposo Final (Label 2)
    end_capture_x = WINDOWS_PER_CAPTURE * window_size
    ax.axvspan(gesture_end_x, end_capture_x, alpha=0.2, color='green', label='Final rest (2)')

    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right')
    ax.set_xlim(0, len(x_values))
    plt.tight_layout()
    
    plt.show()

def count_csv(ruta_carpeta):
    """Counts the number of CSV files in a folder"""

    if not os.path.exists(ruta_carpeta):
        print(f"Error: path doesn't exists")
        return 0

    # 2. List all the elements in the folder
    elementos = os.listdir(ruta_carpeta)
    
    contador_csv = 0
    
    # 3. Count '.csv' ending files
    for elemento in elementos:        
        ruta_completa = os.path.join(ruta_carpeta, elemento)
        if os.path.isfile(ruta_completa) and elemento.lower().endswith('.csv'):
            contador_csv += 1
            
    return contador_csv   


class LimbSensorClient:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.client = None
        self.running = False

        self.emg_buffer = []
        self.imu_buffer = []
        
        self.exp_emg_seq = None
        self.exp_imu_seq = None

        self.subject_name = None

    # ---------------------------------------------------------
    # BLE HANDLERS (PRODUCER)
    # ---------------------------------------------------------
    async def _handle_emg(self, data):
        try:
            *payload, seq = struct.unpack(EMG_PACKET_FORMAT, data)
        except Exception:
            self.exp_emg_seq = None; self.emg_buffer.clear(); return

        if self.exp_emg_seq is None:
            if seq % CHUNKS_PER_WINDOW == 0:
                self.emg_buffer = list(payload)
                self.exp_emg_seq = seq + 1
        elif seq == self.exp_emg_seq:
            self.emg_buffer.extend(payload)
            self.exp_emg_seq += 1
            
            if len(self.emg_buffer) == EMG_WINDOW_SAMPLES:
                await self.queue.put(('EMG', list(self.emg_buffer)))
                self.exp_emg_seq = None 
        else:
            self.exp_emg_seq = None; self.emg_buffer.clear()

    async def _handle_imu(self, data):
        try:
            *payload, seq = struct.unpack(IMU_PACKET_FORMAT, data)
        except Exception:
            self.exp_imu_seq = None; self.imu_buffer.clear(); return

        if self.exp_imu_seq is None:
            if seq % CHUNKS_PER_WINDOW == 0:
                self.imu_buffer = list(payload)
                self.exp_imu_seq = seq + 1
        elif seq == self.exp_imu_seq:
            self.imu_buffer.extend(payload)
            self.exp_imu_seq += 1
            
            if len(self.imu_buffer) == (IMU_WINDOW_SAMPLES * 9):
                await self.queue.put(('IMU', list(self.imu_buffer)))
                self.exp_imu_seq = None
        else:
            self.exp_imu_seq = None; self.imu_buffer.clear()

    async def notification_handler(self, sender, data):
        if str(sender.uuid) == EMG_CHAR_UUID:
            await self._handle_emg(data)
        elif str(sender.uuid) == IMU_CHAR_UUID:
            await self._handle_imu(data)

    # ---------------------------------------------------------
    # MODE 1: REAL TIME CONSUMER
    # ---------------------------------------------------------
    async def run_realtime(self):
        print("\n--- REAL TIME MODE STARTED (Ctrl+C to stop) ---")
        print("Waiting for data stream...")
        
        while self.running:
            sensor_type, data = await self.queue.get()
            # Proccesing layer
            print(f"\r[RT] Received: {sensor_type} Window (Size: {len(data)})", end="")
            self.queue.task_done()

    # ---------------------------------------------------------
    # MODE 2: DATASET CONSUMER 
    # ---------------------------------------------------------
    async def run_dataset(self):
        print("\n--- DATASET COLLECTION MODE ---")
        
        while self.running:
            print("\n" + "-"*40)
            
            label = await asyncio.to_thread(input, "Enter Movement Label (1:Holding, 2:rest, q:quit): ")
            if label.lower() in ['q', 'exit']:
                self.running = False
                break

            if label not in ['1', '2']:
                print("Error: wrong label")
                continue
            
            #Create filename
            timestamp_raw = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_file_label = f"{label}_{timestamp_raw}" # Ej: 1_20251210_091500

            target_count = WINDOWS_PER_CAPTURE
            
            await asyncio.to_thread(input, f">> Get ready. Press ENTER to record {target_count} windows <<")
            
            while not self.queue.empty():
                try: self.queue.get_nowait()
                except: break

            print("--- RECORDING ---")
                
            captured_emg = []
            captured_imu = []
            count_emg = 0
            count_imu = 0
            
            while count_emg < target_count:
                try:
                    sensor_type, data = await asyncio.wait_for(self.queue.get(), timeout=2.0)
                    
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

                    if sensor_type == 'EMG' and count_emg < target_count:
                        # [Label_Original, Timestamp, v0, v1...]
                        captured_emg.append([raw_file_label, timestamp] + data)
                        count_emg += 1
                        
                    elif sensor_type == 'IMU' and count_imu < target_count:
                        if count_imu < count_emg: 
                            for i in range(0, len(data), 9):
                                sample = data[i:i+9]
                                # [Label_Original, Timestamp, ax, ay...]
                                captured_imu.append([raw_file_label, timestamp] + sample)
                            count_imu += 1
                        
                    print(f"\rProgress: EMG {count_emg}/{target_count} | IMU {count_imu}/{target_count}", end="")
                    
                except asyncio.TimeoutError:
                    print("\nWarning: No data received (Timeout). Checking connection...")
                    break

            print(f"\nDone capturing '{label}'.")

            plot_emg_capture(captured_emg, raw_file_label, label)
     
            self._save_raw_to_csv(raw_file_label, captured_emg, captured_imu, True) 

    def _save_raw_to_csv(self, label, emg_data, imu_data, save):
        """Save capture in folder 'data/raw_data'."""
        
        base_folder = "data/raw_data"
        
        # 1. Save EMG Data
        if emg_data:
            folder = os.path.join(base_folder, self.subject_name, "EMG")
            if save:
                if not os.path.exists(folder): os.makedirs(folder)
                path = os.path.join(folder, f"{label}.csv")
                with open(path, 'w', newline='') as f:
                    w = csv.writer(f)
                    header = ["Raw_Label", "Timestamp"] + [f"v{i}" for i in range(len(emg_data[0])-2)]
                    w.writerow(header)
                    w.writerows(emg_data)
                print(f"RAW files saved in '{self.subject_name}/EMG'.")

            files_count = count_csv(folder)
            print(f"Dir: '{self.subject_name}/EMG' contains {files_count} CSV files.")

        # 2. Save IMU Data
        if imu_data:
            folder = os.path.join(base_folder, self.subject_name, "IMU")
            if save:    
                if not os.path.exists(folder): os.makedirs(folder)
                path = os.path.join(folder, f"{label}.csv")
                with open(path, 'w', newline='') as f:
                    w = csv.writer(f)
                    w.writerow(["Raw_Label", "Timestamp"] + IMU_HEADER)
                    w.writerows(imu_data)
                print(f"RAW files saved in '{self.subject_name}/IMU.")

            files_count = count_csv(folder)
            print(f"Dir: '{self.subject_name}/IMU contains {files_count} CSV files.")

    # ---------------------------------------------------------
    # MAIN ENTRY POINT
    # ---------------------------------------------------------
    async def start(self):

        # print("Select Mode:")
        # print("1. Realtime Pipeline")
        print("2. Dataset Collection")
        # opt = await asyncio.to_thread(input, "Option: ")
        
        # mode = 'realtime' if opt == '1' else 'dataset'
        mode = 'dataset'

        self.subject_name = await asyncio.to_thread(input, "Enter Subject Name (e.g., S0, S1): ")
        if not self.subject_name:
            print("Error: Subject name cannot be empty.")
            return
        
        print(f"Scanning for device '{TARGET_NAME}'...")
        device = await BleakScanner.find_device_by_name(TARGET_NAME)
        
        if not device:
            print(f"Error: Device '{TARGET_NAME}' not found.")
            return

        print(f"Connecting to {device.address}...")
        async with BleakClient(device) as client:
            self.client = client
            await client.start_notify(EMG_CHAR_UUID, self.notification_handler)
            await client.start_notify(IMU_CHAR_UUID, self.notification_handler)
            
            self.running = True
            if mode == 'realtime':
                await self.run_realtime()
            else:
                await self.run_dataset()
                
            await client.stop_notify(EMG_CHAR_UUID)
            await client.stop_notify(IMU_CHAR_UUID)

if __name__ == "__main__":
    client = LimbSensorClient()
    try:
        asyncio.run(client.start())
    except KeyboardInterrupt:
        print("\nProgram stopped by user.")