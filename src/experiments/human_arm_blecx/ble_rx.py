import asyncio
import struct
import csv
import sys
import os
from bleak import BleakScanner, BleakClient
from datetime import datetime 

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

class LimbSensorClient:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.client = None
        self.running = False

        self.emg_buffer = []
        self.imu_buffer = []
        
        self.exp_emg_seq = None
        self.exp_imu_seq = None

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
            # ... (Aquí iría tu lógica de procesamiento) ...
            print(f"\r[RT] Received: {sensor_type} Window (Size: {len(data)})", end="")
            self.queue.task_done()

    # ---------------------------------------------------------
    # MODE 2: DATASET CONSUMER (MODIFICADO)
    # ---------------------------------------------------------
    async def run_dataset(self):
        print("\n--- DATASET COLLECTION MODE ---")
        
        while self.running:
            print("\n" + "-"*40)
            label = await asyncio.to_thread(input, "1. Enter Movement Label (or 'q' to quit): ")
            if label.lower() in ['q', 'exit']:
                self.running = False
                break
            
            num_str = await asyncio.to_thread(input, "2. Number of windows to capture (Enter=10): ")
            target_count = int(num_str) if num_str.isdigit() else 10
            
            await asyncio.to_thread(input, f">> Get ready. Press ENTER to record {target_count} windows of '{label}' <<")
            
            while not self.queue.empty():
                try: self.queue.get_nowait()
                except: break

            print("--- RECORDING ---")
            
            captured_emg = []
            captured_imu = []
            count_emg = 0
            count_imu = 0
            
            while count_emg < target_count or count_imu < target_count:
                try:
                    sensor_type, data = await asyncio.wait_for(self.queue.get(), timeout=2.0)
                    
                    # <--- 2. CAPTURAR FECHA Y HORA EXACTA --->
                    # Formato: Año-Mes-Dia Hora:Min:Seg.Microsegundos
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

                    if sensor_type == 'EMG' and count_emg < target_count:
                        # Estructura: [Label, Timestamp, v0, v1...]
                        captured_emg.append([label, timestamp] + data)
                        count_emg += 1
                        
                    elif sensor_type == 'IMU' and count_imu < target_count:
                        for i in range(0, len(data), 9):
                            sample = data[i:i+9]
                            # Estructura: [Label, Timestamp, ax, ay...]
                            captured_imu.append([label, timestamp] + sample)
                        count_imu += 1
                    
                    print(f"\rProgress: EMG {count_emg}/{target_count} | IMU {count_imu}/{target_count}", end="")
                        
                except asyncio.TimeoutError:
                    print("\nWarning: No data received (Timeout). Checking connection...")
                    break

            print(f"\nDone capturing '{label}'. Saving to disk...")
            self._save_to_csv(label, captured_emg, captured_imu)

    def _save_to_csv(self, label, emg_data, imu_data):
        if not os.path.exists("data"): os.makedirs("data")
        
        # 1. Save EMG Data
        if emg_data:
            path = f"data/{label}_EMG.csv"
            file_exists = os.path.isfile(path)
            
            with open(path, 'a', newline='') as f:
                w = csv.writer(f)
                if not file_exists:
                    # <--- 3. ACTUALIZAR HEADER DEL EMG --->
                    header = ["Label", "Timestamp"] + [f"v{i}" for i in range(len(emg_data[0])-2)]
                    w.writerow(header)
                w.writerows(emg_data)

        # 2. Save IMU Data
        if imu_data:
            path = f"data/{label}_IMU.csv"
            file_exists = os.path.isfile(path)
            
            with open(path, 'a', newline='') as f:
                w = csv.writer(f)
                if not file_exists:
                    # <--- 3. ACTUALIZAR HEADER DEL IMU --->
                    w.writerow(["Label", "Timestamp"] + IMU_HEADER)
                w.writerows(imu_data)
        
        print(f"Files updated in 'data/' folder.")

    # ---------------------------------------------------------
    # MAIN ENTRY POINT
    # ---------------------------------------------------------
    async def start(self):
        print("Select Mode:")
        print("1. Realtime Pipeline")
        print("2. Dataset Collection")
        opt = await asyncio.to_thread(input, "Option: ")
        
        mode = 'realtime' if opt == '1' else 'dataset'
        
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