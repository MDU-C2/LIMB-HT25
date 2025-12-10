# pip install -r requirements.txt
import asyncio
import struct
import csv
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

# --- DATASET SEGMENTATION CONSTANTS ---
WINDOWS_PER_CAPTURE = 80 
# X: Fin del primer bloque de reposo (Ventanas 0 a X-1)
REST_START_WINDOWS = 20 
# Y: Inicio del último bloque de reposo (Ventanas Y a 49)
GESTURE_END_WINDOWS = 60 
# Por lo tanto, el gesto activo va de la ventana 10 a la 39 (30 ventanas)

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
    # BLE HANDLERS (PRODUCER) - Lógica sin cambios
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
            
            # Pedir Label principal (1 o 2) y número de repetición (para el archivo raw)
            label = await asyncio.to_thread(input, "1. Enter Movement Label (1:Holding, 2:Release, q:quit): ")
            if label.lower() in ['q', 'exit']:
                self.running = False
                break
            
            # Validar que el label sea 1 o 2
            if label not in ['1', '2']:
                print("Error: Label debe ser '1' o '2'.")
                continue

            # Obtener número de repetición para el archivo RAW
            timestamp_raw = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_file_label = f"{label}_{timestamp_raw}" # Ej: 1_20251210_091500

            target_count = WINDOWS_PER_CAPTURE
            
            await asyncio.to_thread(input, f">> Get ready. Press ENTER to record {target_count} windows of LABEL {label} <<")
            
            # Limpiar cola de paquetes antiguos
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
                        # Aseguramos que IMU y EMG se capturen para el mismo número de ventana
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

            print(f"\nDone capturing '{label}'. Saving to disk...")
            
            # 1. Guardar datos crudos (RAW) en 'data/no_labels'
            self._save_raw_to_csv(raw_file_label, captured_emg, captured_imu)
            
            # 2. Guardar datos segmentados y etiquetados en 'data/dataset_labels'
            if count_emg == WINDOWS_PER_CAPTURE:
                self._save_segmented_to_csv(label, captured_emg, captured_imu)
            else:
                print("Skipping segmented save: Incomplete capture.")


    def _save_raw_to_csv(self, label, emg_data, imu_data):
        """Guarda los datos tal como fueron capturados en 'data/no_labels'."""
        folder = "data/no_labels"
        if not os.path.exists(folder): os.makedirs(folder)
        
        # 1. Save EMG Data
        if emg_data:
            path = os.path.join(folder, f"{label}_EMG.csv")
            with open(path, 'w', newline='') as f: # 'w' para sobrescribir (una captura por archivo)
                w = csv.writer(f)
                header = ["Raw_Label", "Timestamp"] + [f"v{i}" for i in range(len(emg_data[0])-2)]
                w.writerow(header)
                w.writerows(emg_data)

        # 2. Save IMU Data
        if imu_data:
            path = os.path.join(folder, f"{label}_IMU.csv")
            with open(path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(["Raw_Label", "Timestamp"] + IMU_HEADER)
                w.writerows(imu_data)
        
        print(f"RAW files saved in '{folder}/'.")

    def _save_segmented_to_csv(self, main_label, emg_data, imu_data):
        """Aplica la segmentación (0-GESTO-0) y guarda en 'data/dataset_labels'."""
        
        folder = "data/dataset_labels"
        if not os.path.exists(folder): os.makedirs(folder)

        # 1. Inicializar listas para las clases finales
        idle_emg, gesture_emg = [], []
        idle_imu, gesture_imu = [], []
        
        # 2. Aplicar la segmentación ventana por ventana (EMG)
        for i, row in enumerate(emg_data):
            # El label final es 0 (Idle) o el main_label (1 o 2)
            final_label = '0' 
            if REST_START_WINDOWS <= i < GESTURE_END_WINDOWS:
                final_label = main_label
            
            # [Final_Label, Timestamp, v0, v1...]
            processed_row = [final_label, row[1]] + row[2:]
            
            if final_label == '0':
                idle_emg.append(processed_row)
            else:
                gesture_emg.append(processed_row)

        # 3. Aplicar la segmentación muestra por muestra (IMU)
        # NOTA: Asumimos que la temporalidad es la misma para IMU (10 muestras IMU = 1 ventana EMG)
        imu_samples_per_window = IMU_WINDOW_SAMPLES 

        for i in range(len(imu_data) // imu_samples_per_window):
            # Determinar el label final de la ventana
            final_label = '0' 
            if REST_START_WINDOWS <= i < GESTURE_END_WINDOWS:
                final_label = main_label
                
            # Procesar las 10 filas (muestras) IMU de esta ventana
            start_idx = i * imu_samples_per_window
            end_idx = start_idx + imu_samples_per_window
            
            for row in imu_data[start_idx:end_idx]:
                # [Final_Label, Timestamp, ax, ay...]
                processed_row = [final_label, row[1]] + row[2:]
                
                if final_label == '0':
                    idle_imu.append(processed_row)
                else:
                    gesture_imu.append(processed_row)

        # 4. Guardar los datos segmentados y consolidados (Append 'a')
        
        # --- EMG ---
        self._append_data(folder, '0', idle_emg, "EMG", ["Label", "Timestamp"] + [f"v{i}" for i in range(EMG_WINDOW_SAMPLES)], True)
        self._append_data(folder, main_label, gesture_emg, "EMG", ["Label", "Timestamp"] + [f"v{i}" for i in range(EMG_WINDOW_SAMPLES)], True)
        
        # --- IMU ---
        self._append_data(folder, '0', idle_imu, "IMU", ["Label", "Timestamp"] + IMU_HEADER, False)
        self._append_data(folder, main_label, gesture_imu, "IMU", ["Label", "Timestamp"] + IMU_HEADER, False)

        print(f"SEGMENTED files (Labels 0 and {main_label}) updated in '{folder}/'.")

    def _append_data(self, folder, label, data, sensor_type, header, is_emg):
        """Función auxiliar para añadir datos a los archivos consolidados."""
        if not data: return
        
        # Nombre del archivo: Idle_EMG.csv, Holding_EMG.csv, Release_EMG.csv
        class_name = "Idle" if label == '0' else ("Holding" if label == '1' else "Release")
        path = os.path.join(folder, f"{class_name}_{sensor_type}.csv")
        file_exists = os.path.isfile(path)
        
        with open(path, 'a', newline='') as f:
            w = csv.writer(f)
            if not file_exists:
                w.writerow(header)
            w.writerows(data)

    # ---------------------------------------------------------
    # MAIN ENTRY POINT - Lógica sin cambios
    # ---------------------------------------------------------
    async def start(self):
        # print("Select Mode:")
        # print("1. Realtime Pipeline")
        print("2. Dataset Collection")
        # opt = await asyncio.to_thread(input, "Option: ")
        
        # mode = 'realtime' if opt == '1' else 'dataset'
        mode = 'dataset'
        
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