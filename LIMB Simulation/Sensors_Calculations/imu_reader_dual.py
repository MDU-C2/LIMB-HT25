import serial
import json
import numpy as np
import time
import threading
from queue import Queue, Empty

class DualIMUReader:
    def __init__(self, port, baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.running = False
        self.data_queue = Queue(maxsize=1) # On garde seulement le paquet le plus récent

    def activate(self):
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=0.1)
            self.running = True
            threading.Thread(target=self._read_loop, daemon=True).start()
            print(f"✅ Reader connecté sur {self.port}")
            return True
        except Exception as e:
            print(f"❌ Erreur connexion: {e}")
            return False

    def deactivate(self):
        self.running = False
        if self.serial_conn: self.serial_conn.close()

    def get_latest(self):
        try:
            return self.data_queue.get_nowait()
        except Empty:
            return None

    def _read_loop(self):
        buffer = ""
        while self.running:
            if self.serial_conn.in_waiting:
                try:
                    chunk = self.serial_conn.read(self.serial_conn.in_waiting).decode('utf-8', errors='ignore')
                    buffer += chunk
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        self._parse_json(line.strip())
                except Exception:
                    pass
            time.sleep(0.005)

    def _parse_json(self, line):
        if not line.startswith('{'): return
        try:
            data = json.loads(line)
            # On vérifie qu'on a bien les deux capteurs
            if 'imu1' in data and 'imu2' in data:
                parsed = {}
                # IMU 1 (BRAS) -> C'est lui qui dirige l'épaule
                parsed['accel_1'] = np.array([data['imu1']['accel']['x'], data['imu1']['accel']['y'], data['imu1']['accel']['z']])
                parsed['gyro_1']  = np.array([data['imu1']['gyro']['x'],  data['imu1']['gyro']['y'],  data['imu1']['gyro']['z']])
                
                # IMU 2 (AVANT-BRAS) -> Sert pour le coude
                parsed['accel_2'] = np.array([data['imu2']['accel']['x'], data['imu2']['accel']['y'], data['imu2']['accel']['z']])
                parsed['gyro_2']  = np.array([data['imu2']['gyro']['x'],  data['imu2']['gyro']['y'],  data['imu2']['gyro']['z']])
                
                # On pousse dans la queue
                if self.data_queue.full(): self.data_queue.get_nowait()
                self.data_queue.put(parsed)
        except:
            pass