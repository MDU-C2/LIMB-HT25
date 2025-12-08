import pybullet as p
import pybullet_data
import time
import os
import math
import pygame
import numpy as np
import serial
import json
import threading
from queue import Queue, Empty

# =============================================================================
# --- 1. CONFIGURATION ---
# =============================================================================
PORT_BRAS      = "COM5"   
PORT_AVANTBRAS = "COM6"   
BAUDRATE       = 115200

# Parameters
LISSAGE          = 0.15     
GYRO_SENSITIVITY = 0.015    
GYRO_DEADZONE    = 0.05     

# SENSE (Change 1 to -1 if it moves the other way)
# If when you lift the arm, the robot lowers it, change DIR_EPAULE_Y
DIR_EPAULE_Y = 1   
DIR_EPAULE_Z = 1   
DIR_COUDE    = 1   

MAX_FORCE = 150 

# Limits to avoid breaking the arm
LIMIT_SHOULDER_Y = (-90, 160)
LIMIT_SHOULDER_Z = (-90, 90)
LIMIT_ELBOW      = (-120, 0) # Elbow 

def clamp(val, min_val, max_val):
    return max(min_val, min(max_val, val))

# =============================================================================
# --- 2. USB READER ---
# =============================================================================
class USBReader:
    def __init__(self, name, port, baudrate):
        self.name = name
        self.port = port
        self.baudrate = baudrate
        self.running = False
        self.serial_conn = None
        self.data_queue = Queue(maxsize=1)
        self.reconnecting = False

    def activate(self):
        self.running = True
        threading.Thread(target=self._read_loop, daemon=True).start()
        return True

    def _connect(self):
        try:
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.close()
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=0.1)
            print(f"[INFO] {self.name} connected on {self.port}")
            return True
        except Exception as e:
            # We don't spam the error, we just show it once if needed
            return False

    def deactivate(self):
        self.running = False
        if self.serial_conn: self.serial_conn.close()

    def get_latest(self):
        try: return self.data_queue.get_nowait()
        except Empty: return None

    def _read_loop(self):
        while self.running:
            # 1. Try to connect if not connected
            if self.serial_conn is None or not self.serial_conn.is_open:
                if not self.reconnecting:
                    print(f"[{self.name}] Searching for connection...")
                    self.reconnecting = True
                
                if self._connect():
                    self.reconnecting = False
                else:
                    time.sleep(1.0) # We wait 1s before retrying
                    continue

            # 2. Reading
            try:
                if self.serial_conn.in_waiting:
                    chunk = self.serial_conn.read(self.serial_conn.in_waiting).decode('utf-8', errors='ignore')
                    # ... (rest of parsing unchanged) ...
                    # If you want, I can put the full parsing back here
                    if '\n' in chunk:
                        lines = chunk.split('\n')
                        for line in lines:
                            self._parse(line.strip())
                else:
                    time.sleep(0.005)
            except Exception as e:
                print(f"[{self.name}] Connection lost! Attempting reconnection...")
                self.serial_conn.close()
                self.serial_conn = None
                time.sleep(0.5)

    def _parse(self, line):
        if not line.startswith('{'): return
        try:
            data = json.loads(line)
            if 'accel' in data:
                res = {}
                res['acc'] = np.array([data['accel']['x'], data['accel']['y'], data['accel']['z']])
                res['gyr'] = data['gyro']['z']
                if self.data_queue.full(): self.data_queue.get_nowait()
                self.data_queue.put(res)
        except: pass

# =============================================================================
# --- 3. MAIN ---
# =============================================================================
def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    
    project_dir = os.path.abspath(os.path.dirname(__file__))
    urdf_path = os.path.join(project_dir, "arm", "right_arm.urdf")
    robot = p.loadURDF(urdf_path, [0, 0, 1.0], useFixedBase=True)

    target_joints = {
        "jRightShoulder_roty": "sh_y",
        "jRightShoulder_rotz": "sh_z",
        "jRightElbow_roty":    "elb"
    }
    joints = {}
    for i in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, i)
        name = info[1].decode('utf-8')
        if name in target_joints: joints[target_joints[name]] = i
        else: p.setJointMotorControl2(robot, i, p.POSITION_CONTROL, targetPosition=0, force=MAX_FORCE)

    reader_bras = USBReader("BRAS", PORT_BRAS, BAUDRATE)
    reader_avt  = USBReader("AVT-BRAS", PORT_AVANTBRAS, BAUDRATE)
    
    ema_acc1 = np.array([0., 0., 9.81])
    ema_acc2 = np.array([0., 0., 9.81])
    current_yaw = 0.0
    
    offset_bras = 0.0
    offset_avt  = 0.0
    need_calibration = False 

    pygame.init()
    screen = pygame.display.set_mode((400, 350))
    pygame.display.set_caption("CONTROLE Y-AXIS ONLY")
    font = pygame.font.SysFont("Arial", 16)
    clock = pygame.time.Clock()

    running = True
    is_live = False
    
    # Values for display
    raw_bras_deg = 0.0
    raw_avt_deg = 0.0
    final_bras_deg = 0.0
    final_coude_deg = 0.0

    print(">>> READY. Press 'L'.")

    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT: running = False
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_l:
                    if not is_live:
                        if reader_bras.activate() and reader_avt.activate(): 
                            is_live = True; current_yaw = 0.0; need_calibration = True
                        else: reader_bras.deactivate(); reader_avt.deactivate()
                    else: reader_bras.deactivate(); reader_avt.deactivate(); is_live = False

        if is_live:
            d1 = reader_bras.get_latest()
            d2 = reader_avt.get_latest()
            
            if d1:
                ema_acc1 = (d1['acc'] * LISSAGE) + (ema_acc1 * (1 - LISSAGE))
                gz = d1['gyr']
                if abs(gz) > GYRO_DEADZONE: current_yaw += gz * GYRO_SENSITIVITY * DIR_EPAULE_Z
            
            if d2:
                ema_acc2 = (d2['acc'] * LISSAGE) + (ema_acc2 * (1 - LISSAGE))

            # --- THE MAGIC FORMULA (Y vs Z) ---
            # atan2(Y, Z) gives the forward/backward tilt angle.
            # We completely ignore X.
            
            # Arm Angle
            # Note: If your sensor is mounted vertically, you may need to swap [1] and [2]
            # Here we assume: [1]=Y (Local Up/Down), [2]=Z (Gravity)
            raw_angle_bras = math.atan2(ema_acc1[1], ema_acc1[2])

            # Forearm Angle
            raw_angle_avt  = math.atan2(ema_acc2[1], ema_acc2[2])

            if need_calibration:
                offset_bras = raw_angle_bras
                offset_avt  = raw_angle_avt
                need_calibration = False
                print(">>> CALIBRATED !")

            # --- FINAL CALCULATIONS ---
            
            # Shoulder
            angle_bras_final = (raw_angle_bras - offset_bras) * DIR_EPAULE_Y
            
            # Absolute Forearm
            angle_avt_final = (raw_angle_avt - offset_avt)

            # Elbow = Difference
            # If DIR_COUDE = 1: The more the forearm rises relative to the arm, the more it bends
            angle_coude_final = (angle_avt_final - angle_bras_final) * DIR_COUDE
            
            # --- CLAMPING ---
            current_yaw = clamp(current_yaw, math.radians(LIMIT_SHOULDER_Z[0]), math.radians(LIMIT_SHOULDER_Z[1]))
            angle_bras_final = clamp(angle_bras_final, math.radians(LIMIT_SHOULDER_Y[0]), math.radians(LIMIT_SHOULDER_Y[1]))
            angle_coude_final = clamp(angle_coude_final, math.radians(LIMIT_ELBOW[0]), math.radians(LIMIT_ELBOW[1]))

            # --- MOTORS ---
            p.setJointMotorControl2(robot, joints['sh_z'], p.POSITION_CONTROL, targetPosition=current_yaw, force=MAX_FORCE)
            p.setJointMotorControl2(robot, joints['sh_y'], p.POSITION_CONTROL, targetPosition=angle_bras_final, force=MAX_FORCE)
            p.setJointMotorControl2(robot, joints['elb'],  p.POSITION_CONTROL, targetPosition=angle_coude_final, force=MAX_FORCE)

            # Update Display
            raw_bras_deg = math.degrees(raw_angle_bras)
            raw_avt_deg = math.degrees(raw_angle_avt)
            final_bras_deg = math.degrees(angle_bras_final)
            final_coude_deg = math.degrees(angle_coude_final)

        p.stepSimulation()

        # DIAGNOSTIC UI
        screen.fill((20,20,20))
        c = (0,255,0) if is_live else (255,50,50)
        screen.blit(font.render(f"STATE: {'LIVE' if is_live else 'PAUSE (L)'}", True, c), (10,10))
        
        # Left Column : Raw Values (Is the IMU reacting?)
        screen.blit(font.render("--- RAW (Sensors) ---", True, (150,150,150)), (10,50))
        screen.blit(font.render(f"Arm Y/Z: {raw_bras_deg:.1f}", True, (200,200,200)), (10,70))
        screen.blit(font.render(f"Forearm Y/Z: {raw_avt_deg:.1f}", True, (200,200,200)), (10,90))
        
        # Right Column : Robot Result
        screen.blit(font.render("--- ROBOT ---", True, (150,150,150)), (200,50))
        screen.blit(font.render(f"Shoulder: {final_bras_deg:.1f}", True, (255,255,255)), (200,70))
        screen.blit(font.render(f"Elbow:  {final_coude_deg:.1f}", True, (255,255,0)), (200,90))

        pygame.display.flip()
        clock.tick(60)

    reader_bras.deactivate()
    reader_avt.deactivate()
    p.disconnect()
    pygame.quit()

if __name__ == "__main__":
    main()