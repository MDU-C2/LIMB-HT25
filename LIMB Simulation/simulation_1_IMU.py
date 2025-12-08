import pybullet as p
import pybullet_data
import time
import os
import math
import pygame
import numpy as np
from Sensors_Calculations.imu_reader import IMUReader

# =============================================================================
# --- FINAL CONFIGURATION ---
# =============================================================================

PORT_IMU = "COM5"          # Your port
LISSAGE  = 0.20            # Slightly increased for more stability

# --- SENSITIVITY SETTINGS (This is where the magic happens) ---
SCALE_PITCH = 1.2          # Slightly amplified for comfort
SCALE_ROLL  = 1.5          # Reduced compared to your previous "2.5" because atan2 is wider
DEADZONE    = 0.08         # Ignore very small movements (neutral stability)

# DIRECTIONS (1 or -1)
DIR_HAUT_BAS     = 1       
DIR_GAUCHE_DROITE = 1      

# NEW GYRO SETTINGS (LEFT/RIGHT)
GYRO_SENSITIVITY = 0.01    # Reaction speed for left/right (increase to go faster)
GYRO_DEADZONE    = 0.05    # If the rotation speed is low, ignore it (prevents drift)
DIR_GYRO_Z       = 1       # Put -1 if it turns inverted

# MAPPING 
# If left/right doesn't work, try swapping 'y' and 'z' here if your sensor is mounted vertically
AXIS_MAP = {'x': 'x', 'y': 'y', 'z': 'z'}

LISSAGE     = 0.15         # 0.20 was fine, 0.15 is more reactive, 0.10 very smoothed.

current_yaw = 0.0

# =============================================================================

def apply_deadzone(value, threshold):
    """Set the value to 0 if it is too small (prevents drift)."""
    if abs(value) < threshold:
        return 0.0
    # Smooth the recovery after the deadzone
    return value - (math.copysign(threshold, value))

def get_angles_joystick(accel_vec):
    """
    Convert gravity (Accelerometer) into tilt angles (Pitch/Roll).
    Uses atan2 for absolute stability over 360°.
    """
    ax, ay, az = accel_vec

    # 1. PITCH (Up/Down) - Rotation around the Y axis
    # We compare the X axis (front/back) relative to vertical (Y and Z combined)
    # This keeps Pitch correct even if you tilt to the side.
    pitch_rad = math.atan2(ax, math.sqrt(ay**2 + az**2))

    # 2. ROLL (Left/Right) - Rotation around the X axis
    # Major correction here: we use atan2(ay, az).
    # This computes the absolute angle of the gravity vector projected on the YZ plane.
    roll_rad = math.atan2(ay, az)
    
    return pitch_rad, roll_rad

def main():
    # --- 1. PYBULLET SETUP ---
    try:
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        
        project_dir = os.path.abspath(os.path.dirname(__file__))
        urdf_path = os.path.join(project_dir, "arm", "right_arm.urdf")
        robot = p.loadURDF(urdf_path, [0, 0, 1.0], useFixedBase=True)
    except Exception as e:
        print(f"PYBULLET ERROR: {e}")
        return

    # Motor Mapping
    joints = {}
    for i in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, i)
        name = info[1].decode('utf-8')
        if "Shoulder_roty" in name: joints['shoulder_y'] = i # Up/Down
        if "Shoulder_rotx" in name: joints['shoulder_x'] = i # Locked
        if "Shoulder_rotz" in name: joints['shoulder_z'] = i # Left/Right (Base)
        if "Elbow_roty"    in name: joints['elbow_x']    = i # Elbow

    # --- 2. IMU SETUP ---
    reader = IMUReader(port=PORT_IMU, baudrate=115200)
    ema_acc = np.array([0., 0., 9.81]) 
    
    # --- 3. PYGAME HUD ---
    pygame.init()
    width, height = 500, 300
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("ARM CONTROL - [L] Start")
    font = pygame.font.SysFont("Consolas", 18)
    font_big = pygame.font.SysFont("Consolas", 30, bold=True)
    clock = pygame.time.Clock()

    print(">>> SYSTEM READY. Press 'L' to activate.")
    
    is_live = False
    running = True
    
    final_pitch = 0.0
    final_roll = 0.0
    current_yaw = 0.0
    
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT: running = False
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE: running = False
                if e.key == pygame.K_l:
                    if not is_live: 
                        if reader.activate(): 
                            is_live = True
                            current_yaw = 0.0  # <--- ADD THIS LINE (Reset to zero)
                    else: 
                        reader.deactivate(); is_live = False

        if is_live:
            data = reader.get_latest_data_blocking(timeout=0.01)
            if data:
                # --- PART 1: UP / DOWN (We keep the accelerometer, it's very good) ---
                # We get the raw acceleration
                vec = np.array([data.linear_acceleration[0], data.linear_acceleration[1], data.linear_acceleration[2]])
                
                # We smooth (filter) so it won't jitter
                ema_acc = (vec * LISSAGE) + (ema_acc * (1 - LISSAGE))
                
                # Simple tilt calculation
                raw_pitch = math.atan2(ema_acc[0], math.sqrt(ema_acc[1]**2 + ema_acc[2]**2))
                final_pitch = apply_deadzone(raw_pitch, DEADZONE) * SCALE_PITCH * DIR_HAUT_BAS

                # --- PART 2: LEFT / RIGHT (This is where we change everything) ---
                # We read the Z rotation SPEED (gyroscope)
                gyro_z = data.angular_velocity[2] 
                
                # If the speed is very low (noise), we say it's 0
                if abs(gyro_z) < GYRO_DEADZONE:
                    gyro_z = 0.0
                
                # MAGIC FORMULA: New Position = Old Position + (Speed * Sensitivity)
                # You can change 0.03 to go faster or slower
                current_yaw += gyro_z * GYRO_SENSITIVITY * DIR_GYRO_Z
                
                # Safety: We clamp at -90° and +90° (about 1.5 rad) to avoid breaking the arm
                current_yaw = max(-1.5, min(1.5, current_yaw))
                
                # We send that to the motor
                final_roll = current_yaw

        # --- C. MOTOR OUTPUT ---
        # Shoulder Y -> Up / Down
        p.setJointMotorControl2(robot, joints['shoulder_y'], p.POSITION_CONTROL, 
                                targetPosition=final_pitch, force=500)
        
        # Shoulder Z -> Left / Right (Base Rotation)
        # Note: We clamp to avoid the arm doing a weird full 360 turn
        safe_roll = max(-1.5, min(1.5, final_roll))
        p.setJointMotorControl2(robot, joints['shoulder_z'], p.POSITION_CONTROL, 
                                targetPosition=safe_roll, force=500)
        
        # The rest locked
        p.setJointMotorControl2(robot, joints['shoulder_x'], p.POSITION_CONTROL, targetPosition=0)
        p.setJointMotorControl2(robot, joints['elbow_x'], p.POSITION_CONTROL, targetPosition=0)

        p.stepSimulation()

        # --- D. VISUAL DEBUG (PYGAME) ---
        screen.fill((30, 30, 30))
        
        # Screen center
        cx, cy = width // 2, height // 2
        
        # Draw a "Crosshair" to visualize the virtual joystick position
        # Frame
        pygame.draw.rect(screen, (100,100,100), (cx-100, cy-100, 200, 200), 2)
        pygame.draw.line(screen, (50,50,50), (cx, cy-100), (cx, cy+100), 1)
        pygame.draw.line(screen, (50,50,50), (cx-100, cy), (cx+100, cy), 1)

        # Control point (Joystick)
        # We invert Y for display (because screen Y goes down)
        joy_x = cx + int(math.degrees(final_roll) * 2) 
        joy_y = cy - int(math.degrees(final_pitch) * 2)
        
        # Color changes if active
        col = (0, 255, 0) if is_live else (255, 0, 0)
        pygame.draw.circle(screen, col, (joy_x, joy_y), 10)
        
        # Debug text
        txt_status = font_big.render("LIVE" if is_live else "PAUSE (Press L)", True, col)
        screen.blit(txt_status, (20, 20))

        # Display raw values for diagnostics
        str_vals = f"PITCH (U/D): {math.degrees(final_pitch):.1f}° | ROLL (L/R): {math.degrees(final_roll):.1f}°"
        screen.blit(font.render(str_vals, True, (200, 200, 200)), (20, height - 40))

        # Raw sensor gauge (to check if axis Y moves)
        if is_live:
             raw_y_info = f"Sensor Y Raw: {ema_acc[1]:.2f}"
             screen.blit(font.render(raw_y_info, True, (255, 255, 0)), (20, height - 60))

        pygame.display.flip()
        clock.tick(60)

    if is_live: reader.deactivate()
    p.disconnect()
    pygame.quit()

if __name__ == "__main__":
    main()
