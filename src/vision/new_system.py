# To do : pip install -r requirements.txt
# python new_system.py

import cv2
import depthai as dai
import time
import numpy as np
import math
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# --- CONFIGURATION ---
# =============================================================================
BLOB_PATH = "cup.blob"
LABEL_NAME = "Cup"

# --- PHYSICAL OFFSETS (TO ADJUST) ---
# Gap between camera and shoulder
OFFSET_CAM_X = 0.0    # Lateral
OFFSET_CAM_Y = 0.0    # Depth
OFFSET_CAM_Z = 0.20   # Height

# Fixed table height (arm at 41cm = 0.41m)
TABLE_HEIGHT = 0.41  # Fixed height where the cup is always placed

# =============================================================================
# --- ARM KINEMATICS CONSTANTS ---
# =============================================================================

# Shoulder position in the world (x, y, z)
# Corresponds to line 231 of simulation.py (and URDF setup)
SHOULDER_POS_BASE = (0.0, 0.0, 0.0)

# Segment lengths (based on right_arm.urdf + your measurements)
L1_ARM = 0.305        # Shoulder -> Elbow
L2_FOREARM = 0.310  # Elbow -> Wrist
L3_HAND_GRIP = 0.120   # Distance Wrist -> Gripper Center

# Effective total length of the 2nd segment
L_TOTAL_FOREARM = L2_FOREARM + L3_HAND_GRIP # 0.430 m

# --- REACH LIMITS (SHELL) ---
# R_MAX : Arm fully extended (Straight line)
R_MAX = L1_ARM + L_TOTAL_FOREARM # = 0.735 m

# R_MIN : Arm folded to maximum (Dead Zone)
# HARDWARE CONSTRAINT: The elbow is mechanically limited to -90 degrees.
# This means the minimum internal angle is 90 degrees.
# The minimum distance is therefore the hypotenuse of the right triangle formed by the two segments.
R_MIN = math.sqrt(L1_ARM**2 + L_TOTAL_FOREARM**2) # = 0.527 m

# --- ANGULAR LIMITS (BRAIN) ---
# Must correspond exactly to the Shoulder class in simulation.py
AZIMUT_MIN = -95.0   # Z Plane (Left/Right)
AZIMUT_MAX = 50.0
ELEVATION_MIN = -85.0  # Y Plane (Up/Down)
ELEVATION_MAX = 150.0

# NEMA 17 Specs
STEPS_PER_REV = 200          # Standard 1.8 degrees per step
MICROSTEPS = 16              # Driver setting (e.g., A4988 / DRV8825)

# Mechanical Reduction (Gear Ratio)
# !!! CRITICAL: UPDATE THESE VALUES TO MATCH THE REAL HARDWARE (Need the value once the arm is fully done) !!!
# Example: 1.0 = Direct Drive, 5.0 = 5:1 Planetary Gearbox
RATIO_SHOULDER = 5.0         
RATIO_ELBOW = 1.0

# Servo Constants
SERVO_PWM_MIN = 1000  # Microseconds for -90 degrees (approx)
SERVO_PWM_MAX = 2000  # Microseconds for +90 degrees (approx)
SERVO_PWM_CENTER = 1500 # Neutral (0 degrees)
SERVO_ANGLE_RANGE = 180 # Total range in degrees

# =============================================================================
# --- ARM KINEMATICS FUNCTIONS ---
# =============================================================================

def is_reachable(cup_pos: tuple[float, float, float]) -> tuple[bool, str]:
    """
    Checks if the cup is physically reachable by the robot.
    Takes into account the action sphere and joint limits.
    Returns: (True/False, "Explanatory message")
    """
    # A. Calculation of Vector V (Shoulder -> Cup) relative to the world
    try:
        Vx = cup_pos[0] - SHOULDER_POS_BASE[0]
        Vy = cup_pos[1] - SHOULDER_POS_BASE[1]
        Vz = cup_pos[2] - SHOULDER_POS_BASE[2]
    except Exception as e:
        return (False, f"Vectorial error: {e}")

    # B. DISTANCE Verification (Spherical Shell)
    total_dist = math.sqrt(Vx**2 + Vy**2 + Vz**2)
    
    if total_dist > R_MAX:
        return (False, f"TOO FAR : Target {total_dist:.2f}m (Max: {R_MAX:.2f}m)")
    
    if total_dist < R_MIN:
        # This is where the hardware constraint hits
        return (False, f"DEAD ZONE : Target {total_dist:.2f}m (Min: {R_MIN:.2f}m due to the elbow 90°)")

    # C. Z PLANE Verification (Azimuth / Base Rotation)
    # atan2(y, x) gives the World angle.
    angle_world_rad = math.atan2(Vy, Vx)
    angle_world_deg = math.degrees(angle_world_rad)
    
    # URDF Correction: The robot is mounted at +90° (Base Rotated)
    # Motor Angle = World Angle - 90°
    motor_angle_z = angle_world_deg - 90.0
    
    # Normalization (-180 to 180)
    while motor_angle_z <= -180: motor_angle_z += 360
    while motor_angle_z > 180: motor_angle_z -= 360
    
    if not (AZIMUT_MIN <= motor_angle_z <= AZIMUT_MAX):
        return (False, f"ANGLE Z OUT OF LIMIT : {motor_angle_z:.1f}° (Min: {AZIMUT_MIN}, Max: {AZIMUT_MAX})")

    # D. Y PLANE Verification (Approximate Elevation)
    # Note: A precise elevation check would require calculating the full IK
    # because the Y angle depends on elbow flexion. Here we do a rough check.
    dist_xy = math.sqrt(Vx**2 + Vy**2)
    slope_angle = math.degrees(math.atan2(Vz, dist_xy))
    
    # If the target is very high or very low, it's suspicious, but we let the IK handle fine details.
    # We just verify it's not aberrant.
    if slope_angle > ELEVATION_MAX or slope_angle < ELEVATION_MIN:
         return (False, f"ANGLE Y DIFFICULT : Slope {slope_angle:.1f}°")

    # E. Success
    return (True, f"OK : Achievable Target  ({total_dist:.2f}m)")


def calculate_triangle_angles(horizontal_distance, height_diff):
    """
    Calculates Inverse Kinematics (IK) specifically for the 'right_arm' robot.
    Forces a 'V' configuration (Shoulder low, Elbow high) to grab low objects.
    
    Args:
        horizontal_distance (float): Ground distance between shoulder and target.
        height_diff (float): Relative height (Target Z - Shoulder Z).
        
    Returns:
        tuple: (motor_shoulder_y_angle, motor_elbow_angle) in DEGREES.
    """
    # Robot constants
    L1 = 0.305
    L2 = 0.430 
    
    # 1. Calculation of total distance (Hypotenuse)
    total_dist = math.sqrt(horizontal_distance**2 + height_diff**2)
    
    # 2. Safety (Clamping)
    # We ensure the target is reachable (neither too close nor too far)
    # R_MIN ~0.53m (due to locked elbow) | R_MAX ~0.73m
    safe_dist = max(0.528, min(total_dist, 0.730))
    
    # 3. Law of Cosines: Internal shoulder angle (Alpha)
    # Angle between humerus (L1) and direct line to target
    num_alpha = L1**2 + safe_dist**2 - L2**2
    den_alpha = 2 * L1 * safe_dist
    cos_alpha = max(-1.0, min(1.0, num_alpha / den_alpha))
    alpha_deg = math.degrees(math.acos(cos_alpha)) # Ex: 47°
    
    # 4. Law of Cosines: Internal elbow angle (Gamma)
    num_gamma = L1**2 + L2**2 - safe_dist**2
    den_gamma = 2 * L1 * L2
    cos_gamma = max(-1.0, min(1.0, num_gamma / den_gamma))
    gamma_deg = math.degrees(math.acos(cos_gamma)) # Ex: 100°
    
    # 5. Calculation of Target Slope (Phi)
    # Angle of the direct line relative to the horizon
    # Ex: If the cup is lower, atan2 returns negative (ex: -20°)
    slope_deg = math.degrees(math.atan2(height_diff, horizontal_distance))
    
    # --- 6. "ROBOT SPECIFIC" LOGIC (V-SHAPE) ---
    
    # A. SHOULDER (Y)
    # To make a "V", the shoulder must point downwards.
    # Geometrically (standard frame), the angle would be: Slope - Alpha (ex: -20 - 47 = -67°)
    # BUT your robot has inverted Y axis (Positive = Down).
    # So we invert the sign: -(-67) = +67°.
    # Simplified formula: Alpha - Slope
    final_shoulder_angle = alpha_deg - slope_deg
    
    # B. ELBOW
    # Geometrically, to bend the arm, the standard motor angle is: Gamma - 180
    # Ex: 100 - 180 = -80°.
    # On your robot, Negative = Raise forearm (what we want for the V).
    # So we keep this result as is.
    final_elbow_angle = gamma_deg - 180.0
    
    return final_shoulder_angle, final_elbow_angle


def get_motor_commands(target_pos_xyz, current_angles_deg):
    """
    HIGH-LEVEL FUNCTION:
    Takes a 3D target -> Calculates IK Angles -> Returns Motor Steps/Dir.
    
    Args:
        target_pos_xyz (tuple): (x, y, z) target in meters.
        current_angles_deg (dict): {'shoulder': float, 'elbow': float} current positions.
        
    Returns:
        dict: Command dictionary containing steps and direction for each motor.
              Returns None if target is unreachable or calculation fails.
    """
    x, y, z = target_pos_xyz
    
    # 1. Pre-Check Reachability
    reachable, msg = is_reachable(target_pos_xyz)
    if not reachable:
        print(f"[WARNING] Movement Aborted: {msg}")
        return None

    # 2. Prepare Math Data for IK
    base_x, base_y, base_z = SHOULDER_POS_BASE
    dx = x - base_x
    dy = y - base_y
    dz = z - base_z
    
    dist_h = math.sqrt(dx**2 + dy**2)
    diff_h = dz
    
    # 3. Calculate Target Angles (Using existing function)
    try:
        target_s, target_e = calculate_triangle_angles(dist_h, diff_h)
    except Exception as e:
        print(f"[ERROR] Math Calculation Failed: {e}")
        return None
        
    # 4. Convert Angles to Motor Steps
    cmd_shoulder = _convert_angle_to_steps(
        target_s, 
        current_angles_deg['shoulder'], 
        RATIO_SHOULDER
    )
    
    cmd_elbow = _convert_angle_to_steps(
        target_e, 
        current_angles_deg['elbow'], 
        RATIO_ELBOW
    )
    
    return {
        'shoulder': cmd_shoulder,
        'elbow': cmd_elbow,
        'meta': {'target_angles': (target_s, target_e)}
    }

def _convert_angle_to_steps(target_angle, current_angle, gear_ratio):
    """
    Internal Helper: Converts an angle difference into steps + direction.
    """
    delta = target_angle - current_angle
    
    # Determine Direction (1 = Positive, 0 = Negative)
    # Check your wiring! You might need to swap 1 and 0 here.
    direction = 1 if delta >= 0 else 0
    
    # Calculate Step Count
    # Steps = (Degrees / 360) * (StepsPerRev * Microsteps * GearRatio)
    total_steps_per_360 = STEPS_PER_REV * MICROSTEPS * gear_ratio
    steps_needed = int(round((abs(delta) / 360.0) * total_steps_per_360))
    
    return {
        'steps': steps_needed,
        'dir': direction,
        'final_angle_theoretical': target_angle
    }


def get_azimuth_command(target_pos_xyz):
    """
    Specific function for the HV2060MG Servo (Shoulder Rotation/Azimuth).
    
    Args:
        target_pos_xyz (tuple): Target (x, y, z) in meters.
        
    Returns:
        dict: {'angle_deg': float, 'pwm_us': int} or None if unreachable.
    """
    # 1. Extract Target Coordinates
    tx, ty, tz = target_pos_xyz
    bx, by, bz = SHOULDER_POS_BASE
    
    vx = tx - bx
    vy = ty - by
    
    # 2. Calculate Angle in World Frame (Radians -> Degrees)
    angle_world_rad = math.atan2(vy, vx)
    angle_world_deg = math.degrees(angle_world_rad)
    
    # 3. Adjust for Robot Mounting Offset
    # The robot's neutral (0 deg) is usually facing forward (X axis).
    # Based on 'is_reachable', the robot is mounted with a 90 deg offset.
    angle_servo_deg = angle_world_deg - 90.0
    
    # Normalize to -180...180 range
    while angle_servo_deg <= -180: angle_servo_deg += 360
    while angle_servo_deg > 180: angle_servo_deg -= 360
    
    # 4. Safety Limit Check
    if not (AZIMUT_MIN <= angle_servo_deg <= AZIMUT_MAX):
        print(f"[ERROR] Servo Azimuth {angle_servo_deg:.2f}° is out of bounds ({AZIMUT_MIN} to {AZIMUT_MAX})")
        return None
        
    # 5. Convert Angle to PWM (Microseconds)
    # Mapping: Angle -> [1000us ... 2000us]
    # We assume linear mapping centered at 1500us = 0deg
    # Formula: PWM = 1500 + (Angle * (1000/180)) approx
    # More precise: PWM = Center + (Angle / (Range/2)) * (RangePWM/2)
    
    pwm_per_degree = (SERVO_PWM_MAX - SERVO_PWM_MIN) / SERVO_ANGLE_RANGE # Approx 5.55 us/deg
    pwm_signal = SERVO_PWM_CENTER + (angle_servo_deg * pwm_per_degree)
    
    return {
        'angle_deg': angle_servo_deg,
        'pwm_us': int(pwm_signal)
    }

# =============================================================================
# --- VISION ANGLE CALCULATOR ---
# =============================================================================

def get_raw_angles(x_mm, y_mm, z_mm):
    """
    Returns the angles (Sh_Z, Sh_Y, Elb_X) without checking if it's reachable.
    Ignores the Z axis (height) and uses a fixed table height.
    """
    # 1. Base Conversion (mm -> m)
    x_cam = x_mm / 1000.0
    y_cam = y_mm / 1000.0
    z_cam = z_mm / 1000.0

    # 2. Coordinate Frame Transformation (Camera -> Robot)
    # Robot X (Forward) = Camera Depth + Offset
    rob_x = z_cam + OFFSET_CAM_Y
    
    # Robot Y (Left) = -Camera Right + Offset
    rob_y = -x_cam + OFFSET_CAM_X
    
    # Robot Z (Up) = FIXED table height (ignores detected height)
    # The cup is always placed on the table at constant height
    rob_z = TABLE_HEIGHT - OFFSET_CAM_Z

    # 3. Calculate AZIMUTH (Shoulder Z) - Base Rotation
    # atan2(y, x) gives the angle in the plane
    # The robot is mounted such that 0° = X (Forward). 
    # However, motors often have an offset (e.g., 90°).
    # Here is raw trigonometric calculation:
    angle_rad_z = math.atan2(rob_y, rob_x)
    shoulder_z_deg = math.degrees(angle_rad_z)
    
    # Optional correction according to your motor (e.g., if 0° is to the right)
    # shoulder_z_deg -= 90 

    # 4. Calculate TRIANGLE (Shoulder Y + Elbow)
    # Horizontal distance projected on the ground
    dist_h = math.sqrt(rob_x**2 + rob_y**2)
    diff_h = rob_z
    
    # Call the mathematical function
    # Note: It contains an internal clamp to avoid math crash (acos > 1), 
    # but it will always return something.
    try:
        shoulder_y_deg, elbow_deg = calculate_triangle_angles(dist_h, diff_h)
    except Exception as e:
        # If math really crashes (e.g., distance 0), return 0
        shoulder_y_deg, elbow_deg = 0.0, 0.0

    return {
        "x_m": rob_x, "y_m": rob_y, "z_m": rob_z,
        "sh_z": shoulder_z_deg,
        "sh_y": shoulder_y_deg,
        "elb": elbow_deg
    }

# =============================================================================
# --- OAK-D PIPELINE SETUP ---
# =============================================================================

if not Path(BLOB_PATH).exists():
    raise FileNotFoundError(f"❌ Missing file : {BLOB_PATH} is not there!")

print(f">>> Loading model {LABEL_NAME}...")
pipeline = dai.Pipeline()

# 1. Color Camera
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setPreviewSize(640, 640)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

# 2. Depth (Stereo) - OAK-D LITE
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)

# 480p is needed for depth on OAK-D Lite
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

# 3. AI (YOLO)
# We are using the specific node YOLO (Please use the v2.28, more stable)
nn = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
nn.setBlobPath(BLOB_PATH)
nn.setConfidenceThreshold(0.6)
nn.input.setBlocking(False)
nn.setBoundingBoxScaleFactor(0.5)
nn.setDepthLowerThreshold(100)
nn.setDepthUpperThreshold(5000)

# Parameters YOLOv8 Nano
nn.setNumClasses(1)
nn.setCoordinateSize(4)
nn.setAnchors([10,14, 23,27, 37,58, 81,82, 135,169, 344,319])
nn.setAnchorMasks({"side26": [1,2,3], "side13": [3,4,5]})
nn.setIouThreshold(0.5)

# Links
camRgb.preview.link(nn.input)
stereo.depth.link(nn.inputDepth)

# Outputs
xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")
nn.passthrough.link(xoutRgb.input)

xoutNN = pipeline.create(dai.node.XLinkOut)
xoutNN.setStreamName("detections")
nn.out.link(xoutNN.input)

# =============================================================================
# --- EXECUTION ---
# =============================================================================
print(">>> Starting Vision...")
print("Connection to the OAK-D Lite...")
try:
    with dai.Device(pipeline) as device:
        # USB Configuration
        try:
            device.setIrLaserDotProjectorBrightness(0) # Comment out if crash at this line, LITE models don't have laser
        except:
            pass
        
        qRgb = device.getOutputQueue("rgb", 4, False)
        qDet = device.getOutputQueue("detections", 4, False)
        
        print(f"\n✅ Ready ! {LABEL_NAME}.")
        print("\n✅ READY. Waiting for cup...")
        
        while True:
            inRgb = qRgb.get()
            inDet = qDet.get()
            
            if inRgb is not None:
                frame = inRgb.getCvFrame()
                
                if inDet is not None:
                    # Get the best detection
                    detections = sorted(inDet.detections, key=lambda d: d.confidence, reverse=True)
                    
                    if detections:
                        d = detections[0] # Most confident
                        
                        # Visual coordinates
                        x1, y1 = int(d.xmin * 640), int(d.ymin * 640)
                        x2, y2 = int(d.xmax * 640), int(d.ymax * 640)
                        
                        # Spatial coordinates (mm)
                        x_mm = int(d.spatialCoordinates.x)
                        y_mm = int(d.spatialCoordinates.y)
                        z_mm = int(d.spatialCoordinates.z)
                        
                        # --- ANGLE CALCULATION ---
                        if z_mm > 0: # Avoid division by zero
                            res = get_raw_angles(x_mm, y_mm, z_mm)
                            
                            # Display
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Line 1: Robot Position (Meters)
                            txt_pos = f"Pos: X{res['x_m']:.2f} Y{res['y_m']:.2f} Z{res['z_m']:.2f}"
                            cv2.putText(frame, txt_pos, (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
                            
                            # Line 2: Angles (Degrees)
                            txt_ang = f"ANG: Z{res['sh_z']:.0f} | Y{res['sh_y']:.0f} | Elb{res['elb']:.0f}"
                            cv2.putText(frame, txt_ang, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                            
                            # Debug Console
                            print(f"🎯 CUP: {txt_pos} || {txt_ang}")
                        else:
                            # Fallback: Simple display without angles
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"X:{x_mm} Y:{y_mm} Z:{z_mm}", (x1, y1+20), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                            print(f"🎯 Cup found : X={x_mm} Y={y_mm} Z={z_mm}")

                cv2.imshow("Vision & Angles", frame)

            if cv2.waitKey(1) == ord('q'):
                break

except Exception as e:
    print(f"\n❌ ERROR : {e}")

