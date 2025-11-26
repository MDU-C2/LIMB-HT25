import math

# --- 1. PHYSICAL & GEOMETRIC CONSTANTS ---

# Shoulder position in the world (x, y, z)
# Corresponds to line 231 of simulation.py (and URDF setup)
EPAULE_POS_BASE = (0.0, 0.0, 0.0)

# Segment lengths (based on right_arm.urdf + your measurements)
L1_BRAS = 0.305        # Shoulder -> Elbow
L2_AVANT_BRAS = 0.310  # Elbow -> Wrist
L3_MAIN_GRIP = 0.120   # Distance Wrist -> Gripper Center

# Effective total length of the 2nd segment
L_TOTAL_AVANT_BRAS = L2_AVANT_BRAS + L3_MAIN_GRIP # 0.430 m

# --- 2. REACH LIMITS (SHELL) ---

# R_MAX : Arm fully extended (Straight line)
R_MAX = L1_BRAS + L_TOTAL_AVANT_BRAS # = 0.735 m

# R_MIN : Arm folded to maximum (Dead Zone)
# HARDWARE CONSTRAINT: The elbow is mechanically limited to -90 degrees.
# This means the minimum internal angle is 90 degrees.
# The minimum distance is therefore the hypotenuse of the right triangle formed by the two segments.
R_MIN = math.sqrt(L1_BRAS**2 + L_TOTAL_AVANT_BRAS**2) # = 0.527 m

# --- 3. ANGULAR LIMITS (BRAIN) ---
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


# --- 4. VERIFICATION FUNCTION ---

def is_reachable(tasse_pos: tuple[float, float, float]) -> tuple[bool, str]:
    """
    Checks if the cup is physically reachable by the robot.
    Takes into account the action sphere and joint limits.
    Returns: (True/False, "Explanatory message")
    """
    # A. Calculation of Vector V (Shoulder -> Cup) relative to the world
    try:
        Vx = tasse_pos[0] - EPAULE_POS_BASE[0]
        Vy = tasse_pos[1] - EPAULE_POS_BASE[1]
        Vz = tasse_pos[2] - EPAULE_POS_BASE[2]
    except Exception as e:
        return (False, f"Vectorial error: {e}")

    # B. DISTANCE Verification (Spherical Shell)
    dist_totale = math.sqrt(Vx**2 + Vy**2 + Vz**2)
    
    if dist_totale > R_MAX:
        return (False, f"TOO FAR : Target {dist_totale:.2f}m (Max: {R_MAX:.2f}m)")
    
    if dist_totale < R_MIN:
        # This is where the hardware constraint hits
        return (False, f"DEAD ZONE : Target {dist_totale:.2f}m (Min: {R_MIN:.2f}m due to the elbow 90°)")

    # C. Z PLANE Verification (Azimuth / Base Rotation)
    # atan2(y, x) gives the World angle.
    angle_monde_rad = math.atan2(Vy, Vx)
    angle_monde_deg = math.degrees(angle_monde_rad)
    
    # URDF Correction: The robot is mounted at +90° (Base Rotated)
    # Motor Angle = World Angle - 90°
    angle_moteur_z = angle_monde_deg - 90.0
    
    # Normalization (-180 to 180)
    while angle_moteur_z <= -180: angle_moteur_z += 360
    while angle_moteur_z > 180: angle_moteur_z -= 360
    
    if not (AZIMUT_MIN <= angle_moteur_z <= AZIMUT_MAX):
        return (False, f"ANGLE Z OUT OF LIMIT : {angle_moteur_z:.1f}° (Min: {AZIMUT_MIN}, Max: {AZIMUT_MAX})")

    # D. Y PLANE Verification (Approximate Elevation)
    # Note: A precise elevation check would require calculating the full IK
    # because the Y angle depends on elbow flexion. Here we do a rough check.
    dist_xy = math.sqrt(Vx**2 + Vy**2)
    angle_pente = math.degrees(math.atan2(Vz, dist_xy))
    
    # If the target is very high or very low, it's suspicious, but we let the IK handle fine details.
    # We just verify it's not aberrant.
    if angle_pente > ELEVATION_MAX or angle_pente < ELEVATION_MIN:
         return (False, f"ANGLE Y DIFFICULT : Slope {angle_pente:.1f}°")

    # E. Success
    return (True, f"OK : Achievable Target  ({dist_totale:.2f}m)")


def calculer_angles_triangle(distance_horizontale, diff_hauteur):
    """
    Calculates Inverse Kinematics (IK) specifically for the 'right_arm' robot.
    Forces a 'V' configuration (Shoulder low, Elbow high) to grab low objects.
    
    Args:
        distance_horizontale (float): Ground distance between shoulder and target.
        diff_hauteur (float): Relative height (Target Z - Shoulder Z).
        
    Returns:
        tuple: (angle_moteur_epaule_y, angle_moteur_coude) in DEGREES.
    """
    # Robot constants
    L1 = 0.305
    L2 = 0.430 
    
    # 1. Calculation of total distance (Hypotenuse)
    dist_totale = math.sqrt(distance_horizontale**2 + diff_hauteur**2)
    
    # 2. Safety (Clamping)
    # We ensure the target is reachable (neither too close nor too far)
    # R_MIN ~0.53m (due to locked elbow) | R_MAX ~0.73m
    dist_safe = max(0.528, min(dist_totale, 0.730))
    
    # 3. Law of Cosines: Internal shoulder angle (Alpha)
    # Angle between humerus (L1) and direct line to target
    num_alpha = L1**2 + dist_safe**2 - L2**2
    den_alpha = 2 * L1 * dist_safe
    cos_alpha = max(-1.0, min(1.0, num_alpha / den_alpha))
    alpha_deg = math.degrees(math.acos(cos_alpha)) # Ex: 47°
    
    # 4. Law of Cosines: Internal elbow angle (Gamma)
    num_gamma = L1**2 + L2**2 - dist_safe**2
    den_gamma = 2 * L1 * L2
    cos_gamma = max(-1.0, min(1.0, num_gamma / den_gamma))
    gamma_deg = math.degrees(math.acos(cos_gamma)) # Ex: 100°
    
    # 5. Calculation of Target Slope (Phi)
    # Angle of the direct line relative to the horizon
    # Ex: If the cup is lower, atan2 returns negative (ex: -20°)
    pente_deg = math.degrees(math.atan2(diff_hauteur, distance_horizontale))
    
    # --- 6. "ROBOT SPECIFIC" LOGIC (V-SHAPE) ---
    
    # A. SHOULDER (Y)
    # To make a "V", the shoulder must point downwards.
    # Geometrically (standard frame), the angle would be: Slope - Alpha (ex: -20 - 47 = -67°)
    # BUT your robot has inverted Y axis (Positive = Down).
    # So we invert the sign: -(-67) = +67°.
    # Simplified formula: Alpha - Slope
    angle_epaule_final = alpha_deg - pente_deg
    
    # B. ELBOW
    # Geometrically, to bend the arm, the standard motor angle is: Gamma - 180
    # Ex: 100 - 180 = -80°.
    # On your robot, Negative = Raise forearm (what we want for the V).
    # So we keep this result as is.
    angle_coude_final = gamma_deg - 180.0
    
    return angle_epaule_final, angle_coude_final


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
    base_x, base_y, base_z = EPAULE_POS_BASE
    dx = x - base_x
    dy = y - base_y
    dz = z - base_z
    
    dist_h = math.sqrt(dx**2 + dy**2)
    diff_h = dz
    
    # 3. Calculate Target Angles (Using existing function)
    try:
        target_s, target_e = calculer_angles_triangle(dist_h, diff_h)
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

# Servo Constants
SERVO_PWM_MIN = 1000  # Microseconds for -90 degrees (approx)
SERVO_PWM_MAX = 2000  # Microseconds for +90 degrees (approx)
SERVO_PWM_CENTER = 1500 # Neutral (0 degrees)
SERVO_ANGLE_RANGE = 180 # Total range in degrees

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
    bx, by, bz = EPAULE_POS_BASE
    
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


# --- QUICK TEST (UPDATED) ---
if __name__ == "__main__":
    print(f"--- ROBOT ARM CONTROLLER (FULL) ---")
    
    # 1. INITIAL STATE (Simulating robot memory)
    #Must update these variables after every physical move!
    robot_memory = {
        'shoulder': 0.0,
        'elbow': 0.0
    }
    
    # 2. DEFINE TARGET (Cup Position)
    target_cup = (0.2, 0.5, 0.5) # X=20cm, Y=50cm, Z=50cm
    
    print(f"Current State: {robot_memory}")
    print(f"Target Pos   : {target_cup}")
    
    # 3. GET STEPPER COMMANDS (Shoulder Y + Elbow)
    stepper_cmds = get_motor_commands(target_cup, robot_memory)
    
    # 4. GET SERVO COMMAND (Shoulder Z / Azimuth)
    servo_cmd = get_azimuth_command(target_cup)
    
    print("\n>>> INSTRUCTIONS FOR DRIVERS (ARDUINO/ESP32):")
    
    if stepper_cmds:
        s_data = stepper_cmds['shoulder']
        e_data = stepper_cmds['elbow']
        
        print(f"[NEMA17 - SHOULDER Y] PIN_DIR={s_data['dir']} | PIN_STEP={s_data['steps']} pulses")
        print(f"[NEMA17 - ELBOW     ] PIN_DIR={e_data['dir']} | PIN_STEP={e_data['steps']} pulses")
        
        # Update Memory
        robot_memory['shoulder'] = s_data['final_angle_theoretical']
        robot_memory['elbow'] = e_data['final_angle_theoretical']
    else:
        print("[NEMA17] No movement (Unreachable)")

    if servo_cmd:
        print(f"[HV2060 - AZIMUTH Z ] PWM_PIN={servo_cmd['pwm_us']} µs  (Angle: {servo_cmd['angle_deg']:.2f}°)")
    else:
        print("[HV2060] No movement (Unreachable)")

    print(f"\n[INFO] Final Robot Memory: {robot_memory}")