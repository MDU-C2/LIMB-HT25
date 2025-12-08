import pybullet as p
import pybullet_data
import time
import os
import math
import pygame
import tkinter as tk
from arm_kinamatics import calculer_angles_triangle, is_reachable
from Sensors_Calculations.sensor_logic import ArmController

# --- Shoulder (Y/X/Z): Up/Down (Y) | Left/Right (X) | C/V (Z)
# --- Elbow (Flexion): Z (close) / S (open)
# --- Wrist (Flex/Dev): Q/D (Flexion) | W/X (Deviation)


# --- Utility functions for angles ---
def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))

def deg_to_rad(deg: float) -> float:
    return deg * math.pi / 180.0

def rad_to_deg(rad: float) -> float:
    return rad * 180.0 / math.pi

# --- "Brain" Classes (LimbArm) ---
class Shoulder:
    """Shoulder Joint 3-DoF"""
    def __init__(self):
        self.angle_x = 0.0
        self.angle_y = 0.0
        self.angle_z = 0.0
        # Limits (in degrees)
        self.min_angle_x = -120
        self.max_angle_x = 40
        self.min_angle_y = -85
        self.max_angle_y = 150.0
        self.min_angle_z = -95
        self.max_angle_z = 50

class Elbow:
    """Elbow Joint 2-DoF"""
    def __init__(self):
        self.angle_x = 0.0  # Flexion
        self.angle_y = 0.0  # Pronation
        # Limits (in degrees)
        self.min_angle_x = -90.0
        self.max_angle_x = 0.0
        self.min_angle_y = -90.0
        self.max_angle_y = 90.0
        
class Wrist:
    """Wrist Joint 2-DoF"""
    def __init__(self):
        self.angle_x = 0.0  # Flexion
        self.angle_z = 0.0  # Deviation
        # Limits (in degrees)
        self.min_angle_x = -70.0
        self.max_angle_x = 80.0
        self.min_angle_z = -20.0
        self.max_angle_z = 45.0

class AngleSmoother: # Renamed LisseurAngle to AngleSmoother
    def __init__(self, start_value, max_speed=2.0): # Renamed valeur_depart, vitesse_max
        """
        max_speed : Maximum degrees per 'frame' (movement speed)
        """
        self.current_value = start_value # Renamed valeur_actuelle
        self.max_speed = max_speed # Renamed vitesse_max

    def update(self, target): # Renamed cible
        """Calculates the next step to smoothly approach the target"""
        diff = target - self.current_value # Renamed cible
        
        # Limit the displacement to max speed (clamping)
        step = max(-self.max_speed, min(self.max_speed, diff)) # Renamed pas
        
        self.current_value += step
        return self.current_value

class Hand:
    """Manages the 4 long fingers (Index, Middle, Ring, Pinky)"""
    def __init__(self):
        self.curl = 0.0  # 0.0 = Open (Flat hand), 1.0 = Closed (Fist)
        # Limits (in radians, because it's simpler for fingers)
        self.min_curl = 0.0
        self.max_curl = 1.5  # ~90 degrees, closed fist

        self.easteregg = False
        self.gesture_progress = 0.0


class LimbArm:
    """The complete 'Brain' of the arm"""
    def __init__(self):
        self.shoulder = Shoulder()
        self.elbow = Elbow()
        self.wrist = Wrist()
        self.hand = Hand()

# --- "Setter" Functions for the brain (applies limits) ---
def set_shoulder(arm, x: float = None, y: float = None, z: float = None, mode: str = "abs") -> None:
    if x is not None:
        target = (arm.shoulder.angle_x + x) if mode == "rel" else x
        arm.shoulder.angle_x = clamp(target, arm.shoulder.min_angle_x, arm.shoulder.max_angle_x)
    if y is not None:
        target = (arm.shoulder.angle_y + y) if mode == "rel" else y
        arm.shoulder.angle_y = clamp(target, arm.shoulder.min_angle_y, arm.shoulder.max_angle_y)
    if z is not None:
        target = (arm.shoulder.angle_z + z) if mode == "rel" else z
        arm.shoulder.angle_z = clamp(target, arm.shoulder.min_angle_z, arm.shoulder.max_angle_z)

def set_elbow(arm, x: float = None, y: float = None, mode: str = "abs") -> None:
    if x is not None:
        target = (arm.elbow.angle_x + x) if mode == "rel" else x
        arm.elbow.angle_x = clamp(target, arm.elbow.min_angle_x, arm.elbow.max_angle_x)
    if y is not None:
        # Note: Keyboard control does not use 'y' (pronation) by default
        target = (arm.elbow.angle_y + y) if mode == "rel" else y
        arm.elbow.angle_y = clamp(target, arm.elbow.min_angle_y, arm.elbow.max_angle_y)

def set_wrist(arm, x: float = None, z: float = None, mode: str = "abs") -> None:
    if x is not None:
        target = (arm.wrist.angle_x + x) if mode == "rel" else x
        arm.wrist.angle_x = clamp(target, arm.wrist.min_angle_x, arm.wrist.max_angle_x)
    if z is not None:
        target = (arm.wrist.angle_z + z) if mode == "rel" else z
        arm.wrist.angle_z = clamp(target, arm.wrist.min_angle_z, arm.wrist.max_angle_z)

# --- "Getter" Functions (read brain state) ---
def get_angles_deg(arm):
    return {
        "shoulder_x": arm.shoulder.angle_x,
        "shoulder_y": arm.shoulder.angle_y,
        "shoulder_z": arm.shoulder.angle_z,
        "elbow_x": arm.elbow.angle_x,
        "elbow_y": arm.elbow.angle_y,
        "wrist_x": arm.wrist.angle_x,
        "wrist_z": arm.wrist.angle_z,
    }

def get_angles_rad(arm):
    degs = get_angles_deg(arm)
    angles_rad = {k: deg_to_rad(v) for k, v in degs.items()}
    
    # 1. Current state requested by F and G
    normal_value = arm.hand.curl # Renamed valeur_normale
    
    # 2. "Middle Finger" animation progress (0.0 to 1.0)
    mix = arm.hand.gesture_progress 
    
    fingers = ["index", "middle", "ring", "pinky"] # Renamed doigts

    for d in fingers:
        # --- A. CALCULATION OF THEORETICAL "CURL" ---
        # What curvature do we want for this finger (ignoring URDF bugs)?
        
        if d == "middle":
            # The middle finger wants to be straight (-0.5) during the insult
            insult_target = -0.5 # Renamed cible_insulte 
        else:
            # The others want to be fully closed (1.5)
            insult_target = 1.5 # Renamed cible_insulte
            
        # We mix the normal position (F/G) with the insult position (!)
        # If mix = 0, we are in normal mode. If mix = 1, we are in insult mode.
        local_curl = (normal_value * (1 - mix)) + (insult_target * mix)
        # Renamed valeur_normale, cible_insulte, local_curl

        # --- B. APPLICATION OF CORRECTIONS (Your original F/G code) ---
        # This is where we handle the phalanx 1 bug for Ring and Pinky
        
        if d in ["ring", "pinky"]:
            # Apply the offset to the mixed value
            angles_rad[f"{d}_1"] = local_curl - 1.4 
        else:
            angles_rad[f"{d}_1"] = local_curl

        # Phalanges 2 (and 3) have no offset
        angles_rad[f"{d}_2"] = local_curl
        
        # We force phalanx 3 for ring/pinky (as in your original code)
        # or for all if we are in insult mode to fully close the fist
        if d not in ["index", "middle"] or mix > 0: 
             angles_rad[f"{d}_3"] = local_curl
            
    return angles_rad

# --- PyBullet Synchronization Functions ---
def sync_to_pybullet(arm, body_id, joint_name_to_index, client=None, use_motors=True):
    cli = client if client is not None else p
    
    # Connection safety
    try:
        if not cli.getConnectionInfo().get("isConnected", 0): return
    except: return
        
    a = get_angles_rad(arm)
    
    # --- FINE MOTOR SETTINGS ---
    # ARM: Strong (200N) but smooth (Gain 0.05) to avoid jerks
    ARM_FORCE = 200.0
    ARM_POS_GAIN = 0.05  # Drastically lowered (was often at 0.5 or 1.0)
    ARM_VEL_GAIN = 1.0   # Maximum damping
    
    # FINGERS: Weak (2N) and precise (Gain 0.1)
    FINGER_FORCE = 2.0
    FINGER_POS_GAIN = 0.1
    FINGER_VEL_GAIN = 1.0
    
    for name, angle in a.items():
        if name not in joint_name_to_index: continue
        j = joint_name_to_index[name]
        
        # Detection: Is it a finger or the arm?
        is_finger = any(f in name for f in ["thumb", "index", "middle", "ring", "pinky"])
        
        # Value assignment
        force = FINGER_FORCE if is_finger else ARM_FORCE
        p_gain = FINGER_POS_GAIN if is_finger else ARM_POS_GAIN
        v_gain = FINGER_VEL_GAIN if is_finger else ARM_VEL_GAIN
        
        if use_motors:
            cli.setJointMotorControl2(
                bodyIndex=body_id, 
                jointIndex=j,
                controlMode=cli.POSITION_CONTROL,
                targetPosition=angle,
                positionGain=p_gain, # Stiffness (Spring)
                velocityGain=v_gain, # Damping (Damper)
                force=force          # Max Force
            )
        else:
            cli.resetJointState(bodyUniqueId=body_id, jointIndex=j, targetValue=angle)

# --- Mapping Functions (URDF -> Brain) ---
def build_human_joint_map(robot_id, client):
    """
    Scans the robot to find the right arm joint indices
    and associates them with logical names (e.g., 'shoulder_x').
    """
    print("Building joint mapping (Joint Map)...")
    
    # Mapping: URDF Joint Name -> Brain Logical Name
    URDF_NAME_TO_BRAIN_NAME = {
        "jRightShoulder_rotx": "shoulder_x",
        "jRightShoulder_roty": "shoulder_y",
        "jRightShoulder_rotz": "shoulder_z",
        
        "jRightElbow_roty": "elbow_x",
        "jRightElbow_rotz": "elbow_y",
        
        "jRightWrist_rotx": "wrist_x",
        "jRightWrist_rotz": "wrist_z",

        # --- MODIFICATION HERE: REMOVING JOINT_3 ---
        "index_joint_1": "index_1", "index_joint_2": "index_2", # "index_joint_3": "index_3",  <-- REMOVED
        "middle_joint_1": "middle_1", "middle_joint_2": "middle_2", # "middle_joint_3": "middle_3", <-- REMOVED
        
        # The other fingers keep their 3 phalanges if you want, or you do the same
        "ring_joint_1": "ring_1", "ring_joint_2": "ring_2", "ring_joint_3": "ring_3",
        "pinky_joint_1": "pinky_1", "pinky_joint_2": "pinky_2", "pinky_joint_3": "pinky_3",
    }
    
    BRAIN_NAME_TO_URDF_NAME = {v: k for k, v in URDF_NAME_TO_BRAIN_NAME.items()}
    joint_map = {}
    
    num_joints = client.getNumJoints(robot_id)
    for i in range(num_joints):
        info = client.getJointInfo(robot_id, i)
        joint_name = info[1].decode('UTF-8')
        
        if joint_name in URDF_NAME_TO_BRAIN_NAME:
            brain_name = URDF_NAME_TO_BRAIN_NAME[joint_name]
            joint_map[brain_name] = info[0] # Map: 'shoulder_x' -> index 1

    print("--- Joint Map (Brain -> PyBullet) ---")
    for brain_name, index in joint_map.items():
        urdf_name = BRAIN_NAME_TO_URDF_NAME[brain_name]
        print(f"  {brain_name} -> joint {index} ({urdf_name})")
    print("----------------------------------------------")
    
    if len(joint_map) != 7:
        print(f"Warning: {len(joint_map)}/7 arm joints found. Check names in URDF.")
        
    return joint_map

# --- Mapping Functions (to find real positions) ---
def build_link_name_index_map(robot_id, client):
    """Builds a map: link_name -> link_index."""
    link_map = {}
    try:
        base_name = client.getBodyInfo(robot_id)[0].decode("UTF-8")
        link_map[base_name] = -1 # Base
    except Exception:
        pass
    try:
        num_joints = client.getNumJoints(robot_id)
        for i in range(num_joints):
            info = client.getJointInfo(robot_id, i)
            child_link_name = info[12].decode('UTF-8')
            link_map[child_link_name] = i
    except Exception:
        pass
    return link_map

def resolve_link_index(link_name_map: dict, base_names: list, heuristic_key: str) -> tuple:
    """Generic function to find a link."""
    for name in base_names:
        if name in link_name_map:
            return name, link_name_map[name]
    # Heuristic
    for name, idx in link_name_map.items():
        if isinstance(name, str) and heuristic_key in name.lower():
            return name, idx
    # Fallback (last link)
    if heuristic_key == "hand":
        filtered = {n: i for n, i in link_name_map.items() if i is not None and i >= 0}
        if filtered:
            name_idx = max(filtered.items(), key=lambda kv: kv[1])
            return name_idx[0], name_idx[1]
            
    raise KeyError(f"No link found for '{heuristic_key}'")


# --- B. Main Script (based on simulation.py) ---

# --- 1. Path Configuration ---
project_dir = os.path.abspath(os.path.dirname(__file__)) # Renamed projet_dir
# IMPORTANT: The script expects 'right_arm.urdf' to be in 'arm/'
urdf_path = os.path.join(project_dir, "arm", "right_arm.urdf") 

print(f"Project directory (expected) : {project_dir}")
print(f"Path to URDF : {urdf_path}")

# --- 2. Initialization ---
try:
    clientId = p.connect(p.GUI)
except p.error as e:
    print(f"\n--- PYBULLET ERROR --- \n {e}")
    exit()

# --- 3. Environment Configuration ---
p.setAdditionalSearchPath(project_dir) # Allows finding meshes in 'arm/'
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

p.setPhysicsEngineParameter(
    fixedTimeStep=1.0/240.0,  # Standard time step
    numSolverIterations=100,  # VERY IMPORTANT: Increases collision precision (Default 10)
    numSubSteps=4  
           )           # Divides each frame into 4 sub-calculations to stabilize joints
planeId = p.loadURDF("plane.urdf")

tableId = p.loadURDF("table/table.urdf", [0, 0.8, -0.2], useFixedBase=True)

# --- Cup Position ---
cup_pos = [0.2, 0.5, 0.5] # Renamed tasse_pos
        #We can change the coordinates to check if the program is Orientated on the shoulder and not on the world


cup_orn = p.getQuaternionFromEuler([0, 0, 0]) # Renamed tasse_orn

# --- Loading the Cup ---
try:
    # Tries to load the real cup
    # ADDED 'useFixedBase=True' to freeze it in space
    cupId = p.loadURDF("dinnerware/cup_small.urdf", cup_pos, cup_orn, useFixedBase=True) # Renamed tasseId
    print(f"Cup (URDF) loaded (ID: {cupId}) at {cup_pos}") # Renamed tasseId

except Exception:
    # If it fails, creates a red cube WITH COLLISION
    print("Cup not found, creating a red cube (with collision).")
    
    # Cube dimensions (half-size)
    half_extents = [0.03, 0.03, 0.05]
    
    # 1. Create COLLISION shape
    collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                              halfExtents=half_extents)
    
    # 2. Create VISUAL shape
    visualShapeId = p.createVisualShape(shapeType=p.GEOM_BOX,
                                        halfExtents=half_extents,
                                        rgbaColor=[1, 0.2, 0.2, 1])
    
    # 3. Create Object (MultiBody)
    cupId = p.createMultiBody( # Renamed tasseId
        baseMass=0.1,  # <--- IMPORTANT : 0.0 means infinite/static mass (will let the cup floating)
        baseCollisionShapeIndex=collisionShapeId, 
        baseVisualShapeIndex=visualShapeId,    
        basePosition=cup_pos
    )
    print(f"Replacement cube (ID: {cupId}) placed at {cup_pos}")


    # --- MAGLEV SYSTEM (Making the cup floating for testing purposes) ---
# We attach the cup to the "world" (-1) so it doesn't fall immediately.
#cid_floating_cup = p.createConstraint( # Renamed cid_tasse_flottante
    #parentBodyUniqueId=cupId, # Renamed tasseId
    #parentLinkIndex=-1,
    #childBodyUniqueId=-1, # -1 = The World
    #childLinkIndex=-1,
    #jointType=p.JOINT_FIXED, # Fixed in space
    #jointAxis=[0, 0, 0],
    #parentFramePosition=[0, 0, 0],
    #childFramePosition=cup_pos # Fixed where it spawned
#)f


###NO COLLISION THANKS TO THE ONE LINE BELOW, COMMENT IT IF NEEDED
###p.setCollisionFilterGroupMask(cupId, -1, collisionFilterGroup=0, collisionFilterMask=0) # Renamed tasseId
p.resetDebugVisualizerCamera(
    cameraDistance=1.0,
    cameraYaw=45,
    cameraPitch=-30,
    cameraTargetPosition=[0, 0, 0.5]
)



# --- 4. Loading the Arm ---
try:
    print("Attempting to load robot...")
    
    # Resetting orientation to zero
    base_orientation = p.getQuaternionFromEuler([0, 0, 0]) 

    robotId = p.loadURDF(
        urdf_path,
        [0, 0, 0.7], # Base position of the shoulder
                #We can change the coordinates to check if the program is Orientated on the shoulder and not on the world

        base_orientation, # <-- Using orientation [0,0,0]
        useFixedBase=True
    )

    print("\nRobot loaded successfully!")

    # --- GRIP MODIFICATION: INCREASE FRICTION ---
    # 1. The Cup: Make it less slippery
    p.changeDynamics(cupId, -1, lateralFriction=2.5, spinningFriction=0.1, rollingFriction=0.1) # Renamed tasseId

    # 2. The Fingers: Apply friction to ALL robot joints to be sure
    for i in range(p.getNumJoints(robotId)):
        p.changeDynamics(robotId, i, lateralFriction=2.5, spinningFriction=0.1, rollingFriction=0.001)

except Exception as e:
    print("\n--- ERROR LOADING ROBOT ---")
    print(f"Check that the path is correct and .stl files are in 'arm/'")
    print(f"PyBullet Error : {e}")
    p.disconnect()
    exit()

# --- 5. Controller Initialization ---
print("Initializing 'brain'...")

# 5.a. Brain and Mapping
my_arm = LimbArm() # Renamed mon_bras
human_joint_map = build_human_joint_map(robotId, p)
link_name_map = build_link_name_index_map(robotId, p)

shoulder_y_smoother = AngleSmoother(0.0, max_speed=1.5) # Renamed lisseur_epaule_y
shoulder_x_smoother = AngleSmoother(0.0, max_speed=1.5) # Renamed lisseur_epaule_x
shoulder_z_smoother = AngleSmoother(0.0, max_speed=1.5) # Renamed lisseur_epaule_z
elbow_smoother    = AngleSmoother(0.0, max_speed=2.0) # Renamed lisseur_coude

# 5.b. Resolve links for display
try:
    hand_link_name, hand_link_index = resolve_link_index(link_name_map, ["right_hand", "RightHand"], "hand")
    print(f"Link 'Hand' found: {hand_link_name} (Index: {hand_link_index})")
    
    forearm_link_name, forearm_link_index = resolve_link_index(link_name_map, ["right_forearm", "RightForeArm"], "forearm")
    print(f"Link 'Forearm' (wrist) found: {forearm_link_name} (Index: {forearm_link_index})")
    
    upperarm_link_name, upperarm_link_index = resolve_link_index(link_name_map, ["right_upper_arm", "RightUpperArm"], "upper")
    print(f"Link 'Upper Arm' (elbow) found: {upperarm_link_name} (Index: {upperarm_link_index})")
    
except KeyError as e:
    print(f"ERROR: {e}. Real position display will fail.")
    print("Links found:", list(link_name_map.keys()))
    hand_link_index = forearm_link_index = upperarm_link_index = -1 # Fallback

# --- 5.c. NEW : TKINTER Window Initialization (BEFORE PYGAME) ---
print("Initializing sensor window (Tkinter)...")
sensor_window = tk.Tk()
sensor_window.title("LIMB Sensor Dashboard")
sensor_window.geometry("450x450+50+50") # Size and position (X, Y)
sensor_window.attributes('-topmost', True) # Keep on top of PyBullet

# Dictionary to store text "variables"
sensor_vars = {}
tk_font = ("Consolas", 11)
tk_font_bold = ("Consolas", 12, "bold")

# --- Frame 1 : Joint Sensors (Potentiometers & Torque) ---
joint_frame = tk.Frame(sensor_window, padx=10, pady=10)
joint_frame.pack(fill='x')

tk.Label(joint_frame, text="--- JOINT SENSORS ---", font=tk_font_bold).pack(anchor='w')

# Map: Short Name (display) -> Long Key (data)
joint_names_map = {
    "Sh_x": "shoulder_x",
    "Sh_y": "shoulder_y",
    "Sh_z": "shoulder_z",
    "Elb_x": "elbow_x",
    "Elb_y": "elbow_y",
    "Wr_x": "wrist_x",
    "Wr_z": "wrist_z"
}

# Column Headers
row = tk.Frame(joint_frame)
tk.Label(row, text="", width=6, font=("Consolas", 11, "bold")).pack(side=tk.LEFT)
tk.Label(row, text="Angle", width=12, anchor='w', font=("Consolas", 11, "underline")).pack(side=tk.LEFT)
tk.Label(row, text="Torque", width=12, anchor='w', font=("Consolas", 11, "underline")).pack(side=tk.LEFT)
row.pack(anchor='w')

for name, key in joint_names_map.items():
    sensor_vars[f"{key}_angle"] = tk.StringVar(value="--.- deg")
    sensor_vars[f"{key}_torque"] = tk.StringVar(value="--.- Nm")
    
    # Label Line
    row = tk.Frame(joint_frame)
    tk.Label(row, text=f"{name}:", width=6, font=tk_font_bold).pack(side=tk.LEFT)
    tk.Label(row, textvariable=sensor_vars[f"{key}_angle"], width=12, anchor='w', font=tk_font).pack(side=tk.LEFT)
    tk.Label(row, textvariable=sensor_vars[f"{key}_torque"], width=10, anchor='w', font=tk_font).pack(side=tk.LEFT)
    row.pack(anchor='w')

# --- Frame 2 : IMU Sensor (Hand) ---
imu_frame = tk.Frame(sensor_window, padx=10, pady=10)
imu_frame.pack(fill='x')

tk.Label(imu_frame, text="--- IMU SENSOR (Hand) ---", font=tk_font_bold).pack(anchor='w')
sensor_vars["imu_roll"] = tk.StringVar(value="Roll:  --.-")
sensor_vars["imu_pitch"] = tk.StringVar(value="Pitch: --.-")
sensor_vars["imu_yaw"] = tk.StringVar(value="Yaw:   --.-")

tk.Label(imu_frame, textvariable=sensor_vars["imu_roll"], font=tk_font).pack(anchor='w')
tk.Label(imu_frame, textvariable=sensor_vars["imu_pitch"], font=tk_font).pack(anchor='w')
tk.Label(imu_frame, textvariable=sensor_vars["imu_yaw"], font=tk_font).pack(anchor='w')

# --- Frame 3 : Bio-Sensors (Simulated) ---
bio_frame = tk.Frame(sensor_window, padx=10, pady=10)
bio_frame.pack(fill='x')

tk.Label(bio_frame, text="--- BIO-SENSORS (Simulated) ---", font=tk_font_bold).pack(anchor='w')
sensor_vars["emg_shoulder"] = tk.StringVar(value="EMG Shoulder: --.-")
sensor_vars["pressure_hand"] = tk.StringVar(value="Hand Pressure: N/A")

tk.Label(bio_frame, textvariable=sensor_vars["emg_shoulder"], font=tk_font).pack(anchor='w')
tk.Label(bio_frame, textvariable=sensor_vars["pressure_hand"], font=tk_font).pack(anchor='w')


# --- 5.d. Pygame Initialization (AFTER TKINTER) ---
print("Initializing Pygame interface...")
pygame.init()
screen = pygame.display.set_mode((700, 250)) # Slightly larger
pygame.display.set_caption("Arm Control (LIMB) - [ESC] to quit")
font = pygame.font.SysFont("Consolas", 16)
clock = pygame.time.Clock()
step_deg = 1.5  # Control speed (degrees)

print("\n--- Keyboard Controls ---")
print("Shoulder (Y/X/Z): Up/Down (Y) | Left/Right (X) | C/V (Z)")
print("Elbow (flexion):  Z (close) / S (open)")
print("Wrist (flex/dev): Q/D (flexion) | W/X (deviation)")
print("Modifiers: Shift (fast), Ctrl (slow)")

# --- APPROACH PARAMETERS ---
OFFSET_G_AZIMUTH = 15.0  # Degrees to the right for approach (G)
OFFSET_H_CORRECTION = 0.0 # Final correction to center gripper (H)
OFFSET_Z_GRAB = 0.02    # Small height adjustment to "bite" the cup (meters)

# --- 6. Simulation Loop (Controlled) ---
running = True
hud_visible = True
cid_attach = None
cid_floating_cup = None # Renamed cid_tasse_flottante


camera_mode = "orbit"  # Starts in normal view
# Save default external view parameters if needed later
default_cam_yaw = 45 
default_cam_pitch = -30
default_cam_target = [0, 0, 0.5]


while running and p.isConnected():
    
    # In simulation.py, inside the while running loop:
    # 1. Get "World" positions (World Frame)
    try:
        # Code that might crash if PyBullet connection is lost
        cup_pos_world, _ = p.getBasePositionAndOrientation(cupId) # Renamed pos_tasse_world, tasseId
        robot_pos_world, _ = p.getBasePositionAndOrientation(robotId) # Renamed pos_robot_world
    except p.error:
        # If PyBullet is brutally closed (clicking the X), 
        # catch the error and force loop exit.
        running = False
        break

    # IMPORTANT : The shoulder is not at the robot's base (0,0,0), but at the top of the body.
    # In your current setup, the shoulder is offset by +0.7 in Z relative to the robot's base.
    shoulder_z_offset = 0.0 # Renamed offset_epaule_z 

    # 2. Calculate "Relative" position (Local Frame)
    # The robot becomes the center (0,0,0)
    cup_x_rel = cup_pos_world[0] - robot_pos_world[0] # Renamed tasse_x_rel
    cup_y_rel = cup_pos_world[1] - robot_pos_world[1] # Renamed tasse_y_rel
    cup_z_rel = cup_pos_world[2] - (robot_pos_world[2] + shoulder_z_offset) # Renamed tasse_z_rel

    cup_pos_relative = (cup_x_rel, cup_y_rel, cup_z_rel) # Renamed pos_tasse_relative


    # --- A. Events and Controls ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_t:
                hud_visible = not hud_visible

            if event.key == pygame.K_TAB:
                if camera_mode == "orbit":
                    camera_mode = "shoulder"
                    print(">>> VIEW: Shoulder (Embedded) ACTIVATED.")
                else:
                    camera_mode = "orbit"
                    # Reset camera to default
                    p.resetDebugVisualizerCamera(
                        cameraDistance=1.0, 
                        cameraYaw=default_cam_yaw, 
                        cameraPitch=default_cam_pitch, 
                        cameraTargetPosition=default_cam_target
                    )

    keys = pygame.key.get_pressed()
    if keys[pygame.K_ESCAPE]:
        running = False

    # Add an event running = False
    # When closing the PyBullet simulation abruptly
    # Understand how to add the event
        
    # Reset (SPACE Key)
    if keys[pygame.K_SPACE]:
        # 1. Robot Reset (Existing code)
        set_shoulder(my_arm, x=0, y=0, z=0, mode="abs") # Renamed mon_bras
        set_elbow(my_arm, x=0, y=0, mode="abs") # Renamed mon_bras
        set_wrist(my_arm, x=0, z=0, mode="abs") # Renamed mon_bras
        if p.isConnected():
            for name, idx in human_joint_map.items():
                p.resetJointState(bodyUniqueId=robotId, jointIndex=idx, targetValue=0.0)

        # 2. Cup Reset (NEW)
        # Put it back to its starting position
        p.resetBasePositionAndOrientation(cupId, cup_pos, cup_orn) # Renamed tasseId, tasse_pos, tasse_orn
        # Stop its speed (linear and angular) so it's immobile
        p.resetBaseVelocity(cupId, [0, 0, 0], [0, 0, 0]) # Renamed tasseId

        # 3. Grip Safety: If we were holding it, cleanly release everything
        if cid_attach is not None:
            p.removeConstraint(cid_attach)
            cid_attach = None
            # Reset normal physics (mass + collisions)
            p.changeDynamics(cupId, -1, mass=0.1) # Renamed tasseId
            p.setCollisionFilterGroupMask(cupId, -1, collisionFilterGroup=1, collisionFilterMask=1) # Renamed tasseId
            print(">>> TOTAL RESET (Robot + Cup)")


# --- Key H : HIT WITH RECALIBRATION ---
    if keys[pygame.K_h]:
        try:
            # 1. Use the RELATIVE position already calculated at the top of the loop
            # No need for p.getBasePosition... or re-hardcoding 0.7!
            Tx, Ty, Tz = cup_pos_relative # Renamed pos_tasse_relative 
            
            # For IK, the shoulder is now the center (0,0,0)
            horizontal_dist = math.sqrt(Tx**2 + Ty**2) # Renamed dist_horizontale
            height_diff = Tz # Renamed diff_hauteur 
            
            # 2. Angle Calculation (V-Shape Function)
            target_y, target_elbow = calculer_angles_triangle(horizontal_dist, height_diff) # Renamed target_coude

            # 3. Azimuth Calculation (Z) - Also use relative coordinates
            world_angle_z = math.atan2(Ty, Tx) # Renamed angle_monde_z

            # 4. Calculate Azimuth (Z)
            world_angle_z = math.atan2(Ty, Tx) # Renamed angle_monde_z
            target_z = math.degrees(world_angle_z - 1.5708)
            while target_z > 180: target_z -= 360
            while target_z <= -180: target_z += 360
            
            # 5. Application with RECALIBRATION
            speed = 0.8 # Renamed vitesse - Smooth speed to avoid ragdoll
            
            # A. IK Movements (Y and Elbow)
            diff_y = target_y - my_arm.shoulder.angle_y # Renamed mon_bras
            diff_elb = target_elbow - my_arm.elbow.angle_x # Renamed target_coude, mon_bras.elbow.angle_x
            
            # B. Azimuth Movement (Z)
            diff_z = target_z - my_arm.shoulder.angle_z # Renamed mon_bras
            if diff_z > 180: diff_z -= 360
            elif diff_z < -180: diff_z += 360
            
            # C. X-AXIS RECALIBRATION (The fix is here!)
            # We want the shoulder straight (0) for IK calculation to be valid
            target_x = 0.0 
            diff_x = target_x - my_arm.shoulder.angle_x # Renamed mon_bras
            
            # D. WRIST RECALIBRATION (Optional but recommended)
            # We put the wrist flat to grab properly
            target_wrist_x = 0.0
            diff_wrist_x = target_wrist_x - my_arm.wrist.angle_x # Renamed mon_bras
            
            # 6. Send Commands (All at once)
            
            set_shoulder(my_arm, # Renamed mon_bras
                         z=my_arm.shoulder.angle_z + max(-speed, min(speed, diff_z)), # Renamed mon_bras, vitesse
                         y=my_arm.shoulder.angle_y + max(-speed, min(speed, diff_y)), # Renamed mon_bras, vitesse
                         x=my_arm.shoulder.angle_x + max(-speed, min(speed, diff_x)), # <--- X Correction - Renamed mon_bras, vitesse
                         mode="abs")
                         
            set_elbow(my_arm, # Renamed mon_bras
                      x=my_arm.elbow.angle_x + max(-speed, min(speed, diff_elb)), # Renamed mon_bras, vitesse
                      mode="abs")

            set_wrist(my_arm, # Renamed mon_bras
                      x=my_arm.wrist.angle_x + max(-speed, min(speed, diff_wrist_x)), # Renamed mon_bras, vitesse
                      mode="abs")
                      
        except Exception as e:
            print(f"Error H: {e}")



    if keys[pygame.K_1] and (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]):
        my_arm.hand.easteregg = True # Renamed mon_bras
    else:
        my_arm.hand.easteregg = False # Renamed mon_bras       

    # --- GESTURE ANIMATION (F/G Style) ---
    # If the mode is activated, increase 'gesture_progress' towards 1.0
    # Otherwise, decrease towards 0.0
    if my_arm.hand.easteregg: # Renamed mon_bras
        my_arm.hand.gesture_progress += 0.05  # Opening speed # Renamed mon_bras
    else:
        my_arm.hand.gesture_progress -= 0.05  # Closing speed # Renamed mon_bras
    
    # Keep the value between 0.0 and 1.0 (Clamping)
    my_arm.hand.gesture_progress = max(0.0, min(1.0, my_arm.hand.gesture_progress)) # Renamed mon_bras      

    # --- HAND CONTROL & GRIP LOGIC (CORRECTED & STABILIZED) ---
    delta_hand = (0.05 if keys[pygame.K_f] else 0) + (-0.05 if keys[pygame.K_g] else 0)
    
    # 1. Finger Animation
    if delta_hand:
        new_value = my_arm.hand.curl + delta_hand # Renamed nouvelle_valeur, mon_bras
        my_arm.hand.curl = max(0.0, min(1.5, new_value)) # Renamed mon_bras, nouvelle_valeur

    # 2. GRIP (F) : Grab WHERE IT TOUCHES
    if keys[pygame.K_f] and cid_attach is None and hand_link_index != -1:
        
        # Distance Hand <-> Cup
        dx = cup_pos_world[0] - real_hand_pos[0] # Renamed pos_tasse_world, pos_reelle_main
        dy = cup_pos_world[1] - real_hand_pos[1] # Renamed pos_tasse_world, pos_reelle_main
        dz = cup_pos_world[2] - real_hand_pos[2] # Renamed pos_tasse_world, pos_reelle_main
        contact_dist = math.sqrt(dx**2 + dy**2 + dz**2) # Renamed dist_contact
        
        # Tolerance (20cm)
        if contact_dist < 0.20:
            print(">>> DYNAMIC GRIP ACTIVATED")
            
            # --- NO NEED TO CLEAN cid_floating_cup HERE ---
            
            # A. CALCULATION OF RELATIVE POSE (To fix the position)
            # 1. Hand State
            hand_state = p.getLinkState(robotId, hand_link_index)
            hand_pos_world = hand_state[0] # Renamed pos_main_world
            hand_orn_world = hand_state[1] # Renamed orn_main_world
            
            # 2. Cup State
            cup_pos_w, cup_orn_w = p.getBasePositionAndOrientation(cupId) # Renamed pos_tasse_w, orn_tasse_w, tasseId
            
            # 3. Inverse Transform Calculation (Cup view from hand)
            inv_hand_pos, inv_hand_orn = p.invertTransform(hand_pos_world, hand_orn_world) # Renamed inv_pos_main, inv_orn_main
            cup_pos_local, cup_orn_local = p.multiplyTransforms( # Renamed pos_local_tasse, orn_local_tasse
                inv_hand_pos, inv_hand_orn, # Renamed inv_pos_main, inv_orn_main, 
                cup_pos_w, cup_orn_w # Renamed pos_tasse_w, orn_tasse_w
            )

            # B. PHYSICS SAFETY (Anti-Explosion)
            # 1. Make the cup ultra-light (10g) to avoid stressing the wrist
            p.changeDynamics(cupId, -1, mass=0.01) # Renamed tasseId
            # 2. Disable collisions (Ghost Mode) so fingers don't push it away
            p.setCollisionFilterGroupMask(cupId, -1, collisionFilterGroup=0, collisionFilterMask=0) # Renamed tasseId

            # C. CONSTRAINT CREATION
            cid_attach = p.createConstraint(
                parentBodyUniqueId=robotId,
                parentLinkIndex=hand_link_index,
                childBodyUniqueId=cupId, # Renamed tasseId
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=cup_pos_local, # Renamed pos_local_tasse
                parentFrameOrientation=cup_orn_local, # Renamed orn_local_tasse
                childFramePosition=[0, 0, 0],
                childFrameOrientation=[0, 0, 0, 1]
            )
            
            # Reduced force to avoid breaking the robot with sudden movement
            p.changeConstraint(cid_attach, maxForce=200)

    # 3. RELEASE (G)
    if keys[pygame.K_g] and cid_attach is not None:
        print(">>> RELEASE")
        p.removeConstraint(cid_attach)
        cid_attach = None
        
        # Reset normal physics
        p.changeDynamics(cupId, -1, mass=0.1) # Real mass # Renamed tasseId
        p.setCollisionFilterGroupMask(cupId, -1, collisionFilterGroup=1, collisionFilterMask=1) # Collisions ON # Renamed tasseId
        
        # Small nudge downwards to detach
        p.resetBaseVelocity(cupId, linearVelocity=[0, 0, -0.2]) # Renamed tasseId



    # Speed
    speed_mult = 1.0
    if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: speed_mult = 4.0
    if keys[pygame.K_LCTRL] or keys[pygame.K_RSHIFT]: speed_mult = 0.25
    step = step_deg * speed_mult

    # Shoulder
    delta_should_y = (step if keys[pygame.K_UP] else 0) + (-step if keys[pygame.K_DOWN] else 0)
    delta_should_x = (-step if keys[pygame.K_RIGHT] else 0) + (step if keys[pygame.K_LEFT] else 0)
    delta_should_z = (step if keys[pygame.K_c] else 0) + (-step if keys[pygame.K_v] else 0)
    if delta_should_x or delta_should_y or delta_should_z:
        set_shoulder(my_arm, x=delta_should_x, y=delta_should_y, z=delta_should_z, mode="rel") # Renamed mon_bras

    # Elbow (flexion)
    delta_elbow_x = (step if keys[pygame.K_z] else 0) + (-step if keys[pygame.K_s] else 0)
    if delta_elbow_x:
        set_elbow(my_arm, x=delta_elbow_x, mode="rel") # Renamed mon_bras
        
    # Wrist
    delta_wrist_x = (step if keys[pygame.K_q] else 0) + (-step if keys[pygame.K_d] else 0)
    delta_wrist_z = (step if keys[pygame.K_w] else 0) + (-step if keys[pygame.K_x] else 0)
    if delta_wrist_x or delta_wrist_z:
        set_wrist(my_arm, x=delta_wrist_x, z=delta_wrist_z, mode="rel") # Renamed mon_bras

    # --- B. Synchronization and Physics ---
    if not p.isConnected():
        running = False
        continue

        # --- MOVEMENT SMOOTHING (Trajectory Generation) ---
    # 1. Get the raw target (calculated by your math or the keyboard)
    target_sh_y = my_arm.shoulder.angle_y # Renamed cible_sh_y, mon_bras
    target_sh_x = my_arm.shoulder.angle_x # Renamed cible_sh_x, mon_bras
    target_sh_z = my_arm.shoulder.angle_z # Renamed cible_sh_z, mon_bras
    target_elb_x = my_arm.elbow.angle_x # Renamed cible_elb_x, mon_bras

    # 2. Calculate the fluid intermediate step
    # THIS is the value you will give to your colleague
    cmd_sh_y = shoulder_y_smoother.update(target_sh_y) # Renamed lisseur_epaule_y
    cmd_sh_x = shoulder_x_smoother.update(target_sh_x) # Renamed lisseur_epaule_x
    cmd_sh_z = shoulder_z_smoother.update(target_sh_z) # Renamed lisseur_epaule_z
    cmd_elb_x = elbow_smoother.update(target_elb_x) # Renamed lisseur_coude

    # 3. EXPORT TO HARDWARE (This is where you validate your statement!)
    # Create a clean dictionary for your colleague
    motor_commands = { # Renamed commandes_moteurs
        "shoulder_y": cmd_sh_y,
        "shoulder_x": cmd_sh_x,
        "shoulder_z": cmd_sh_z,
        "elbow_x":    cmd_elb_x,
    }

    sync_to_pybullet(
        arm=my_arm, # Renamed mon_bras
        body_id=robotId,
        joint_name_to_index=human_joint_map,
        client=p,
        use_motors=True
    )

    # =========================================================
    # --- CAMERA MODE: X-AXIS VIEW (COMPENSATED BIOMETRIC) ---
    # =========================================================
    if camera_mode == "shoulder":
        
        if 'robotId' in locals():
            # 1. POSITION : STRICT OFFSET (0, 0.05, 0.05)
            pos_base, _ = p.getBasePositionAndOrientation(robotId)
            
            # This is the camera pivot point (on the shoulder)
            cam_target_pos = [
                pos_base[0] - 0.05,    
                pos_base[1] + 0.05,   
                pos_base[2] + 0.05    
            ]

            # 2. ROBOT ANGLE TRACKING
            current_rot_z = real_angles_deg.get('shoulder_z', 0) # Left/Right (Yaw) # Renamed rot_z_actuelle
            current_rot_y = real_angles_deg.get('shoulder_y', 0) # Up/Down (Pitch) # Renamed rot_y_actuelle

            # 3. YAW CALCULATION (LOOKING IN X)
            fps_yaw = 0 + current_rot_z # Renamed rot_z_actuelle
            
            # 4. PITCH CALCULATION (COMPENSATED UP/DOWN)
            # INVERT THE SIGN : When the arm goes up (+rot_y), the camera dives (-rot_y)
            fps_pitch = -15 - current_rot_y # Renamed rot_y_actuelle

            # 5. APPLICATION
            p.resetDebugVisualizerCamera(
                cameraDistance=0.2, 
                cameraYaw=fps_yaw, 
                cameraPitch=fps_pitch, 
                cameraTargetPosition=cam_target_pos
            )

    p.stepSimulation()

    # --- C. Data Reading and Display ---
    
    # 1. Read actual angles/torques
    torques = {}
    real_angles_deg = {}
    default_torque = 0.0
    
    try:
        for name, index in human_joint_map.items():
            state = p.getJointState(robotId, index)
            real_angle_rad = state[0]
            torques[name] = state[3] 
            real_angles_deg[name] = rad_to_deg(real_angle_rad)
            
    except Exception:
        for name in human_joint_map.keys():
            torques[name] = default_torque
            real_angles_deg[name] = 0.0 # Fallback
            
    # 2. Read real positions AND IMU
    default_real_pos = (0.0, 0.0, 0.0)
    default_imu_rad = (0.0, 0.0, 0.0)
    try:
        if hand_link_index != -1:
            hand_state = p.getLinkState(robotId, hand_link_index)
            real_hand_pos = hand_state[0] # Renamed pos_reelle_main
            imu_quaternion = hand_state[1] # <--- IMU READING
            imu_euler_rad = p.getEulerFromQuaternion(imu_quaternion) # <---
        else:
             real_hand_pos = default_real_pos
             imu_euler_rad = default_imu_rad

        real_wrist_pos = p.getLinkState(robotId, forearm_link_index)[0] if forearm_link_index != -1 else default_real_pos # Renamed pos_reelle_poignet
        real_elbow_pos = p.getLinkState(robotId, upperarm_link_index)[0] if upperarm_link_index != -1 else default_real_pos # Renamed pos_reelle_coude
    except Exception as e:
        real_hand_pos = real_wrist_pos = real_elbow_pos = default_real_pos # Renamed pos_reelle_main, pos_reelle_poignet, pos_reelle_coude
        imu_euler_rad = default_imu_rad

    # 3. Desired angles (from "brain")
    current_angles = get_angles_deg(my_arm) # Renamed mon_bras

    # 4. Pygame Display
    screen.fill((18, 18, 18))
    
    text_lines = [
        f"Shoulder (x,y,z): {real_angles_deg.get('shoulder_x', 0):.1f}, {real_angles_deg.get('shoulder_y', 0):.1f}, {real_angles_deg.get('shoulder_z', 0):.1f} deg",
        f"Elbow (x,y):      {real_angles_deg.get('elbow_x', 0):.1f}, {real_angles_deg.get('elbow_y', 0):.1f} deg",
        f"Wrist (x,z):      {real_angles_deg.get('wrist_x', 0):.1f}, {real_angles_deg.get('wrist_z', 0):.1f} deg",
        "---",
        f"Torque Sh(x,y,z): {torques.get('shoulder_x', 0):.2f}, {torques.get('shoulder_y', 0):.2f}, {torques.get('shoulder_z', 0):.2f} Nm",
        f"Torque Elb(x,y):  {torques.get('elbow_x', 0):.2f}, {torques.get('elbow_y', 0):.2f} | Wr(x,z): {torques.get('wrist_x', 0):.2f}, {torques.get('wrist_z', 0):.2f} Nm",
        "---",
        f"ELB_Real: x={real_elbow_pos[0]:.3f} y={real_elbow_pos[1]:.3f} z={real_elbow_pos[2]:.3f}", # Renamed pos_reelle_coude
        f"WRI_Real: x={real_wrist_pos[0]:.3f} y={real_wrist_pos[1]:.3f} z={real_wrist_pos[2]:.3f}", # Renamed pos_reelle_poignet
        f"HAND_Real:x={real_hand_pos[0]:.3f} y={real_hand_pos[1]:.3f} z={real_hand_pos[2]:.3f}", # Renamed pos_reelle_main
        "---",
        "Keys: Arrows+C/V | Elbow z/s | Wrist q/d, w/x | Space reset | ESC quit",
    ]
    y = 10
    for line in text_lines:
        surf = font.render(line, True, (220, 220, 220))
        screen.blit(surf, (10, y))
        y += 18 

    # --- ADDED : REACHABILITY INDICATOR (GRAPPABLE / UNGRAPPABLE) ---
    # 1. Check current cup position
    cup_pos_visual, _ = p.getBasePositionAndOrientation(cupId) # Renamed pos_tasse_visuel, tasseId
    
    # 2. Ask kinematics module if it's good
    is_reachable_status, info_message = is_reachable(cup_pos_relative) # Renamed est_atteignable, message_info, pos_tasse_relative
    
    # 3. Choose color and text
    if is_reachable_status:
        status_color = (0, 255, 0) # GREEN # Renamed couleur_status
        status_text = f"GRAPPABLE : {info_message}" # Renamed texte_status, message_info
    else:
        status_color = (255, 0, 0) # RED # Renamed couleur_status
        status_text = f"UNGRAPPABLE : {info_message}" # Renamed texte_status, message_info
    
    # 4. Display at bottom of window
    surf_status = font.render(status_text, True, status_color) # Renamed texte_status, couleur_status
    # Display slightly lower than other texts (y + 10 pixels)
    screen.blit(surf_status, (10, y + 10))


    # --- TACTICAL VISUALIZATION (HUD) ---
    
    # Check if tactical mode is active AND if the hand was found
    if hud_visible and hand_link_index != -1:
        
        # 1. DYNAMIC DISTANCE CALCULATION (Hand <-> Cup)
        # Calculate Pythagoras between current hand position and cup position
        dx = cup_pos_world[0] - real_hand_pos[0] # Renamed pos_tasse_world, pos_reelle_main
        dy = cup_pos_world[1] - real_hand_pos[1] # Renamed pos_tasse_world, pos_reelle_main
        dz = cup_pos_world[2] - real_hand_pos[2] # Renamed pos_tasse_world, pos_reelle_main
        hand_cup_dist = math.sqrt(dx**2 + dy**2 + dz**2) # Renamed dist_main_tasse

        # 2. Line of sight (Red)
        p.addUserDebugLine(real_hand_pos, cup_pos_world, lineColorRGB=[1, 0, 0], lifeTime=0.1, lineWidth=2) # Renamed pos_reelle_main, pos_tasse_world
            
        # 3. Dynamic text (The number will change when you move the arm)
        p.addUserDebugText(f"TARGET: CUP_ALPHA [DIST: {hand_cup_dist:.2f}m]", # Renamed dist_main_tasse
                           [cup_pos_world[0], cup_pos_world[1], cup_pos_world[2]+0.15], # Renamed pos_tasse_world
                           textColorRGB=[1, 1, 0], 
                           textSize=1.2, 
                           lifeTime=0.1)

    pygame.display.flip()

    # --- 5. NEW : Tkinter Window Update ---
    try:
        # Update joint sensors
        for name, key in joint_names_map.items():
            sensor_vars[f"{key}_angle"].set(f"{real_angles_deg.get(key, 0):.1f} deg")
            sensor_vars[f"{key}_torque"].set(f"{torques.get(key, 0):.2f} Nm")
        
        # Update IMU
        sensor_vars["imu_roll"].set(f"Roll:  {rad_to_deg(imu_euler_rad[0]):.1f}")
        sensor_vars["imu_pitch"].set(f"Pitch: {rad_to_deg(imu_euler_rad[1]):.1f}")
        sensor_vars["imu_yaw"].set(f"Yaw:   {rad_to_deg(imu_euler_rad[2]):.1f}")

        # Update bio-sensors (simulated)
        # Simulate EMG as average torque (effort) of shoulder
        emg_sim = (abs(torques.get('shoulder_x',0)) + abs(torques.get('shoulder_y',0)) + abs(torques.get('shoulder_z',0))) / 3
        sensor_vars["emg_shoulder"].set(f"EMG Shoulder: {emg_sim:.2f} (Sim-Torque)")
        # Pressure (we have no object, so N/A)
        sensor_vars["pressure_hand"].set("Hand Pressure: N/A (No Object)")
        
        # Update window
        sensor_window.update()
        
    except tk.TclError:
        # User closed the Tkinter window
        print("Sensor window closed.")
        running = False
    
    # --- D. Timing ---
    clock.tick(60) # Target 60 FPS


# --- 7. Shutdown ---
print("Simulation finished.")
try:
    sensor_window.destroy() # Close Tkinter
except: # Catch all errors
    pass
    
try:
    pygame.quit() # Close Pygame
except:
    pass
    
try:
    p.disconnect() # Close PyBullet
except:
    pass