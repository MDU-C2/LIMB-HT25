"""
OAK-D Vision System using DepthAI v2.28 API with YOLO cup detection and kinematics.

This module provides a modular vision system for detecting cups, computing their 3D positions,
and calculating robot arm angles for reaching the detected cups.

Based on new_system.py with class-based architecture for integration with control layers.
"""

from typing import Optional, Dict, List, Tuple
import depthai as dai
import numpy as np
import cv2
import time
import threading
import math
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# --- ARM KINEMATICS CONSTANTS AND FUNCTIONS ---
# =============================================================================

# Shoulder position in the world (x, y, z)
SHOULDER_POS_BASE = (0.0, 0.0, 0.0)

# Segment lengths
L1_ARM = 0.305        # Shoulder -> Elbow
L2_FOREARM = 0.310    # Elbow -> Wrist
L3_HAND_GRIP = 0.120  # Distance Wrist -> Gripper Center
L_TOTAL_FOREARM = L2_FOREARM + L3_HAND_GRIP  # 0.430 m

# Reach limits
R_MAX = L1_ARM + L_TOTAL_FOREARM  # = 0.735 m
R_MIN = math.sqrt(L1_ARM**2 + L_TOTAL_FOREARM**2)  # = 0.527 m

# Angular limits
AZIMUT_MIN = -95.0
AZIMUT_MAX = 50.0
ELEVATION_MIN = -85.0
ELEVATION_MAX = 150.0

# NEMA 17 Specs
STEPS_PER_REV = 200
MICROSTEPS = 16
RATIO_SHOULDER = 5.0
RATIO_ELBOW = 1.0

# Servo Constants
SERVO_PWM_MIN = 1000
SERVO_PWM_MAX = 2000
SERVO_PWM_CENTER = 1500
SERVO_ANGLE_RANGE = 180


def is_reachable(cup_pos: tuple[float, float, float]) -> tuple[bool, str]:
    """Checks if the cup is physically reachable by the robot."""
    try:
        Vx = cup_pos[0] - SHOULDER_POS_BASE[0]
        Vy = cup_pos[1] - SHOULDER_POS_BASE[1]
        Vz = cup_pos[2] - SHOULDER_POS_BASE[2]
    except Exception as e:
        return (False, f"Vectorial error: {e}")

    total_dist = math.sqrt(Vx**2 + Vy**2 + Vz**2)
    
    if total_dist > R_MAX:
        return (False, f"TOO FAR : Target {total_dist:.2f}m (Max: {R_MAX:.2f}m)")
    
    if total_dist < R_MIN:
        return (False, f"DEAD ZONE : Target {total_dist:.2f}m (Min: {R_MIN:.2f}m due to the elbow 90°)")

    angle_world_rad = math.atan2(Vy, Vx)
    angle_world_deg = math.degrees(angle_world_rad)
    motor_angle_z = angle_world_deg - 90.0
    
    while motor_angle_z <= -180: motor_angle_z += 360
    while motor_angle_z > 180: motor_angle_z -= 360
    
    if not (AZIMUT_MIN <= motor_angle_z <= AZIMUT_MAX):
        return (False, f"ANGLE Z OUT OF LIMIT : {motor_angle_z:.1f}° (Min: {AZIMUT_MIN}, Max: {AZIMUT_MAX})")

    dist_xy = math.sqrt(Vx**2 + Vy**2)
    slope_angle = math.degrees(math.atan2(Vz, dist_xy))
    
    if slope_angle > ELEVATION_MAX or slope_angle < ELEVATION_MIN:
         return (False, f"ANGLE Y DIFFICULT : Slope {slope_angle:.1f}°")

    return (True, f"OK : Achievable Target  ({total_dist:.2f}m)")


def calculate_triangle_angles(horizontal_distance, height_diff):
    """Calculates Inverse Kinematics (IK) for the robot arm."""
    L1 = 0.305
    L2 = 0.430 
    
    total_dist = math.sqrt(horizontal_distance**2 + height_diff**2)
    safe_dist = max(0.528, min(total_dist, 0.730))
    
    num_alpha = L1**2 + safe_dist**2 - L2**2
    den_alpha = 2 * L1 * safe_dist
    cos_alpha = max(-1.0, min(1.0, num_alpha / den_alpha))
    alpha_deg = math.degrees(math.acos(cos_alpha))
    
    num_gamma = L1**2 + L2**2 - safe_dist**2
    den_gamma = 2 * L1 * L2
    cos_gamma = max(-1.0, min(1.0, num_gamma / den_gamma))
    gamma_deg = math.degrees(math.acos(cos_gamma))
    
    slope_deg = math.degrees(math.atan2(height_diff, horizontal_distance))
    
    final_shoulder_angle = alpha_deg - slope_deg
    final_elbow_angle = gamma_deg - 180.0
    
    return final_shoulder_angle, final_elbow_angle


def get_raw_angles(x_mm, y_mm, z_mm, offset_cam_x=0.0, offset_cam_y=0.0, offset_cam_z=0.20, table_height=0.41):
    """
    Returns the angles (Sh_Z, Sh_Y, Elb_X) without checking if it's reachable.
    Ignores the Z axis (height) and uses a fixed table height.
    """
    # Base Conversion (mm -> m)
    x_cam = x_mm / 1000.0
    y_cam = y_mm / 1000.0
    z_cam = z_mm / 1000.0

    # Coordinate Frame Transformation (Camera -> Robot)
    rob_x = z_cam + offset_cam_y
    rob_y = -x_cam + offset_cam_x
    rob_z = table_height - offset_cam_z

    # Calculate AZIMUTH (Shoulder Z) - Base Rotation
    angle_rad_z = math.atan2(rob_y, rob_x)
    shoulder_z_deg = math.degrees(angle_rad_z)

    # Calculate TRIANGLE (Shoulder Y + Elbow)
    dist_h = math.sqrt(rob_x**2 + rob_y**2)
    diff_h = rob_z
    
    try:
        shoulder_y_deg, elbow_deg = calculate_triangle_angles(dist_h, diff_h)
    except Exception as e:
        shoulder_y_deg, elbow_deg = 0.0, 0.0

    return {
        "x_m": rob_x, "y_m": rob_y, "z_m": rob_z,
        "sh_z": shoulder_z_deg,
        "sh_y": shoulder_y_deg,
        "elb": elbow_deg
    }


# =============================================================================
# --- VISION SYSTEM CLASS ---
# =============================================================================

class VisionSystem:
    """
    Modular vision system for OAK-D Lite camera with YOLO cup detection.
    
    Handles cup detection, depth estimation, and robot arm angle calculation.
    Uses non-blocking pipeline for sensor fusion integration.
    """
    
    def __init__(
        self,
        blob_path: str = "cup.blob",
        label_name: str = "Cup",
        confidence_threshold: float = 0.6,
        depth_lower_threshold: int = 100,
        depth_upper_threshold: int = 5000,
        offset_cam_x: float = 0.0,
        offset_cam_y: float = 0.0,
        offset_cam_z: float = 0.20,
        table_height: float = 0.41,
        enable_visualization: bool = True,
    ):
        """
        Initialize the vision system.
        
        Args:
            blob_path: Path to YOLO model blob file
            label_name: Label name for detected objects
            confidence_threshold: Minimum confidence for detections
            depth_lower_threshold: Minimum depth in mm
            depth_upper_threshold: Maximum depth in mm
            offset_cam_x: Camera offset in X direction (lateral) in meters
            offset_cam_y: Camera offset in Y direction (depth) in meters
            offset_cam_z: Camera offset in Z direction (height) in meters
            table_height: Fixed table height in meters
            enable_visualization: Whether to show visualization windows
        """
        self.blob_path = blob_path
        self.label_name = label_name
        self.confidence_threshold = confidence_threshold
        self.depth_lower_threshold = depth_lower_threshold
        self.depth_upper_threshold = depth_upper_threshold
        self.offset_cam_x = offset_cam_x
        self.offset_cam_y = offset_cam_y
        self.offset_cam_z = offset_cam_z
        self.table_height = table_height
        self.enable_visualization = enable_visualization
        
        # Pipeline components
        self.pipeline: Optional[dai.Pipeline] = None
        self.device: Optional[dai.Device] = None
        
        # Output queues
        self.detection_queue: Optional[dai.DataOutputQueue] = None
        self.rgb_queue: Optional[dai.DataOutputQueue] = None
        
        # Latest detections cache
        self.latest_cup_detections: List = []
        self.latest_angles: Optional[Dict] = None
        
        # Thread-safe lock for updates
        self.update_lock = threading.Lock()
        
        # Running state
        self._running = False
        
        print(f"VisionSystem2 initialized (blob={blob_path}, visualization={'ON' if enable_visualization else 'OFF'})")
    
    def _create_and_build_pipeline(self):
        """Create and configure the DepthAI pipeline (non-blocking mode)."""
        if not Path(self.blob_path).exists():
            raise FileNotFoundError(f"❌ Missing file : {self.blob_path} is not there!")
        
        print(f">>> Loading model {self.label_name}...")
        self.pipeline = dai.Pipeline()
        
        # 1. Color Camera
        camRgb = self.pipeline.create(dai.node.ColorCamera)
        camRgb.setPreviewSize(640, 640)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        
        # 2. Depth (Stereo) - OAK-D LITE
        monoLeft = self.pipeline.create(dai.node.MonoCamera)
        monoRight = self.pipeline.create(dai.node.MonoCamera)
        stereo = self.pipeline.create(dai.node.StereoDepth)
        
        # 480p is needed for depth on OAK-D Lite
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)
        
        # 3. AI (YOLO)
        nn = self.pipeline.create(dai.node.YoloSpatialDetectionNetwork)
        nn.setBlobPath(self.blob_path)
        nn.setConfidenceThreshold(self.confidence_threshold)
        nn.input.setBlocking(False)
        nn.setBoundingBoxScaleFactor(0.5)
        nn.setDepthLowerThreshold(self.depth_lower_threshold)
        nn.setDepthUpperThreshold(self.depth_upper_threshold)
        
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
        xoutRgb = self.pipeline.create(dai.node.XLinkOut)
        xoutRgb.setStreamName("rgb")
        nn.passthrough.link(xoutRgb.input)
        
        xoutNN = self.pipeline.create(dai.node.XLinkOut)
        xoutNN.setStreamName("detections")
        nn.out.link(xoutNN.input)
        
        # Create output queues (non-blocking access for sensor fusion)
        self.rgb_queue = xoutRgb.createOutputQueue(maxSize=4, blocking=False)
        self.detection_queue = xoutNN.createOutputQueue(maxSize=4, blocking=False)
        
        print("Pipeline created successfully")
    
    def start_pipeline(self) -> bool:
        """
        Start the vision pipeline (non-blocking).
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self._create_and_build_pipeline()
            self.device = dai.Device(self.pipeline)
            
            # USB Configuration
            try:
                self.device.setIrLaserDotProjectorBrightness(0)  # LITE models don't have laser
            except:
                pass
            
            self.pipeline.start()
            self._running = True
            
            print("Pipeline started successfully")
            print(f"\n✅ Ready ! {self.label_name}.")
            print("\n✅ READY. Waiting for cup...")
            return True
        except Exception as e:
            print(f"Error starting pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_pipeline(self):
        """
        Run the pipeline (blocking mode).
        """
        try:
            self._create_and_build_pipeline()
            self.device = dai.Device(self.pipeline)
            
            try:
                self.device.setIrLaserDotProjectorBrightness(0)
            except:
                pass
            
            self.pipeline.run()
            self._running = True
            
            print("Pipeline running successfully")
            return True
        except Exception as e:
            print(f"Error running pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def is_pipeline_running(self) -> bool:
        """Check if the pipeline is running."""
        return self._running and self.pipeline is not None and self.pipeline.isRunning()
    
    def update(self):
        """
        Update vision system - poll queues and process data.
        Call this in your main loop to get latest detections and angle calculations.
        """
        if not self.is_pipeline_running():
            return
        
        # Poll detection queue
        detection_msg = self.detection_queue.tryGet()
        if detection_msg is not None:
            detections = sorted(detection_msg.detections, key=lambda d: d.confidence, reverse=True)
            
            with self.update_lock:
                self.latest_cup_detections = detections
                
                # Calculate angles for the best detection
                if detections:
                    d = detections[0]  # Most confident
                    x_mm = int(d.spatialCoordinates.x)
                    y_mm = int(d.spatialCoordinates.y)
                    z_mm = int(d.spatialCoordinates.z)
                    
                    if z_mm > 0:  # Avoid division by zero
                        self.latest_angles = get_raw_angles(
                            x_mm, y_mm, z_mm,
                            self.offset_cam_x,
                            self.offset_cam_y,
                            self.offset_cam_z,
                            self.table_height
                        )
                    else:
                        self.latest_angles = None
                else:
                    self.latest_angles = None
        
        # Handle visualization if enabled
        if self.enable_visualization:
            self._update_visualization()
    
    def _update_visualization(self):
        """Update visualization windows."""
        rgb_msg = self.rgb_queue.tryGet()
        if rgb_msg is None:
            return
        
        frame = rgb_msg.getCvFrame()
        
        with self.update_lock:
            if self.latest_cup_detections:
                d = self.latest_cup_detections[0]  # Most confident
                
                # Visual coordinates
                x1, y1 = int(d.xmin * 640), int(d.ymin * 640)
                x2, y2 = int(d.xmax * 640), int(d.ymax * 640)
                
                # Spatial coordinates (mm)
                x_mm = int(d.spatialCoordinates.x)
                y_mm = int(d.spatialCoordinates.y)
                z_mm = int(d.spatialCoordinates.z)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                if self.latest_angles:
                    # Line 1: Robot Position (Meters)
                    txt_pos = f"Pos: X{self.latest_angles['x_m']:.2f} Y{self.latest_angles['y_m']:.2f} Z{self.latest_angles['z_m']:.2f}"
                    cv2.putText(frame, txt_pos, (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
                    
                    # Line 2: Angles (Degrees)
                    txt_ang = f"ANG: Z{self.latest_angles['sh_z']:.0f} | Y{self.latest_angles['sh_y']:.0f} | Elb{self.latest_angles['elb']:.0f}"
                    cv2.putText(frame, txt_ang, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                else:
                    # Fallback: Simple display without angles
                    cv2.putText(frame, f"X:{x_mm} Y:{y_mm} Z:{z_mm}", (x1, y1+20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        
        cv2.imshow("Vision & Angles", frame)
        
        if cv2.waitKey(1) == ord('q'):
            self.shutdown()
    
    def get_latest_detection(self) -> Optional[Dict]:
        """
        Get the latest cup detection.
        
        Returns:
            Dictionary containing detection info, or None if no detection
        """
        with self.update_lock:
            if not self.latest_cup_detections:
                return None
            
            d = self.latest_cup_detections[0]
            return {
                'x_mm': int(d.spatialCoordinates.x),
                'y_mm': int(d.spatialCoordinates.y),
                'z_mm': int(d.spatialCoordinates.z),
                'confidence': d.confidence,
                'bbox': {
                    'xmin': d.xmin,
                    'ymin': d.ymin,
                    'xmax': d.xmax,
                    'ymax': d.ymax
                }
            }
    
    def get_latest_angles(self) -> Optional[Dict]:
        """
        Get the latest calculated robot arm angles.
        
        Returns:
            Dictionary containing:
                - x_m, y_m, z_m: Robot position in meters
                - sh_z: Shoulder Z angle (azimuth) in degrees
                - sh_y: Shoulder Y angle (elevation) in degrees
                - elb: Elbow angle in degrees
            Or None if no valid detection
        """
        with self.update_lock:
            return self.latest_angles.copy() if self.latest_angles else None
    
    def get_latest_cup_position(self) -> Optional[Tuple[float, float, float]]:
        """
        Get the latest cup position in robot frame (meters).
        
        Returns:
            Tuple (x, y, z) in meters, or None if no detection
        """
        angles = self.get_latest_angles()
        if angles is None:
            return None
        
        return (angles['x_m'], angles['y_m'], angles['z_m'])
    
    def shutdown(self):
        """Shutdown the vision system and release resources."""
        print("Shutting down vision system...")
        self._running = False
        
        if self.enable_visualization:
            cv2.destroyAllWindows()
        
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except:
                pass
        
        if self.device is not None:
            try:
                self.device.close()
            except:
                pass
        
        print("Vision system shut down.")

