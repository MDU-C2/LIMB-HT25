"""
OAK-D Vision System using DepthAI v3 API.

This module provides a modular vision system for detecting objects (cups) and AprilTags,
computing their 3D positions, and calculating relative poses between them.

Updated to use non-blocking pipeline.start() for sensor fusion integration.
"""

# IMPORTS
from typing import Optional, Dict, List, Tuple
import depthai as dai
import numpy as np
import cv2
import time
import threading
import json


class VisionSystem:
    """
    Modular vision system for OAK-D camera.
    
    Handles cup detection, AprilTag detection, depth estimation,
    and relative pose computation. Uses non-blocking pipeline for sensor fusion.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        apriltag_family: str = "TAG36H11",
        confidence_threshold: float = 0.5,
        spatial_threshold: int = 3000,  # Max depth in mm
        apriltag_quad_decimate: float = 1.5,
        apriltag_quad_sigma: float = 1.0,
        apriltag_refine_edges: bool = True,
        apriltag_max_hamming: int = 1,
        tag_size: float = 0.05,  # Tag size in meters
        enable_visualization: bool = True,
        full_frame_tracking: bool = False,
    ):
        """
        Initialize the vision system.
        
        Args:
            model_path: Path to object detection model (if None, uses MobileNet SSD from zoo)
            apriltag_family: AprilTag family to detect (TAG36H11, TAG25H9, TAG16H5)
            confidence_threshold: Minimum confidence for object detections
            spatial_threshold: Maximum depth threshold in mm
            tag_size: Physical size of AprilTag in meters (default 0.1 = 10cm)
            enable_visualization: Whether to show visualization windows
        """
        self.model_path = model_path
        self.apriltag_family = apriltag_family
        self.confidence_threshold = confidence_threshold
        self.spatial_threshold = spatial_threshold
        self.apriltag_quad_decimate = apriltag_quad_decimate
        self.apriltag_quad_sigma = apriltag_quad_sigma
        self.apriltag_refine_edges = apriltag_refine_edges
        self.apriltag_max_hamming = apriltag_max_hamming
        self.tag_size = tag_size
        self.enable_visualization = enable_visualization
        self.full_frame_tracking = full_frame_tracking
        
        # Pipeline components
        self.pipeline: Optional[dai.Pipeline] = None
        self.device: Optional[dai.Device] = None
        
        # Output queues
        self.detection_queue: Optional[dai.DataOutputQueue] = None
        self.apriltag_queue: Optional[dai.DataOutputQueue] = None
        self.depth_queue: Optional[dai.DataOutputQueue] = None
        self.preview_queue: Optional[dai.DataOutputQueue] = None
        self.apriltag_passthrough_queue: Optional[dai.DataOutputQueue] = None
        
        # Latest detections cache
        self.latest_cup_detections: List = []
        self.latest_apriltag_detections: List = []
        
        # Latest pose estimate (for sensor fusion)
        self.latest_pose = {
            "position": None,
            "orientation": None,
            "timestamp": None,
            "tag_id": None,
            "valid": False
        }
        
        # Thread-safe lock for pose updates
        self.pose_lock = threading.Lock()

        # Running state
        self._running = False
        
        # Camera intrinsics
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        
        print(f"VisionSystem initialized (tag_size={tag_size}m, visualization={'ON' if enable_visualization else 'OFF'})")
    
    def _create_and_build_pipeline(self):
        """
        Create and configure the DepthAI pipeline (non-blocking mode).
        """
        
        #model_desc = dai.NNModelDescription("mobilenet-ssd")
        FPS = 30
        self.device = dai.Device()

        self.pipeline = dai.Pipeline(self.device)

        # 1. Create camera nodes
        cam_rgb = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        mono_left = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        mono_right = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
        
        # 2. Create stereo depth node
        stereo = self.pipeline.create(dai.node.StereoDepth)
        #stereo.setExtendedDisparity(True)
        left_output = mono_left.requestOutput((640, 400))
        right_output = mono_right.requestOutput((640, 400))
        left_output.link(stereo.left)
        right_output.link(stereo.right)

        #platform = self.pipeline.getDefaultDevice().getPlatform()
        #if platform == dai.Platform.RVC2:
        #    stereo.setOutputSize(480, 320)

        # 3. Create spatial detection network for cup detection
        #model_desc_v6 = dai.NNModelDescription("yolov6-nano")
        model_desc_v10 = dai.NNModelDescription("yolov6-nano")
        spatial_detection_network = self.pipeline.create(dai.node.SpatialDetectionNetwork).build(cam_rgb, stereo, model_desc_v10, fps=FPS)
        spatial_detection_network.setConfidenceThreshold(0.6)
        spatial_detection_network.input.setBlocking(False)
        spatial_detection_network.setBoundingBoxScaleFactor(0.5)
        spatial_detection_network.setDepthLowerThreshold(100)
        spatial_detection_network.setDepthUpperThreshold(5000)

        object_tracker = self.pipeline.create(dai.node.ObjectTracker)
        object_tracker.setDetectionLabelsToTrack([41])  # Track cups (label 41 in COCO)
        object_tracker.setTrackerType(dai.TrackerType.SHORT_TERM_IMAGELESS)
        object_tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

        #preview = object_tracker.passthroughTrackerFrame.createOutputQueue()
        #tracklets = object_tracker.out.createOutputQueue()

        if self.full_frame_tracking:
            cam_rgb.requestFullResolutionOutput().link(object_tracker.inputTrackerFrame)
            object_tracker.inputTrackerFrame.setBlocking(False)
            object_tracker.inputTrackerFrame.setMaxSize(1)
        else:
            spatial_detection_network.passthrough.link(object_tracker.inputTrackerFrame)

        # Link detection frame and detections to object tracker
        spatial_detection_network.passthrough.link(object_tracker.inputDetectionFrame)
        spatial_detection_network.out.link(object_tracker.inputDetections)

        # 4. Create AprilTag detection node
        apriltag_node = self.pipeline.create(dai.node.AprilTag)
        apriltag_node.initialConfig.setFamily(self._get_apriltag_family())
        apriltag_node.initialConfig.quadDecimate = 2 # Could add more settings here.
        
        # Configure AprilTag detector
        #try:
        #    apriltag_node.initialConfig.setQuadDecimate(self.apriltag_quad_decimate)
        #    apriltag_node.initialConfig.setQuadSigma(self.apriltag_quad_sigma)
        #    apriltag_node.initialConfig.setRefineEdges(self.apriltag_refine_edges)
        #    apriltag_node.initialConfig.setMaxHammingDistance(self.apriltag_max_hamming)
        #except Exception as e:
        #    print(f"Warning: Could not set AprilTag config: {e}")
        
        # 5. Link cameras
        #mono_left.requestOutput((640, 400)).link(stereo.left)
        #mono_right.requestOutput((640, 400)).link(stereo.right)
        
        # Preprocessing for AprilTag: resize and convert to GRAY8
        manip = self.pipeline.create(dai.node.ImageManip)
        manip.initialConfig.setOutputSize(640, 400)
        manip.initialConfig.setFrameType(dai.ImgFrame.Type.GRAY8)
        cam_rgb.requestOutput((640, 400)).link(manip.inputImage)
        manip.out.link(apriltag_node.inputImage)

        # 6. Create output queues (non-blocking access for sensor fusion)
        self.detection_queue = spatial_detection_network.out.createOutputQueue()
        self.apriltag_queue = apriltag_node.out.createOutputQueue()
        self.apriltag_passthrough_queue = apriltag_node.passthroughInputImage.createOutputQueue()
        self.preview_queue = spatial_detection_network.passthrough.createOutputQueue()
        self.depth_queue = spatial_detection_network.passthroughDepth.createOutputQueue()
        
        # 7. Create queues for object tracker visualization (if enabled)
        if self.enable_visualization:
            print(f"Visualizer enabled: {self.enable_visualization}")
            self.tracker_frame_queue = object_tracker.passthroughTrackerFrame.createOutputQueue(maxSize=4, blocking=False)
            self.tracklets_queue = object_tracker.out.createOutputQueue(maxSize=4, blocking=False)

        # 8. Load camera intrinsics
        self._load_camera_intrinsics("camera_calibration.json")

        print("Pipeline created successfully")

    def _load_camera_intrinsics(self, calibration_file):
        """Load camera intrinsics from OAK-D calibration."""
        with open(calibration_file, "r") as f:
            calibration_data = json.load(f)
            self.camera_matrix = np.array(calibration_data["camera_matrix"])
            self.dist_coeffs = np.array(calibration_data["distortion_coefficients"])
    
    def _get_apriltag_family(self) -> dai.AprilTagConfig.Family:
        """Convert string family name to DepthAI enum."""
        family_map = {
            "TAG36H11": dai.AprilTagConfig.Family.TAG_36H11,
            "TAG25H9": dai.AprilTagConfig.Family.TAG_25H9,
            "TAG16H5": dai.AprilTagConfig.Family.TAG_16H5,
        }
        return family_map.get(self.apriltag_family.upper(), dai.AprilTagConfig.Family.TAG_16H5)

    def start_pipeline(self) -> bool:
        """
        Start the vision pipeline (non-blocking).
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self._create_and_build_pipeline()
            self.pipeline.start()
            self._running = True
            
            print("Pipeline started successfully")
            return True
        except Exception as e:
            print(f"Error starting pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False 
    
    def run_pipeline(self):
        """
        Run the pipeline with visualization loop.
        """
        try:
            self._create_and_build_pipeline()
            self.pipeline.start()
            self._running = True
            
            print("Pipeline running successfully")
            
            # Run visualization loop if enabled
            if self.enable_visualization:
                visualize_tracker(self.pipeline, self.tracker_frame_queue, self.tracklets_queue)
            
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
        Call this in your main loop to get latest detections and pose estimates.
        """
        if not self.is_pipeline_running():
            return
        
        # Poll AprilTag queue
        apriltag_msg = self.apriltag_queue.tryGet()
        if apriltag_msg is not None:
            tags = apriltag_msg.aprilTags
            self.latest_apriltag_detections = tags
            
            # Update pose from AprilTags
            for tag in tags:
                pose = self._estimate_pose_from_apriltag(tag)
                if pose is not None and pose['valid']:
                    with self.pose_lock:
                        self.latest_pose = pose
                    break  # Use first valid tag
        
        # Poll detection queue
        detection_msg = self.detection_queue.tryGet()
        if detection_msg is not None:
            self.latest_cup_detections = [
                d for d in detection_msg.detections 
                if d.labelName == "cup"
            ]
    
    def get_latest_pose(self, tag_id: Optional[int] = None) -> Optional[Dict]:
        """
        Get the latest pose estimate from AprilTag detection.
        
        Args:
            tag_id: AprilTag ID to get pose for (None = any tag)
        
        Returns:
            Dictionary containing:
                - position: np.ndarray [x, y, z] in mm
                - orientation: np.ndarray [w, x, y, z] quaternion
                - timestamp: float
                - tag_id: int
                - valid: bool
        """
        with self.pose_lock:
            if not self.latest_pose['valid']:
                return None
            
            if tag_id is not None and self.latest_pose['tag_id'] != tag_id:
                return None
            
            return self.latest_pose.copy()
    
    def _estimate_pose_from_apriltag(self, tag) -> Optional[Dict]:
        """
        Estimate 6-DOF pose from AprilTag detection.
        
        Args:
            tag: AprilTag detection from DepthAI
        
        Returns:
            Dictionary with position and orientation, or None if estimation fails
        """
        #if self.camera_matrix is None or self.dist_coeffs is None:
        #    print("")
        #    return None
        
        try:

            if self.camera_matrix is None or self.dist_coeffs is None:
                print("Camera intrinsics not loaded, using default values")
                self.camera_matrix = np.array([
                    [500, 0, 320],
                    [0, 500, 200],
                    [0, 0, 1]
                ], dtype=np.float32)
                self.dist_coeffs = np.zeros(5, dtype=np.float32)
            else:
                print(f"Camera intrinsics loaded, using loaded values: {self.camera_matrix}, {self.dist_coeffs}")
            
            # Extract corners
            corners = np.array([
                [tag.topLeft.x, tag.topLeft.y],
                [tag.topRight.x, tag.topRight.y],
                [tag.bottomRight.x, tag.bottomRight.y],
                [tag.bottomLeft.x, tag.bottomLeft.y]
            ], dtype=np.float32)
            
            # Define 3D tag corners in tag coordinate frame
            half_size = self.tag_size / 2.0
            object_points = np.array([
                [-half_size,  half_size, 0],  # Top-left
                [ half_size,  half_size, 0],  # Top-right
                [ half_size, -half_size, 0],  # Bottom-right
                [-half_size, -half_size, 0],  # Bottom-left
            ], dtype=np.float32)
            
            # Solve PnP to get pose
            success, rvec, tvec = cv2.solvePnP(
                object_points,
                corners,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            
            if not success:
                return None
            
            # Convert position to mm
            position = tvec.flatten() * 1000.0  # meters to mm
            
            # Convert rvec to quaternion
            orientation = self._rvec_to_quaternion(rvec)
            
            return {
                'position': position,
                'orientation': orientation,
                'timestamp': time.time(),
                'tag_id': tag.id,
                'valid': True
            }
        
        except Exception as e:
            print(f"Error estimating pose: {e}")
            return None
    
    @staticmethod
    def _rvec_to_quaternion(rvec: np.ndarray) -> np.ndarray:
        """
        Convert rotation vector to quaternion.
        
        Args:
            rvec: 3x1 rotation vector from cv2.solvePnP
        
        Returns:
            Quaternion [w, x, y, z]
        """
        # Convert to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # Convert rotation matrix to quaternion
        trace = np.trace(R)
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s
        
        return np.array([w, x, y, z])
    
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