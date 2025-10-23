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

class SpatialVisualizer(dai.node.HostNode):
    
    def __init__(self, camera_matrix=None, dist_coeffs=None, tag_size=0.1):
        dai.node.HostNode.__init__(self)
        self.sendProcessingToPipeline(True)
        
        
        # Camera parameters for 3D axes drawing
        #self.camera_matrix = camera_matrix
        #self.dist_coeffs = dist_coeffs
        #self.tag_size = tag_size  # Tag size in meters

    # CHECK!
    def build(self, rgb: dai.Node.Output, depth: dai.Node.Output, cup_detections: dai.Node.Output, apriltags: dai.Node.Output, apriltag_passthrough: dai.Node.Output):
        self.link_args(rgb, depth, cup_detections, apriltags, apriltag_passthrough)

    # CHECK!
    def process(self, rgb_preview, depth_preview, cup_detections, apriltags, apriltag_passthrough_preview):
        # Load camera intrinsics if not already loaded
        #if self.camera_matrix is None:
        #    self._load_camera_intrinsics_from_device()
        
        depth_frame = depth_preview.getCvFrame()
        rgb_frame = rgb_preview.getCvFrame()
        apriltags_frame = apriltag_passthrough_preview.getCvFrame()

        depth_frame_color = self.process_depth_frame(depth_frame)

        cups = [d for d in cup_detections.detections if d.labelName == "cup"]
        tags = apriltags.aprilTags

        self.display_results(rgb_frame, depth_frame_color, apriltags_frame, cups, tags)

    # CHECK!
    def process_depth_frame(self, depth_frame):
        depth_downscaled = depth_frame[::4]
        if np.all(depth_downscaled == 0):
            min_depth = 0
        else:
            min_depth = np.percentile(depth_downscaled[depth_downscaled != 0], 1)

        max_depth = np.percentile(depth_downscaled, 99)
        depth_frame_color = np.interp(depth_frame, (min_depth, max_depth), (0, 255)).astype(np.uint8)
        return cv2.applyColorMap(depth_frame_color, cv2.COLORMAP_HOT)

    # CHECK!
    def display_results(self, rgb_frame, depth_frame_color, apriltags_frame, cup_detections, tag_detections):
        height, width, _ = rgb_frame.shape
        
        # Draw cup detections
        for cup in cup_detections:
            self.draw_bounding_box(depth_frame_color, cup)
            self.draw_detections(rgb_frame, cup, width, height)

        # Draw AprilTag detections
        num_tags = len(tag_detections)
        #if num_tags > 0:
        #    print(f"Detected {num_tags} AprilTag(s)")
        
        for tag in tag_detections:
            self.draw_apriltags(apriltags_frame, tag)

        # Display tag count on frame
        cv2.putText(apriltags_frame, f"Tags: {num_tags}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Depth", depth_frame_color)
        cv2.imshow("RGB", rgb_frame)
        cv2.imshow("AprilTags", apriltags_frame)

        if cv2.waitKey(1) == ord('q'):
            self.stopPipeline()

    # CHECK!
    def draw_bounding_box(self, depth_frame_color, detection):
        roi_data = detection.boundingBoxMapping
        roi = roi_data.roi
        roi = roi.denormalize(depth_frame_color.shape[1], depth_frame_color.shape[0])
        top_left = roi.topLeft()
        bottom_right = roi.bottomRight()
        cv2.rectangle(depth_frame_color, (int(top_left.x), int(top_left.y)), (int(bottom_right.x), int(bottom_right.y)), (255, 255, 255), 1)

    # CHECK!
    def draw_detections(self, frame, detection, frame_width, frame_height):
        x1 = int(detection.xmin * frame_width)
        x2 = int(detection.xmax * frame_width)
        y1 = int(detection.ymin * frame_height)
        y2 = int(detection.ymax * frame_height)
        label = detection.labelName
        color = (255, 255, 255)
        cv2.putText(frame, str(label), (x1+10,y1+20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, "{:.2f}".format(detection.confidence*100), (x1+10,y1+35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x)} mm", (x1+10,y1+50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y)} mm", (x1+10,y1+65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z)} mm", (x1+10,y1+80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 1)

    # CHECK!
    def draw_apriltags(self, frame, tag):
        """Draw AprilTag corners, ID, and 3D axes on frame"""
        # Draw corners
        corners = [(int(tag.topLeft.x), int(tag.topLeft.y)), 
                (int(tag.topRight.x), int(tag.topRight.y)), 
                (int(tag.bottomRight.x), int(tag.bottomRight.y)), 
                (int(tag.bottomLeft.x), int(tag.bottomLeft.y))]

        # Draw quadrilateral
        for i in range(4):
            cv2.line(frame, corners[i], corners[(i+1)%4], (0, 255, 0), 2)

        # Draw tag ID
        center_x = int((tag.topLeft.x + tag.bottomRight.x)/2)
        center_y = int((tag.topLeft.y + tag.bottomRight.y)/2)
        cv2.putText(frame, f"ID: {tag.id}", (center_x, center_y), 
                    cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        
        
        # Draw 3D coordinate axes
        #self.draw_3d_axes(frame, tag, corners)
        
    
    def _load_camera_intrinsics_from_device(self):
        """Load camera intrinsics from the connected device."""
        try:
            # Get the device from the pipeline
            device = dai.Device()
            calib = device.readCalibration()
            
            # Get intrinsics for RGB camera (CAM_A) at 640x400 resolution
            intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, 640, 400)
            
            # Build camera matrix
            self.camera_matrix = np.array([
                [intrinsics[0][0], 0, intrinsics[0][2]],
                [0, intrinsics[1][1], intrinsics[1][2]],
                [0, 0, 1]
            ], dtype=np.float32)
            
            # Get distortion coefficients
            self.dist_coeffs = np.array(
                calib.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A), 
                dtype=np.float32
            )
            
            print(f"SpatialVisualizer: Camera intrinsics loaded - fx={self.camera_matrix[0,0]:.1f}, fy={self.camera_matrix[1,1]:.1f}")
            
        except Exception as e:
            print(f"SpatialVisualizer: Error loading camera intrinsics: {e}")
            # Set default values if loading fails
            self.camera_matrix = np.array([
                [500, 0, 320],
                [0, 500, 200],
                [0, 0, 1]
            ], dtype=np.float32)
            self.dist_coeffs = np.zeros(5, dtype=np.float32)
    

    
    def draw_3d_axes(self, frame, tag, corners, length=0.05):
        """
        Draw 3D coordinate axes on an AprilTag.
        
        Args:
            frame: Image to draw on
            tag: AprilTag detection from DepthAI
            length: Length of axes in meters (default 5cm)
        """
        
        try:
            # Extract corners as numpy array
            corners = np.array(corners, dtype=np.float32)
            
            # Define 3D tag corners in tag coordinate frame
            half_size = 10 / 2.0 # 10 cm
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
                print("No success in solvePnP")
                return
            
            # Define axis endpoints in tag frame
            axis_points = np.array([
                [0, 0, 0],           # Origin
                [length, 0, 0],      # X-axis (red)
                [0, length, 0],      # Y-axis (green)
                [0, 0, length]       # Z-axis (blue)
            ], dtype=np.float32)
            
            # Project to image plane
            img_points, _ = cv2.projectPoints(
                axis_points, rvec, tvec, self.camera_matrix, self.dist_coeffs
            )
            img_points = img_points.reshape(-1, 2).astype(int)
            
            # Draw axes with thicker lines
            origin = tuple(img_points[0])
            cv2.line(frame, origin, tuple(img_points[1]), (0, 0, 255), 3)    # X: red
            cv2.line(frame, origin, tuple(img_points[2]), (0, 255, 0), 3)    # Y: green
            cv2.line(frame, origin, tuple(img_points[3]), (255, 0, 0), 3)    # Z: blue
            
            # Add axis labels
            cv2.putText(frame, "X", tuple(img_points[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, "Y", tuple(img_points[2]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, "Z", tuple(img_points[3]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
        except Exception as e:
            # Silently skip if axes drawing fails
            pass
    

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
        model_desc = dai.NNModelDescription("yolov6-nano")
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
        stereo.setExtendedDisparity(True)
        platform = self.pipeline.getDefaultDevice().getPlatform()
        if platform == dai.Platform.RVC2:
            stereo.setOutputSize(480, 320)

        # 3. Create spatial detection network for cup detection
        spatial_detection_network = self.pipeline.create(dai.node.SpatialDetectionNetwork).build(
            cam_rgb, stereo, model_desc, fps=FPS
        )
        spatial_detection_network.input.setBlocking(False)
        spatial_detection_network.setBoundingBoxScaleFactor(0.5)
        spatial_detection_network.setDepthLowerThreshold(100)
        spatial_detection_network.setDepthUpperThreshold(5000)

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
        mono_left.requestOutput((480, 320)).link(stereo.left)
        mono_right.requestOutput((480, 320)).link(stereo.right)
        
        # Preprocessing for AprilTag: resize and convert to GRAY8
        manip = self.pipeline.create(dai.node.ImageManip)
        manip.initialConfig.setOutputSize(480, 320)
        manip.initialConfig.setFrameType(dai.ImgFrame.Type.GRAY8)
        cam_rgb.requestOutput((480, 320)).link(manip.inputImage)
        manip.out.link(apriltag_node.inputImage)

        # 6. Create output queues (non-blocking access for sensor fusion)
        self.detection_queue = spatial_detection_network.out.createOutputQueue()
        self.apriltag_queue = apriltag_node.out.createOutputQueue()
        self.apriltag_passthrough_queue = apriltag_node.passthroughInputImage.createOutputQueue()
        self.preview_queue = spatial_detection_network.passthrough.createOutputQueue()
        self.depth_queue = spatial_detection_network.passthroughDepth.createOutputQueue()
        
        # 7. Create SpatialVisualizer for real-time display (if enabled)
        if self.enable_visualization:
            print(f"Visualizer enabled: {self.enable_visualization}")
            # Create visualizer (camera params will be set after device connection)
            self.visualizer = self.pipeline.create(SpatialVisualizer)
            
            # Link data streams to visualizer
            self.visualizer.build(
                rgb=spatial_detection_network.passthrough,
                depth=spatial_detection_network.passthroughDepth,
                cup_detections=spatial_detection_network.out,
                apriltags=apriltag_node.out,
                apriltag_passthrough=apriltag_node.passthroughInputImage
            )

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
        Run the pipeline.
        """
        try:
            self._create_and_build_pipeline()
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