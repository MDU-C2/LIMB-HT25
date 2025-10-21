"""
OAK-D Vision System using DepthAI v3 API.

This module provides a modular vision system for detecting objects (cups) and AprilTags,
computing their 3D positions, and calculating relative poses between them.
"""

# IMPORTS
from typing import Optional, Dict, List, Tuple
import depthai as dai
import numpy as np
import cv2
import time

class SpatialVisualizer(dai.node.HostNode):
    
    def __init__(self):
        dai.node.HostNode.__init__(self)
        self.sendProcessingToPipeline(True)

    def build(self, rgb: dai.Node.Output, depth: dai.Node.Output, cup_detections: dai.Node.Output, apriltags: dai.Node.Output, apriltag_passthrough: dai.Node.Output):
        self.link_args(rgb, depth, cup_detections, apriltags, apriltag_passthrough)

    def process(self, rgb_preview, depth_preview, cup_detections, apriltags, apriltag_passthrough_preview):
        depth_frame = depth_preview.getCvFrame()
        rgb_frame = rgb_preview.getCvFrame()
        apriltags_frame = apriltag_passthrough_preview.getCvFrame()

        depth_frame_color = self.process_depth_frame(depth_frame)

        cups = [d for d in cup_detections.detections if d.labelName == "cup"]
        tags = apriltags.aprilTags

        self.display_results(rgb_frame, depth_frame_color, apriltags_frame, cups, tags)

    def process_depth_frame(self, depth_frame):
        depth_downscaled = depth_frame[::4]
        if np.all(depth_downscaled == 0):
            min_depth = 0
        else:
            min_depth = np.percentile(depth_downscaled[depth_downscaled != 0], 1)

        max_depth = np.percentile(depth_downscaled, 99)
        depth_frame_color = np.interp(depth_frame, (min_depth, max_depth), (0, 255)).astype(np.uint8)
        return cv2.applyColorMap(depth_frame_color, cv2.COLORMAP_HOT)

    def display_results(self, rgb_frame, depth_frame_color, apriltags_frame, cup_detections, tag_detections):
        height, width, _ = rgb_frame.shape
        
        # Draw cup detections
        for cup in cup_detections:
            self.draw_bounding_box(depth_frame_color, cup)
            self.draw_detections(rgb_frame, cup, width, height)

        # Draw AprilTag detections
        num_tags = len(tag_detections)
        if num_tags > 0:
            print(f"Detected {num_tags} AprilTag(s)")
        
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

    def draw_bounding_box(self, depth_frame_color, detection):
        roi_data = detection.boundingBoxMapping
        roi = roi_data.roi
        roi = roi.denormalize(depth_frame_color.shape[1], depth_frame_color.shape[0])
        top_left = roi.topLeft()
        bottom_right = roi.bottomRight()
        cv2.rectangle(depth_frame_color, (int(top_left.x), int(top_left.y)), (int(bottom_right.x), int(bottom_right.y)), (255, 255, 255), 1)

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

    def draw_apriltags(self, frame, tag):
        """Draw AprilTag corner and info on frame"""
        # Draw corners
        #print(tag)
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

class VisionSystem:
    """
    Modular vision system for OAK-D camera.
    
    Handles cup detection, AprilTag detection, depth estimation,
    and relative pose computation.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        apriltag_family: str = "TAG36H11",
        confidence_threshold: float = 0.5,
        spatial_threshold: int = 3000,  # Max depth in mm
        apriltag_quad_decimate: float = 2.0,
        apriltag_quad_sigma: float = 0.0,
        apriltag_refine_edges: bool = True,
        apriltag_max_hamming: int = 1,
    ):
        """
        Initialize the vision system.
        
        Args:
            model_path: Path to object detection model (if None, uses MobileNet SSD from zoo)
            apriltag_family: AprilTag family to detect (TAG36H11, TAG25H9, TAG16H5)
            confidence_threshold: Minimum confidence for object detections
            spatial_threshold: Maximum depth threshold in mm
        """
        self.model_path = model_path
        self.apriltag_family = apriltag_family
        self.confidence_threshold = confidence_threshold
        self.spatial_threshold = spatial_threshold
        self.apriltag_quad_decimate = apriltag_quad_decimate
        self.apriltag_quad_sigma = apriltag_quad_sigma
        self.apriltag_refine_edges = apriltag_refine_edges
        self.apriltag_max_hamming = apriltag_max_hamming
        
        # Pipeline components
        self.pipeline: Optional[dai.Pipeline] = None
        self.device: Optional[dai.Device] = None
        
        # Output queues
        self.detection_queue: Optional[dai.DataOutputQueue] = None
        self.apriltag_queue: Optional[dai.DataOutputQueue] = None
        self.depth_queue: Optional[dai.DataOutputQueue] = None
        self.preview_queue: Optional[dai.DataOutputQueue] = None
        
        # Latest detections cache
        self.latest_cup_detections: List = []
        self.latest_apriltag_detections: List = []
        
        # Running state
        self._running = False
        
        #logger.info("VisionSystem initialized")
    
    def _create_and_run_pipeline(self):
        """
        Create and configure the DepthAI pipeline, then start it.
        """

        model_desc = dai.NNModelDescription("yolov6-nano")
        FPS = 30
        self.device = dai.Device()

        with dai.Pipeline(self.device) as pipeline:

            # 1. Create camera nodes
            cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
            mono_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
            mono_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
            
            # 2. Create stereo depth node
            stereo = pipeline.create(dai.node.StereoDepth)
            stereo.setExtendedDisparity(True)
            platform = pipeline.getDefaultDevice().getPlatform()
            if platform == dai.Platform.RVC2:
                stereo.setOutputSize(640, 400)

            # 3. Create spatial detection network for cup detection
            spatial_detection_network = pipeline.create(dai.node.SpatialDetectionNetwork).build(cam_rgb, stereo, model_desc, fps=FPS)
            spatial_detection_network.input.setBlocking(False)
            spatial_detection_network.setBoundingBoxScaleFactor(0.5)
            spatial_detection_network.setDepthLowerThreshold(100)
            spatial_detection_network.setDepthUpperThreshold(5000)

            # 4. Create AprilTag detection node
            apriltag_node = pipeline.create(dai.node.AprilTag)
            # Set AprilTag family and thresholds
            apriltag_node.initialConfig.setFamily(self._get_apriltag_family())
            # Basic detector thresholds
            try:
                apriltag_node.initialConfig.setQuadDecimate(self.apriltag_quad_decimate)
                apriltag_node.initialConfig.setQuadSigma(self.apriltag_quad_sigma)
                apriltag_node.initialConfig.setRefineEdges(self.apriltag_refine_edges)
                apriltag_node.initialConfig.setMaxHammingDistance(self.apriltag_max_hamming)
            except Exception:
                pass
            # Create output queue for AprilTag
            self.apriltag_queue = apriltag_node.out.createOutputQueue()
            self.apriltag_passthrough_queue = apriltag_node.passthroughInputImage.createOutputQueue()
            
            # 5. Link cameras and april node
            mono_left.requestOutput((640, 400)).link(stereo.left)
            mono_right.requestOutput((640, 400)).link(stereo.right)
            # Insert lightweight preprocessing: resize and convert to GRAY8
            manip = pipeline.create(dai.node.ImageManip)
            manip.initialConfig.setOutputSize(640, 400)
            manip.initialConfig.setFrameType(dai.ImgFrame.Type.GRAY8)
            cam_rgb.requestOutput((640, 400)).link(manip.inputImage)
            manip.out.link(apriltag_node.inputImage)

            

            # 6. Create visualizer
            visualizer = pipeline.create(SpatialVisualizer)
            visualizer.build(rgb=spatial_detection_network.passthrough,
                             depth=spatial_detection_network.passthroughDepth, 
                             cup_detections=spatial_detection_network.out, 
                             apriltags=apriltag_node.out,
                             apriltag_passthrough=apriltag_node.passthroughInputImage)
            
            
            
            self.pipeline = pipeline
            self.pipeline.run()

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
        Start the vision pipeline.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self._create_and_run_pipeline()
            return True
        except Exception as e:
            print(f"Error starting pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False 
    
    def is_pipeline_running(self) -> bool:
        """Check if the pipeline is running."""
        return self.pipeline.isRunning()
    
    def shutdown(self):
        """Shutdown the vision system and release resources."""
        print("Shutting down vision system...")
        if self.device:
            self.device.close()
        print("Vision system shut down.")