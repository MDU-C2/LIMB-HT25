#!/usr/bin/env python3
"""
Extrinsic Calibration Script for OAK-D Camera

This script performs extrinsic calibration by detecting AprilTags and computing
the transformation from the camera coordinate frame to a world/reference frame.

The reference frame is defined by placing AprilTag ID 0 at the origin with:
- Tag plane as the XY plane
- Z-axis pointing out of the tag (towards camera)

Usage:
    python extrinsic_calibration.py

Controls:
    - Press 's' to save current calibration
    - Press 'r' to reset calibration
    - Press 'q' to quit
"""

import depthai as dai
import numpy as np
import cv2
import time
import json
from pathlib import Path
from typing import Optional, Tuple, Dict


class ExtrinsicCalibrator:
    """
    Extrinsic calibration using AprilTags to establish camera-to-world transform.
    """
    
    def __init__(
        self,
        tag_size: float = 0.1,  # Tag size in meters (10 cm default)
        reference_tag_id: int = 0,  # Which tag defines the world origin
        apriltag_family: str = "TAG36H11",
        output_file: str = "extrinsic_calibration.json"
    ):
        """
        Initialize the extrinsic calibrator.
        
        Args:
            tag_size: Physical size of AprilTag in meters
            reference_tag_id: ID of the tag that defines world origin
            apriltag_family: AprilTag family to detect
            output_file: Where to save calibration results
        """
        self.tag_size = tag_size
        self.reference_tag_id = reference_tag_id
        self.apriltag_family = apriltag_family
        self.output_file = output_file
        
        # Calibration data
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.T_world_camera: Optional[np.ndarray] = None  # 4x4 transformation matrix
        self.calibration_valid = False
        
        # Statistics for averaging multiple measurements
        self.rotation_samples = []
        self.translation_samples = []
        self.max_samples = 30
        
    def _get_apriltag_family(self) -> dai.AprilTagConfig.Family:
        """Convert string family name to DepthAI enum."""
        family_map = {
            "TAG36H11": dai.AprilTagConfig.Family.TAG_36H11,
            "TAG25H9": dai.AprilTagConfig.Family.TAG_25H9,
            "TAG16H5": dai.AprilTagConfig.Family.TAG_16H5,
        }
        return family_map.get(self.apriltag_family.upper(), dai.AprilTagConfig.Family.TAG_36H11)
    
    def create_pipeline(self) -> Tuple[dai.Pipeline, dai.Device]:
        """Create and configure the DepthAI pipeline for AprilTag detection."""
        device = dai.Device()
        pipeline = dai.Pipeline(device)
        
        # Create RGB camera
        cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        
        # Create AprilTag detector with optimized settings
        apriltag_node = pipeline.create(dai.node.AprilTag)
        apriltag_node.initialConfig.setFamily(self._get_apriltag_family())
        
        # Optimize for better detection
        apriltag_node.initialConfig.setQuadDecimate(1.5)
        apriltag_node.initialConfig.setQuadSigma(1.0)
        apriltag_node.initialConfig.setRefineEdges(True)
        apriltag_node.initialConfig.setMaxHammingDistance(1)
        
        # Preprocessing: convert to grayscale
        manip = pipeline.create(dai.node.ImageManip)
        manip.initialConfig.setOutputSize(1280, 800)  # Higher res for calibration
        manip.initialConfig.setFrameType(dai.ImgFrame.Type.GRAY8)
        
        # Link pipeline
        cam_rgb.requestOutput((1280, 800)).link(manip.inputImage)
        manip.out.link(apriltag_node.inputImage)
        
        # Create output queues
        apriltag_queue = apriltag_node.out.createOutputQueue()
        passthrough_queue = apriltag_node.passthroughInputImage.createOutputQueue()
        
        return pipeline, device, apriltag_queue, passthrough_queue
    
    def get_camera_intrinsics(self, device: dai.Device) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract camera intrinsics from the OAK-D device calibration.
        
        Returns:
            camera_matrix: 3x3 intrinsic camera matrix
            dist_coeffs: Distortion coefficients
        """
        calib = device.readCalibration()
        
        # Get intrinsics for RGB camera (CAM_A)
        intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, 1280, 800)
        
        # Build camera matrix
        camera_matrix = np.array([
            [intrinsics[0][0], 0, intrinsics[0][2]],
            [0, intrinsics[1][1], intrinsics[1][2]],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Get distortion coefficients
        dist_coeffs = np.array(calib.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A), dtype=np.float32)
        
        return camera_matrix, dist_coeffs
    
    def estimate_tag_pose(
        self,
        corners: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Estimate 6-DOF pose of an AprilTag.
        
        Args:
            corners: 4x2 array of tag corner pixel coordinates
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: Distortion coefficients
            
        Returns:
            rvec: Rotation vector (None if estimation fails)
            tvec: Translation vector (None if estimation fails)
        """
        # Define 3D coordinates of tag corners in tag's coordinate frame
        # Tag center is at origin, corners ordered: TL, TR, BR, BL
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
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        
        if success:
            return rvec, tvec
        return None, None
    
    def compute_transformation_matrix(
        self,
        rvec: np.ndarray,
        tvec: np.ndarray
    ) -> np.ndarray:
        """
        Convert rotation vector and translation to 4x4 transformation matrix.
        
        Args:
            rvec: 3x1 rotation vector
            tvec: 3x1 translation vector
            
        Returns:
            4x4 homogeneous transformation matrix
        """
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        return T
    
    def process_frame(
        self,
        tags: list,
        frame: np.ndarray
    ) -> bool:
        """
        Process detected AprilTags and update calibration.
        
        Args:
            tags: List of detected AprilTags from DepthAI
            frame: Current video frame for visualization
            
        Returns:
            True if reference tag was detected and processed
        """
        reference_tag_found = False
        
        for tag in tags:
            if tag.id == self.reference_tag_id:
                reference_tag_found = True
                
                # Extract corners
                corners = np.array([
                    [tag.topLeft.x, tag.topLeft.y],
                    [tag.topRight.x, tag.topRight.y],
                    [tag.bottomRight.x, tag.bottomRight.y],
                    [tag.bottomLeft.x, tag.bottomLeft.y]
                ], dtype=np.float32)
                
                # Estimate pose
                rvec, tvec = self.estimate_tag_pose(
                    corners,
                    self.camera_matrix,
                    self.dist_coeffs
                )
                
                if rvec is not None and tvec is not None:
                    # Store samples for averaging
                    self.rotation_samples.append(rvec.copy())
                    self.translation_samples.append(tvec.copy())
                    
                    # Keep only recent samples
                    if len(self.rotation_samples) > self.max_samples:
                        self.rotation_samples.pop(0)
                        self.translation_samples.pop(0)
                    
                    # Compute T_camera_tag (tag pose in camera frame)
                    T_camera_tag = self.compute_transformation_matrix(rvec, tvec)
                    
                    # Invert to get T_tag_camera (camera pose in tag/world frame)
                    # Since tag defines world origin, T_world_camera = T_tag_camera
                    self.T_world_camera = np.linalg.inv(T_camera_tag)
                    self.calibration_valid = True
                    
                    # Draw coordinate axes on tag
                    self.draw_axes(frame, rvec, tvec, self.camera_matrix, self.dist_coeffs)
                
                # Draw tag outline
                self.draw_tag(frame, corners, tag.id, is_reference=True)
            else:
                # Draw non-reference tags in different color
                corners = np.array([
                    [tag.topLeft.x, tag.topLeft.y],
                    [tag.topRight.x, tag.topRight.y],
                    [tag.bottomRight.x, tag.bottomRight.y],
                    [tag.bottomLeft.x, tag.bottomLeft.y]
                ], dtype=np.float32)
                self.draw_tag(frame, corners, tag.id, is_reference=False)
        
        return reference_tag_found
    
    def draw_tag(
        self,
        frame: np.ndarray,
        corners: np.ndarray,
        tag_id: int,
        is_reference: bool
    ):
        """Draw tag outline and ID on frame."""
        color = (0, 255, 0) if is_reference else (255, 0, 0)
        
        # Draw quadrilateral
        for i in range(4):
            pt1 = tuple(corners[i].astype(int))
            pt2 = tuple(corners[(i + 1) % 4].astype(int))
            cv2.line(frame, pt1, pt2, color, 2)
        
        # Draw tag ID
        center = corners.mean(axis=0).astype(int)
        label = f"ID {tag_id}" + (" (REF)" if is_reference else "")
        cv2.putText(
            frame, label, tuple(center),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )
    
    def draw_axes(
        self,
        frame: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        length: float = 0.05
    ):
        """Draw 3D coordinate axes on the tag."""
        # Define axis endpoints in tag frame
        axis_points = np.array([
            [0, 0, 0],           # Origin
            [length, 0, 0],      # X-axis (red)
            [0, length, 0],      # Y-axis (green)
            [0, 0, length]       # Z-axis (blue)
        ], dtype=np.float32)
        
        # Project to image plane
        img_points, _ = cv2.projectPoints(
            axis_points, rvec, tvec, camera_matrix, dist_coeffs
        )
        img_points = img_points.reshape(-1, 2).astype(int)
        
        # Draw axes
        origin = tuple(img_points[0])
        cv2.line(frame, origin, tuple(img_points[1]), (0, 0, 255), 3)  # X: red
        cv2.line(frame, origin, tuple(img_points[2]), (0, 255, 0), 3)  # Y: green
        cv2.line(frame, origin, tuple(img_points[3]), (255, 0, 0), 3)  # Z: blue
    
    def save_calibration(self):
        """Save extrinsic calibration to JSON file."""
        if not self.calibration_valid:
            print("No valid calibration to save!")
            return
        
        # Average rotation and translation samples for stability
        avg_rvec = np.mean(self.rotation_samples, axis=0)
        avg_tvec = np.mean(self.translation_samples, axis=0)
        
        # Recompute final transformation with averaged values
        T_camera_tag = self.compute_transformation_matrix(avg_rvec, avg_tvec)
        T_world_camera = np.linalg.inv(T_camera_tag)
        
        # Prepare calibration data
        calib_data = {
            "tag_size_meters": self.tag_size,
            "reference_tag_id": self.reference_tag_id,
            "apriltag_family": self.apriltag_family,
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "T_world_camera": T_world_camera.tolist(),
            "num_samples": len(self.rotation_samples),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save to file
        output_path = Path(self.output_file)
        with open(output_path, 'w') as f:
            json.dump(calib_data, f, indent=2)
        
        print(f"Calibration saved to {output_path}")
        print(f"   Averaged over {len(self.rotation_samples)} samples")
    
    def reset_calibration(self):
        """Reset calibration samples."""
        self.rotation_samples.clear()
        self.translation_samples.clear()
        self.T_world_camera = None
        self.calibration_valid = False
        print("Calibration reset")
    
    def display_status(self, frame: np.ndarray):
        """Display calibration status on frame."""
        h, w = frame.shape[:2]
        
        # Status background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Status text
        status = "CALIBRATED" if self.calibration_valid else "NOT CALIBRATED"
        color = (0, 255, 0) if self.calibration_valid else (0, 0, 255)
        cv2.putText(frame, f"Status: {status}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        cv2.putText(frame, f"Reference Tag: ID {self.reference_tag_id}", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(frame, f"Samples: {len(self.rotation_samples)}/{self.max_samples}", (20, 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(frame, "Press 's' to save | 'r' to reset | 'q' to quit", (20, 125),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def run(self):
        """Main calibration loop."""
        print("=" * 60)
        print("Extrinsic Calibration - OAK-D Camera")
        print("=" * 60)
        print(f"\nPlace AprilTag ID {self.reference_tag_id} to define world origin")
        print(f"   Tag size: {self.tag_size * 100:.1f} cm")
        print(f"   Family: {self.apriltag_family}")
        print("\nInstructions:")
        print("   1. Position the reference tag in view")
        print("   2. Move camera to different angles for better calibration")
        print("   3. Press 's' when ready to save calibration")
        print("   4. Press 'r' to reset and start over")
        print("   5. Press 'q' to quit")
        print("\n" + "=" * 60 + "\n")
        
        # Create pipeline
        pipeline, device, apriltag_queue, passthrough_queue = self.create_pipeline()
        
        # Get camera intrinsics
        self.camera_matrix, self.dist_coeffs = self.get_camera_intrinsics(device)
        print(f"Camera intrinsics loaded:")
        print(f" fx={self.camera_matrix[0,0]:.2f}, fy={self.camera_matrix[1,1]:.2f}")
        print(f" cx={self.camera_matrix[0,2]:.2f}, cy={self.camera_matrix[1,2]:.2f}\n")
        
        # Start pipeline
        pipeline.start()
        
        try:
            while pipeline.isRunning():
                # Get AprilTag detections
                apriltag_msg = apriltag_queue.get()
                passthrough_msg = passthrough_queue.get()
                
                # Convert frame to color for visualization
                frame = passthrough_msg.getCvFrame()
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                
                # Process tags
                tags = apriltag_msg.aprilTags
                reference_found = self.process_frame(tags, frame)
                
                # Display status
                self.display_status(frame)
                
                # Show frame
                cv2.imshow("Extrinsic Calibration", frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nExiting calibration...")
                    break
                elif key == ord('s'):
                    self.save_calibration()
                elif key == ord('r'):
                    self.reset_calibration()
                
        except KeyboardInterrupt:
            print("\n\nInterrupted by user (Ctrl+C)")
        finally:
            pipeline.stop()
            cv2.destroyAllWindows()
            print("Calibration session ended")


def main():
    """Run extrinsic calibration."""
    # Configuration
    TAG_SIZE = 0.10  # 10 cm tags
    REFERENCE_TAG_ID = 0
    APRILTAG_FAMILY = "TAG36H11"
    OUTPUT_FILE = "extrinsic_calibration.json"
    
    # Create and run calibrator
    calibrator = ExtrinsicCalibrator(
        tag_size=TAG_SIZE,
        reference_tag_id=REFERENCE_TAG_ID,
        apriltag_family=APRILTAG_FAMILY,
        output_file=OUTPUT_FILE
    )
    
    calibrator.run()


if __name__ == "__main__":
    main()
