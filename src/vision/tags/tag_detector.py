from dataclasses import dataclass
from typing import List, Optional, Tuple
import time

import cv2
import numpy as np


@dataclass
class TagDetectionResult:
    tag_ids: np.ndarray               # shape (N,)
    rvecs: Optional[np.ndarray]       # shape (N, 1, 3)
    tvecs: Optional[np.ndarray]       # shape (N, 1, 3)
    corners: Optional[List[np.ndarray]]
    transforms: Optional[List[np.ndarray]]  # shape (N, 4, 4)
    reproj_errors: Optional[np.ndarray]     # shape (N,)
    timestamps: Optional[List[float]]       # optional: useful for sync


class TagDetector:
    """
    Detects ArUco markers and estimates pose.

    - camera_matrix: 3x3 intrinsics matrix
    - dist_coeffs: distortion coefficients array
    - marker_length_meters: physical size of the marker in meters (edge length)
    - aruco_dictionary: cv2.aruco predefined dictionary name, e.g., cv2.aruco.DICT_4X4_50
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        marker_length_meters: float,
        aruco_dictionary: int = cv2.aruco.DICT_4X4_50,
        refine: bool = True,
    ) -> None:
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.marker_length_meters = marker_length_meters
        self.dictionary = cv2.aruco.getPredefinedDictionary(aruco_dictionary)
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.detector_params)


    def detect_and_estimate(self, frame_bgr: np.ndarray) -> TagDetectionResult:
        '''
        Detect and estimate the pose of the ArUco markers in the frame.

        Args:
            frame_bgr: The input frame in BGR format.

        Returns:
            TagDetectionResult: The detection result containing the tag IDs, rvecs, tvecs, and corners.
        '''
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    
        corners, ids, rejected = self.detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            return TagDetectionResult(
                tag_ids=np.array([]), 
                rvecs=None, 
                tvecs=None, 
                corners=None,
                transforms=None,
                reproj_errors=None,
                timestamps=None
            )

        rvecs = None
        tvecs = None
        reproj_errors = None
        
        if len(corners) > 0:
            rvecs, tvecs, reproj_errors = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_length_meters, self.camera_matrix, self.dist_coeffs
            )

        # Convert rvecs and tvecs to 4x4 transform matrices
        transforms = None
        if rvecs is not None and tvecs is not None:
            transforms = []
            for i in range(len(rvecs)):
                transform = self.rvec_tvec_to_matrix(rvecs[i], tvecs[i])
                transforms.append(transform)

        # Create timestamps (optional - can be None if not needed)
        timestamps = [time.time()] * len(ids) if ids is not None else None

        return TagDetectionResult(
            tag_ids=ids.flatten(),  # Flatten from (N, 1) to (N,)
            rvecs=rvecs, 
            tvecs=tvecs, 
            corners=corners,
            transforms=transforms,
            reproj_errors=reproj_errors,
            timestamps=timestamps
        )

    @staticmethod
    def rvec_tvec_to_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """Convert Rodrigues rvec and tvec to a 4x4 transform matrix from tag to camera."""
        rot_mat, _ = cv2.Rodrigues(rvec.reshape(3,))
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rot_mat
        T[:3, 3] = tvec.reshape(3,)
        return T

    @staticmethod
    def invert_transform(T: np.ndarray) -> np.ndarray:
        """Invert a 4x4 transform matrix."""
        R = T[:3, :3]
        t = T[:3, 3]
        T_inv = np.eye(4, dtype=T.dtype)
        T_inv[:3, :3] = R.T
        T_inv[:3, 3] = -R.T @ t
        return T_inv


