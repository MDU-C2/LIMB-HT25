from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class TagDetectionResult:
    tag_ids: List[int]
    rvecs: Optional[np.ndarray]  # shape (N, 1, 3)
    tvecs: Optional[np.ndarray]  # shape (N, 1, 3)
    corners: Optional[List[np.ndarray]]


class TagDetector:
    """
    Detects ArUco (and optionally AprilTag if available in OpenCV build) markers and estimates pose.

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
        # DetectorParameters API changed across versions: try new, fallback to old create
        try:
            self.detector_params = cv2.aruco.DetectorParameters()
        except Exception:
            # Older API
            self.detector_params = cv2.aruco.DetectorParameters_create()

        # Disable refine by default for compatibility; can be enabled if available
        self.refine = refine and hasattr(cv2.aruco, "refineDetectedMarkers")
        self._has_refine = self.refine
        self.refine_params = None
        if self._has_refine:
            try:
                self.refine_params = getattr(cv2.aruco, "RefineParameters", lambda: None)()
            except Exception:
                self.refine_params = None

        # ArUcoDetector exists in newer OpenCV; fallback to legacy detectMarkers otherwise
        self.detector = None
        try:
            self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.detector_params)
        except Exception:
            self.detector = None

    def detect_and_estimate(self, frame_bgr: np.ndarray) -> TagDetectionResult:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        # Detect markers (new API if available, else legacy)
        try:
            if self.detector is not None:
                corners, ids, rejected = self.detector.detectMarkers(gray)
            else:
                corners, ids, rejected = cv2.aruco.detectMarkers(
                    gray, self.dictionary, parameters=self.detector_params
                )
        except Exception:
            return TagDetectionResult(tag_ids=[], rvecs=None, tvecs=None, corners=[])

        # Optional refinement (skip if unavailable)
        if self.refine and self._has_refine and self.refine_params is not None:
            if isinstance(rejected, (list, tuple)) and len(rejected) > 0 and len(corners) > 0:
                try:
                    cv2.aruco.refineDetectedMarkers(
                        image=gray,
                        board=None,
                        detectedCorners=corners,
                        detectedIds=ids,
                        rejectedCorners=rejected,
                        cameraMatrix=self.camera_matrix,
                        distCoeffs=self.dist_coeffs,
                    )
                except Exception:
                    pass

        if ids is None or len(ids) == 0:
            return TagDetectionResult(tag_ids=[], rvecs=None, tvecs=None, corners=[])

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, self.marker_length_meters, self.camera_matrix, self.dist_coeffs
        )

        # ids is Nx1; flatten to list of ints
        tag_ids = [int(x[0]) for x in ids]
        return TagDetectionResult(tag_ids=tag_ids, rvecs=rvecs, tvecs=tvecs, corners=corners)

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


