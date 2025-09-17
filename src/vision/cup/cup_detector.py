from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class CupDetectionResult:
    detected: bool
    pixel_center: Optional[Tuple[int, int]]
    bounding_box: Optional[Tuple[int, int, int, int]]  # x, y, w, h
    distance_m: Optional[float]


class CupDetector:
    """
    Simple markerless cup detection using color/edge/contour heuristics.

    Assumes a roughly cylindrical cup with near-elliptical rim. Distance is approximated via apparent width and known cup width, or via pinhole model if focal length is provided.
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        assumed_cup_diameter_m: float = 0.08,
    ) -> None:
        self.camera_matrix = camera_matrix
        self.assumed_cup_diameter_m = assumed_cup_diameter_m
        self.fx = float(camera_matrix[0, 0])

    def detect(self, frame_bgr: np.ndarray) -> CupDetectionResult:
        img = frame_bgr
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, None, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_bbox = None
        best_score = -1.0
        best_center = None

        for cnt in contours:
            if len(cnt) < 5:
                continue
            area = cv2.contourArea(cnt)
            if area < 400:  # heuristic to filter noise
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / max(h, 1)
            circularity = 4 * np.pi * area / (cv2.arcLength(cnt, True) ** 2 + 1e-6)

            # Heuristic scoring: prefer near-vertical rectangles with moderate circularity
            score = 0.0
            score += max(0.0, 1.0 - abs(aspect - 0.8))  # cups are taller than wide typically
            score += min(1.0, circularity) * 0.5
            score += min(1.0, area / 5000.0)

            if score > best_score:
                best_score = score
                best_bbox = (int(x), int(y), int(w), int(h))
                best_center = (int(x + w / 2), int(y + h / 2))

        if best_bbox is None:
            return CupDetectionResult(False, None, None, None)

        # Approx distance using pinhole: Z = fx * real_width / pixel_width
        _, _, w, _ = best_bbox
        if w > 0:
            distance_m = self.fx * self.assumed_cup_diameter_m / float(w)
        else:
            distance_m = None

        return CupDetectionResult(True, best_center, best_bbox, distance_m)


