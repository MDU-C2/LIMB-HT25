from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import cv2

from .cup_detector import CupDetectionResult


class UltralyticsYOLOCupDetector:
    """
    YOLOv11 via Ultralytics API. Optimized for Jetson Orin; set device appropriately.
    Filters predictions to the 'cup' class and estimates distance from bbox width.
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        weights_path: str,
        assumed_cup_diameter_m: float = 0.08,
        conf: float = 0.35,
        iou: float = 0.45,
        device: Optional[str] = None,
    ) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Ultralytics is required for YOLOv11. Install with 'pip install ultralytics'."
            ) from e

        self._YOLO = YOLO
        self.model = YOLO(weights_path)
        self.conf = float(conf)
        self.iou = float(iou)
        self.device = device  # e.g., 'cuda:0' on Orin
        self.camera_matrix = camera_matrix
        self.fx = float(camera_matrix[0, 0])
        self.assumed_cup_diameter_m = assumed_cup_diameter_m

        # Resolve cup class id from model names (if available), else fallback to 'cup'
        self._cup_class_ids: List[int] = []
        names = getattr(self.model.model, 'names', None)
        if isinstance(names, dict):
            for k, v in names.items():
                if str(v).lower() == 'cup':
                    try:
                        self._cup_class_ids.append(int(k))
                    except Exception:
                        pass
        # Fallback to common COCO id for cup (41)
        if not self._cup_class_ids:
            self._cup_class_ids = [41]

    def detect(self, frame_bgr: np.ndarray) -> CupDetectionResult:
        # Run inference
        try:
            results = self.model(
                source=frame_bgr,  # ndarray input
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                verbose=False,
                stream=False,
            )
        except Exception:
            return CupDetectionResult(False, None, None, None)

        if not results:
            return CupDetectionResult(False, None, None, None)

        result = results[0]
        boxes = getattr(result, 'boxes', None)
        if boxes is None or boxes.xyxy is None:
            return CupDetectionResult(False, None, None, None)

        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else np.array(boxes.xyxy)
        cls = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, 'cls') else None
        confs = boxes.conf.cpu().numpy() if hasattr(boxes, 'conf') else None

        best_idx = -1
        best_conf = -1.0
        if cls is not None:
            for i, c in enumerate(cls):
                if c in self._cup_class_ids:
                    score = float(confs[i]) if confs is not None else 1.0
                    if score > best_conf:
                        best_conf = score
                        best_idx = i

        if best_idx < 0:
            return CupDetectionResult(False, None, None, None)

        x1, y1, x2, y2 = xyxy[best_idx]
        x = int(max(0, x1))
        y = int(max(0, y1))
        w = int(max(1, x2 - x1))
        h = int(max(1, y2 - y1))
        center = (int(x + w / 2), int(y + h / 2))

        distance_m = self.fx * self.assumed_cup_diameter_m / float(max(w, 1))
        return CupDetectionResult(True, center, (x, y, w, h), distance_m)


