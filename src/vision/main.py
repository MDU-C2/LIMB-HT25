import json
import time
import argparse
from typing import Optional

import cv2
import numpy as np

from system import VisionSystem


def load_camera_calibration() -> tuple[np.ndarray, np.ndarray]:
    # Placeholder camera intrinsics (fx, fy, cx, cy). Replace with real calibration.
    fx, fy = 800.0, 800.0
    cx, cy = 640.0, 360.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((5, 1), dtype=np.float64)
    return K, dist


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true", help="Display camera frames in a window")
    parser.add_argument("--detector", choices=["contour", "yolo11"], default="contour", help="Cup detector backend")
    parser.add_argument("--yolo-model", type=str, default=None, help="Path to YOLOv11 .pt model (used with --detector yolo11)")
    parser.add_argument("--yolo-device", type=str, default=None, help="Device for yolo11 (e.g., cuda:0 on Jetson)")
    args = parser.parse_args()
    camera_matrix, dist_coeffs = load_camera_calibration()
    system = VisionSystem(
        camera_matrix,
        dist_coeffs,
        marker_length_m=0.03,
        detector_kind=args.detector,
        yolo_model_path=args.yolo_model,
        yolo_device=args.yolo_device,
    )

    # Prefer AVFoundation on macOS for stability
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    # Reduce internal buffering to lower latency and potential threading issues
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera 0")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # Optional: Provide IMU delta transform per frame. Using identity here as placeholder.
            imu_delta = np.eye(4, dtype=np.float64)
            try:
                result = system.process_frame(frame, imu_delta)
                #print(json.dumps(result), end="\r")
            except Exception:
                # Skip frame on unexpected errors to avoid crashing
                continue

            if args.show:
                try:
                    # Draw cup bounding box for visualization
                    try:
                        cup_det = system.cup_detector.detect(frame)
                        if cup_det.detected and cup_det.bounding_box is not None:
                            x, y, w, h = cup_det.bounding_box
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            if cup_det.pixel_center is not None:
                                cx, cy = cup_det.pixel_center
                                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
                        # Draw detection flag in top-right corner
                        flag_text = "True" if cup_det.detected else "False"
                        color = (0, 200, 0) if cup_det.detected else (0, 0, 255)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        scale = 0.8
                        thickness = 2
                        (text_w, text_h), _ = cv2.getTextSize(flag_text, font, scale, thickness)
                        margin = 10
                        x_right = frame.shape[1] - text_w - margin
                        y_top = margin + text_h
                        # Optional background for readability
                        bg_margin = 6
                        cv2.rectangle(
                            frame,
                            (x_right - bg_margin, y_top - text_h - bg_margin),
                            (x_right + text_w + bg_margin, y_top + bg_margin // 2),
                            (0, 0, 0),
                            -1,
                        )
                        cv2.putText(frame, flag_text, (x_right, y_top), font, scale, color, thickness, cv2.LINE_AA)
                    except Exception:
                        pass
                    cv2.imshow("Vision", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except Exception:
                    # If GUI not available, continue headless
                    pass

            # Throttle loop slightly
            time.sleep(0.005 if args.show else 0.01)
    finally:
        cap.release()
        if 'args' in locals() and args.show:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


if __name__ == "__main__":
    main()


