import json
import time
import argparse
import os
from typing import Optional

import cv2
import numpy as np

from system import VisionSystem
from tags.camera_calibration import load_calibration_json


def load_camera_calibration(calibration_path: Optional[str] = None) -> tuple[np.ndarray, np.ndarray]:
    # Try to load from provided path or default utils/camera_calibration.json; fallback to placeholder
    if calibration_path is None:
        default_path = os.path.join(os.path.dirname(__file__), "tags", "camera_calibration.json")
    else:
        default_path = calibration_path
    try:
        if os.path.isfile(default_path):
            return load_calibration_json(default_path)
    except Exception:
        pass
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
    parser.add_argument("--mode", choices=["cup", "tag"], default="cup", help="Select detection pipeline: cup or tag")
    parser.add_argument("--calib", type=str, default=None, help="Path to calibration JSON (defaults to utils/camera_calibration.json)")
    args = parser.parse_args()
    camera_matrix, dist_coeffs = load_camera_calibration(args.calib)
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
                result = system.process_frame(frame, imu_delta, mode=args.mode)
                #print(json.dumps(result), end="\r")
            except Exception:
                # Skip frame on unexpected errors to avoid crashing
                continue

            if args.show:
                try:
                    if args.mode == "cup":
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
                    else:
                        # Visualize detected tags (draw axes and ids)
                        try:
                            tag_result = system.tag_detector.detect_and_estimate(frame)
                            if tag_result.corners and len(tag_result.corners) > 0:
                                cv2.aruco.drawDetectedMarkers(frame, tag_result.corners)
                            if tag_result.rvecs is not None and tag_result.tvecs is not None:
                                for i in range(len(tag_result.tag_ids)):
                                    rvec = tag_result.rvecs[i]
                                    tvec = tag_result.tvecs[i]
                                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)
                                    # draw id and RPY near the first corner
                                    c = tag_result.corners[i][0]
                                    x, y = int(c[0][0]), int(c[0][1])
                                    cv2.putText(frame, str(tag_result.tag_ids[i]), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                                    try:
                                        R, _ = cv2.Rodrigues(rvec.reshape(3,))
                                        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
                                        singular = sy < 1e-6
                                        if not singular:
                                            roll = np.arctan2(R[2, 1], R[2, 2])
                                            pitch = np.arctan2(-R[2, 0], sy)
                                            yaw = np.arctan2(R[1, 0], R[0, 0])
                                        else:
                                            roll = np.arctan2(-R[1, 2], R[1, 1])
                                            pitch = np.arctan2(-R[2, 0], sy)
                                            yaw = 0.0
                                        # degrees
                                        rpy = (np.degrees(roll), np.degrees(pitch), np.degrees(yaw))
                                        txt = f"R:{rpy[0]:.0f} P:{rpy[1]:.0f} Y:{rpy[2]:.0f}"
                                        cv2.putText(frame, txt, (x, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1, cv2.LINE_AA)
                                    except Exception:
                                        pass
                            # Top-right flag: aruco detected yes/no
                            try:
                                detected_any = tag_result is not None and hasattr(tag_result, 'tag_ids') and len(tag_result.tag_ids) > 0
                            except Exception:
                                detected_any = False
                            label = "aruco detected: "
                            status_text = "yes" if detected_any else "no"
                            color = (0, 200, 0) if detected_any else (0, 0, 255)
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            scale = 0.7
                            thickness = 2
                            (label_w, label_h), _ = cv2.getTextSize(label, font, scale, thickness)
                            (stat_w, stat_h), _ = cv2.getTextSize(status_text, font, scale, thickness)
                            margin = 10
                            bg_margin = 6
                            total_w = label_w + stat_w + 10
                            total_h = max(label_h, stat_h)
                            x_right = frame.shape[1] - total_w - margin
                            y_top = margin + total_h
                            # background
                            cv2.rectangle(
                                frame,
                                (x_right - bg_margin, y_top - total_h - bg_margin),
                                (x_right + total_w + bg_margin, y_top + bg_margin // 2),
                                (0, 0, 0),
                                -1,
                            )
                            # draw label and status
                            cv2.putText(frame, label, (x_right, y_top), font, scale, (200, 200, 200), thickness, cv2.LINE_AA)
                            cv2.putText(frame, status_text, (x_right + label_w + 5, y_top), font, scale, color, thickness, cv2.LINE_AA)
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


