# Vision System Scaffold

## Overview
This scaffold provides:
- Tag detection with ArUco using OpenCV
- Markerless cup detection via contour heuristics
- IMU + vision fusion placeholder (complementary filter style)
- Orchestrator that outputs structured dict/JSON per frame

## Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python main.py
```

Press 'q' to quit. Replace camera intrinsics in `main.py` with calibrated values. Adjust `marker_length_m` and `assumed_cup_diameter_m` to match hardware.

## Jetson Orin (YOLOv11) quickstart
- Prefer NVIDIA's PyTorch for Jetson. Install following NVIDIA instructions for your JetPack version, then:
  ```bash
  pip install ultralytics
  ```
- If OpenCV is not present, either install the system OpenCV (`sudo apt install python3-opencv`) or use the wheel in `requirements.txt`.
- Run with YOLOv11 (nano) on GPU:
  ```bash
  python main.py --detector yolo11 --yolo-model /path/to/yolo11n.pt --yolo-device cuda:0 --show
  ```

Notes:
- `ultralytics` is only needed when using `--detector yolo11`. If you stick to contour or ONNX-based YOLO, you can skip it.
- For CPU-only or non-GPU scenarios, omit `--yolo-device`.

## Output structure
```json
{
  "cup_detected": true,
  "cup_relative_position": {"x": 0.12, "y": -0.03, "z": 0.45},
  "arm_pose_corrected": {"matrix": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}
}
```

## Notes
- AprilTag support requires OpenCV built with `aruco` including AprilTag; otherwise this uses ArUco dictionaries.
- IMU fusion uses a simple matrix blend as a placeholder. Replace with Lie-algebra complementary or EKF for production.

