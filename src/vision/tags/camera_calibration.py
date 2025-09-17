from typing import NamedTuple, List, Tuple, Optional, Any
import argparse
import json
import os

import cv2 as cv
import numpy as np


class BoardDetectionResults(NamedTuple):
    charuco_corners: np.ndarray
    charuco_ids: np.ndarray
    aruco_corners: list
    aruco_ids: Optional[np.ndarray]


class PointReferences(NamedTuple):
    object_points: np.ndarray
    image_points: np.ndarray


class CameraCalibrationResults(NamedTuple):
    rep_error: float
    cam_matrix: np.ndarray
    dist_coeffs: np.ndarray
    rvecs: List[np.ndarray]
    tvecs: List[np.ndarray]


def get_aruco_dictionary(name: str) -> int:
    name = name.strip().upper()
    if hasattr(cv.aruco, name):
        return getattr(cv.aruco, name)
    aliases = {
        "DICT_4X4_50": "DICT_4X4_50",
        "DICT_5X5_50": "DICT_5X5_50",
        "DICT_6X6_250": "DICT_6X6_250",
        "ARUCO_ORIGINAL": "DICT_ARUCO_ORIGINAL",
    }
    resolved = aliases.get(name)
    if resolved and hasattr(cv.aruco, resolved):
        return getattr(cv.aruco, resolved)
    raise ValueError(f"Unknown ArUco dictionary: {name}")


def create_charuco_board(
    squares_x: int,
    squares_y: int,
    square_length: float,
    marker_length: float,
    dict_id: int,
):
    dictionary = cv.aruco.getPredefinedDictionary(dict_id)
    try:
        board = cv.aruco.CharucoBoard((int(squares_x), int(squares_y)), float(square_length), float(marker_length), dictionary)
    except Exception:
        board = cv.aruco.CharucoBoard_create(int(squares_x), int(squares_y), float(square_length), float(marker_length), dictionary)
    return board, dictionary


def render_charuco_board_image(board, out_w: int, out_h: int, margin_px: int = 20, border_bits: int = 1) -> np.ndarray:
    try:
        return board.draw((int(out_w), int(out_h)), marginSize=int(margin_px), borderBits=int(border_bits))
    except Exception:
        # Fallbacks for different OpenCV versions
        try:
            img = board.generateImage((int(out_w), int(out_h)), None, int(margin_px), int(border_bits))
            return img
        except Exception:
            img = np.zeros((int(out_h), int(out_w)), dtype=np.uint8)
            try:
                cv.aruco.drawPlanarBoard(board, (int(out_w), int(out_h)), img, int(margin_px), int(border_bits))
                return img
            except Exception as e:
                raise RuntimeError(f"Failed to render ChArUco board: {e}")


def detect_charuco(image_gray: np.ndarray, board) -> BoardDetectionResults:
    # Prefer CharucoDetector if available
    try:
        detector = cv.aruco.CharucoDetector(board)
        charuco_corners, charuco_ids, aruco_corners, aruco_ids = detector.detectBoard(image_gray)
    except Exception:
        # Fallback: detect markers then interpolate
        try:
            params = cv.aruco.DetectorParameters()
        except Exception:
            params = cv.aruco.DetectorParameters_create()
        dictionary = board.dictionary if hasattr(board, "dictionary") else cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
        try:
            aruco_detector = cv.aruco.ArucoDetector(dictionary, params)
            aruco_corners, aruco_ids, _ = aruco_detector.detectMarkers(image_gray)
        except Exception:
            aruco_corners, aruco_ids, _ = cv.aruco.detectMarkers(image_gray, dictionary, parameters=params)
        ok, charuco_corners, charuco_ids = cv.aruco.interpolateCornersCharuco(aruco_corners, aruco_ids, image_gray, board)
        if not ok:
            charuco_corners, charuco_ids = None, None
    return BoardDetectionResults(
        charuco_corners=charuco_corners if charuco_corners is not None else np.empty((0, 1, 2), dtype=np.float32),
        charuco_ids=charuco_ids if charuco_ids is not None else np.empty((0, 1), dtype=np.int32),
        aruco_corners=aruco_corners if aruco_corners is not None else [],
        aruco_ids=aruco_ids if aruco_ids is not None else None,
    )


def _match_image_points(board, charuco_corners: np.ndarray, charuco_ids: np.ndarray) -> PointReferences:
    obj_pts, img_pts = board.matchImagePoints(charuco_corners, charuco_ids)
    return PointReferences(object_points=obj_pts, image_points=img_pts)


def _capture_frames(device: int, samples: int, show: bool = True) -> List[np.ndarray]:
    cap = cv.VideoCapture(device, cv.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera {device}")
    frames: List[np.ndarray] = []
    try:
        if show:
            print("Press SPACE to capture a frame with the ChArUco board visible; press Q to quit.")
            print("Captured: 0/{}".format(samples), end="\r", flush=True)
        while len(frames) < samples:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            if show:
                cv.imshow("Calibration capture", frame)
                key = cv.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                if key == ord(' '):
                    frames.append(frame.copy())
                    print("Captured: {}/{}".format(len(frames), samples), end="\r", flush=True)
            else:
                frames.append(frame.copy())
    finally:
        cap.release()
        if show:
            try:
                cv.destroyAllWindows()
            except Exception:
                pass
            # Move to next line after final update
            print()
    return frames


def calibrate_camera_from_images(images_bgr: List[np.ndarray], board) -> CameraCalibrationResults:
    object_points_all: List[np.ndarray] = []
    image_points_all: List[np.ndarray] = []
    img_size: Optional[Tuple[int, int]] = None

    for img_bgr in images_bgr:
        img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
        img_size = img_gray.shape[::-1]
        det = detect_charuco(img_gray, board)
        if det.charuco_corners.size == 0 or det.charuco_ids.size == 0:
            continue
        refs = _match_image_points(board, det.charuco_corners, det.charuco_ids)
        if refs.object_points.size == 0 or refs.image_points.size == 0:
            continue
        object_points_all.append(refs.object_points)
        image_points_all.append(refs.image_points)

    if not object_points_all or img_size is None:
        raise RuntimeError("Insufficient valid ChArUco detections for calibration")

    rep_error, cam_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
        object_points_all, image_points_all, img_size, None, None
    )

    return CameraCalibrationResults(
        rep_error=float(rep_error),
        cam_matrix=cam_matrix,
        dist_coeffs=dist_coeffs,
        rvecs=rvecs,
        tvecs=tvecs,
    )


def save_calibration_json(
    path: str,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    reprojection_error: Optional[float] = None,
    metadata: Optional[dict] = None,
) -> None:
    data: dict[str, Any] = {
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist(),
    }
    if reprojection_error is not None:
        data["reprojection_error"] = float(reprojection_error)
    if metadata:
        data["metadata"] = metadata
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_calibration_json(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    K = np.array(data["camera_matrix"], dtype=np.float64)
    dist = np.array(data["dist_coeffs"], dtype=np.float64)
    return K, dist


def get_calibration(
    device: int,
    samples: int,
    squares_x: int,
    squares_y: int,
    square_length_m: float,
    marker_length_m: float,
    dict_name: str,
    output_json: str,
    show: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float]:
    dict_id = get_aruco_dictionary(dict_name)
    board, _ = create_charuco_board(
        squares_x=squares_x,
        squares_y=squares_y,
        square_length=square_length_m,
        marker_length=marker_length_m,
        dict_id=dict_id,
    )

    # Save a printable board image (pixel-based) into tags/charuco
    charuco_dir = os.path.join(os.path.dirname(__file__), "charuco")
    os.makedirs(charuco_dir, exist_ok=True)
    printable_square_px = 200  # pixels per square for printing/display
    out_w = int(squares_x * printable_square_px)
    out_h = int(squares_y * printable_square_px)
    board_img = render_charuco_board_image(board, out_w, out_h, margin_px=20, border_bits=1)
    board_filename = f"charuco_{dict_name}_{squares_x}x{squares_y}_{printable_square_px}px.png"
    board_path = os.path.join(charuco_dir, board_filename)
    try:
        cv.imwrite(board_path, board_img)
    except Exception:
        pass

    images = _capture_frames(device=device, samples=samples, show=show)
    results = calibrate_camera_from_images(images, board)
    save_calibration_json(
        output_json,
        results.cam_matrix,
        results.dist_coeffs,
        reprojection_error=results.rep_error,
        metadata={
            "squares_x": squares_x,
            "squares_y": squares_y,
            "square_length": square_length_m,
            "marker_length": marker_length_m,
            "dict": dict_name,
            "device": device,
            "samples": len(images),
        },
    )
    return results.cam_matrix, results.dist_coeffs, results.rep_error


def _cli() -> None:
    parser = argparse.ArgumentParser(description="ChArUco camera calibration")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--squares-x", type=int, default=8)
    parser.add_argument("--squares-y", type=int, default=11)
    parser.add_argument("--square-length", type=float, default=0.025)
    parser.add_argument("--marker-length", type=float, default=0.018)
    parser.add_argument("--dict", type=str, default="DICT_4X4_50")
    parser.add_argument("--output", type=str, default=os.path.join(os.path.dirname(__file__), "camera_calibration.json"))
    parser.add_argument("--save-board", action="store_true", help="Also save a printable ChArUco board image to tags/charuco")
    args = parser.parse_args()

    K, dist, err = get_calibration(
        device=args.device,
        samples=args.samples,
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length_m=args.square_length,
        marker_length_m=args.marker_length,
        dict_name=args.dict,
        output_json=args.output,
        show=True,
    )
    print(f"Saved calibration to {args.output}. Reprojection error: {err:.4f} px")


if __name__ == "__main__":
    _cli()


