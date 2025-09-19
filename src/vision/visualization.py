import cv2
import numpy as np

from tags.tag_detector import TagDetectionResult
from cup import CupDetectionResult

def draw_axis_on_tag(frame, camera_matrix, dist_coeffs, rvecs, tvecs, length=0.01):
    """
    Draw coordinate axes on the frame.
    
    Args:
        frame: Input frame
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        rvec: Rotation vector
        tvec: Translation vector
        length: Length of the axes
    """

    if rvecs is None or tvecs is None:
        return frame
    try:
        # Define the axis points in 3D space
        axis_points = np.float32([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, -length]]).reshape(-1, 3)
        
        # Reshape rvec and tvec to the correct format for projectPoints
        # estimatePoseSingleMarkers returns arrays with shape (1, 1, 3), so we need to flatten them first
        rvec = rvecs[0].flatten().reshape(3, 1)  # Convert from (1, 1, 3) to (3, 1)
        tvec = tvecs[0].flatten().reshape(3, 1)  # Convert from (1, 1, 3) to (3, 1)

        # Project 3D points to 2D
        imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
        imgpts = np.int32(imgpts).reshape(-1, 2)
        
        # Draw the axes
        frame = cv2.line(frame, tuple(imgpts[0]), tuple(imgpts[1]), (0, 0, 255), 3)  # X axis - Red
        frame = cv2.line(frame, tuple(imgpts[0]), tuple(imgpts[2]), (0, 255, 0), 3)  # Y axis - Green
        frame = cv2.line(frame, tuple(imgpts[0]), tuple(imgpts[3]), (255, 0, 0), 3)  # Z axis - Blue
    except Exception as e:
        print(f"Error drawing axes: {e}")

    return frame


def visualize_tag_detection(frame_bgr: np.ndarray, tag_detection_result: TagDetectionResult, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> np.ndarray:
    '''
    Visualize the tag detection result.
    '''

    try:
        
        if tag_detection_result.tag_ids is not None and len(tag_detection_result.tag_ids) > 0 and tag_detection_result.corners is not None and len(tag_detection_result.corners) > 0:
            # Draw if tags and corners are not None
            cv2.aruco.drawDetectedMarkers(frame_bgr, corners=tag_detection_result.corners, ids=np.array(tag_detection_result.tag_ids))
        
            if tag_detection_result.rvecs is not None and tag_detection_result.tvecs is not None:
                frame_bgr = draw_axis_on_tag(frame_bgr, camera_matrix, dist_coeffs, rvecs=tag_detection_result.rvecs, tvecs=tag_detection_result.tvecs, length=0.01)

    except Exception as e:
        print(f"Error visualizing tag detection: {e}")

    return frame_bgr

def visualize_cup_detection(frame_bgr: np.ndarray, cup_detection_result: CupDetectionResult) -> np.ndarray:

    if cup_detection_result.detected and cup_detection_result.bounding_box is not None:
        x, y, w, h = cup_detection_result.bounding_box
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if cup_detection_result.pixel_center is not None:
            cx, cy = cup_detection_result.pixel_center
            cv2.circle(frame_bgr, (cx, cy), 3, (0, 255, 0), -1)
    # Draw detection flag in top-right corner
    flag_text = "True" if cup_detection_result.detected else "False"
    color = (0, 200, 0) if cup_detection_result.detected else (0, 0, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(flag_text, font, scale, thickness)
    margin = 10
    x_right = frame_bgr.shape[1] - text_w - margin
    y_top = margin + text_h
    bg_margin = 6
    cv2.rectangle(
        frame_bgr,
        (x_right - bg_margin, y_top - text_h - bg_margin),
        (x_right + text_w + bg_margin, y_top + bg_margin // 2),
        (0, 0, 0),
        -1,
    )
    cv2.putText(frame_bgr, flag_text, (x_right, y_top), font, scale, color, thickness, cv2.LINE_AA)

    return frame_bgr