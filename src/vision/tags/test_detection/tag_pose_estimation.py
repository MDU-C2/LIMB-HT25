import numpy as np
import cv2
import argparse
import sys
import json
from typing import Optional
import os


def draw_axis(frame, camera_matrix, dist_coeffs, rvec, tvec, length=0.01):
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
    # Define the axis points in 3D space
    axis_points = np.float32([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]]).reshape(-1, 3)
    
    # Reshape rvec and tvec to the correct format for projectPoints
    # estimatePoseSingleMarkers returns arrays with shape (1, 1, 3), so we need to flatten them first
    rvec = rvec[0].flatten().reshape(3, 1)  # Convert from (1, 1, 3) to (3, 1)
    tvec = tvec[0].flatten().reshape(3, 1)  # Convert from (1, 1, 3) to (3, 1)

    # Project 3D points to 2D
    imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = np.int32(imgpts).reshape(-1, 2)
    
    # Draw the axes
    frame = cv2.line(frame, tuple(imgpts[0]), tuple(imgpts[1]), (0, 0, 255), 3)  # X axis - Red
    frame = cv2.line(frame, tuple(imgpts[0]), tuple(imgpts[2]), (0, 255, 0), 3)  # Y axis - Green
    frame = cv2.line(frame, tuple(imgpts[0]), tuple(imgpts[3]), (255, 0, 0), 3)  # Z axis - Blue
    
    return frame
        

def pose_estimation(frame: np.ndarray, aruco_dict: int, matrix_coefficients: np.ndarray, distortion_coefficients: np.ndarray) -> np.ndarray:
    
    '''
    Pose estimation using ArUco markers.

    Args:
        frame: The input frame.
        aruco_dict: The ArUco dictionary.
        matrix_coefficients: The matrix coefficients.
        distortion_coefficients: The distortion coefficients.

    Returns:
        Frame with the pose estimation.
    '''

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict)
    detector_params = cv2.aruco.DetectorParameters()

    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
    
    corners, ids, rejected = detector.detectMarkers(gray)
    
    if len(corners) > 0:
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.02, matrix_coefficients, distortion_coefficients)
        # Draw square around the marker
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        # Draw axis of the marker
        frame = draw_axis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)
    return frame

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    #parser.add_argument("-t", "--type", type=str, default="DICT_4X4_50", help="ArUco dictionary")
    parser.add_argument("-c", "--camera-calibration", type=str, default="camera_calibration.json", help="Camera calibration")

    args = parser.parse_args()

    ARUCO_DICT = cv2.aruco.DICT_4X4_50

    #if ARUCO_DICT.get(args["type"], None) is None:
    #    print("[INFO] ArUCo tag type '{}' is not supported".format(args["type"]))
    #    sys.exit(0)

    aruco_dict_type = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    #calibration_matrix, distortion_coefficients = load_calibration_json("calibration.json")

    # Hardcoded calibration matrix and distortion coefficients
    K = np.array([[
      952.5707541721167,
      0.0,
      658.4498324003038
    ],
    [
      0.0,
      975.7839999332679,
      387.07676314277694
    ],
    [
      0.0,
      0.0,
      1.0
    ]])
    d = np.array([
      0.07808309235991324,
      -0.1401252684293918,
      -0.012108901162419902,
      0.011931374485882535,
      -0.04680128610403889
    ])

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = pose_estimation(frame, ARUCO_DICT, K, d)
        
        cv2.imshow("Estimated Pose", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()