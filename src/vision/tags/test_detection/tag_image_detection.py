import cv2
import numpy as np

import argparse
import imutils
import sys

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-t", "--type", type=str, default="DICT_4X4_50", help="type of ArUco tag to detect")

args = vars(ap.parse_args())


image = cv2.imread(args["image"])
resized_image = imutils.resize(image, width=600)

#blurred = cv2.GaussianBlur(resized_image, (5, 5), 0)
gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
}

if ARUCO_DICT.get(args["type"], None) is None:
    print("[INFO] ArUCo tag type '{}' is not supported".format(args["type"]))
    sys.exit(0)

aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args["type"]])
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

corners, ids, rejected = detector.detectMarkers(gray)

# Verify *at least* one ArUco marker was detected
if len(corners) > 0:
    ids = ids.flatten()

    for (markerCorner, markerID) in zip(corners, ids):
        corners = markerCorner.reshape((4, 2))
        (topLeft, topRight, bottomRight, bottomLeft) = corners
        
        topRight = (int(topRight[0]), int(topRight[1]))
        bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
        bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
        topLeft = (int(topLeft[0]), int(topLeft[1]))

        cv2.line(resized_image, topLeft, topRight, (0, 255, 0), 2)
        cv2.line(resized_image, topRight, bottomRight, (0, 255, 0), 2)
        cv2.line(resized_image, bottomRight, bottomLeft, (0, 255, 0), 2)
        cv2.line(resized_image, bottomLeft, topLeft, (0, 255, 0), 2)

        cX = int((topLeft[0] + bottomRight[0]) / 2.0)
        cY = int((topLeft[1] + bottomRight[1]) / 2.0)
        cv2.circle(resized_image, (cX, cY), 4, (0, 0, 255), -1)
        
        cv2.putText(resized_image, str(markerID), (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        print("[INFO] ArUco marker ID: {}".format(markerID))
else:
    print("[INFO] No ArUco markers detected")

cv2.imshow("Image", resized_image)
cv2.waitKey(0)

