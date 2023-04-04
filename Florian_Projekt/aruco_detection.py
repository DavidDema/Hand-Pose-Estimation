import cv2
from cv2 import aruco
from cvzone.HandTrackingModule import HandDetector
import yaml
import math
import numpy as np
import cvzone
import matplotlib.pyplot as plt
from helper_functions import draw_hand
from object_detector import *

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)
# Webcam
camera = cv2.VideoCapture(0)
#camera.set(3, 1280)
#camera.set(4, 720)
ret, img = camera.read()

with open('calibration.yaml') as f:
    loadeddict = yaml.safe_load(f)
mtx = loadeddict.get('camera_matrix')
dist = loadeddict.get('dist_coeff')
mtx = np.array(mtx)
dist = np.array(dist)

ret, img = camera.read()
img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
h,  w = img_gray.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

aruco_params = aruco.DetectorParameters_create()

# Define the length of the ArUco marker side in meters
marker_length = 0.04

detector = HomogeneousBgDetector()

while True:
    # Capture a frame from the camera
    ret, frame = camera.read()
    if not ret:
        break

    # Detect ArUco markers in the frame
    corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)
    if ids:
        # Draw the detected markers on the frame
        frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Estimate the pose of each detected marker
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, mtx, dist)

        # Calculate the distance to each detected marker
        distances = []
        for i in range(len(ids)):
            distance = np.linalg.norm(tvecs[i])
            distances.append(distance)
            org = (corners[i][0][0][0].astype(int), corners[i][0][0][1].astype(int))
            cv2.putText(frame, f"ID: {ids[i]} Koordinaten:  x: {tvecs[i][0][0]:.2f}m", org,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame,
                        f"y: {tvecs[i][0][1]:.2f}m",(corners[i][0][0][0].astype(int)+185, corners[i][0][0][1].astype(int)+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame,
                        f"z: {tvecs[i][0][2]:.2f}m",
                        (corners[i][0][0][0].astype(int) + 185, corners[i][0][0][1].astype(int) + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


            # Draw the pose axes of the detected markers
            for rvec, tvec in zip(rvecs, tvecs):
                # Project the axes points onto the image
                axis_points, _ = cv2.projectPoints(np.float32([[0.005, 0, 0], [0, 0.005, 0], [0, 0, -0.005]]), rvec, tvec,
                                                   mtx, dist)

                # Draw the axes lines on the image
                image = cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, 0.05)

        # Show the frame
        # contours = detector.detect_objects(frame)
        #
        # # Draw objects boundaries
        # for cnt in contours:
        #     # Get rect
        #     rect = cv2.minAreaRect(cnt)
        #     (x, y), (w, h), angle = rect
        #     marker_corner = np.array(corners[0][0][0])
        #     if (x - w/2 < marker_corner[0] < x + w/2) and (y - h/2 < marker_corner[1] < y + h/2):
        #         # Display rectangle
        #         box = cv2.boxPoints(rect)
        #         box = np.intp(box)
        #         cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
        #         cv2.polylines(frame, [box], True, (255, 0, 0), 2)


    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:  # Press ESC to exit
        break

# Clean up
camera.release()
cv2.destroyAllWindows()