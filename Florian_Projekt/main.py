import cv2
from cv2 import aruco
from cvzone.HandTrackingModule import HandDetector
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import yaml
import math
import numpy as np
import cvzone
import matplotlib.pyplot as plt
from helper_functions import *


thumb_landmark_index = 8
# Webcam
cap = cv2.VideoCapture(0)
#cap.set(3, 1280)
#cap.set(4, 720)
ret, img = cap.read()

mtx, dist, aruco_params, aruco_dict = start_aruco()
pixel_size = (mtx[0][0] + mtx[1][1]) / 2
marker_size = 0.04

# Hand Detector
#detector = HandDetector(detectionCon=0.8, maxHands=2)
mp_hands = mp.solutions.hands.Hands()

fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(projection='3d')
# Loop
while True:
    success, img = cap.read()
    #hands, img = detector.findHands(img)
    #results = mp_hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #landmarks = results.multi_hand_landmarks
    #landmarks = results.multi_hand_landmarks
    #print(landmarks)
    #corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img, aruco_dict, parameters=aruco_params)
    #if hands:
        # lmList = hands[0]['lmList']
        # ax.cla()
        # ax.set_xlim3d(0, 1300)
        # ax.set_ylim3d(0, 1300)
        # ax.set_zlim3d(-200, 200)
        # draw_hand(ax,np.array(lmList),'k-', 'first', 's')
        # if len(hands) == 2:
        #     lmList2 = hands[1]['lmList']
        #     #draw_hand(ax, np.array(lmList2), 'r-', 'second', 'o')
        # plt.pause(.001)
    # Detect Mediapipe hand landmarks
    results = mp_hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Get the position of the thumb
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]
        thumb_x = landmarks.landmark[thumb_landmark_index].x * img.shape[1]
        thumb_y = landmarks.landmark[thumb_landmark_index].y * img.shape[0]

        # Detect ArUco markers
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img, aruco_dict, parameters=aruco_params)

        # Get the position of the thumb relative to the ArUco marker
        if ids is not None and len(ids) > 0:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, mtx, dist)
            marker_index = np.where(ids == ids.min())[0][0]  # choose the closest marker
            marker_center_x = (corners[marker_index][0][0][0] + corners[marker_index][0][2][0]) / 2
            marker_center_y = (corners[marker_index][0][0][1] + corners[marker_index][0][2][1]) / 2
            thumb_relative_x = thumb_x - marker_center_x
            thumb_relative_y = thumb_y - marker_center_y

            # Calculate the world coordinate of the thumb
            thumb_world_x = thumb_relative_x * marker_size / img.shape[1]
            thumb_world_y = thumb_relative_y * marker_size / img.shape[0]
            thumb_world_z = tvecs[0][0][2]

            # Print the world coordinate of the thumb
            print(
                "Thumb world coordinate: ({:.3f}, {:.3f}, {:.3f})".format(thumb_world_x, thumb_world_y, thumb_world_z))


    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == 27:  # Press ESC to exit
        break


