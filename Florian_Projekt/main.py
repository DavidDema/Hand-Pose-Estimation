import cv2
from cv2 import aruco
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from helper_functions import *

thumb_landmark_index = 4
index_landmark_index = 8
# Webcam
cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 720)
ret, img = cap.read()

mtx, dist, aruco_params, aruco_dict = start_aruco()
pixel_size = (mtx[0][0] + mtx[1][1]) / 2
marker_size = 0.04
pixel_distance = 50
mp_hands = mp.solutions.hands.Hands()

fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(projection='3d')
# Loop
while True:
    success, img = cap.read()
    # Detect Mediapipe hand landmarks
    results = mp_hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Get the position of the thumb
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]
        thumb_x = landmarks.landmark[thumb_landmark_index].x * img.shape[1]
        thumb_y = landmarks.landmark[thumb_landmark_index].y * img.shape[0]
        thumb_z = landmarks.landmark[thumb_landmark_index].z
        index_x = landmarks.landmark[index_landmark_index].x * img.shape[1]
        index_y = landmarks.landmark[index_landmark_index].y * img.shape[0]
        index_z = landmarks.landmark[index_landmark_index].z
        # Detect ArUco markers
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img, aruco_dict, parameters=aruco_params)
        if corners:
            # calculate pixel distance from aruco marker
            pixel_distance = np.sqrt(np.sum(np.square(corners[0][0][0] - corners[0][0][2])))

        # Get the position of the thumb relative to the ArUco marker
        if ids is not None and len(ids) > 0:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, mtx, dist)
            marker_index = np.where(ids == ids.min())[0][0]  # choose the closest marker
            marker_center_x = (corners[marker_index][0][0][0] + corners[marker_index][0][2][0]) / 2
            marker_center_y = (corners[marker_index][0][0][1] + corners[marker_index][0][2][1]) / 2
            thumb_relative_x = thumb_x - marker_center_x
            thumb_relative_y = thumb_y - marker_center_y

            # Calculate the world coordinate of the thumb and index finger
            thumb_world_x = thumb_x * marker_size / pixel_distance
            thumb_world_y = thumb_y * marker_size / pixel_distance
            thumb_world_z = tvecs[0][0][2]
            index_world_x = index_x * marker_size / pixel_distance
            index_world_y = index_y * marker_size / pixel_distance
            index_world_z = thumb_world_z + (index_z - thumb_z)  # Estimation depth from index finger
            thumb_coordinates = np.array(
                [landmarks.landmark[thumb_landmark_index].x, landmarks.landmark[thumb_landmark_index].y,
                 landmarks.landmark[thumb_landmark_index].z])
            index_coordinates = np.array(
                [landmarks.landmark[index_landmark_index].x, landmarks.landmark[index_landmark_index].y,
                 landmarks.landmark[index_landmark_index].z])
            # Print the world coordinate of the thumb
            print(
                "Thumb world coordinate: ({:.3f}, {:.3f}, {:.3f})".format(thumb_world_x, thumb_world_y, thumb_world_z))
            print(
                "Index world coordinate: ({:.3f}, {:.3f}, {:.3f})".format(index_world_x, index_world_y, index_world_z))
            img = show_coordinates(img, (thumb_world_x * 100., thumb_world_y * 100., thumb_world_z * 100.),
                                   (np.float32(thumb_x), np.float32(thumb_y)), "Thumb")
            img = show_coordinates(img, (index_world_x * 100., index_world_y * 100., index_world_z * 100.),
                                   (np.float32(index_x), np.float32(index_y)), "Index")

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == 27:  # Press ESC to exit
        break
