import matplotlib.pyplot as plt
import numpy as np
import cv2
from cv2 import aruco
import math
import cvzone
import yaml


def draw_hand(ax, points, color, label, marker):
    """
    This method is used to plot the recognized hand in 3d space
    :param ax: plot
    :param points: landmarks of the hand
    :param color: color of the plot
    :param label: label of the hand
    :param marker: marker of the landmarks
    """
    thumb_idx = [0, 1, 2, 3, 4]
    index_idx = [0, 5, 6, 7, 8]
    middle_idx = [9, 10, 11, 12]
    ring_idx = [13, 14, 15, 16]
    pinky_idx = [17, 18, 19, 20]
    palm_idx = [5, 9, 13, 17, 0]
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], marker = marker, label = label)
    thumb_x, thumb_y, thumb_z = [], [], []
    for i in thumb_idx:
        thumb_x.append(points[i, 0])
        thumb_y.append(points[i, 1])
        thumb_z.append(points[i, 2])
    index_x, index_y, index_z = [], [], []
    for i in index_idx:
        index_x.append(points[i, 0])
        index_y.append(points[i, 1])
        index_z.append(points[i, 2])
    middle_x, middle_y, middle_z = [], [], []
    for i in middle_idx:
        middle_x.append(points[i, 0])
        middle_y.append(points[i, 1])
        middle_z.append(points[i, 2])
    ring_x, ring_y, ring_z = [], [], []
    for i in ring_idx:
        ring_x.append(points[i, 0])
        ring_y.append(points[i, 1])
        ring_z.append(points[i, 2])
    pinky_x, pinky_y, pinky_z = [], [], []
    for i in pinky_idx:
        pinky_x.append(points[i, 0])
        pinky_y.append(points[i, 1])
        pinky_z.append(points[i, 2])
    palm_x, palm_y, palm_z = [], [], []
    for i in palm_idx:
        palm_x.append(points[i, 0])
        palm_y.append(points[i, 1])
        palm_z.append(points[i, 2])
    plt.plot(thumb_x, thumb_y, thumb_z, color)
    plt.plot(index_x, index_y, index_z, color)
    plt.plot(middle_x, middle_y, middle_z, color)
    plt.plot(ring_x, ring_y, ring_z, color)
    plt.plot(pinky_x, pinky_y, pinky_z, color)
    plt.plot(palm_x, palm_y, palm_z, color)


def calc_handDistance(lmList, hands, img):
    x = [220, 200, 170, 155, 150, 140, 130, 120, 110, 105, 100, 95, 90]
    y = [40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    coff = np.polyfit(x, y, 2)  # y = Ax^2 + Bx + C
    x, y, w, h = hands[0]['bbox']
    x1, y1, z1 = lmList[5]
    x2, y2, z2 = lmList[17]
    distance = int(math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))
    print(distance)
    A, B, C = coff
    distanceCM = A * distance ** 2 + B * distance + C
    print(distanceCM)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 3)
    cvzone.putTextRect(img, f'{int(distanceCM)} cm', (x + 5, y - 10))


def start_aruco():
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)
    with open('calibration.yaml') as f:
        loadeddict = yaml.safe_load(f)
    mtx = loadeddict.get('camera_matrix')
    dist = loadeddict.get('dist_coeff')
    mtx = np.array(mtx)
    dist = np.array(dist)
    aruco_params = aruco.DetectorParameters_create()
    return mtx, dist, aruco_params, aruco_dict

def detect_marker(frame, mtx, dist, aruco_params, aruco_dict, marker_length):
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
            frame = show_coordinates(frame, tvecs[i][0], corners[i][0][0], ids[i])

            # Project the axes points onto the image
            for tvec,rvec in zip(tvecs,rvecs):
                axis_points, _ = cv2.projectPoints(np.float32([[0.005, 0, 0], [0, 0.005, 0], [0, 0, -0.005]]), rvec,
                                                   tvec,
                                                   mtx, dist)

                # Draw the axes lines on the image
                frame = cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, 0.01)
    return frame, tvecs, rvecs, corners, ids


def show_coordinates(frame, tvec, corners, id = 0):
    """
    This function prints the coordinates on the image
    :param frame: image where the text will be printed on
    :param tvec: coordinates of the object
    :param corners: pixel coordinates where the text will be printed
    :param id: ID of Aruco marker
    :return: image with text on it
    """
    org = (corners[0].astype(int), corners[1].astype(int))
    cv2.putText(frame, f"ID: {id} Koordinaten:  x: {tvec[0]:.2f}cm", org,
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame,
                f"y: {tvec[1]:.2f}cm",
                (corners[0].astype(int) + 185, corners[1].astype(int) + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame,
                f"z: {tvec[2]:.2f}cm",
                (corners[0].astype(int) + 185, corners[1].astype(int) + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return frame

