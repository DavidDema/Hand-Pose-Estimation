from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import cvzone


n_el_mean = 50 # Number of values that are considered for the mean value
finger_lengths = np.zeros(shape=n_el_mean, dtype=np.float32) # Numpy array to save all the previous values of the finger_lengths
def get_finger_length(points, finger_idx=[9, 10, 11, 12]) -> Tuple[float, float, float]:
    """
    Find the z-coordinate of the wrist using the length of a finger (default middle finger = longest)
    :param img: RGB-image
    :param points: landmark-points (eg. landmark[i].x) i=0 is wrist
    :return: finger length (float) [mm]
    """

    finger_length = 0  # in cm
    point1 = np.zeros(shape=3, dtype=np.float32)
    point2 = np.zeros(shape=3, dtype=np.float32)

    for i, idx in enumerate(finger_idx):
        # for each finger joint
        if i == 0:
            # init the first point
            point1[:] = np.array([points[idx].x, points[idx].y, points[idx].z], dtype=np.float32)
            continue
        # assign the second point
        point2[:] = np.array([points[idx].x, points[idx].y, points[idx].z], dtype=np.float32)
        # calculate length using the eukledian-norm (3D)
        finger_length += np.linalg.norm(point2 - point1) * 1e3
        # assign the point2 to the point1 for the next calculation
        point1 = point2.copy()

    # Pop out the oldest/last entry of the array and shift the array
    finger_lengths[:-1] = finger_lengths[1:]
    # Append the newest measurement value to the array
    finger_lengths[-1] = finger_length

    # Create mean value for finger length
    mu = float(np.mean(finger_lengths))
    # Create variance of the finger length
    sigma = float(np.var(finger_lengths))

    # TODO: ignore outliers and wait for the array to be filled up first
    return finger_length, mu, sigma

measured_flag = False
finger_length = 0
def calibrate_space(img, lm_world, lm):
    """

    :param img: RGB-image
    :param lm_world: world-landmark-points
    :param lm: landmark-points
    :return:
    """
    # Define global variables
    global measured_flag
    global finger_length

    # TODO: adapt to also work with other fingers (for now only middle performs "good"/ok)
    #middle_idx = [9, 10, 11, 12]
    middle_idx = [9, 12]
    #index_idx = [5, 6]

    finger_idx = middle_idx
    img_height, img_width, _ = img.shape

    # Measure the finger (given landmark points) length
    if not measured_flag:
        finger_length, mu, sigma = get_finger_length(lm_world, finger_idx)

        # If the variance is small enough we finish measuring and start calculating the z-value
        sigma_eps = 0.5
        if sigma > sigma_eps:
            return False, 0, 0, finger_length, mu, sigma

    # Continue calibration process (if sigma is small enough)
    measured_flag = True

    # Measurement point (by hand with a ruler)
    # This is used to identify the camera properties, the convert a number of pixels to the unit length [mm]
    # at a distance z0 from the webcam, I measured an image width of x0 and image height of y0
    x0 = 0.31
    z0 = 0.31
    y0 = 0.235

    # Measurement point (emipirical with projected finger_length_2D, see above)
    # at a distance z1 from the webcam, I measured the finger_scale
    finger_scale = 0.6 # finger_length_2D/finger_length
    z1 = 0.5

    # Assuming a linear model of the camera
    # Calculate the linear properties
    dz = z1-z0
    dy = finger_length * (finger_scale-1) # y1-y0
    k = dz/dy # Slope of fct
    d = -k * finger_length + z0

    # Get projected finger length in 2D image space
    # x,y-coordinates are normalized to the frame width/height thus we scale according to the measurement !
    # Using the first and last index of the index list
    point1 = np.array(
        [lm[finger_idx[-1]].x * x0, lm[finger_idx[-1]].y * y0],
        dtype=np.float32)
    point2 = np.array(
        [lm[finger_idx[0]].x * x0, lm[finger_idx[0]].y * y0],
        dtype=np.float32)

    # Projected finger length [mm] using the 2D eukledian norm
    finger_length_2D = np.linalg.norm(point2 - point1) * 1e3

    # Calculate the z-value using the linear model
    z = k * finger_length_2D + d

    # TODO: This model is not that accurate especially when the finger is not perpendicular to the camera rays (3D length is different to the projected 2D values)
    # - We could use another length of the hand that is more accurate and stable
    # - During the grasping process, the hand is partially occluded, therefore we should simultaneously
    # measure different parts of the hand and take some type of best estimate of that

    return True, z, finger_length_2D, finger_length, 0, 0

def draw_hand3(ax, points, color, label, marker):
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

    scale = 10 # 1 is meter; 100 is centimeter
    scale_x = scale
    scale_y = -scale
    scale_z = scale

    thumb_x, thumb_y, thumb_z = [], [], []
    for i in thumb_idx:
        thumb_x.append(points[i].x*scale_x)
        thumb_y.append(points[i].y*scale_y)
        thumb_z.append(points[i].z*scale_z)
    index_x, index_y, index_z = [], [], []
    for i in index_idx:
        index_x.append(points[i].x*scale_x)
        index_y.append(points[i].y*scale_y)
        index_z.append(points[i].z*scale_z)
    middle_x, middle_y, middle_z = [], [], []
    for i in middle_idx:
        middle_x.append(points[i].x*scale_x)
        middle_y.append(points[i].y*scale_y)
        middle_z.append(points[i].z*scale_z)
    ring_x, ring_y, ring_z = [], [], []
    for i in ring_idx:
        ring_x.append(points[i].x*scale_x)
        ring_y.append(points[i].y*scale_y)
        ring_z.append(points[i].z*scale_z)
    pinky_x, pinky_y, pinky_z = [], [], []
    for i in pinky_idx:
        pinky_x.append(points[i].x*scale_x)
        pinky_y.append(points[i].y*scale_y)
        pinky_z.append(points[i].z*scale_z)
    palm_x, palm_y, palm_z = [], [], []
    for i in palm_idx:
        palm_x.append(points[i].x*scale_x)
        palm_y.append(points[i].y*scale_y)
        palm_z.append(points[i].z*scale_z)

    finger_length = 0 # in cm
    point1 = np.zeros(shape=3, dtype=np.float32)
    point2 = np.zeros(shape=3, dtype=np.float32)

    finger_idx = middle_idx
    for i, idx in enumerate(finger_idx):
        if i == 0:
            point1[:] = np.array([points[idx].x, points[idx].y, points[idx].z], dtype=np.float32)
            continue
        point2[:] = np.array([points[idx].x, points[idx].y, points[idx].z], dtype=np.float32)
        finger_length += np.linalg.norm(point2 - point1)*100
        point1 = point2.copy()
    print(finger_length)

    plt.plot(thumb_x, thumb_z, thumb_y, color)
    plt.plot(index_x, index_z, index_y, color)
    plt.plot(middle_x, middle_z, middle_y, color)
    plt.plot(ring_x, ring_z, ring_y, color)
    plt.plot(pinky_x, pinky_z, pinky_y, color)
    plt.plot(palm_x, palm_z, palm_y, color)
    plt.plot(0, 0, 0, color="red", markersize=10, marker="o")

def draw_hand2(ax, points, color, label, marker):
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

def draw_gripper_coordinate(ax, points):
    """
    This method is used to plot the coordinate system of the gripper according to the position of selected finger positions
    :param ax: plot
    :param points: landmarks of the hand
    """
    p_thumb = points[4, :]
    p_index = points[8, :]
    p_wrist = points[0, :]

    d1 = p_index-p_thumb
    p_center = p_thumb+d1/2

    ax.plot(p_thumb[0], p_thumb[1], p_thumb[2], color="orange", markersize=3, marker="s")
    ax.plot(p_index[0], p_index[1], p_index[2], color="orange", markersize=3, marker="s")
    ax.plot(p_wrist[0], p_wrist[1], p_wrist[2], color="orange", markersize=3, marker="s")
    ax.plot(p_center[0], p_center[1], p_center[2], color="orange", markersize=5, marker="o")

    #ax.Arrow3D(p_center, [0,0,0])
    scale = 50

    pz = (p_center-p_wrist)
    pz = scale*pz/np.linalg.norm(pz)

    px = np.cross(pz, d1)
    px = scale * px / np.linalg.norm(px)

    py = np.cross(pz, px)
    py = scale * py / np.linalg.norm(py)

    #ax.plot(pz[0], pz[1], pz[2], color="orange", markersize=5, marker="o")
    ax.quiver(p_center[0], p_center[1], p_center[2], px[0], px[1], px[2], colors='r')
    ax.quiver(p_center[0], p_center[1], p_center[2], py[0], py[1], py[2], colors='g')
    ax.quiver(p_center[0], p_center[1], p_center[2], pz[0], pz[1], pz[2], colors='b')