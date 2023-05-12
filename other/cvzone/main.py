import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import matplotlib.pyplot as plt
from mediapipe.helper_functions_mediapipe import draw_hand2, draw_gripper_coordinate

# Webcam
cap = cv2.VideoCapture(0)
success, img = cap.read()
print(img.shape)
cap.set(3, 1280)
cap.set(4, 720)

# Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(projection='3d')
# Loop
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        lmList = hands[0]['lmList']

        ax.cla()
        ax.set_xlim3d(0, 1300)
        ax.set_ylim3d(0, 1300)
        ax.set_zlim3d(-200, 200)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        draw_hand2(ax, np.array(lmList), 'k-', 'first', 's')
        draw_gripper_coordinate(ax, np.array(lmList))
        if len(hands) == 2:
            lmList2 = hands[1]['lmList']
            draw_hand2(ax, np.array(lmList2), 'r-', 'second', 'o')
        plt.pause(.001)

    #cv2.imshow("Image", img)
    #if cv2.waitKey(5) & 0xFF == 27:
    #  break
cap.release()

