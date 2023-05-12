import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

from mediapipe.helper_functions_mediapipe import calibrate_space

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(projection='3d')

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    results = hands.process(image)

    if results.multi_hand_world_landmarks and results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        hand_world_landmarks = results.multi_hand_world_landmarks[0]
        if False:
            ax.cla()
            ax.set_xlim3d(-1, 1)
            ax.set_ylim3d(-1, 1)
            ax.set_zlim3d(-1, 1)
            ax.set_xlabel('x')
            ax.set_ylabel('z')
            ax.set_zlabel('y')

            draw_hand3(ax, hand_world_landmarks.landmark, 'k-', 'first', 's')
            plt.pause(.001)
        else:
            # For the calibration process, hold the opened hand in front of the camera, trying to keep the fingers
            # fully stretched and perpendicular to the camera direction
            # After the hand is measured the z-value shows up as z value

            flag, z1, finger_length_2D, finger_length, mu, sigma = calibrate_space(
                img=image,
                lm_world=hand_world_landmarks.landmark,
                lm=hand_landmarks.landmark)

            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # ------
            # Plot text on the image
            # font
            font = cv2.FONT_HERSHEY_SIMPLEX
            # org
            org = (50, 50)
            org2 = (50, 100)
            # fontScale
            fontScale = 0.5
            # Blue color in BGR
            color = (255, 0, 0)
            color2 = (255, 0, 0)
            # Line thickness of 2 px
            thickness = 1

            # Using cv2.putText() method
            image = cv2.putText(image, "fl=%.2f (mu=%.2f/sigma=%.2f)" %(finger_length, mu, sigma), org, font,
                                fontScale, color, thickness, cv2.LINE_AA)
            image = cv2.putText(image, "z=%.2f (fl_2D=%.2f)" % (z1, finger_length_2D), org2, font,
                                fontScale, color2, thickness, cv2.LINE_AA)

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', image)
    #cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
