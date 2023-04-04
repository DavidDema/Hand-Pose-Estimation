import cv2
import numpy as np

# Define the size of the Aruco markers and the number of squares in the calibration pattern
marker_size = 0.034 # meters
num_markers = (4, 5)

# Generate the Aruco calibration pattern
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
board = cv2.aruco.CharucoBoard_create(num_markers[0], num_markers[1], marker_size, 0.004, aruco_dict)

# Create an array to store the calibration images and corresponding object points
calibration_images = []
object_points = []

# Capture images for calibration
capture = cv2.VideoCapture(0) # Change the index to use a different camera
while len(calibration_images) < 10: # Capture 10 images for calibration
    ret, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the Aruco markers in the image
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)

    # If the markers are detected, refine the corners and detect the charuco board
    if ids is not None:
        corners, ids, _ = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
        if corners is not None and ids is not None and len(corners) > 3:
            cv2.aruco.drawDetectedCornersCharuco(frame, corners, ids, (0,255,0))
            calibration_images.append(gray)
            object_points.append(board.chessboardCorners.reshape(-1, 3))

    cv2.imshow('Calibration', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

# Perform camera calibration using the captured images and object points
image_size = calibration_images[0].shape[::-1]
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, calibration_images, image_size, None, None)

# Save the camera matrix and distortion coefficients to a file
np.save('camera_matrix.npy', camera_matrix)
np.save('dist_coeffs.npy', dist_coeffs)

print('Camera calibration successful!')