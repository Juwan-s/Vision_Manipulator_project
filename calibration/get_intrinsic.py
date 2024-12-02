import numpy as np
import cv2 as cv
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--num", type=int, default=0, help="Cam device number is required.", required=True)

args = parser.parse_args()


# Termination criteria for corner refinement
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points based on the size of the chessboard (7x9, with 20mm squares)
square_size = 20  # in millimeters
objp = np.zeros((7*9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:7].T.reshape(-1, 2) * square_size

# Arrays to store object points and image points
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Open the webcam (device number 4)
cap = cv.VideoCapture(args.num)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'c' to capture the frame with detected corners.")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv.findChessboardCorners(gray, (9, 7), None)

    if ret:
        # Refine corners for subpixel accuracy
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Draw and display the corners
        cv.drawChessboardCorners(frame, (9, 7), corners2, ret)

    cv.imshow('Webcam Calibration', frame)

    key = cv.waitKey(1) & 0xFF
    if key == ord('c') and ret:
        # Save the current frame's points
        objpoints.append(objp)
        imgpoints.append(corners2)
        print("Captured frame with corners.")
    elif key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

if len(objpoints) > 0:
    # Calibrate the camera using the collected points
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    if ret:
        print("Camera calibration successful.")
        print("Camera matrix:")
        print(camera_matrix)
        print("Distortion coefficients:")
        print(dist_coeffs)
    else:
        print("Camera calibration failed.")
else:
    print("No frames captured for calibration.")

