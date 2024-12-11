import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2 as cv
import argparse

CHECKERBOARD = (9, 7)
square_size = 20  # mm

class ChessboardCalibrator(Node):
    def __init__(self):
        super().__init__('chessboard_calibrator')
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',  # 사용할 토픽 이름
            self.image_callback,
            10
        )

        self.bridge = CvBridge()
        self.current_frame = None

        self.objp = np.zeros((CHECKERBOARD[1]*CHECKERBOARD[0], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * square_size

        self.objpoints = []
        self.imgpoints = []

        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def image_callback(self, msg):
        try:
            self.current_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge Error: {e}")

    def process_frame(self):
        if self.current_frame is None:
            return True  # 아직 프레임이 없으면 계속 진행

        frame = self.current_frame.copy()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, None)
        corners2 = None
        if ret:
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
            cv.drawChessboardCorners(frame, CHECKERBOARD, corners2, ret)

        cv.imshow('Camera Calibration', frame)
        key = cv.waitKey(1) & 0xFF

        if key == ord('c') and ret:
            self.objpoints.append(self.objp)
            self.imgpoints.append(corners2)
            print("Captured frame with corners.")
        elif key == ord('q'):
            return False  # 종료 신호

        return True

    def calibrate_camera(self, gray_shape):
        if len(self.objpoints) > 0:
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
                self.objpoints, self.imgpoints, gray_shape, None, None
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", type=str, default="/camera/camera/color/image_raw", help="Image topic name")
    args = parser.parse_args()

    rclpy.init()
    node = ChessboardCalibrator()

    print("Press 'c' in the image window to capture corners.")
    print("Press 'q' to quit and perform calibration.")

    running = True
    gray_shape = None
    while rclpy.ok() and running:
        # spin_once를 사용하여 콜백 처리
        rclpy.spin_once(node, timeout_sec=0.01)

        if node.current_frame is not None:
            gray_shape = (node.current_frame.shape[1], node.current_frame.shape[0])

        running = node.process_frame()

    cv.destroyAllWindows()

    if gray_shape is not None:
        node.calibrate_camera(gray_shape)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
