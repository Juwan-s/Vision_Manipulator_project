import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge

class HandEyeCalibration(Node):
    def __init__(self):
        super().__init__('hand_eye_calibration')

        # Subscribe to Base -> EFF transformation matrix
        self.create_subscription(
            Float64MultiArray,
            '/base_to_eff_matrix',
            self.base_to_eff_callback,
            10
        )

        # Subscribe to RealSense Camera Info
        self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.camera_info_callback,
            10
        )

        # Subscribe to Camera Image
        self.create_subscription(
            Image,
            '/camera/camera/color/image_rect_raw',
            self.image_callback,
            10
        )
        
        self.solve_pnp_counter = 0
        
        self.current_T_base_to_eff = None

        self.A_matrices = []  # Base to EFF matrices
        self.B_matrices = []  # Camera to Marker matrices

        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_info_received = False  # Flag to check if CameraInfo is received

        self.bridge = CvBridge()
        self.current_frame = None
        self.c_pressed = False

    def base_to_eff_callback(self, msg):
        try:
            T_base_to_eff = np.array(msg.data).reshape((4, 4))
            if self.current_T_base_to_eff is None:
                self.get_logger().info(f"Initial T_base_to_eff is setted")
            self.current_T_base_to_eff = T_base_to_eff
                
        except Exception as e:
            self.get_logger().error(f"Error processing Base -> EFF matrix: {e}")

    def camera_info_callback(self, msg):
        if not self.camera_info_received:
            try:
                # Extract camera matrix and distortion coefficients
                self.camera_matrix = np.array(msg.k).reshape((3, 3))
                self.dist_coeffs = np.array(msg.d)
                self.camera_info_received = True
                self.get_logger().info(f"Camera Matrix:\n{self.camera_matrix}")
                self.get_logger().info(f"Distortion Coefficients:\n{self.dist_coeffs}")
            except Exception as e:
                self.get_logger().error(f"Error processing CameraInfo: {e}")

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            self.current_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # self.get_logger().info(f"Receiving frames is working...")
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")

    def capture_pose(self):
        if self.camera_matrix is None or self.dist_coeffs is None:
            self.get_logger().error("Camera Matrix or Distortion Coefficients not set.")
            return False

        self.get_logger().info("Press 'c' to capture a frame and detect chessboard.")
        while rclpy.ok():
            rclpy.spin_once(self)
            if self.current_frame is None:
                self.get_logger().warning("No image received yet. Waiting...")
                continue

            frame = self.current_frame.copy()
            cv2.imshow("Live Feed", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):
                self.get_logger().info("'c' pressed. Attempting to detect chessboard.")
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, (7, 9), None)
                if ret:
                    object_points = np.zeros((7 * 9, 3), dtype=np.float32)
                    object_points[:, :2] = np.mgrid[0:7, 0:9].T.reshape(-1, 2) * 20

                    success, rvec, tvec = cv2.solvePnP(object_points, corners, self.camera_matrix, self.dist_coeffs)
                    if success:
                        self.A_matrices.append(self.current_T_base_to_eff)
                        rotation_matrix, _ = cv2.Rodrigues(rvec)
                        T_cam_to_marker = np.eye(4)
                        T_cam_to_marker[:3, :3] = rotation_matrix
                        T_cam_to_marker[:3, 3] = tvec.flatten()

                        self.B_matrices.append(T_cam_to_marker)
                        self.get_logger().info(f"Captured Camera -> Marker Matrix:\n{T_cam_to_marker}")
                        cv2.drawChessboardCorners(frame, (7, 9), corners, ret)
                        cv2.imshow("Detected Chessboard", frame)
                        cv2.waitKey(0)
                        cv2.destroyWindow("Detected Chessboard")
                        self.solve_pnp_counter += 1
                    else:
                        self.get_logger().error("solvePnP failed.")
                        break
                else:
                    self.get_logger().warning("Chessboard corners not found. Try again.")
            elif self.solve_pnp_counter == 10:
                break
        cv2.destroyAllWindows()
        return

    def calibrate_hand_eye(self):
        if len(self.A_matrices) < 3 or len(self.B_matrices) < 3:
            self.get_logger().error("Not enough data for hand-eye calibration. At least 3 poses are required.")
            return None

        A_rotations = [A[:3, :3] for A in self.A_matrices]
        A_translations = [A[:3, 3] for A in self.A_matrices]
        B_rotations = [B[:3, :3] for B in self.B_matrices]
        B_translations = [B[:3, 3] for B in self.B_matrices]

        R_cam_to_eff, t_cam_to_eff = cv2.calibrateHandEye(
            R_gripper2base=A_rotations,
            t_gripper2base=A_translations,
            R_target2cam=B_rotations,
            t_target2cam=B_translations,
            method=cv2.CALIB_HAND_EYE_TSAI
        )

        T_eff_to_cam = np.eye(4)
        T_eff_to_cam[:3, :3] = R_cam_to_eff
        T_eff_to_cam[:3, 3] = t_cam_to_eff.flatten()

        self.get_logger().info(f"End-Effector → Camera Transform:\n{T_eff_to_cam}")
        return T_eff_to_cam


def main():
    rclpy.init()
    hand_eye_calibration = HandEyeCalibration()
    # print("Ckpt 3")
    
    while hand_eye_calibration.current_T_base_to_eff is None\
        or hand_eye_calibration.camera_matrix is None \
        or hand_eye_calibration.dist_coeffs is None:
            rclpy.spin_once(hand_eye_calibration)

    hand_eye_calibration.capture_pose()
    
    T_eff_to_cam = hand_eye_calibration.calibrate_hand_eye()
    
    hand_eye_calibration.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()