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
            '/camera/camera/color/image_raw',
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
            # if self.current_T_base_to_eff is None:
            self.get_logger().info(f"T_base_to_eff is setted")
            self.current_T_base_to_eff = T_base_to_eff
                
        except Exception as e:
            self.get_logger().error(f"Error processing Base -> EFF matrix: {e}")

    def camera_info_callback(self, msg):
    
        try:
            # Extract camera matrix and distortion coefficients
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d)
            self.get_logger().info(f"Camera Intrinsic setted")
            if not self.camera_info_received:
                self.camera_info_received = True
                self.get_logger().info(f"Camera Matrix:\n{self.camera_matrix}")
                self.get_logger().info(f"Distortion Coefficients:\n{self.dist_coeffs}")
        except Exception as e:
            self.get_logger().error(f"Error processing CameraInfo: {e}")

    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        try:
            raw_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if self.camera_matrix is not None and self.dist_coeffs is not None:
                # Apply undistortion
                self.current_frame = cv2.undistort(raw_frame, self.camera_matrix, self.dist_coeffs, None)
            else:
                self.current_frame = raw_frame
            self.get_logger().info(f"Receiving and undistorting frames is working...")
        except Exception as e:
            self.get_logger().error(f"Error converting or undistorting image: {e}")
    
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
            cv2.imshow("Undistorted Live Feed", frame)
            key = cv2.waitKey(1) & 0xFF
    
            if key == ord('c'):
                self.get_logger().info("'c' pressed. Attempting to detect chessboard.")
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, (7, 9), None)
                
           
    
                if ret:
                    object_points = np.zeros((7 * 9, 3), dtype=np.float32)
                    object_points[:, :2] = np.mgrid[0:7, 0:9].T.reshape(-1, 2) * 20
    
                    success, rvec, tvec = cv2.solvePnP(object_points, corners, self.camera_matrix, self.dist_coeffs)
                    
                    print("@@@@@@@@@@@@@@@@@@@@tvec:", tvec)
                    
                    projected_points, _ = cv2.projectPoints(object_points, rvec, tvec, self.camera_matrix, self.dist_coeffs)

                    # Draw the projected points on the frame
                    for point in projected_points:
                        x, y = point.ravel()  # Flatten to (x, y)
                        top_left = (int(x) - 5, int(y) - 5)  # Top-left corner of the square
                        bottom_right = (int(x) + 5, int(y) + 5)  # Bottom-right corner of the square
                        cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), -1)
                    
                    if success:
                        rotation_matrix, _ = cv2.Rodrigues(rvec)
                        T_cam_to_marker = np.eye(4)
                        T_cam_to_marker[:3, :3] = rotation_matrix
                        T_cam_to_marker[:3, 3] = tvec.flatten()
    
                        frame = self.draw_axes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, scale=100)
                        cv2.drawChessboardCorners(frame, (7, 9), corners, ret)
    
                        for idx, corner in enumerate(corners):
                            x, y = corner[0]
                            
                            print(f"idx: {idx}, corner: {(x, y)}")
                            
                            cv2.putText(frame, str(idx), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        self.get_logger().warning(f"self.current_T_base_to_eff: \n{self.current_T_base_to_eff}")
                        cv2.imshow("Detected Chessboard with Axes", frame)
                        key = cv2.waitKey(0)
    
                        if key == ord('t'):
                            self.get_logger().info("'t' pressed. Abort current scene.")
                            cv2.destroyWindow("Detected Chessboard with Axes")
    
                            continue
    
                        cv2.destroyWindow("Detected Chessboard with Axes")
                        
                        self.B_matrices.append(T_cam_to_marker)
                        self.A_matrices.append(self.current_T_base_to_eff)
                        
                        self.get_logger().info(f"Captured Camera -> Marker Matrix:\n{T_cam_to_marker}")
    
                        self.solve_pnp_counter += 1
                        if self.solve_pnp_counter == 10:
                            break
                    else:
                        self.get_logger().error("solvePnP failed.")
                else:
                    self.get_logger().warning("Chessboard corners not found. Try again.")
        cv2.destroyAllWindows()
        return

    
    def draw_axes(self, image, camera_matrix, dist_coeffs, rvec, tvec, scale=50):
        """
        Draw XYZ axes on the image.
        :param image: The input image to draw on.
        :param camera_matrix: Camera intrinsic matrix.
        :param dist_coeffs: Camera distortion coefficients.
        :param rvec: Rotation vector.
        :param tvec: Translation vector.
        :param scale: Scale of the axes.
        """
        axis_points = np.float32([
            [scale, 0, 0],  # X-axis endpoint
            [0, scale, 0],  # Y-axis endpoint
            [0, 0, -scale]  # Z-axis endpoint
        ])
        origin = np.float32([[0, 0, 0]])  # Origin point

        # Project the points to 2D image space
        axis_points_2d, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
        origin_2d, _ = cv2.projectPoints(origin, rvec, tvec, camera_matrix, dist_coeffs)

        # Draw the axes
        origin_2d = tuple(origin_2d[0].ravel().astype(int))
        image = cv2.line(image, origin_2d, tuple(axis_points_2d[0].ravel().astype(int)), (0, 0, 255), 2)  # X-axis (Red)
        image = cv2.line(image, origin_2d, tuple(axis_points_2d[1].ravel().astype(int)), (0, 255, 0), 2)  # Y-axis (Green)
        image = cv2.line(image, origin_2d, tuple(axis_points_2d[2].ravel().astype(int)), (255, 0, 0), 2)  # Z-axis (Blue)
        return image    

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
    
    # rclpy.spin(hand_eye_calibration)
    
    while hand_eye_calibration.current_T_base_to_eff is None \
        or hand_eye_calibration.camera_matrix is None \
        or hand_eye_calibration.dist_coeffs is None \
        or hand_eye_calibration.current_frame is None:
            
            # print(f"hand_eye_calibration.current_T_base_to_eff is {(hand_eye_calibration.current_T_base_to_eff is None)}")
            # print(f"hand_eye_calibration.camera_matrix is {(hand_eye_calibration.camera_matrix is None)}")
            # print(f"hand_eye_calibration.dist_coeffs is {(hand_eye_calibration.dist_coeffs is None)}")
            # print(f"hand_eye_calibration.current_frame is {(hand_eye_calibration.current_frame is None)}")
            
            
            rclpy.spin_once(hand_eye_calibration)
    # print('Reached!!!!!!!!!!')
    hand_eye_calibration.capture_pose()
    
    T_eff_to_cam = hand_eye_calibration.calibrate_hand_eye()
    
    hand_eye_calibration.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()