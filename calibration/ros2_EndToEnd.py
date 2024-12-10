import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float64MultiArray
import tf_transformations


class PositionRotationCalculator(Node):
    def __init__(self):
        super().__init__('position_rotation_calculator')

        # Subscribe to position and rotation topics
        self.create_subscription(
            Float64MultiArray,  # 메시지 타입
            '/dsr01/msg/current_posx',  # 위치 및 자세 토픽
            self.position_rotation_callback,  # 콜백 함수
            10
        )

        self.transformation_matrix = None

    def position_rotation_callback(self, msg):
        self.get_logger().info("PositionRotationCalculator callback invoked!")

        try:
            # Extract data from message
            x, y, z, rx, ry, rz = msg.data

            # Calculate rotation matrix using ZYZ Euler angles
            R_matrix = self.rotation_matrix_from_rzyz(rx, ry, rz)

            # Construct transformation matrix
            self.transformation_matrix = np.eye(4)
            self.transformation_matrix[:3, :3] = R_matrix
            self.transformation_matrix[:3, 3] = [x, y, z]

            self.get_logger().info(f"Transformation Matrix (Base -> End Effector):\n{self.transformation_matrix}")
        except Exception as e:
            self.get_logger().error(f"Error processing message: {e}")

    def rotation_matrix_from_rzyz(self, rx, ry, rz):
        # Convert degrees to radians
        rx, ry, rz = np.deg2rad([rx, ry, rz])

        # Create rotation matrices for ZYZ order
        R_z1 = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])

        R_y = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])

        R_z2 = np.array([
            [np.cos(rx), -np.sin(rx), 0],
            [np.sin(rx), np.cos(rx), 0],
            [0, 0, 1]
        ])

        # Combine rotations (ZYZ order)
        R_matrix = np.dot(R_z1, np.dot(R_y, R_z2))
        return R_matrix

    def get_transform_matrix(self):
        return self.transformation_matrix


class HandEyeCalibration(Node):
    def __init__(self, position_rotation_calculator):
        super().__init__('hand_eye_calibration')
        self.position_rotation_calculator = position_rotation_calculator
        self.A_matrices = []
        self.B_matrices = []

        # 이미지 구독용 변수 및 구독자 설정
        self.current_frame = None
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',  # 실제 카메라 이미지 토픽명에 맞게 수정
            self.image_callback,
            10
        )

        # 체커보드 파라미터
        self.chessboard_size = (7, 9)
        self.square_size = 20.0  # mm
        self.object_points = np.zeros((self.chessboard_size[0]*self.chessboard_size[1], 3), dtype=np.float32)
        self.object_points[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2) * self.square_size

        # 카메라 행렬 및 왜곡 계수 (예시값)
        self.camera_matrix = np.array([
            [905.37653593,   0.        , 653.92759642],
            [  0.        , 903.06919   , 376.95680054],
            [  0.        ,   0.        ,   1.        ],
        ], dtype=np.float64)

        self.dist_coeffs = np.array([0.1608557,  -0.55173363,  0.00210995,  0.0035083,   0.57101619])

    def image_callback(self, msg):
        try:
            self.current_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge Error: {e}")

    def capture_pose(self):
        # 로봇 포즈 계산
        rclpy.spin_once(self.position_rotation_calculator)
        T_base_to_eff = self.position_rotation_calculator.transformation_matrix
        if T_base_to_eff is not None:
            # 카메라 이미지에서 solvePnP 시도
            success, T_cam_to_marker = self.solve_pnp()
            if success:
                self.A_matrices.append(T_base_to_eff)
                self.B_matrices.append(T_cam_to_marker)
                self.get_logger().info(f"Captured Pose {len(self.A_matrices)}: Success")

    def solve_pnp(self):
        self.get_logger().info("Press 'c' to detect chessboard, 'q' to quit, and 'd' to close the detected chessboard window.")
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)
            
            if self.current_frame is None:
                continue

            frame = self.current_frame.copy()
            cv2.imshow("Camera Feed", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                return False, None

            if key == ord('c'):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
                if ret:
                    corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                                                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                    cv2.drawChessboardCorners(frame, self.chessboard_size, corners2, ret)
                    cv2.imshow("Detected Chessboard", frame)

                    # 'd' 키가 눌릴 때까지 대기하여 체커보드 화면 유지
                    while True:
                        k = cv2.waitKey(1) & 0xFF
                        if k == ord('d'):
                            cv2.destroyWindow("Detected Chessboard")
                            break

                    success, rvec, tvec = cv2.solvePnP(self.object_points, corners2, self.camera_matrix, self.dist_coeffs)
                    if success:
                        rotation_matrix, _ = cv2.Rodrigues(rvec)
                        T_cam_to_marker = np.eye(4)
                        T_cam_to_marker[:3, :3] = rotation_matrix
                        T_cam_to_marker[:3, 3] = tvec.flatten()
                        return True, T_cam_to_marker
                    else:
                        self.get_logger().error("solvePnP failed.")
                        return False, None

    def calibrate_hand_eye(self):
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
    position_rotation_calculator = PositionRotationCalculator()
    hand_eye_calibration = HandEyeCalibration(position_rotation_calculator)
    try:
        for _ in range(10):  # 예를 들어 10번 포즈 캡쳐
            hand_eye_calibration.capture_pose()
        T_eff_to_cam = hand_eye_calibration.calibrate_hand_eye()
    except KeyboardInterrupt:
        pass

    position_rotation_calculator.destroy_node()
    hand_eye_calibration.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
