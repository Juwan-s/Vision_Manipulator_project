import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float32MultiArray, Float64MultiArray
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

    def capture_pose(self):
        rclpy.spin_once(self.position_rotation_calculator)
        T_base_to_eff = self.position_rotation_calculator.transformation_matrix
        if T_base_to_eff is not None:
            self.A_matrices.append(T_base_to_eff)
            success, T_cam_to_marker = self.solve_pnp()
            if success:
                self.B_matrices.append(T_cam_to_marker)
                self.get_logger().info(f"Captured Pose {len(self.A_matrices)}: Success")

    def solve_pnp(self):
        object_points = np.zeros((7 * 9, 3), dtype=np.float32)
        object_points[:, :2] = np.mgrid[0:7, 0:9].T.reshape(-1, 2) * 20
        camera_matrix = np.array([
            [744.58805716,   0.      ,   346.66750372],
            [  0.         , 745.78693653, 280.83647637],
            [  0.         ,  0.         ,  1.        ]
        ], dtype=np.float64)
        dist_coeffs = np.array([-0.43147739,  0.23392883,  0.00118724, -0.00219554, -0.0411546])

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.get_logger().error("Error: Could not open video.")
            return False, None

        self.get_logger().info("Press 'c' to capture a frame and detect chessboard corners.")
        while True:
            ret, frame = cap.read()
            if not ret:
                self.get_logger().error("Failed to grab frame. Exiting...")
                break

            cv2.imshow("Video Feed", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, (7, 9), None)
                if ret:
                    cv2.drawChessboardCorners(frame, (7, 9), corners, ret)
                    success, rvec, tvec = cv2.solvePnP(object_points, corners, camera_matrix, dist_coeffs)
                    
                    if success:
                        rotation_matrix, _ = cv2.Rodrigues(rvec)
                        T_cam_to_marker = np.eye(4)
                        T_cam_to_marker[:3, :3] = rotation_matrix
                        T_cam_to_marker[:3, 3] = tvec.flatten()
                        
                        cv2.imshow("Detected Chessboard", frame)
                        cv2.waitKey(0)  # 사용자 확인 후 아무 키나 눌러 닫기

                        cap.release()
                        cv2.destroyAllWindows()
                        return True, T_cam_to_marker
                    else:
                        self.get_logger().error("solvePnP failed.")
                        break

        cap.release()
        cv2.destroyAllWindows()
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
        for _ in range(10):
            hand_eye_calibration.capture_pose()
        T_eff_to_cam = hand_eye_calibration.calibrate_hand_eye()
    except KeyboardInterrupt:
        pass

    position_rotation_calculator.destroy_node()
    hand_eye_calibration.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
