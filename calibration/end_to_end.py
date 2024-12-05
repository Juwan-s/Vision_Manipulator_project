import cv2
import numpy as np
import rclpy
import time
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from tf2_msgs.msg import TFMessage

# 기존의 TFChainCalculator 클래스 정의
class TFChainCalculator(Node):
    def __init__(self):
        super().__init__('tf_chain_calculator')
        self.subscriber = self.create_subscription(
            TFMessage,
            '/tf',
            self.tf_callback,
            10
        )
        self.transforms = {}  # 저장된 변환 데이터
        self.target_frame = 'link_6'
        self.base_frame = 'base_link'

    def tf_callback(self, msg):
        # 각 TransformStamped 메시지 처리
        for transform in msg.transforms:
            parent_frame = transform.header.frame_id
            child_frame = transform.child_frame_id

            # 변환 정보 저장
            self.transforms[(parent_frame, child_frame)] = transform.transform

        # Base → End-Effector 변환 행렬 계산
        self.get_transform_matrix(self.base_frame, self.target_frame)

    def get_transform_matrix(self, base_frame, target_frame):
        # 부모 → 자식 경로 추적
        current_frame = target_frame
        chain = []

        while current_frame != base_frame:
            # 부모-자식 관계를 역추적
            for (parent, child), transform in self.transforms.items():
                if child == current_frame:
                    chain.append((parent, child, transform))
                    current_frame = parent
                    break
            else:
                self.get_logger().error(f"No parent frame found for {current_frame}")
                return

        # 변환 행렬 계산
        transform_matrix = np.eye(4)
        for parent, child, transform in reversed(chain):
            # Translation (이동)
            translation = [
                transform.translation.x,
                transform.translation.y,
                transform.translation.z
            ]

            # Rotation (쿼터니언)
            quaternion = [
                transform.rotation.x,
                transform.rotation.y,
                transform.rotation.z,
                transform.rotation.w
            ]

            # 쿼터니언 → 회전 행렬
            rotation_matrix = R.from_quat(quaternion).as_matrix()

            # 변환 행렬 생성
            T = np.eye(4)
            T[:3, :3] = rotation_matrix
            T[:3, 3] = translation

            # 누적 곱
            transform_matrix = transform_matrix @ T
            
        return transform_matrix

class HandEyeCalibration(Node):
    def __init__(self, tf_calculator):
        super().__init__('hand_eye_calibration')
        self.tf_calculator = tf_calculator  # TFChainCalculator를 활용
        self.A_matrices = []  # Base → End-Effector 변환 리스트
        self.B_matrices = []  # Camera → Marker 변환 리스트

    def capture_pose(self):
        # Base → End-Effector 변환 가져오기
        rclpy.spin_once(self.tf_calculator)
        T_base_to_eff = self.tf_calculator.get_transform_matrix('base_link', 'link_6')
        
        self.A_matrices.append(T_base_to_eff)

        # solvePnP로 Camera → Marker 변환 계산
        success, T_cam_to_marker = self.solve_pnp()
        if success:
            self.B_matrices.append(T_cam_to_marker)
            self.get_logger().info(f"Captured Pose {len(self.A_matrices)}: Success")

    def solve_pnp(self):
        # solvePnP 계산
        object_points = np.zeros((7 * 9, 3), dtype=np.float32)
        object_points[:, :2] = np.mgrid[0:7, 0:9].T.reshape(-1, 2) * 20

        camera_matrix = np.array([
            [756.79624248, 0, 300.79559778],
            [0, 758.18809564, 283.84914722],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.array([-0.41748243, 0.2726393, 0.00340657, 0.00165751, -0.39339633])

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

            if key == ord('c'):  # 'c'를 눌러 캡처
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, (7, 9), None)

                if ret:
                    _, rvec, tvec = cv2.solvePnP(object_points, corners, camera_matrix, dist_coeffs)
                    rotation_matrix, _ = cv2.Rodrigues(rvec)
                    T_cam_to_marker = np.eye(4)
                    T_cam_to_marker[:3, :3] = rotation_matrix
                    T_cam_to_marker[:3, 3] = tvec.flatten()
                    cap.release()
                    cv2.destroyAllWindows()
                    return True, T_cam_to_marker

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

    def compute_object_in_base(self, T_eff_to_cam, T_cam_to_obj):
        T_base_to_eff = self.tf_calculator.get_transform_matrix('base_link', 'link_6')
        T_base_to_obj = T_base_to_eff @ T_eff_to_cam @ T_cam_to_obj
        self.get_logger().info(f"Base → Object Transform:\n{T_base_to_obj}")
        self.get_logger().info(f"Object Position in Base Frame: {T_base_to_obj[:3, 3]}")


def main():
    rclpy.init()

    # TFChainCalculator 노드 생성
    tf_calculator = TFChainCalculator()
    # rclpy.spin(tf_calculator)

    # HandEyeCalibration 노드 생성
    hand_eye_calibration = HandEyeCalibration(tf_calculator)

    try:
        # 포즈 캡처
        for _ in range(5):
            hand_eye_calibration.capture_pose()

        # Hand-Eye 캘리브레이션 수행
        T_eff_to_cam = hand_eye_calibration.calibrate_hand_eye()

        # Camera → Object 변환 계산 (solvePnP로 가정)
        _, T_cam_to_obj = hand_eye_calibration.solve_pnp()

        # Base → Object 변환 계산
        hand_eye_calibration.compute_object_in_base(T_eff_to_cam, T_cam_to_obj)

    except KeyboardInterrupt:
        pass

    tf_calculator.destroy_node()
    hand_eye_calibration.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
