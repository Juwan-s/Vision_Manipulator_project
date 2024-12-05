import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from scipy.spatial.transform import Rotation as R
import numpy as np


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
        self.compute_chain_transform(self.base_frame, self.target_frame)

    def compute_chain_transform(self, base_frame, target_frame):
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

        self.get_logger().info(f"Base → {target_frame} Transform Matrix:\n{transform_matrix}")


def main():
    rclpy.init()
    node = TFChainCalculator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()