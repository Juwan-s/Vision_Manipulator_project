import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np
import math

class PositionRotationCalculator(Node):
    def __init__(self):
        super().__init__('position_rotation_calculator')

        # 로봇 포즈 구독
        self.create_subscription(
            Float64MultiArray,
            '/dsr01/msg/current_posx',
            self.position_rotation_callback,
            10
        )

        # 계산된 변환행렬 퍼블리셔
        self.pub = self.create_publisher(Float64MultiArray, '/base_to_eff_matrix', 10)
        self.transformation_matrix = None

    def rotation_matrix_from_rzyz(self, rx, ry, rz):
        # Convert degrees to radians
        rx, ry, rz = np.deg2rad([rx, ry, rz])

        R_z1 = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz),  np.cos(rz), 0],
            [0,           0,          1]
        ])

        R_y = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0,          1, 0        ],
            [-np.sin(ry),0, np.cos(ry)]
        ])

        R_z2 = np.array([
            [np.cos(rx), -np.sin(rx), 0],
            [np.sin(rx),  np.cos(rx), 0],
            [0,           0,          1]
        ])

        R_matrix = R_z1 @ R_y @ R_z2
        return R_matrix

    def position_rotation_callback(self, msg):
        self.get_logger().info("PositionRotationCalculator callback invoked!")
        try:
            x, y, z, rx, ry, rz = msg.data
            R_matrix = self.rotation_matrix_from_rzyz(rx, ry, rz)
            self.transformation_matrix = np.eye(4)
            self.transformation_matrix[:3, :3] = R_matrix
            self.transformation_matrix[:3, 3] = [x, y, z]

            # 토픽으로 변환행렬 퍼블리시
            mat_msg = Float64MultiArray()
            mat_msg.data = self.transformation_matrix.flatten().tolist()
            self.pub.publish(mat_msg)

            self.get_logger().info(f"Published T_base_to_eff:\n{self.transformation_matrix}")

        except Exception as e:
            self.get_logger().error(f"Error processing message: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = PositionRotationCalculator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
