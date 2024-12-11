import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

# Object Detection Model Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True).to(device)
model.eval()

CUP_CLASS_ID = 47  # COCO dataset에서 cup 클래스 ID


class PositionRotationCalculator(Node):
    def __init__(self):
        super().__init__('position_rotation_calculator')

        # Subscribe to position and rotation topics
        self.create_subscription(
            Float64MultiArray,
            '/dsr01/msg/current_posx',
            self.position_rotation_callback,
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

        # Rotation matrices for ZYZ order
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

        R_matrix = np.dot(R_z1, np.dot(R_y, R_z2))
        return R_matrix

    def get_transform_matrix(self):
        return self.transformation_matrix


class ObjectDetectionNode(Node):
    def __init__(self, position_rotation_calculator, T_eff_to_cam):
        super().__init__('object_detection_node')
        self.cv_bridge = CvBridge()
        self.position_rotation_calculator = position_rotation_calculator
        self.T_eff_to_cam = T_eff_to_cam

        # Subscribe to color and depth image topics
        self.color_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.color_callback,
            10
        )

        self.depth_sub = self.create_subscription(
            Image,
            '/camera/camera/depth/image_rect_raw',
            self.depth_callback,
            10
        )

        self.color_image = None
        self.depth_image = None

        # 주기적으로 detect_and_transform를 호출하기 위한 타이머 설정
        self.timer = self.create_timer(0.1, self.detect_and_transform)

    def color_callback(self, msg):
        try:
            self.color_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"Color image conversion error: {e}")

    def depth_callback(self, msg):
        try:
            self.depth_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except CvBridgeError as e:
            self.get_logger().error(f"Depth image conversion error: {e}")

    def detect_and_transform(self):
        # color_image와 depth_image가 모두 준비되어 있어야 함
        if self.color_image is None or self.depth_image is None:
            return

        frame = self.color_image.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = F.to_tensor(frame_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(frame_tensor)[0]

        labels = outputs["labels"].cpu().numpy()
        boxes = outputs["boxes"].cpu().numpy()
        scores = outputs["scores"].cpu().numpy()

        # cup(47) 클래스만 필터링
        cup_indices = np.where(labels == CUP_CLASS_ID)[0]
        if len(cup_indices) == 0:
            self.get_logger().info("No cup detected")
            cv2.imshow("Object Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                rclpy.shutdown()
            return

        # 가장 자신도가 높은 cup 선택
        cup_scores = scores[cup_indices]
        max_cup_idx = cup_indices[np.argmax(cup_scores)]

        box = boxes[max_cup_idx]
        center_pixel = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))

        # depth 이미지에서 해당 픽셀의 깊이 값 추출
        depth_value = self.depth_image[center_pixel[1], center_pixel[0]]
        

        # depth 값이 유효하지 않은 경우 처리
        if depth_value == 0:
            self.get_logger().warn("Invalid depth value at detected cup center.")
            return

        # Camera Intrinsics
        camera_intrinsics = np.array([
            [905.37653593,   0.        , 653.92759642],
            [  0.        , 903.06919   , 376.95680054],
            [  0.        ,   0.        ,   1.        ],
        ], dtype=np.float64)

        uv = np.array([center_pixel[0], center_pixel[1], 1.0])
        # depth_value가 mm 단위라 가정
        camera_coords = np.linalg.inv(camera_intrinsics) @ (uv * depth_value)
        camera_coords_h = np.append(camera_coords, 1.0)  # Homogeneous coordinates

        # Camera to End-Effector Coordinates
        eff_coords = np.linalg.inv(self.T_eff_to_cam) @ camera_coords_h

        # End-Effector to Base Coordinates
        rclpy.spin_once(self.position_rotation_calculator)
        T_base_to_eff = self.position_rotation_calculator.get_transform_matrix()
        if T_base_to_eff is None:
            self.get_logger().error("Could not compute T_base_to_eff")
            return

        self.get_logger().info(f"Base -> Eff: \n{T_base_to_eff}")

        base_coords = T_base_to_eff @ eff_coords
        self.get_logger().info(f"Cup center in Base Frame: {base_coords[:3]}")

        # Visualization
        cv2.circle(frame, center_pixel, 5, (0, 0, 255), -1)
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        cv2.putText(frame, "Cup", (center_pixel[0] - 20, center_pixel[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rclpy.shutdown()

    def __del__(self):
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)

    # PositionRotationCalculator 초기화
    position_rotation_calculator = PositionRotationCalculator()

    # End-Effector → Camera Transform
    T_eff_to_cam = np.array(
    [[ 8.27249618e-01, -2.29716623e-01, -5.12726382e-01,  2.26548780e+02],
     [ 2.80572465e-01 , 9.59562668e-01,  2.27723192e-02, -2.80346639e+02],
     [ 4.86761915e-01 ,-1.62695297e-01,  8.58250009e-01,  2.51757560e+02],
     [ 0.00000000e+00 , 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
    )

    # Object Detection Node 초기화
    detection_node = ObjectDetectionNode(position_rotation_calculator, T_eff_to_cam)

    # ROS 스핀
    try:
        while rclpy.ok():
            rclpy.spin_once(detection_node, timeout_sec=0.01)
            cv2.waitKey(1)  # OpenCV 이벤트 처리
    except KeyboardInterrupt:
        pass
    finally:
        detection_node.destroy_node()
        position_rotation_calculator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
