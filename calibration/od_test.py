import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import tf_transformations
import rclpy


# Object Detection Model Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True).to(device)
model.eval()


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



class ObjectDetectionNode(Node):
    def __init__(self, position_rotation_calculator, T_eff_to_cam):
        super().__init__('object_detection_node')
        self.bridge = cv2.VideoCapture(0)  # Camera Index
        self.position_rotation_calculator = position_rotation_calculator
        self.T_eff_to_cam = T_eff_to_cam

    def detect_and_transform(self):
        ret, frame = self.bridge.read()
        if not ret:
            self.get_logger().error("Could not read frame")
            return

        # Object Detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = F.to_tensor(frame_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(frame_tensor)[0]

        if len(outputs["boxes"]) == 0:
            self.get_logger().info("No objects detected")
            return

        # Get the most confident detection
        max_idx = torch.argmax(outputs["scores"])
        box = outputs["boxes"][max_idx].cpu().numpy()
        center_pixel = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))

        # Camera Intrinsics
        camera_intrinsics = np.array([
            [744.58805716,   0.      ,   346.66750372],
            [  0.         , 745.78693653, 280.83647637],
            [  0.         ,  0.         ,  1.        ]
        ], dtype=np.float64)

        # Pixel to Camera Coordinates
        depth = 400  # Assume depth in mm (replace with actual depth sensor data)
        uv = np.array([center_pixel[0], center_pixel[1], 1.0])
        camera_coords = np.linalg.inv(camera_intrinsics) @ (uv * depth)
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

        self.get_logger().info(f"Object center in Base Frame: {base_coords[:3]}")

        # Visualization
        cv2.circle(frame, center_pixel, 5, (0, 0, 255), -1)
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        cv2.putText(frame, "Object", (center_pixel[0] - 20, center_pixel[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rclpy.shutdown()

    def __del__(self):
        self.bridge.release()


def main(args=None):
    rclpy.init(args=args)

    # PositionRotationCalculator 초기화
    position_rotation_calculator = PositionRotationCalculator()

    # End-Effector → Camera Transform
    T_eff_to_cam = np.array(
[[ 9.36617093e-01,  3.17745660e-02,  3.48910873e-01,  7.04658814e+01],
 [ 1.65273478e-01,  8.38037453e-01, -5.19978754e-01,  5.70715047e+01],
 [-3.08922479e-01,  5.44686703e-01,  7.79668710e-01,  5.05987400e+02],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]


)

    # Object Detection Node 초기화
    detection_node = ObjectDetectionNode(position_rotation_calculator, T_eff_to_cam)

    try:
        while rclpy.ok():
            detection_node.detect_and_transform()
    except KeyboardInterrupt:
        pass
    finally:
        detection_node.destroy_node()
        position_rotation_calculator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
