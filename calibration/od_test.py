import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from tf2_msgs.msg import TFMessage
import rclpy


# Object Detection ?? ??
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True).to(device)
model.eval()


class TFChainCalculator(Node):
    def __init__(self):
        super().__init__('tf_chain_calculator')
        self.subscriber = self.create_subscription(
            TFMessage,
            '/tf',
            self.tf_callback,
            10
        )
        self.transforms = {}
        self.target_frame = 'link_6'
        self.base_frame = 'base_link'

    def tf_callback(self, msg):
        for transform in msg.transforms:
            parent_frame = transform.header.frame_id
            child_frame = transform.child_frame_id
            self.transforms[(parent_frame, child_frame)] = transform.transform

    def get_transform_matrix(self, base_frame, target_frame):
        current_frame = target_frame
        chain = []

        while current_frame != base_frame:
            for (parent, child), transform in self.transforms.items():
                if child == current_frame:
                    chain.append((parent, child, transform))
                    current_frame = parent
                    break
            else:
                self.get_logger().error(f"No parent frame found for {current_frame}")
                return None

        transform_matrix = np.eye(4)
        for parent, child, transform in reversed(chain):
            translation = [
                transform.translation.x * 1000,
                transform.translation.y * 1000,
                transform.translation.z * 1000
            ]
            quaternion = [
                transform.rotation.x,
                transform.rotation.y,
                transform.rotation.z,
                transform.rotation.w
            ]
            rotation_matrix = R.from_quat(quaternion).as_matrix()
            T = np.eye(4)
            T[:3, :3] = rotation_matrix
            T[:3, 3] = translation
            transform_matrix = transform_matrix @ T

        return transform_matrix


class ObjectDetectionNode(Node):
    def __init__(self, tf_calculator, T_eff_to_cam):
        super().__init__('object_detection_node')
        self.bridge = cv2.VideoCapture(0)  # Camera Index
        self.tf_calculator = tf_calculator
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

        # ?? ?? confidence? ?? ?? ??
        max_idx = torch.argmax(outputs["scores"])
        box = outputs["boxes"][max_idx].cpu().numpy()
        # center_pixel = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        center_pixel = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))


        # Camera Intrinsics (?? ?)
        camera_intrinsics = np.array([
            [756.79624248, 0, 300.79559778],
            [0, 758.18809564, 283.84914722],
            [0, 0, 1]
        ])

        # ?? ?? -> ??? ?? ??
        depth = 1000.0  # Assume depth in mm (replace with actual depth sensor data)
        uv = np.array([center_pixel[0], center_pixel[1], 1.0])
        camera_coords = np.linalg.inv(camera_intrinsics) @ (uv * depth)
        camera_coords_h = np.append(camera_coords, 1.0)  # Homogeneous coordinates

        # ??? -> End-Effector ??
        eff_coords = np.linalg.inv(self.T_eff_to_cam) @ camera_coords_h

        # End-Effector -> Base ??
        rclpy.spin_once(self.tf_calculator)
        T_base_to_eff = self.tf_calculator.get_transform_matrix('base_link', 'link_6')
        if T_base_to_eff is None:
            self.get_logger().error("Could not compute T_base_to_eff")
            return

        base_coords = T_base_to_eff @ eff_coords

        self.get_logger().info(f"Object center in Base Frame: {base_coords[:3]}")

        cv2.circle(frame, center_pixel, 5, (0, 0, 255), -1)  # ???? ?? ? ???
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

    # TFChainCalculator ??
    tf_calculator = TFChainCalculator()
    print("TF Calculator loaded")
    # ?????? ?? (T_eff_to_cam)
    T_eff_to_cam = np.array([[-4.23654902e-02, -1.93093243e-01, 9.80265354e-01, -1.69343083e+02],
        [9.92444481e-01, 1.04945510e-01, 6.35640804e-02, -1.93568921e+02],
        [-1.15148242e-01,  9.75551864e-01,  1.87188255e-01, -3.10265035e+01],
        [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    # Object Detection Node ??
    detection_node = ObjectDetectionNode(tf_calculator, T_eff_to_cam)

    try:
        print("spinnig detection node")
        while rclpy.ok():
            print("detecting start")
            detection_node.detect_and_transform()
    except KeyboardInterrupt:
        pass
    finally:
        detection_node.destroy_node()
        tf_calculator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

