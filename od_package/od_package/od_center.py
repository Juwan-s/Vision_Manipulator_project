import cv2
import torch
from ultralytics import YOLO
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
import numpy as np

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Use YOLOv8n model for object detection

# ROS 2 Node for Object Detection with Depth
class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')
        self.bridge = CvBridge()

        # Subscriptions
        self.color_subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.color_callback,
            10
        )
        self.depth_subscription = self.create_subscription(
            Image,
            '/camera/camera/depth/image_rect_raw',
            self.depth_callback,
            10
        )
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.camera_info_callback,
            10
        )

        self.color_frame = None
        self.depth_frame = None
        self.intrinsics = None

    def camera_info_callback(self, msg):
        # Extract camera intrinsics
        self.intrinsics = {
            "fx": msg.k[0],
            "fy": msg.k[4],
            "ppx": msg.k[2],
            "ppy": msg.k[5]
        }

    def color_callback(self, msg):
        self.color_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.process_frames()

    def depth_callback(self, msg):
        self.depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def process_frames(self):
        if self.color_frame is None or self.depth_frame is None or self.intrinsics is None:
            return

        # Run YOLO model on the color frame
        results = model(self.color_frame)

        # Filter for cell_phone class (class 67 in COCO for YOLO)
        cell_phone_detections = [d for d in results[0].boxes if d.cls == 67]

        if cell_phone_detections:
            # Find the cell_phone with the highest confidence
            best_detection = max(cell_phone_detections, key=lambda d: d.conf)
            box = best_detection.xyxy.cpu().numpy().astype(int).flatten()
            score = best_detection.conf.cpu().item()

            # Calculate center of the bounding box
            center_x = int((box[0] + box[2]) / 2)
            center_y = int((box[1] + box[3]) / 2)

            # Get depth value at the center of the bounding box
            depth_value = self.depth_frame[center_y, center_x]

            # Convert to camera coordinates
            camera_x = (center_x - self.intrinsics["ppx"]) * depth_value / self.intrinsics["fx"]
            camera_y = (center_y - self.intrinsics["ppy"]) * depth_value / self.intrinsics["fy"]
            camera_z = depth_value

            # Log the camera coordinates
            self.get_logger().info(f"cell_phone detected at camera coordinates: X={camera_x:.2f}, Y={camera_y:.2f}, Z={camera_z:.2f}")

            # Visualize the bounding box on the color frame
            self.visualize_with_opencv(self.color_frame, box, score, camera_x, camera_y, camera_z)
        else:
            # Show the frame even if no cell phone is detected
            self.visualize_with_opencv(self.color_frame, None, None, None, None, None)

    def visualize_with_opencv(self, frame, box, score, camera_x, camera_y, camera_z):

        R = np.array([
            [1, 0, 0],  # X는 동일
            [0, 0, 1],  # Z -> Y
            [0, -1, 0]  # Y -> -Z
        ])

        camera_coords = np.array([camera_x, camera_y, camera_z])
        camera_coords = R @ camera_coords

        if box is not None:
            label_text = f"cell_phone {score:.2f}"
            coords_text = f"X={camera_x:.2f}, Y={camera_y:.2f}, Z={camera_z:.2f}"
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(frame, label_text, (box[0], box[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, coords_text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            self.destroy_node()
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()