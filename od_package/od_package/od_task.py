import cv2
import torch
from torchvision.transforms import functional as F
from torchvision.ops import nms
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
import numpy as np
import json

from ament_index_python.packages import get_package_share_directory
import os

# JSON 파일 경로 설정
package_share_directory = get_package_share_directory('od_package')
coco_label_path = os.path.join(package_share_directory, 'resource', 'coco_label.json')

# JSON 파일 읽기
with open(coco_label_path) as f:
    categories = json.load(f)

COCO_INSTANCE_CATEGORY_NAMES = [0 for i in range(91)]
for d in categories:
    COCO_INSTANCE_CATEGORY_NAMES[d['id']] = d['name']

# PyTorch model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True).to(device)
model.eval()

# Apply Non-Maximum Suppression
def apply_nms(predictions, iou_threshold=0.5):
    boxes = predictions["boxes"]
    scores = predictions["scores"]
    labels = predictions["labels"]

    keep_indices = nms(boxes, scores, iou_threshold)

    predictions["boxes"] = boxes[keep_indices]
    predictions["scores"] = scores[keep_indices]
    predictions["labels"] = labels[keep_indices]
    return predictions

# Detect objects
def detect_objects(frame, confidence_threshold=0.5, iou_threshold=0.5):
    frame_tensor = F.to_tensor(frame).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(frame_tensor)[0]

    # Filter predictions by confidence threshold
    mask = outputs["scores"] > confidence_threshold
    predictions = {
        "boxes": outputs["boxes"][mask],
        "scores": outputs["scores"][mask],
        "labels": outputs["labels"][mask],
    }

    return apply_nms(predictions, iou_threshold)

# ROS 2 Node for Object Detection
class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')
        self.publisher_ = self.create_publisher(Image, 'detected_objects_image', 10)
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10
        )
        self.subscription  # prevent unused variable warning

    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Convert BGR to RGB for PyTorch
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predictions = detect_objects(frame_rgb, confidence_threshold=0.6, iou_threshold=0.5)

        # Visualize predictions on the frame
        frame = self.visualize_with_opencv(frame, predictions)

        # Convert frame to ROS Image message
        ros_image = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        ros_image.header = Header()
        ros_image.header.stamp = self.get_clock().now().to_msg()

        # Publish the ROS Image message
        self.publisher_.publish(ros_image)

    def visualize_with_opencv(self, frame, predictions):
        for box, score, label in zip(predictions["boxes"], predictions["scores"], predictions["labels"]):
            box = box.cpu().numpy().astype(int)
            label_text = COCO_INSTANCE_CATEGORY_NAMES[label.item()]
            score_text = f"{label_text} {score:.2f}"

            # Draw bounding box
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(frame, score_text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame

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