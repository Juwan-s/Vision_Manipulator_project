import cv2
import torch
from ultralytics import YOLO
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header, Float64MultiArray
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
        
        self.create_subscription(
            Float64MultiArray,
            '/base_to_eff_matrix',
            self.base_to_eff_callback,
            10
        )

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
        
        self.T_eff_to_cam = np.array(
[[-5.29621954e-01,  3.50601354e-01, -7.72385445e-01,  3.09028722e+02],
 [ 5.33484248e-01,  8.45617707e-01,  1.80347044e-02,  1.79618245e+01],
 [ 6.59465801e-01, -4.02503893e-01, -6.34898002e-01,  6.77169843e+02],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]

            )   
        
    def base_to_eff_callback(self, msg):
        # /base_to_eff_matrix 토픽으로부터 행렬 수신
        mat = np.array(msg.data).reshape(4,4)
        self.T_base_to_eff = mat

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
        cell_phone_detections = [d for d in results[0].boxes if d.cls == 41]

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

        
        if box is not None:
            # 카메라 좌표를 로봇 좌표로 변환
            camera_coords = np.array([camera_x, camera_y, camera_z])
            
            camera_coords_h = np.append(camera_coords, 1.0)
            
            eff_coords = np.linalg.inv(self.T_eff_to_cam) @ camera_coords_h
            
            robot_coords = self.T_base_to_eff @ eff_coords
            
            # 화면에 표시할 정보
            label_text = f"cell_phone {score:.2f}"
            coords_text = f"X={robot_coords[0]:.2f}, Y={robot_coords[1]:.2f}, Z={robot_coords[2]:.2f}"
            
            # 사각형 및 텍스트 그리기
            cv2.circle(frame, (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)), 5, (0, 0, 255), -1)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(frame, label_text, (box[0], box[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, coords_text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 화면에 출력
        cv2.imshow("Detection", frame)
        
        # 'q' 키로 종료
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
