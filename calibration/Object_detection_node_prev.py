import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from rclpy.node import Node
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray

# Object Detection Model Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True).to(device)
model.eval()

CUP_CLASS_ID = 47  # COCO dataset에서 cup 클래스 ID

class ObjectDetectionNode(Node):
    def __init__(self, T_eff_to_cam):
        super().__init__('object_detection_node')
        self.cv_bridge = CvBridge()

        # 이미지 구독
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

        # 로봇 변환행렬 구독
        self.base_to_eff_sub = self.create_subscription(
            Float64MultiArray,
            '/base_to_eff_matrix',
            self.base_to_eff_callback,
            10
        )

        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_info_received = False
        self.color_image = None
        self.depth_image = None
        self.T_base_to_eff = None
        self.T_eff_to_cam = T_eff_to_cam

        # 타이머 콜백을 통해 주기적으로 오브젝트 디텍션 수행
        # self.timer = self.create_timer(0.1, self.detect_and_transform)

    def color_callback(self, msg):
        try:
            self.color_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"Color image conversion error: {e}")
            
    def camera_info_callback(self, msg):
        if not self.camera_info_received:
            try:
                # Camera matrix와 distortion coefficients 추출
                self.camera_matrix = np.array(msg.k).reshape((3, 3))
                self.dist_coeffs = np.array(msg.d)
                self.camera_info_received = True
                self.get_logger().info(f"Camera Matrix:\n{self.camera_matrix}")
                self.get_logger().info(f"Distortion Coefficients:\n{self.dist_coeffs}")
            except Exception as e:
                self.get_logger().error(f"Error processing CameraInfo: {e}")

    def depth_callback(self, msg):
        try:
            self.depth_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except CvBridgeError as e:
            self.get_logger().error(f"Depth image conversion error: {e}")

    def base_to_eff_callback(self, msg):
        # /base_to_eff_matrix 토픽으로부터 행렬 수신
        mat = np.array(msg.data).reshape(4,4)
        self.T_base_to_eff = mat

    def detect_and_transform(self):
        if self.color_image is None or self.depth_image is None or self.T_base_to_eff is None:
            # 필요한 데이터가 준비되지 않았다면 넘어감
            return

        frame = self.color_image.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = F.to_tensor(frame_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(frame_tensor)[0]

        labels = outputs["labels"].cpu().numpy()
        boxes = outputs["boxes"].cpu().numpy()
        scores = outputs["scores"].cpu().numpy()

        cup_indices = np.where(labels == CUP_CLASS_ID)[0]
        if len(cup_indices) == 0:
            # 컵이 감지되지 않았을 경우
            self.get_logger().info("No cup detected")
            cv2.imshow("Object Detection", frame)
            return

        # 가장 자신도가 높은 컵 선택
        cup_scores = scores[cup_indices]
        max_cup_idx = cup_indices[np.argmax(cup_scores)]

        box = boxes[max_cup_idx]

        # center_pixel 계산
        cx = int((box[0] + box[2]) / 2)
        cy = int((box[1] + box[3]) / 2)

        # 이미지 범위 확인 (depth_image 크기)
        h, w = self.depth_image.shape[:2]
        cx = max(0, min(cx, w - 1))
        cy = max(0, min(cy, h - 1))
        center_pixel = (cx, cy)

        depth_value = self.depth_image[center_pixel[1], center_pixel[0]]
        if depth_value == 0:
            self.get_logger().warn("Invalid depth value at detected cup center.")
            return

        # # Camera Intrinsics (예시 값)
        # camera_intrinsics = np.array([
        #     [905.37653593,   0.        , 653.92759642],
        #     [  0.        , 903.06919   , 376.95680054],
        #     [  0.        ,   0.        ,   1.        ],
        # ], dtype=np.float64)

        uv = np.array([center_pixel[0], center_pixel[1], 1.0])
        camera_coords = np.linalg.inv(self.camera_matrix) @ (uv * depth_value)
        camera_coords_h = np.append(camera_coords, 1.0)

        eff_coords = np.linalg.inv(self.T_eff_to_cam) @ camera_coords_h

        base_coords = self.T_base_to_eff @ eff_coords
        self.get_logger().info(f"Cup center in Base Frame: {base_coords[:3]}")

        # Visualization
        cv2.circle(frame, center_pixel, 5, (0, 0, 255), -1)
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        cv2.putText(frame, "Cup", (center_pixel[0] - 20, center_pixel[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("Object Detection", frame)

    def __del__(self):
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)

    # End-Effector → Camera Transform (예시 값)
    T_eff_to_cam = np.array(
[[  0.90177285,   0.33341757,  -0.27502445, -37.14875245],
 [ -0.37440123,   0.92052015,  -0.11165288, -27.04946166],
 [  0.21593852,   0.20365503,   0.95493203,  19.43836865],
 [  0.        ,   0.        ,   0.        ,   1.        ],]
    )

    node = ObjectDetectionNode(T_eff_to_cam)

    try:
        # 메인 루프에서 spin_once를 호출하며 OpenCV 이벤트 처리
        while rclpy.ok():
            while node.camera_matrix is None\
                or node.color_image is None \
                or node.depth is None:
                rclpy.spin_once(node)
            
            rclpy.spin_once(node, timeout_sec=0.01)
            node.detect_and_transform()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()




# rokey-jw@rokeyjw-desktop:~/Vision_Manipulator_project/ca