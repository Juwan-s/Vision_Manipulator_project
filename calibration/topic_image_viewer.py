import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImageViewer(Node):
    def __init__(self):
        super().__init__('image_viewer_node')
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',  # 수신할 이미지 토픽 이름 (환경에 맞게 변경)
            self.image_callback,
            10
        )
        self.bridge = CvBridge()
        self.get_logger().info("ImageViewer node has been started.")

    def image_callback(self, msg):
        # ROS Image -> OpenCV Image 변환
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # OpenCV 윈도우에 이미지 표시
        cv2.imshow("Camera Image", frame)
        cv2.waitKey(1)  # 1ms 대기 (이 값을 줄이면 CPU사용률이 높아질 수 있음)

def main(args=None):
    rclpy.init(args=args)
    node = ImageViewer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
