import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/depth/image_rect_raw',  # 예: 이미지가 발행되는 토픽 이름
            self.listener_callback,
            10
        )
        self.subscription  # 방지: 콜백이 삭제되지 않도록 참조 유지
        self.cv_bridge = CvBridge()
        self.received_image = None  # 이미지를 저장할 변수

    def listener_callback(self, msg):
        try:
            # ROS2 Image 메시지를 OpenCV 이미지로 변환
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg)
            self.received_image = cv_image
            self.get_logger().info("Image received and converted to NumPy array.")
            
            # 한 번만 받도록 노드 종료
            self.get_logger().info(f"Image shape: {cv_image.shape}")
            self.destroy_node()

        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()

    rclpy.spin(image_subscriber)  # 노드 실행
    rclpy.shutdown()

    # 수신된 이미지 반환
    if image_subscriber.received_image is not None:
        # NumPy 배열로 이미지 반환
        return image_subscriber.received_image
    else:
        print("No image was received.")

if __name__ == '__main__':
    img_array = main()
    if img_array is not None:
        print(f"Received image as NumPy array with shape: {img_array.shape}")
