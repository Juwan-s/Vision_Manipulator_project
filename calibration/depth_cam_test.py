import pyrealsense2.pyrealsense2 as rs
import cv2
import numpy as np

def main():
    # RealSense pipeline 초기화
    pipeline = rs.pipeline()
    print("Ckpt_1")
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    print("Ckpt_2")
    # Start the pipeline
    pipeline.start()
    print("Ckpt_3")
    pipeline.start(config)
    print("Ckpt_4")
    try:
        print("Ckpt_4")
        while True:
            # 깊이 데이터 가져오기
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()

            if not depth_frame:
                continue

            # 깊이 데이터를 numpy array로 변환
            depth_image = np.asanyarray(depth_frame.get_data())

            # 깊이 이미지를 8비트로 변환
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )

            # 시각화
            cv2.imshow('RealSense Depth Visualization', depth_colormap)

            # 종료 키 설정
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # 종료 시 파이프라인 정리
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
