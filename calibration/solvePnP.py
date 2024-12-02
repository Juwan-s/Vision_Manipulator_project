import cv2
import numpy as np

# 체스보드의 크기 (가로 7칸, 세로 9칸)
chessboard_size = (7, 9)
square_size = 20  # 각 사각형의 한 변 크기 (단위: mm)

# 체스보드의 3D 세계 좌표 생성 (Z=0, 평면 좌표)
object_points = np.zeros((chessboard_size[0] * chessboard_size[1], 3), dtype=np.float32)
object_points[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
object_points *= square_size  # 20x20 mm로 변환

# 카메라 내부 파라미터 (가정된 값)
camera_matrix = np.array([
    [756.79624248,   0.         ,300.79559778],
    [  0.,         758.18809564, 283.84914722],
    [  0.,           0.,           1.        ]]
, dtype=np.float64)


# 렌즈 왜곡 계수 (가정된 값)
dist_coeffs = np.array([-0.41748243, 0.2726393, 0.00340657, 0.00165751, -0.39339633])

# 비디오 캡처 초기화
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

print("Press 'c' to capture a frame and detect chessboard corners. Press 'q' to quit.")

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    # 프레임 표시
    cv2.imshow("Video Feed", frame)

    # 키 입력 대기
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):  # 'q'를 누르면 종료
        print("Exiting...")
        break
    elif key == ord('c'):  # 'c'를 누르면 현재 프레임에서 처리
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 그레이스케일 변환

        # 체스보드 코너 찾기
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            # 코너를 그려서 시각적으로 확인
            cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)

            # 2D 이미지에서 관찰된 코너 좌표 (픽셀 좌표)
            image_points = corners.reshape(-1, 2)

            # solvePnP 실행
            success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)

            if success:
                # 회전 벡터를 회전 행렬로 변환
                rotation_matrix, _ = cv2.Rodrigues(rvec)

                print("\nPose Estimation Results:")
                print("Rotation Vector (rvec):\n", rvec)
                print("Translation Vector (tvec):\n", tvec)
                print("Rotation Matrix (R):\n", rotation_matrix)

                # 카메라 포즈를 4x4 변환 행렬로 표현
                transformation_matrix = np.eye(4)
                transformation_matrix[:3, :3] = rotation_matrix
                transformation_matrix[:3, 3] = tvec.flatten()
                print("Camera Pose (4x4 Transformation Matrix):\n", transformation_matrix)

            else:
                print("solvePnP failed to estimate pose.")
            
            # 결과 프레임 표시
            cv2.imshow("Detected Chessboard", frame)
            cv2.waitKey(0)  # 사용자 확인 후 아무 키나 눌러 닫기
        else:
            print("Chessboard corners not found. Try again.")

# 자원 해제
cap.release()
cv2.destroyAllWindows()