import cv2
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--num", type=int)

args = parser.parse_args()


# 웹캠 장치 번호 (기본: 0, 필요 시 다른 번호로 변경)
cap = cv2.VideoCapture(args.num)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Could not read frame.")
        break

    # 이미지를 GUI 창에 표시
    cv2.imshow("Webcam Feed", frame)

    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
