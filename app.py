import cv2
import mediapipe as mp

# Mediapipe 설정
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if results.multi_face_landmarks:
        for face in results.multi_face_landmarks:
            h, w, _ = img.shape

            # 왼쪽 눈 중심 좌표 계산
            left_eye = face.landmark[33]
            left_x, left_y = int(left_eye.x * w), int(left_eye.y * h)

            # 오른쪽 눈 중심 좌표 계산
            right_eye = face.landmark[263]
            right_x, right_y = int(right_eye.x * w), int(right_eye.y * h)

            # 안경 프레임 사각형 그리기
            eye_width = abs(right_x - left_x)
            eye_height = int(eye_width * 0.5)

            top_left = (left_x - 10, left_y - eye_height // 2)
            bottom_right = (right_x + 10, right_y + eye_height // 2)

            cv2.rectangle(img, top_left, bottom_right, (0, 0, 0), thickness=3)

            # 안경 다리 (브릿지)
            bridge_x = (left_x + right_x) // 2
            bridge_y = (left_y + right_y) // 2
            cv2.line(img, (bridge_x, bridge_y - 10), (bridge_x, bridge_y + 10), (0, 0, 0), 2)

    cv2.imshow("Simpke AR", img)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC 키로 종료
        break

cap.release()
cv2.destroyAllWindows()
