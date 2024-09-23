import cv2
import mediapipe as mp
import numpy as np
import pickle

# 初始化 Mediapipe 的臉部偵測器
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# 初始化攝像頭
cap = cv2.VideoCapture(0)

# 儲存面部特徵資料
face_data = {}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video")
        break

    # 將影像轉為 RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 使用 Mediapipe 進行面部特徵點偵測
    results = face_mesh.process(rgb_frame)

    # 如果偵測到臉部，保存特徵點
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            ih, iw, _ = frame.shape

            face_points = []
            for i in range(468):
                x = face_landmarks.landmark[i].x
                y = face_landmarks.landmark[i].y
                face_points.append((x, y))

            # 保存 "GU" 的特徵點
            face_data['GU'] = face_points

            # 顯示保存成功並退出
            print("Face data saved!")
            break

    cv2.imshow("Saving Face Data", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 儲存面部特徵資料到文件
with open("face_data.pkl", "wb") as f:
    pickle.dump(face_data, f)

