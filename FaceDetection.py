import cv2
import time
import mediapipe as mp

cap = cv2.VideoCapture(0)

c_time = 0
p_time = 0

while True:
    success, img = cap.read()

    # FPS calculation
    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time

    # Adding FPS Text to the Video
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)