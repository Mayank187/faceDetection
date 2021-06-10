import cv2
import time
import mediapipe as mp

cap = cv2.VideoCapture('Sample/video1.mp4')

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

c_time = 0
p_time = 0

while True:
    success, img = cap.read()

    img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = faceDetection.process(img_RGB)

    if results.detections:
        for id,detection in enumerate(results.detections):
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin*iw), int(bboxC.ymin*ih), \
                    int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img, bbox, (0, 255, 0), 2)
            cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 2, (25, 0, 25), 3)

    # FPS calculation
    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time

    # Adding FPS Text to the Video
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)