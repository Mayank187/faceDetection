import cv2
import time
import mediapipe as mp

class FaceDetector:
    def __init__(self,minDetectionCon=0.5):
        self.minDetectionCon = minDetectionCon

        self. mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection()

    def findFaces(self, img, draw = True):

        img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(img_RGB)
        bboxes = []
        if self.results.detections:
            for id,detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin*iw), int(bboxC.ymin*ih), \
                        int(bboxC.width * iw), int(bboxC.height * ih)
                if draw:
                    cv2.rectangle(img, bbox, (0, 255, 0), 2)
                    cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 2, (25, 0, 25), 3)

                bboxes.append([id, bbox, int(detection.score[0]*100)])
                print(bboxes)

        return img,bboxes



def main():
    cap = cv2.VideoCapture('Sample/video1.mp4')

    c_time = 0
    p_time = 0

    detector = FaceDetector()

    while True:
        success, img = cap.read()

        img, boxList = detector.findFaces(img)

        # FPS calculation
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        # Adding FPS Text to the Video
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()