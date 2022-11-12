import cv2
import os

class MainApplication:
    def __init__(self, face_cascade_classifier_path: os.path):
        self.__face_cascade_classifier = cv2.CascadeClassifier(face_cascade_classifier_path)

    def record_frames(self):
        # capture webcam-stream
        stream = cv2.VideoCapture(0)

        finished: bool = False
        while not finished:
            # get current frame
            (grabbed, frame) = stream.read()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert to RGB

            frame = self.crop_and_process_frame(frame)

            # display frame
            cv2.imshow("face-recognition", frame)

            # check for exit-key
            key = cv2.waitKey(50) & 0xFF
            if key == 0x1B:  # exit with ESC
                finished = True

        # release stream and close window
        stream.release()
        cv2.destroyAllWindows()

    def crop_and_process_frame(self, frame):
        # detect faces
        face_detections = self.__face_cascade_classifier.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)
        # classify every face (and display)
        for (x, y, w, h) in face_detections:
            # this is passed to all services
            region_of_interest = frame[y:y + h, x:x + w]

            color = (255, 2555, 255)  # color in BGR
            stroke = 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, stroke)

        # sends frames to services

        return frame


if __name__ == "__main__":
    face_cascade_classifier_path: os.path = os.path.join('..', 'haar_cascade', 'haarcascade_frontalface_default.xml')
    main = MainApplication(face_cascade_classifier_path)
    main.record_frames()
