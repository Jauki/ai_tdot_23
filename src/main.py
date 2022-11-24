from typing import Generator
import cv2
import os
import numpy as np
from face_recognition.face_recognition_service import FaceRecognitionService, FaceRecognitionResult
from personaldata_recognition.personaldata_recognition_service import PersonalDataRecognitionService, PersonalDataRecognitionResult
from glasses_recognition.glasses_recognition_service import GlassesRecognitionService, GlassessRecognitionResult, Result as GlassesResult

CWD: os.path = os.getcwd()
FACE_CASCADE_CLASSIFIER_PATH: os.path = os.path.join(CWD, 'haar_cascade', 'haarcascade_frontalface_default.xml')
FACE_RECOGNITION_SERVICE_MODEL_PATH: os.path = os.path.join(CWD, 'face_recognition', 'model-storage',
                                                            'face-recognition-model-last-state')
AGE_DETECTION_MODEL_PATH: os.path = os.path.join(CWD, 'personaldata_recognition', 'model-storage',
                                                            'age_model.h5')
GENDER_DETECTION_MODEL_PATH: os.path = os.path.join(CWD, 'personaldata_recognition', 'model-storage',
                                                            'gender_model.h5')
GLASSESS_CLASSIFIER_PATH: os.path = os.path.join(CWD, 'haar_cascade', 'shape_predictor_68_face_landmarks.dat')

class Main:
    def __init__(self, face_cascade_classifier_path: os.path = FACE_CASCADE_CLASSIFIER_PATH):
        self.__face_cascade_classifier = cv2.CascadeClassifier(face_cascade_classifier_path)
        self.__init_services(face_cascade_classifier_path)

    def __init_services(self, face_cascade_classifier_path: os.path):
        # face recognition service
        self.__face_recognition_service = FaceRecognitionService(face_cascade_classifier_path)
        self.__face_recognition_service.load(FACE_RECOGNITION_SERVICE_MODEL_PATH)
        self.__personaldata_recognition_service = PersonalDataRecognitionService()
        self.__personaldata_recognition_service.load(AGE_DETECTION_MODEL_PATH, GENDER_DETECTION_MODEL_PATH)

        self.__glasses_recognition_service = GlassesRecognitionService(GLASSESS_CLASSIFIER_PATH)

    def run(self):
        frame_generator: Generator[np.ndarray, None, None] = self.__record_webcam()
        display_frame_generator: Generator[np.ndarray, None, None] = self.__process_frames(frame_generator)
        self.__display_frames(display_frame_generator)

    def __display_frames(self, frame_generator: Generator[np.ndarray, None, None]):
        for frame in frame_generator:
            # display frame
            cv2.imshow("face-recognition", frame)

    def __record_webcam(self) -> Generator[np.ndarray, None, None]:
        # capture webcam-stream
        stream = cv2.VideoCapture(0)

        finished: bool = False
        while not finished:
            # get current frame
            (grabbed, frame) = stream.read()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert to RGB

            yield frame

            # check for exit-key
            key = cv2.waitKey(1) & 0xFF
            if key == 0x1B:  # exit with ESC
                finished = True

        # release stream and close window
        stream.release()
        cv2.destroyAllWindows()

    def __process_frames(self, frame_generator) -> Generator[np.ndarray, None, None]:
        for frame in frame_generator:

            # detect faces
            face_detections = self.__face_cascade_classifier.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)
            # classify every face (and display)
            for (x, y, w, h) in face_detections:
                # this is passed to all services
                region_of_interest: np.ndarray = np.copy(frame[y:y + h, x:x + w])

                # highlight ROI
                color = (255, 255, 255)  # color in BGR
                stroke = 2
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, stroke)

                self.__execute_face_recognition_service(frame, region_of_interest, x, y)
                self.__execute_personaldata_recognition_service(frame, region_of_interest, x, y, h)
                # TODO: sends ROI to other services

                self.__execute_glasses_recognition_service(frame, region_of_interest, x, y)

                yield frame

    def __execute_face_recognition_service(self, frame: np.ndarray, region_of_interest: np.ndarray, x: int, y: int):
        # use face-recognition-service
        result: FaceRecognitionResult = self.__face_recognition_service.predict_frame(region_of_interest)

        # draw face-recognition-service result
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, int(result.certainty * 255), int((1 - result.certainty) * 255))  # color in BGR
        stroke = 2
        probability_str: str = '%.0f' % (result.certainty * 100)
        cv2.putText(frame, f'{result.label}', (x, y - 40), font, 1, color, stroke, cv2.LINE_AA)
        cv2.putText(frame, f'{probability_str}%', (x, y - 10), font, 1, color, stroke, cv2.LINE_AA)

    def __execute_personaldata_recognition_service(self, frame: np.ndarray, region_of_interest: np.ndarray, x: int,
                                                   y: int, height: int):
        # use face-recognition-service
        result: PersonalDataRecognitionResult = self.__personaldata_recognition_service.predict_frame(region_of_interest)

        # draw face-recognition-service result
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)  # color in BGR
        stroke = 2
        cv2.putText(frame, f'{result.gender}, {result.age} ', (x, y + 40), font, 1, color, stroke, cv2.LINE_AA)

    def __execute_glasses_recognition_service(self, frame: np.ndarray, region_of_interest: np.ndarray, x: int, y: int):
        # use face-recognition-service
        result: GlassessRecognitionResult = self.__glasses_recognition_service.detectGlasses(region_of_interest)

        # draw face-recognition-service result
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)  # color in BGR
        stroke = 2

        if result.result == GlassesResult.NO_FACE:
            return

        cv2.putText(frame, 
            "Glasses" if result.result == GlassesResult.HAS_GLASSES else "no glasses",
             (x, y - 20), font, 1, color, stroke, cv2.LINE_AA)

        for i in range(68):
            xPos = result.landmarks[i][0] + x
            yPos = result.landmarks[i][1] + y
            cv2.circle(frame, (xPos, yPos), radius=2, color=(0, 0, 255), thickness=-1)
            cv2.putText(frame, str(i), (xPos, yPos), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)



if __name__ == "__main__":
    main = Main()
    main.run()
