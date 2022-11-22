import copy
from email.mime import image
from enum import Enum
import os
from time import sleep
import numpy as np
import dlib
import cv2
from PIL import Image
import statistics

class Result(Enum):
    NO_FACE = 1
    NO_GLASSES = 2
    HAS_GLASSES = 3

class GlassessRecognitionResult:
    def __init__(self, result: Result, landmarks: np.array):
        self.__result = result
        self.__landmarks = landmarks

    @property
    def result(self) -> int:
        return self.__result

    @property
    def landmarks(self) -> str:
        return self.__landmarks

class GlassesRecognitionService:
    def __init__(self, path : os.path) -> None:
        self.__detector = dlib.get_frontal_face_detector()
        self.__predictor = dlib.shape_predictor(path)

    def detectGlasses(self, img) -> GlassessRecognitionResult:

        faces = self.__detector(img)

        if len(faces) == 0:
           return GlassessRecognitionResult(Result.NO_FACE, np.array([]))

        face = faces[0]
        # face = [(0, 0), (len(img), len(img.T))]

        # Get the face landmarks
        sp = self.__predictor(img, face)
        landmarks = np.array([[p.x, p.y] for p in sp.parts()])

        # get nose bridge coordinates
        nose_bridge_x = []
        nose_bridge_y = []

        for i in [28,29,30,31,33,34,35]:
            nose_bridge_x.append(landmarks[i][0])
            nose_bridge_y.append(landmarks[i][1])

        # x_min and x_max
        x_min = min(nose_bridge_x)
        x_max = max(nose_bridge_x)
        # ymin (from top eyebrow coordinate)
        y_min = landmarks[20][1]
        y_max = landmarks[29][1]

        # crop the image to only include the nose bridge
        img2 = Image.fromarray(img)
        img2 = img2.crop((x_min,y_min,x_max,y_max))

        # apply filters to image
        img_blur = cv2.GaussianBlur(np.array(img2),(3,3), sigmaX=0, sigmaY=0)
        edges = cv2.Canny(image =img_blur, threshold1=100, threshold2=200)

        # get center vertical line of nose bridge
        edges_center = edges.T[(int(len(edges.T)/2))]

        glasses_present = 255 in edges_center

        result = Result.HAS_GLASSES if glasses_present else Result.NO_GLASSES
        return GlassessRecognitionResult(result, landmarks)