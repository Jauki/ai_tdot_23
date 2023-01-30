from __future__ import annotations
import cv2
import os
import numpy as np
import keras.models
from config import model_config


class GenderEstimationResult:
    def __init__(self, gender: str):
        self.gender = gender

    def to_dto(self) -> dict[str, any]:
        return {
            'gender': self.gender,
        }

    @staticmethod
    def from_dto(dto: dict[str, any]) -> GenderEstimationResult:
        return GenderEstimationResult(
            gender=dto['gender'],
        )


class GenderEstimationService:
    def __init__(self) -> None:
        self.__IMAGE_SIZE = 80
        self.__GENDER_DICT = {0: 'Male', 1: 'Female'}
        self.load(model_config.MODEL_PATH)

    def predict_frame(self, region_of_interest: np.ndarray) -> GenderEstimationResult:
        # resize ROI
        resized_face_image = cv2.resize(region_of_interest, (self.__IMAGE_SIZE, self.__IMAGE_SIZE))
        # color ROI
        resized_face_image = cv2.cvtColor(resized_face_image, cv2.COLOR_RGB2GRAY)

        features = []
        features.append(np.array(resized_face_image))
        features = np.array(features)
        features = features.reshape(len(features), self.__IMAGE_SIZE, self.__IMAGE_SIZE, 1)
        features = features / 255.0
        features = features[0].reshape(1, self.__IMAGE_SIZE, self.__IMAGE_SIZE, 1)

        model_input = features
        predicted_gender = self.__GENDER_DICT[round(self.gender_model.predict(model_input, verbose=0)[0][0][0])]

        return GenderEstimationResult(predicted_gender)

    def load(self, gender_model_path: os.path):
        self.gender_model = keras.models.load_model(gender_model_path)
