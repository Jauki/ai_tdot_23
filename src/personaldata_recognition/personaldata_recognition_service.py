import cv2
import os
import numpy as np
import keras.models

class PersonalDataRecognitionResult:
    def __init__(self, age: int, gender: str):
        self.__age = age
        self.__gender = gender

    @property
    def age(self) -> int:
        return self.__age
    
    @property
    def gender(self) -> str:
        return self.__gender

class PersonalDataRecognitionService:
    def __init__(self) -> None:
        self.__IMAGE_SIZE = 80
        self.__GENDER_DICT = {0:'Male', 1:'Female'}

    def predict_frame(self, region_of_interest: np.ndarray) -> PersonalDataRecognitionResult:
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
        predicted_gender = self.__GENDER_DICT[round(self.gender_model.predict(model_input)[0][0][0])]
        predicted_age = round(self.age_model.predict(model_input)[1][0][0])
        
        return PersonalDataRecognitionResult(predicted_age, predicted_gender)

    def load(self, age_model_path: os.path, gender_model_path: os.path):
        self.age_model = keras.models.load_model(age_model_path)
        self.gender_model = keras.models.load_model(gender_model_path)
