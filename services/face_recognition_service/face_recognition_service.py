from __future__ import annotations
import cv2
import pickle
import os
import numpy as np
from tensorflow import keras
from config import model_config


class FaceRecognitionResult:
    def __init__(self, label: str, certainty: float):
        self.label = label
        self.certainty = certainty

    def to_dto(self) -> dict[str, any]:
        return {
            'label': self.label,
            'certainty': f'{self.certainty}',
        }


class FaceRecognitionService:
    def __init__(self):
        self.__IMAGE_DIM = (self.__IMAGE_WIDTH, self.__IMAGE_HEIGHT) = (224, 224)
        self.__load(model_config.MODEL_PATH)

    def predict_label(self, model_input: np.ndarray) -> FaceRecognitionResult:
        # predict the face-class
        predicted_prob: np.ndarray = self.__model.predict(model_input, verbose=0)
        class_id: int = predicted_prob[0].argmax()
        return FaceRecognitionResult(label=self.__class_labels[class_id], certainty=predicted_prob[0][class_id])

    def predict_frame(self, region_of_interest: np.ndarray) -> FaceRecognitionResult:
        # resize ROI
        resized_face_image = cv2.resize(region_of_interest, self.__IMAGE_DIM)
        face_image_array: np.ndarray = np.array(resized_face_image, "uint8")

        # prepare model input
        model_input = face_image_array.reshape(1, self.__IMAGE_WIDTH, self.__IMAGE_HEIGHT, 3)
        model_input = model_input.astype('float32')
        model_input /= 255

        # predict the face-class
        return self.predict_label(model_input)

    def __load(self, model_directory: str):
        # load model
        model_path: str = os.path.join(model_directory, 'model.h5')
        self.__model: keras.Model = keras.models.load_model(model_path)

        # load class labels
        class_labels_path: str = os.path.join(model_directory, 'class_labels.pickle')
        with open(class_labels_path, 'rb') as f:
            self.__class_labels = pickle.load(f)
