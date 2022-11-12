import os
from face_recognition_service import FaceRecognitionService


def main():
    train_data_directory: os.path = os.path.join(os.getcwd(), 'training-data')
    face_cascade_classifier_path: os.path = os.path.join('..', 'haar_cascade', 'haarcascade_frontalface_default.xml')
    storage_path: os.path = os.path.join(os.getcwd(), 'model-storage')

    model = FaceRecognitionService(face_cascade_classifier_path)

    # model.preprocess_training_data(train_data_directory)

    # model.train(train_data_directory, 3)
    # model.save(storage_path)

    model_storage_path = os.path.join(storage_path, 'face-recognition-model-last-state')
    model.load(model_storage_path)
    model.test_webcam()


if __name__ == "__main__":
    main()
