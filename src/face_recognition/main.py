import os
from face_recognition_service import FaceRecognitionService


def main():
    cwd: os.path = os.getcwd()
    train_data_directory: os.path = os.path.join(cwd, 'face_recognition', 'training-data')
    face_cascade_classifier_path: os.path = os.path.join(cwd, 'haar_cascade', 'haarcascade_frontalface_default.xml')
    storage_path: os.path = os.path.join(os.getcwd(), 'face_recognition', 'model-storage')

    model = FaceRecognitionService(face_cascade_classifier_path)

    # model.preprocess_training_data(train_data_directory)

    # model.train(train_data_directory, 3)
    # model.save(storage_path)

    model_storage_path = os.path.join(storage_path, 'face-recognition-model-last-state')
    model.load(model_storage_path)
    model.test_webcam()


if __name__ == "__main__":
    main()
