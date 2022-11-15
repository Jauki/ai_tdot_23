import cv2
import pickle
from PIL import Image
import os
import numpy as np
import tensorflow.keras as keras
import keras_vggface
import datetime
from matplotlib import pyplot


class FaceRecognitionResult:
    def __init__(self, label: str, certainty: float):
        self.__label = label
        self.__certainty = certainty

    @property
    def label(self) -> str:
        return self.__label

    @property
    def certainty(self) -> float:
        return self.__certainty


class FaceRecognitionService:
    def __init__(self, face_cascade_classifier_path: os.path):
        self.__face_cascade_classifier = cv2.CascadeClassifier(face_cascade_classifier_path)
        self.__IMAGE_DIM = (self.__IMAGE_WIDTH, self.__IMAGE_HEIGHT) = (224, 224)

    def __build_model(self) -> None:
        number_of_classes = len(self.__class_labels.items())

        base_model = keras_vggface.vggface.VGGFace(include_top=False, model='vgg16',
                                                   input_shape=(self.__IMAGE_WIDTH, self.__IMAGE_HEIGHT, 3))

        # add custom NN that uses the extracted face-features to identify the person
        custom_model = base_model.output
        custom_model = keras.layers.GlobalAveragePooling2D()(custom_model)  # average pooling
        custom_model = keras.layers.Dense(1024, activation='relu')(custom_model)  # fully-connected layer with ReLU
        custom_model = keras.layers.Dense(1024, activation='relu')(custom_model)  # fully-connected layer with ReLU
        custom_model = keras.layers.Dense(512, activation='relu')(custom_model)  # fully-connected layer with ReLU
        custom_model = keras.layers.Dense(number_of_classes, activation='softmax')(
            custom_model)  # fully-connected layer with SoftMax

        self.__model: keras.Model = keras.Model(inputs=base_model.inputs, outputs=custom_model)

        # freeze the base model (it is already trained)
        for layer in self.__model.layers[:19]:
            layer.trainable = False

        # unfreeze the custom part that identifies the person
        for layer in self.__model.layers[19:]:
            layer.trainable = True

        self.__model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def preprocess_training_data(self, train_data_directory: os.path):
        # iterate over all people
        for root, _, files in os.walk(train_data_directory):
            class_label = os.path.basename(root).replace(" ", "-").lower()

            for file in files:
                # check if file is an image
                if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
                    image_path = os.path.join(root, file)

                    # load the image
                    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                    face_image_array = np.array(image, "uint8")

                    # detect faces
                    face_detections = self.__face_cascade_classifier.detectMultiScale(image, scaleFactor=1.1,
                                                                                      minNeighbors=5)

                    # delete and skip the image if more than 1 (or 0) face was detected
                    if len(face_detections) != 1:
                        os.remove(image_path)
                        continue  # skip the image

                    # save the detected face
                    for (x, y, w, h) in face_detections:
                        region_of_interest = face_image_array[y: y + h, x: x + w]

                        # resize the ROI
                        resized_face_image = cv2.resize(region_of_interest, self.__IMAGE_DIM)
                        face_image_array = np.array(resized_face_image, "uint8")

                        # remove the original image
                        os.remove(image_path)

                        # save the face-image
                        face_image = Image.fromarray(face_image_array)
                        face_image.save(image_path)

    def train(self, train_data_directory: os.path, epochs: int) -> None:
        image_data_generator = keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=keras.applications.mobilenet.preprocess_input,

            rotation_range=5,  # rotates the images
            # NOTE: (but only 5° because highly rotated faces are not detected by the haar-cascade-classifier anyway)
            fill_mode='nearest',  # account for the empty areas due to rotation

            horizontal_flip=True,
            # NOTE: vertical_flip does not make much sense (face upside down)

            brightness_range=[0.5, 1.5],

            zoom_range=0.2,
        )

        # uses the subdirectory-names as class-names
        train_data_generator = image_data_generator.flow_from_directory(
            directory=train_data_directory,
            target_size=self.__IMAGE_DIM,
            color_mode='rgb',
            batch_size=32,
            class_mode='categorical',
            shuffle=True
        )

        # build a class-label dict
        class_indices: dict[str, int] = train_data_generator.class_indices
        self.__class_labels: dict[int, str] = dict(zip(class_indices.values(), class_indices.keys()))

        self.__build_model()

        self.__model.fit(train_data_generator, batch_size=1, verbose=1, epochs=epochs)

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

    def test_webcam(self):
        # capture webcam-stream
        stream = cv2.VideoCapture(0)

        finished: bool = False
        while not finished:
            # get current frame
            (grabbed, frame) = stream.read()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert to RGB

            # detect faces
            face_detections = self.__face_cascade_classifier.detectMultiScale(rgb_frame,
                                                                              scaleFactor=1.3,
                                                                              minNeighbors=5)

            # classify every face (and display)
            for (x, y, w, h) in face_detections:
                region_of_interest = rgb_frame[y:y + h, x:x + w]

                # highlight ROI (face)
                color: tuple[int, int, int] = (255, 0, 0)  # color in BGR
                stroke: int = 2
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, stroke)

                # predict the face-class
                result: FaceRecognitionResult = self.predict_frame(region_of_interest)

                # display class-label
                font = cv2.FONT_HERSHEY_SIMPLEX
                color = (0, int(result.certainty * 255), int((1 - result.certainty) * 255))  # color in BGR
                stroke = 2
                probability_str: str = '%.0f' % (result.certainty * 100)
                cv2.putText(frame, f'{result.label}', (x, y - 40), font, 1, color, stroke, cv2.LINE_AA)
                cv2.putText(frame, f'{probability_str}%', (x, y - 10), font, 1, color, stroke, cv2.LINE_AA)

            # display frame
            cv2.imshow("face-recognition", frame)

            # check for exit-key
            key = cv2.waitKey(50) & 0xFF
            if key == 0x1B:  # exit with ESC
                finished = True

        # release stream and close window
        stream.release()
        cv2.destroyAllWindows()

    def save(self, storage_directory: os.path):
        # create model directory
        # date_time_str: str = datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')
        model_directory: os.path = os.path.join(storage_directory, f'face-recognition-model-last-state-state')
        os.mkdir(model_directory)

        # store keras model
        model_path: os.path = os.path.join(model_directory, 'model.h5')
        self.__model.save(model_path)

        # store class labels (person names)
        class_labels_path: os.path = os.path.join(model_directory, 'class_labels.pickle')
        with open(class_labels_path, 'wb+') as f:
            pickle.dump(self.__class_labels, f)

    def load(self, model_directory: os.path):
        # load model
        model_path: os.path = os.path.join(model_directory, 'model.h5')
        self.__model: keras.Model = keras.models.load_model(model_path)

        # load class labels
        class_labels_path: os.path = os.path.join(model_directory, 'class_labels.pickle')
        with open(class_labels_path, 'rb') as f:
            self.__class_labels = pickle.load(f)
