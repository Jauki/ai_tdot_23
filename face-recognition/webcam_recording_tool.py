import cv2
import os
import time

person = 'test'

video_path = 0
stream = cv2.VideoCapture(video_path)

if not stream.isOpened():
    print("webcam not available")
    exit()

# create
storage_directory: os.path = os.path.join(os.getcwd(), 'recorded-frames', person)
if not os.path.exists(storage_directory):
    os.mkdir(storage_directory)

# load haar cascade
face_cascade_classifier_path: os.path = os.path.join('..', 'haar_cascade', 'haarcascade_frontalface_default.xml')
face_cascade_classifier = cv2.CascadeClassifier(face_cascade_classifier_path)

image_count: int = 0
draw_color: tuple[int, int, int] = (0, 255, 0)  # color in BGR
warn_color: tuple[int, int, int] = (0, 0, 255)
draw_stroke: int = 1
draw_font = font = cv2.FONT_HERSHEY_SIMPLEX

finished: bool = False
while not finished:
    # get current frame
    (grabbed, frame) = stream.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert to RGB

    # detect faces
    face_detections = face_cascade_classifier.detectMultiScale(rgb_frame, scaleFactor=1.3, minNeighbors=5)

    if len(face_detections) == 1:
        image_count += 1

        # store image
        time_stamp = f'{time.time() : .3f} Zeit'
        image_path: os.path = os.path.join(storage_directory, f'{time_stamp}.jpg')
        cv2.imwrite(image_path, frame)
    else:
        cv2.putText(frame, f'image should not contain exactly 1 face', (20, 60), font, 0.5, warn_color, draw_stroke,
                    cv2.LINE_AA)

    # classify every face (and display)
    for (x, y, w, h) in face_detections:
        region_of_interest = rgb_frame[y:y + h, x:x + w]

        # highlight ROI (face)
        cv2.rectangle(frame, (x, y), (x + w, y + h), draw_color, draw_stroke)

    cv2.putText(frame, f'person: {person}', (20, 40), font, 0.5, draw_color, draw_stroke, cv2.LINE_AA)
    cv2.putText(frame, f'({image_count} / 1000)', (20, 20), font, 0.5, draw_color, draw_stroke, cv2.LINE_AA)

    # display frame
    cv2.imshow(f'face-recognition - {person}', frame)

    # check for exit-key
    key = cv2.waitKey(50) & 0xFF
    if key == 0x1B:  # exit with ESC
        finished = True

# release stream and close window
stream.release()
cv2.destroyAllWindows()
