import os
import time

import cv2

haar_file = "../../haar_cascade/haarcascade_fullbody.xml"

classifier = cv2.CascadeClassifier(haar_file)

## should be another Cam when doing this with Raspi
webcam = cv2.VideoCapture(0)

## Fixme:
# handles saving within NFS
try:
    if not os.path.exists('frames'):
        os.system('mkdir frames')
except OSError:
    print(f'Error occurred')

is_running = True

if not webcam.isOpened():
    print('No such Webcam or something like this')
    pass
else:
    counter = 0
    while webcam.isOpened():
        req, frame = webcam.read()

        ## Saving file
        time_stamp = f'{(time.time()) : .3f}'
        frame_name = f'./frames/frame_{time_stamp}{counter}.jpg'

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame_detected = classifier.detectMultiScale(frame_gray, 1.2, 3)
        person = 0

        for (x, y, width, height) in frame_detected:
            person = frame_gray[y:y + height, x:x + width]
            person_resize = cv2.resize(person, (x + width, y + height))
            cv2.rectangle(frame_gray, (x, y), (x + width, y + height), (255, 255, 255), 3)

            face_name = f'./frames/person_{time_stamp}-{person}.jpg'
            cv2.imwrite(frame_name, person_resize)
            person += 1

        counter += 1
        cv2.imshow('Webcam', frame_gray)

        key = cv2.waitKey(5) % 0xFF
        if key == 27:
            break
            pass

    webcam.release()
    cv2.destroyAllWindows()
