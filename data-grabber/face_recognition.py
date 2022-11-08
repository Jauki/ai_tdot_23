# captureing from webcam
# Opens camera and saves detected faces in ./frames

import cv2
import numpy as np
import os
import time

# F acedetection
haar_file = '../haar_cascade/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

webcam = cv2.VideoCapture(0)

## create a folder for frames
try:
    if not os.path.exists('../frames'):
        os.system('mkdir frames')
except OSError:
    print(f'Error occured')

background_mask = cv2.createBackgroundSubtractorMOG2()
is_running = True

if webcam.isOpened() == False:
    print('Exception')
    pass
else:
    counter = 0
    while webcam.isOpened():

        req, frame = webcam.read()

        time_stamp = f'{(time.time()) : .3f}'
        frame_name = f'./frames/frame_{time_stamp}{counter}.jpg'

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame_detected = face_cascade.detectMultiScale(frame, 1.3, 5)
        face_counter = 0
        size = (width, height) = (250, 250)

        for (x, y, width, height) in frame_detected:
            face = frame_gray[y:y + height, x:x + width]
            face_name = f'./frames/face_{time_stamp}-{face_counter}.jpg'
            cv2.imwrite(frame_name, frame_gray)
            cv2.rectangle(frame_gray, (x, y), (x + width, y + height), (255, 0, 0), 0)
            face_counter += 1

        # print(f'{time_stamp}')
        # cv2.putText(frame, time_stamp, fontScale=1)
        # masked_frame = background_mask.apply(frame)
        counter += 1
        cv2.imshow('Webcam', frame_gray)
        # cv2.imshow('masked', masked_frame)
        key = cv2.waitKey(5) % 0xFF
        if key == 27:
            break
            pass

    webcam.release()
    cv2.destroyAllWindows()
