import cv2
import numpy as np
import os
import pyscreenshot as ImageGrab
import time

haar_file = './haar_cascade/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

try:
    if not os.path.exists('frames'):
        os.system('mkdir frames')
except OSError:
    print(f'Error occured')

background_mask = cv2.createBackgroundSubtractorMOG2()
is_running = True

counter = 0
while True:
    grab_from_screen = ImageGrab.grab(bbox=(200, 0, 2880, 1800))

    frame = np.array(grab_from_screen)
    time_stamp = f'{(time.time()) : .3f}'
    frame_name = f'./frames/frame_{counter}.jpg'

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame_detected = face_cascade.detectMultiScale(frame, 1.3, 5)
    face_counter = 0
    size = (width, height) = (250, 250)

    for (x, y, width, height) in frame_detected:
        print("foo!")
        face_name = f'./frames/face_{time_stamp}-{face_counter}.jpg'
        cv2.imwrite(face_name, frame_gray)
        face_counter += 1

    counter += 1

    key = cv2.waitKey(5) % 0xFF
    if key == 27:
        break
        pass

cv2.destroyAllWindows()
