import cv2
import numpy as np
import os
import time
import youtube_dl


def process_video(video_url, name):
    ydl_opts = {}
    ydl = youtube_dl.YoutubeDL(ydl_opts)
    info_dict = ydl.extract_info(video_url, download=False)

    formats = info_dict.get('formats', None)

    for f in formats:

        if f.get('format_note', None) == '480p':
            url = f.get('url', None)

            cap = cv2.VideoCapture(url)

            if not cap.isOpened():
                print('video not opened')
                exit(-1)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                seek_faces(frame, name)
            print('DONE!')
            cap.release()

    cv2.destroyAllWindows()


def seek_faces(source, name):
    haar_file = './haar_cascade/haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(haar_file)

    frame_detected = face_cascade.detectMultiScale(source, 1.3, 5)
    face_counter = 0
    size = (width, height) = (250, 250)

    for (x, y, width, height) in frame_detected:
        print('detected!')
        face_name = f'./frames/{name}/{name}_{time.time()}.jpg'
        cv2.imwrite(face_name, source)
        face_counter += 1
    pass


if __name__ == '__main__':
    name = "gu"
    try:
        if not os.path.exists(f'frames/{name}'):
            os.system(f'mkdir frames/{name}')
    except OSError:
        print(f'Error occured')
    process_video("https://www.youtube.com/watch?v=J3MSyvJSn9A", name)
