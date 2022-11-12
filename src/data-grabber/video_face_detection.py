import os
import time

import cv2
import youtube_dl


def process_video(video_url, name):
    counter = 0
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

            while True and (counter <= 1200):
                ret, frame = cap.read()
                if not ret:
                    break

                if seek_faces(frame, name):
                    counter += 1

            cap.release()

    cv2.destroyAllWindows()


def seek_faces(source, name):
    found = False
    ## targets the haarcascade file
    haar_file = 'haar_cascade/haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(haar_file)
    frame_detected = face_cascade.detectMultiScale(source, 1.3, 5)

    for face in frame_detected:
        face_name = f'./frames/{name}/{name}_{time.time()}.jpg'
        cv2.imwrite(face_name, source)
        print(face_name)
        found = True

    return found


if __name__ == '__main__':
    videos = [
        ["renate-bauer", "https://www.youtube.com/watch?v=wcG45zXcgtM"],
        ["walter-white", "https://www.youtube.com/watch?v=H1TjhNHlJGs"],
    ]
    for video in videos:
        try:
            if not os.path.exists(f'frames/{video[0]}'):
                os.system(f'mkdir frames/{video[0]}')
        except OSError:
            print(f'Error occured')

        process_video(video[1], video[0])
