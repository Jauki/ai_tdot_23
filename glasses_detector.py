import copy
from email.mime import image
from enum import Enum
from time import sleep
import numpy as np
import dlib
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import statistics

class Result(Enum):
    NO_FACE = 1
    NO_GLASSES = 2
    HAS_GLASSES = 3

# returns true if there are 2 or more occurences of 255 seperated by at least one other value
def hasTwoSpaces(array):
    search = 255
    lastElement = -200
    count = 0
    for ele in array:
        if (lastElement == ele):
            continue

        lastElement = ele
        if (ele == search):
            count = count + 1
    return count >= 2

class GlassesDector:

    def __init__(self) -> None:
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        pass

    def detectGlasses(self, img) -> Result:

        faces = self.detector(img)

        if(len(faces) == 0):
            return Result.NO_FACE

        rect = faces[0]

        sp = self.predictor(img, rect)
        landmarks = np.array([[p.x, p.y] for p in sp.parts()])

        # for x, y in landmarks:
        #     cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (0, 255, 0), 3)
        #     pass

        # cv2.imshow("frame", img)
        # cv2.waitKey(0)

        nose_bridge_x = []
        nose_bridge_y = [] 

        for i in [28,29,30,31,33,34,35]:
            nose_bridge_x.append(landmarks[i][0])
            nose_bridge_y.append(landmarks[i][1])
                
        ### x_min and x_max
        x_min = min(nose_bridge_x)
        x_max = max(nose_bridge_x)### ymin (from top eyebrow coordinate),  ymax
        y_min = landmarks[20][1]
        y_max = landmarks[29][1]


        # crop the image to only include the nose bridge
        img2 = Image.fromarray(img)
        img2 = img2.crop((x_min,y_min,x_max,y_max))


        img_blur = cv2.GaussianBlur(np.array(img2),(3,3), sigmaX=0, sigmaY=0)
        edges = cv2.Canny(image =img_blur, threshold1=100, threshold2=200)
        # plt.imshow(edges, cmap =plt.get_cmap('gray'))
        # plt.waitforbuttonpress()


        # print(f'edges.T {len(edges.T)}')
        # print(f'edges {len(edges)}')
        # print(f'edges[0] {len(edges[0])}')

        #center strip
        edges_center = edges.T[(int(len(edges.T)/2))]

        print(edges_center)

        ##WHAAAT schaut der einfach nur ob in der mittleren "spalte" eines bildes ein weißer pixel ist?
        glassesPresent = 255 in edges_center
        # glassesPresent = hasTwoSpaces(edges_center)

        cv2.line(edges, ((int(len(edges.T)/2)), 0), ((int(len(edges.T)/2)), len(edges)), (255, 0, 0), 1)

        img_blur_big = cv2.resize(img_blur, (400, 400))
        edges_big = cv2.resize(edges, (400, 400))

        cv2.imshow("ai2", img_blur_big)
        cv2.imshow("ai", edges_big)


        for i in range(68):
            cv2.circle(img, (landmarks[i][0], landmarks[i][1]), radius=2, color=(0, 0, 255), thickness=-1)
            cv2.putText(img, str(i), (landmarks[i][0], landmarks[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)

        if glassesPresent:
            return Result.HAS_GLASSES
        else:
            return Result.NO_GLASSES

        return 

def main():

    webcam = cv2.VideoCapture(0)

    detector = GlassesDector()

    while True:
        ret, frame = webcam.read()

        img_blur = cv2.GaussianBlur(np.array(frame),(3,3), sigmaX=0, sigmaY=0)
        edges = cv2.Canny(image =img_blur, threshold1=100, threshold2=200)
        cv2.imshow("canny", edges)

        cv2.imshow("stff", edges.T)

        result = detector.detectGlasses(frame)

        if result == Result.HAS_GLASSES:
            cv2.putText(frame, "glasses", (100, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0)) 

        if result == Result.NO_GLASSES:
            cv2.putText(frame, "no glasses", (100, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0)) 

        if result == Result.NO_FACE:
            cv2.putText(frame, "no face :(", (100, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0)) 

        cv2.imshow("frame", frame)

        
        key = cv2.waitKey(10) 
        if key == 27: 
            break

    pass

if __name__ == '__main__':
    main()

    # edges = np.array([
    #     [255, 0, 255],
    #     [255, 0, 255],
    #     [255, 0, 255]
    # ])

    # edges_center = edges.T[(int(len(edges.T)/2))]

    # print(edges_center)