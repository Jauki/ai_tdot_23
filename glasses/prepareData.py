import os
import cv2
from glassesDetector import GlassesDector


def main():
    folder = "D:\\glasses\\no_glasses"
    output = "D:\\output\\no_glasses"

    os.makedirs(output)

    index = 0
    detector = GlassesDector()

    for fileName in os.listdir(folder):
        file = folder + "\\" + fileName

        face = cv2.imread(file)
        
        noseBridge = detector.getNoseBridgeOutline(face)

        index += 1
        print(f'{index} - {len(noseBridge)} | {len(noseBridge[0])}')

        if noseBridge is not None :
            cv2.imwrite(output + "\\" + fileName, noseBridge)


if __name__ == '__main__':
    main()