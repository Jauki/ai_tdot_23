import os
import cv2
from glassesDetector import GlassesDector

def main():

    outputFolder = "D:\\pavica"
    glassesFolder = f'{outputFolder}\\glasses'
    noGlassesFolder = f'{outputFolder}\\no_glasses'

    currentlyHasGlasses = False
    paused = True

    glassesDetector = GlassesDector()
    webcam = cv2.VideoCapture(0)

    os.makedirs(glassesFolder, exist_ok=True)
    os.makedirs(noGlassesFolder, exist_ok=True)
    fileCount = len(os.listdir(noGlassesFolder)) + len(os.listdir(glassesFolder))

    while True:

        ret, frame = webcam.read()

        if(paused):
            cv2.putText(frame, "paused", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
            
        else:
            output = glassesDetector.getNoseBridgeOutline(frame)
            if(output is None):
                cv2.putText(frame, "no face detected", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
            else:
                cv2.imwrite(f'{glassesFolder if currentlyHasGlasses else noGlassesFolder}\\{fileCount}.jpg', output)
                fileCount += 1

                cv2.imshow("last saved frame", output)


        cv2.putText( frame,
            'Expecting faces with glasses' if currentlyHasGlasses else 'Expecting faces without glasses',
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA
        )
        cv2.imshow("frame", frame)
        key = cv2.waitKey(33)

        
        if key == 27: # esc
            return

        if key == 32: # space
            paused = not paused

        if key == 97: # a
            currentlyHasGlasses = False

        if key == 100: # d
            currentlyHasGlasses = True


if __name__ == '__main__':
    main()