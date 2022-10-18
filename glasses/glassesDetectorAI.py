import cv2

from glassesDetector import GlassesDector
from NeuralNet import NeuronalNet

def main():

    webcam = cv2.VideoCapture(0)

    network = NeuronalNet(0, 0, 0, 0.5)
    network.loadFromFile("epoch_3")

    detector = GlassesDector()

    while True:
        ret, frame = webcam.read()

        noseBridgeOutline = detector.getNoseBridgeOutline(frame)

        if noseBridgeOutline is not None:
            cv2.imshow("thing", noseBridgeOutline)

            flattNose = noseBridgeOutline.flatten()
            prediction = network.query(flattNose)
            result = bool(round(prediction[0][0]))

            cv2.putText(frame, f'{prediction[0][0]:.2f}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

            if(result):
                cv2.putText(frame, "glasses", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, "no glasses", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

        else:
            cv2.putText(frame, "no face detected :(", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)


        cv2.imshow("frame", frame)

        key = cv2.waitKey(10) 
        if key == 27: 
            break

    pass

if __name__ == '__main__':
    main()