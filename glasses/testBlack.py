import os
import cv2

from NeuralNet import NeuronalNet

def main():
    directory = "D:\\combined\\no_glasses"

    network = NeuronalNet(0, 0, 0, 0.5)
    network.loadFromFile("epoch_3")

    blackCounter = 0

    nonBlackWrongGuesses = 0
    blackWrongGuesses = 0

    for file in os.listdir(directory):
        filePath = f'{directory}\\{file}'

        image = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)
        
        hasGlasses = bool(round(network.query(image.flatten())[0][0]))

        if cv2.countNonZero(image) == 0:
            if(hasGlasses):
                blackWrongGuesses += 1

            blackCounter += 1
        else:
            if(hasGlasses):
                cv2.imshow("asdf", image)
                cv2.waitKey(0)
                nonBlackWrongGuesses += 1

    pass

    print(f'{blackCounter} out of {len(os.listdir(directory))} files are entirely black')
    print(f'{blackWrongGuesses} out of {blackCounter} entirely black pictures were falsely identified')
    print(f'{nonBlackWrongGuesses} out of {len(os.listdir(directory)) - blackCounter} non black pictures were falsely identified')

if __name__ == '__main__':
    main()