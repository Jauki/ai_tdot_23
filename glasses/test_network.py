import enum
from fileinput import filename
import cv2
import os
import random
from NeuralNet import NeuronalNet

class Task():
    def __init__(self, fileName, hasGlasses) -> None:
        self.fileName = fileName
        self.hasGlasses = hasGlasses
        pass
    
    def __repr__(self) -> str:
        return f'{self.fileName} -> {self.hasGlasses}'

def testNetwork(network : NeuronalNet, glasses_directory, no_glasses_directory) -> float:

    glassesTasks = [Task(glasses_directory + "\\" + filename, True) for filename in os.listdir(glasses_directory)]
    noGlassesTasks = [Task(no_glasses_directory + "\\" + filename, False) for filename in os.listdir(no_glasses_directory)]

    allTasks = glassesTasks + noGlassesTasks
    random.shuffle(allTasks)


    rightGuesses = 0
    for i, task in enumerate(allTasks):
    
        image = cv2.imread(task.fileName, cv2.IMREAD_GRAYSCALE)
        flatImage = image.flatten()

        output = network.query(flatImage)
        hasGlasses = round(output[0][0])

        # print(f'{bool(hasGlasses)} - expected: {task.hasGlasses}')

        if(bool(hasGlasses) == task.hasGlasses):
            rightGuesses += 1

    return (rightGuesses, len(allTasks))

def main():
    glasses_directory = "D:\\combined\\glasses"
    no_glasses_directory = "D:\\combined\\no_glasses"

    inputNodes = 30 * 30
    hiddenNodes = 1000
    outputNodes = 2
    learningRate = 0.5
    network = NeuronalNet(inputNodes, hiddenNodes, outputNodes, learningRate)

    network.loadFromFile("big_epoch_1")

    rightGuesses, numberOfTasks = testNetwork(network, glasses_directory, no_glasses_directory) 

    print(f'The AI managed to guess {rightGuesses} out of {numberOfTasks} tasks')

    pass

if __name__ == '__main__':
    main()