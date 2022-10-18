import enum
from fileinput import filename
import cv2
import os
import random
from tqdm import tqdm
from NeuralNet import NeuronalNet
from test_network import testNetwork

class Task():
    def __init__(self, fileName, hasGlasses) -> None:
        self.fileName = fileName
        self.hasGlasses = hasGlasses
        pass
    
    def __repr__(self) -> str:
        return f'{self.fileName} -> {self.hasGlasses}'

def main():

    glasses_directory = "D:\\pavica\\glasses"
    no_glasses_directory = "D:\\pavica\\no_glasses"

    glassesTasks = [Task(glasses_directory + "\\" + filename, True) for filename in os.listdir(glasses_directory)]
    noGlassesTasks = [Task(no_glasses_directory + "\\" + filename, False) for filename in os.listdir(no_glasses_directory)]

    allTasks = glassesTasks + noGlassesTasks
    random.shuffle(allTasks)


    inputNodes = 30 * 30
    hiddenNodes = 2_000
    outputNodes = 1
    learningRate = 0.5
    network = NeuronalNet(inputNodes, hiddenNodes, outputNodes, learningRate)

    rightGuesses, totalTasks = testNetwork(network, glasses_directory, no_glasses_directory)
    print(f'{rightGuesses} of {totalTasks}')
    print(f'Inital performance: - {rightGuesses / totalTasks} ( learning rate: {learningRate} )')

    for k, learningRate in enumerate([0.7, 0.5, 0.3, 0.2, 0.1]):

        network.learning_rate = learningRate

        progressbar = iter(tqdm(range(len(allTasks)), desc="Loading…", ascii=False))

        for i, task in enumerate(allTasks):

            progressbar.__next__()
        
            image = cv2.imread(task.fileName, cv2.IMREAD_GRAYSCALE)
            flatImage = image.flatten()

            network.train(flatImage, [task.hasGlasses])
            # print(i)

        
        rightGuesses, totalTasks = testNetwork(network, glasses_directory, no_glasses_directory)
        print(f'{rightGuesses} of {totalTasks}')
        print(f'Epoch {k} - {rightGuesses / totalTasks} ( learning rate: {learningRate} )')

        network.saveToFile(f'pavica_{k}')



        

    network.saveToFile("trainedNetwork4.npy")
    pass

if __name__ == '__main__':
    main()