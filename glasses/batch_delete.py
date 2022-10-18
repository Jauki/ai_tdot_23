import os

dir = "D:\\output\\glasses"

for filePath in [dir + "\\" + file for file in os.listdir(dir)]:
    if filePath.find("face") == -1:
        print(filePath)
        os.remove(filePath)