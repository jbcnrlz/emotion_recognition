from cProfile import label
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def loadTextFile(pathFile):
    returnData = []
    with open(pathFile,'r') as fl:
        for f in fl:
            returnData.append(f.strip())

    return returnData

def loadFrames(frameFile):
    ff = loadTextFile(frameFile)
    ff = list(map(int,ff))
    return ff

def loadAnnFile(annFile):
    af = loadTextFile(annFile)
    af = [s.split(",") for s in af[1:]]
    af = [list(map(float,s)) for s in af]
    return np.array(af)

def main():
    frames = loadFrames("D:/PycharmProjects/emotion_recognition/selectedClips/10-60-1280x720.txt")

    p1mva = loadAnnFile("fc_clip1_1.csv")
    p2mva = loadAnnFile("fc_clip1_2.csv")

    thresSize = min(len(p1mva),len(p2mva))

    p1va = loadAnnFile("D:/Affwild/annotations/VA_Set/Train_Set/10-60-1280x720.txt")[frames][:thresSize]
    p2va = loadAnnFile("D:/Affwild/annotations/VA_Set/Train_Set/10-60-1280x720_right.txt")[frames][:thresSize]

    fig, axs = plt.subplots(2,2)
    for i in range(2):
        axs[0][i].plot(p1mva[:,i][:thresSize],label="Infered Value")
        axs[0][i].plot(p1va[:,i],label="GT Value")
        axs[1][i].plot(p2mva[:,i][:thresSize],label="Infered Value")
        axs[1][i].plot(p2va[:,i],label="GT Value")

    axs[0][1].legend()

    plt.show()

    print('oi')

if __name__ == '__main__':
    main()