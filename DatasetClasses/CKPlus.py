import torch.utils.data as data, os, torch, numpy as np, sys, pandas as pd, random
from PIL import Image as im
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getDirectoriesInPath, getFilesInPath
#from generateDenseOpticalFlow import runDenseOpFlow

class CKPlus(data.Dataset):    
    def __init__(self, ckData, transform=None):
        self.transform = transform
        self.label = []
        self.filesPath = []

        labelsFolder = os.path.join(ckData,'Emotion')
        imagesFolder = os.path.join(ckData,'cohn-kanade-images')
        directoriesInFolder = getDirectoriesInPath(labelsFolder)
        for d in directoriesInFolder:
            sesFiles = getDirectoriesInPath(os.path.join(labelsFolder,d))
            for s in sesFiles:
                filesForLabel = getFilesInPath(os.path.join(labelsFolder,d,s))
                for f in filesForLabel:
                    self.label.append(self.openLabelFile(f))
                    fileName = f.split(os.path.sep)[-1]
                    imageFrameNumber = '_'.join(fileName.split('_')[:-1]) + '.png'
                    self.filesPath.append(os.path.join(imagesFolder,d,s,imageFrameNumber))
        

    def openLabelFile(self,pathFile):
        with open(pathFile,'r') as lf:
            for d in lf:
                return int(float(d.strip()))

    def __len__(self):
        return len(self.filesPath)

    def __getitem__(self, idx):
        path = self.filesPath[idx]
        image = im.open(path)
        label = self.label[idx]
        if self.transform is not None:
            image = self.transform(image)
            label = self.label

        return image, label, self.filesPath[idx]