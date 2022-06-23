import torch.utils.data as data, os, re, torch, numpy as np, sys, pandas as pd
from PIL import Image as im
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getDirectoriesInPath, getFilesInPath

class TeachingDataset(data.Dataset):    
    def __init__(self, teachingData, transform=None):
        self.transform = transform
        self.label = None
        self.filesPath = []

        dirs = getDirectoriesInPath(teachingData)
        for d in dirs:
            folderName = d.split(os.path.sep)[-1]
            faces = getFilesInPath(os.path.join(teachingData,folderName))
            for faceImage in faces:
                fileVideoName = faceImage.split(os.path.sep)[-1]
                if 'jpg' in fileVideoName:                            
                    self.filesPath.append(faceImage)

    def __len__(self):
        return len(self.filesPath)

    def __getitem__(self, idx):
        path = self.filesPath[idx]
        image = im.open(path)
        if self.transform is not None:
            image = self.transform(image)

        return image, [], self.filesPath[idx]