import torch.utils.data as data, os, re, torch, numpy as np, sys, pandas as pd, json
from PIL import Image as im
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getDirectoriesInPath, getFilesInPath
#from generateDenseOpticalFlow import runDenseOpFlow

class AFEWVA(data.Dataset):    
    def __init__(self, afewdata, transform=None):
        self.transform = transform
        self.label = []
        self.filesPath = []

        dirs = getDirectoriesInPath(afewdata)        
        for d in dirs:
            subjectNumber = d.split(os.path.sep)[-1]
            subjectAnn = json.load(open(os.path.join(afewdata,d,subjectNumber+'.json')))
            faceFiles = getFilesInPath(os.path.join(afewdata,d,'faces'),imagesOnly=True)
            for f in faceFiles:
                frameNumber = f.split(os.path.sep)[-1][:-4]
                self.filesPath.append(f)
                self.label.append([subjectAnn['frames'][frameNumber]['arousal'],subjectAnn['frames'][frameNumber]['valence']])

    def __len__(self):
        return len(self.filesPath)

    def __getitem__(self, idx):
        path = self.filesPath[idx]
        image = im.open(path)
        label = torch.from_numpy(np.array(self.label[idx])).to(torch.float32)
        if self.transform is not None:
            image = self.transform(image)

        return image, label, self.filesPath[idx]