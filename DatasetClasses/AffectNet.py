import torch.utils.data as data, os, re, torch, numpy as np, sys, pandas as pd, json
from PIL import Image as im
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getDirectoriesInPath, getFilesInPath
#from generateDenseOpticalFlow import runDenseOpFlow

class AffectNet(data.Dataset):    
    def __init__(self, afectdata, transform=None):
        self.transform = transform
        self.label = []
        self.filesPath = []

        faces = getFilesInPath(os.path.join(afectdata,'images'),imagesOnly=True)
        for f in faces:
            print("Loading face %s" % (f))
            imageNumber = f.split(os.path.sep)[-1][:-4]
            valValue = np.load(os.path.join(afectdata,'annotations','%d_val.npy' % (int(imageNumber))))
            aroValue = np.load(os.path.join(afectdata,'annotations','%d_aro.npy' % (int(imageNumber))))
            self.filesPath.append(f)
            self.label.append([valValue,aroValue])

    def __len__(self):
        return len(self.filesPath)

    def __getitem__(self, idx):
        path = self.filesPath[idx]
        image = im.open(path)
        label = torch.from_numpy(np.array( [self.label[idx][0].astype(np.float32),self.label[idx][1].astype(np.float32)] )).to(torch.float32)
        if self.transform is not None:
            image = self.transform(image)

        return image, label, self.filesPath[idx]