import torch.utils.data as data, os, torch, numpy as np, sys, pandas as pd, random
from PIL import Image as im
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getDirectoriesInPath, getFilesInPath, printProgressBar
#from generateDenseOpticalFlow import runDenseOpFlow

class JoinedDataset(data.Dataset):    
    def __init__(self, pathData, transform=None):
        super(JoinedDataset,self).__init__()
        self.transform = transform
        self.filesPath = []
        self.label = []
        classesData = getDirectoriesInPath(pathData)
        for c in classesData:
            filesForClass = getFilesInPath(os.path.join(pathData,c))
            for idx, f in enumerate(filesForClass):
                printProgressBar(idx,len(filesForClass),length=50,prefix='Loading Faces from class %s...' % (c))
                self.filesPath.append(f)
                self.label.append(int(c))

    def __len__(self):
        return len(self.filesPath)

    def __getitem__(self, idx):
        path = self.filesPath[idx]
        image = im.open(path)
        label = torch.from_numpy(np.array(self.label[idx])).to(torch.long)
        if self.transform is not None:
            image = self.transform(image)

        return image, label, self.filesPath[idx]