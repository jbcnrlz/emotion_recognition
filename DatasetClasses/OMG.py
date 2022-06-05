from fileinput import filename
import torch.utils.data as data, os, re, torch, numpy as np, sys, pandas as pd
from PIL import Image as im
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getDirectoriesInPath, getFilesInPath
#from generateDenseOpticalFlow import runDenseOpFlow

class OMGData(data.Dataset):    
    def __init__(self, omgData, annotationFile, transform=None):
        self.transform = transform
        self.label = []
        self.filesPath = []

        dirs = getDirectoriesInPath(omgData)
        anFile = pd.read_csv(annotationFile)
        for d in dirs:
            folderName = d.split(os.path.sep)[-1]
            if folderName != 'youtube_videos_temp':
                foldersFace = getDirectoriesInPath(os.path.join(omgData,d))
                for f in foldersFace:
                    va = [anFile[anFile["video"] == folderName][anFile[anFile["video"] == folderName]["utterance"] == "%s.mp4" % (f)]["valence"].values[0],anFile[anFile["video"] == folderName][anFile[anFile["video"] == folderName]["utterance"] == "%s.mp4" % (f)]['arousal'].values[0]]
                    faces = getFilesInPath(os.path.join(omgData,d,f))
                    for faceImage in faces:
                        fileVideoName = faceImage.split(os.path.sep)[-1]
                        if 'jpg' in fileVideoName:                            
                            self.filesPath.append(faceImage)
                            self.label.append(va)

    def loadLabels(self,path):
        van = []
        with open(path,'r') as fp:
            fp.readline()
            for f in fp:
               van.append(list(map(float,f.split(','))))

        return van

    def __len__(self):
        return len(self.filesPath)

    def __getitem__(self, idx):
        path = self.filesPath[idx]
        image = im.open(path)
        label = torch.from_numpy(np.array(self.label[idx])).to(torch.float32)
        if self.transform is not None:
            image = self.transform(image)

        return image, label, self.filesPath[idx]