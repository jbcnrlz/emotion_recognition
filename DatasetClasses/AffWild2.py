from fileinput import filename
import torch.utils.data as data, os, re, torch, numpy as np, sys
from PIL import Image as im
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getDirectoriesInPath, getFilesInPath
#from generateDenseOpticalFlow import runDenseOpFlow

class AFF2Data(data.Dataset):    
    def __init__(self, affData, phase, type='VA', transform=None):
        typeData = {"VA" : 'VA_Set','AU' : 'AU_Set', 'EXPR' : 'EXPR_Set'}
        self.dataType = typeData[type]
        self.transform = transform
        self.label = []
        self.filesPath = []
        files = getFilesInPath(os.path.join(affData,'annotations',self.dataType,phase))
        imagePath = os.path.join(affData,'cropped_aligned')
        for r in files:
            fileName = r.split(os.path.sep)[-1]            
            if fileName[-3:] != 'txt':                
                continue

            videoName = os.path.join(imagePath,fileName.split('.')[0])
            frames = getFilesInPath(videoName,imagesOnly=True)            
            if not phase == 'AffWild1_Set':
                labelsForImage = self.loadLabels(r)
                for frm in frames:
                    frameName = int(frm.split(os.path.sep)[-1][:-4]) - 1
                    if frameName > len(labelsForImage) or labelsForImage[frameName][0] < -1:
                        continue
                    self.filesPath.append(frm)
                    self.label.append(labelsForImage[frameName])
            else:
                for frm in frames:
                    frameName = int(frm.split(os.path.sep)[-1][:-4]) - 1
                    self.filesPath.append(frm)
                    self.label.append([-10,-10])

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