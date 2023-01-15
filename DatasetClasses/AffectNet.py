import torch.utils.data as data, os, re, torch, numpy as np, sys, pandas as pd, json
from PIL import Image as im
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getDirectoriesInPath, getFilesInPath
#from generateDenseOpticalFlow import runDenseOpFlow

class AffectNet(data.Dataset):    
    def __init__(self, afectdata, typeExperiment="VA", transform=None, termsQuantity=151, fixLabel=None):
        self.terms = None if typeExperiment != 'TERMS' else self.loadTermsFile(termsQuantity)
        self.transform = transform
        self.label = []
        self.filesPath = []
        self.typeExperiment = typeExperiment
        faces = getFilesInPath(os.path.join(afectdata,'images'),imagesOnly=True)
        for f in faces:
            print("Loading face %s" % (f))
            imageNumber = f.split(os.path.sep)[-1][:-4]
            if typeExperiment == "VA":
                valValue = np.load(os.path.join(afectdata,'annotations','%d_val.npy' % (int(imageNumber))))
                aroValue = np.load(os.path.join(afectdata,'annotations','%d_aro.npy' % (int(imageNumber))))
                self.label.append([valValue,aroValue])
            elif typeExperiment == "EXP":
                currLabel = np.load(os.path.join(afectdata,'annotations' ,'%d_exp.npy' % (int(imageNumber))))
                if fixLabel is not None and int(currLabel) not in fixLabel:
                    continue
                self.label.append(int(currLabel))
            else:
                currLabel = self.loadTermData(os.path.join(afectdata,'annotations_%d' % (termsQuantity),'%d_terms.txt' % (int(imageNumber))))
                self.label.append(np.where(self.terms == currLabel)[0][0])
            self.filesPath.append(f)
            
    def loadTermsFile(self,termsQuantity):
        return np.array(pd.read_csv('joinedWithDistance_%d.csv' % (termsQuantity)))[:,0]

    def __len__(self):
        return len(self.filesPath)

    def loadTermData(self,pathData):
        returnData = ""
        with open(pathData,'r') as pd:
            for p in pd:
                returnData = p

        return returnData

    def __getitem__(self, idx):
        path = self.filesPath[idx]
        image = im.open(path)
        if self.typeExperiment == 'TERMS':
            label = torch.from_numpy(np.array(self.label[idx])).to(torch.float32)
        elif self.typeExperiment == 'EXP':
            label = torch.from_numpy(np.array(self.label[idx])).to(torch.long)
        else:
            label = torch.from_numpy(np.array( [self.label[idx][0].astype(np.float32),self.label[idx][1].astype(np.float32)] )).to(torch.float32)
        if self.transform is not None:
            image = self.transform(image)

        return image, label, self.filesPath[idx]