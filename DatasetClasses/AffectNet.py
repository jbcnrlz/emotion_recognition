import torch.utils.data as data, os, torch, numpy as np, sys, pandas as pd, random
from PIL import Image as im
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getDirectoriesInPath, getFilesInPath, printProgressBar
#from generateDenseOpticalFlow import runDenseOpFlow

class AffectNet(data.Dataset):    
    def __init__(self, afectdata, typeExperiment="VA", transform=None, termsQuantity=151,exchangeLabel=None,loadLastLabel=True):
        self.exchangeLabel = exchangeLabel
        self.terms = None if typeExperiment != 'TERMS' else self.loadTermsFile(termsQuantity)
        self.transform = transform
        self.label = []
        self.filesPath = []
        self.typeExperiment = typeExperiment
        quantitylabels = None
        faces = getFilesInPath(os.path.join(afectdata,'images'),imagesOnly=True)
        for idx, f in enumerate(faces):
            printProgressBar(idx,len(faces),length=50,prefix='Loading Faces...')
            imageNumber = f.split(os.path.sep)[-1][:-4]
            if typeExperiment == "VA":
                valValue = np.load(os.path.join(afectdata,'annotations','%d_val.npy' % (int(imageNumber))))
                aroValue = np.load(os.path.join(afectdata,'annotations','%d_aro.npy' % (int(imageNumber))))
                self.label.append([valValue,aroValue])
            elif typeExperiment == "EXP":
                currLabel = np.load(os.path.join(afectdata,'annotations' ,'%d_exp.npy' % (int(imageNumber))))
                if int(currLabel) == 7 and not loadLastLabel:
                    continue
                self.label.append(int(currLabel))
            else:
                currLabel = self.loadTermData(os.path.join(afectdata,'annotations_%d' % (termsQuantity),'%d_terms.txt' % (int(imageNumber))))
                self.label.append(np.where(self.terms == currLabel)[0][0])
            if quantitylabels is not None and quantitylabels[int(currLabel)] > 500 and not random.randint(0,1):
                continue
            self.filesPath.append(f)

            if quantitylabels is not None and sum(quantitylabels) >= 1000:
                break
            
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
        valenceLabel = None
        if self.typeExperiment == 'TERMS':
            label = torch.from_numpy(np.array(self.label[idx])).to(torch.float32)
        elif self.typeExperiment == 'EXP':            
            if self.exchangeLabel is not None:
                valenceLabel = torch.from_numpy(np.array(self.exchangeLabel[self.label[idx]])).to(torch.long)
            label = torch.from_numpy(np.array(self.label[idx])).to(torch.long)
        else:
            label = torch.from_numpy(np.array( [self.label[idx][0].astype(np.float32),self.label[idx][1].astype(np.float32)] )).to(torch.float32)
        if self.transform is not None:
            image = self.transform(image)

        if valenceLabel is not None:
            return image, label, self.filesPath[idx], valenceLabel
        else:
            return image, label, self.filesPath[idx]
        
    def sample(self,classes,exclude):
        returnValue = [0] * (max(classes) + 1)
        for c in classes:
            availble = [ idxc for idxc, cl in enumerate(self.label) if cl == c]
            sortedIdx = random.randint(0,len(availble)-1)
            while self.filesPath[availble[sortedIdx]] in exclude:
                sortedIdx = random.randint(0,len(availble)-1)

            i,_,_ = self.__getitem__(availble[sortedIdx])
            returnValue[c] = i

        return torch.stack(returnValue)