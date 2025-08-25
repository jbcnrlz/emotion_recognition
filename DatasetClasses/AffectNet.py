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
        self.seconLabel = []
        self.typeExperiment = typeExperiment
        quantitylabels = None
        faces = getFilesInPath(os.path.join(afectdata,'images'),imagesOnly=True)
        print(f"Loading {len(faces)} face images")
        for idx, f in enumerate(faces):
            printProgressBar(idx,len(faces),length=50,prefix='Loading Faces...')
            imageNumber = f.split(os.path.sep)[-1][:-4]
            if typeExperiment == "VA":
                try:
                    valValue = np.load(os.path.join(afectdata,'annotations','%d_val.npy' % (int(imageNumber))))
                    aroValue = np.load(os.path.join(afectdata,'annotations','%d_aro.npy' % (int(imageNumber))))
                except:
                    valValue = np.load(os.path.join(afectdata,'annotations',f'{imageNumber}_val.npy'))
                    aroValue = np.load(os.path.join(afectdata,'annotations',f'{imageNumber}_aro.npy'))
                self.label.append([valValue,aroValue])
            elif typeExperiment == "VAD":
                try:
                    valValue = np.load(os.path.join(afectdata,'annotations','%d_val.npy' % (int(imageNumber))))
                    aroValue = np.load(os.path.join(afectdata,'annotations','%d_aro.npy' % (int(imageNumber))))
                    domValue = np.load(os.path.join(afectdata,'annotations','%d_dom.npy' % (int(imageNumber))))
                except:
                    valValue = np.load(os.path.join(afectdata,'annotations',f'{imageNumber}_val.npy'))
                    aroValue = np.load(os.path.join(afectdata,'annotations',f'{imageNumber}_aro.npy'))
                    try:
                        domValue = np.load(os.path.join(afectdata,'annotations',f'{imageNumber}_dom.npy'))
                    except:
                        continue
                self.label.append([valValue,aroValue,domValue])
            elif typeExperiment == "EXP":
                currLabel = np.load(os.path.join(afectdata,'annotations' ,'%d_exp.npy' % (int(imageNumber))))
                if int(currLabel) == 7 and not loadLastLabel:
                    continue
                self.label.append(int(currLabel))
            elif typeExperiment == "BOTH":
                currLabel = np.load(os.path.join(afectdata,'annotations' ,'%d_exp.npy' % (int(imageNumber))))
                if int(currLabel) == 7 and not loadLastLabel:
                    continue
                try:
                    valValue = np.load(os.path.join(afectdata,'annotations','%d_val.npy' % (int(imageNumber))))
                    aroValue = np.load(os.path.join(afectdata,'annotations','%d_aro.npy' % (int(imageNumber))))
                except:
                    valValue = np.load(os.path.join(afectdata,'annotations',f'{imageNumber}_val.npy'))
                    aroValue = np.load(os.path.join(afectdata,'annotations',f'{imageNumber}_aro.npy'))
                self.label.append(currLabel)
                self.seconLabel.append([valValue,aroValue])
            elif typeExperiment == 'RANK':
                currLabel = os.path.join(afectdata,'annotations' ,'%d_rank.txt' % (int(imageNumber)))
                self.label.append(currLabel)
            elif typeExperiment == 'RANDOM':
                self.label.append(np.random.randint(0,8))
            elif typeExperiment == 'PROBS':
                currLabel = os.path.join(afectdata,'annotations' ,'%d_prob_rank.txt' % (int(imageNumber)))
                self.label.append(currLabel)
            elif typeExperiment == 'PROBS_VA':
                try:
                    currLabel = os.path.join(afectdata,'annotations' ,'%d_prob_rank.txt' % (int(imageNumber)))
                    valValue = np.load(os.path.join(afectdata,'annotations','%d_val.npy' % (int(imageNumber)))).astype(np.float64)
                    aroValue = np.load(os.path.join(afectdata,'annotations','%d_aro.npy' % (int(imageNumber)))).astype(np.float64)
                except:
                    currLabel = os.path.join(afectdata,'annotations' ,f'{imageNumber}_prob_rank.txt')
                    valValue = np.load(os.path.join(afectdata,'annotations',f'{imageNumber}_val.npy' )).astype(np.float64)
                    aroValue = np.load(os.path.join(afectdata,'annotations',f'{imageNumber}_aro.npy')).astype(np.float64)
                self.label.append([currLabel,valValue,aroValue])
            elif typeExperiment == 'PROBS_VAD':
                try:
                    currLabel = os.path.join(afectdata,'annotations' ,'%d_prob_rank.txt' % (int(imageNumber)))
                    valValue = np.load(os.path.join(afectdata,'annotations','%d_val.npy' % (int(imageNumber)))).astype(np.float64)
                    aroValue = np.load(os.path.join(afectdata,'annotations','%d_aro.npy' % (int(imageNumber)))).astype(np.float64)
                    aroValue = np.load(os.path.join(afectdata,'annotations','%d_dom.npy' % (int(imageNumber)))).astype(np.float64)
                except:
                    currLabel = os.path.join(afectdata,'annotations' ,f'{imageNumber}_prob_rank.txt')
                    valValue = np.load(os.path.join(afectdata,'annotations',f'{imageNumber}_val.npy' )).astype(np.float64)
                    aroValue = np.load(os.path.join(afectdata,'annotations',f'{imageNumber}_aro.npy')).astype(np.float64)
                    try:
                        aroValue = np.load(os.path.join(afectdata,'annotations',f'{imageNumber}_dom.npy')).astype(np.float64)
                    except:
                        continue
                self.label.append([currLabel,valValue,aroValue])
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

    def loadRankFile(self,rankPath):
        dataOutput = None
        with open(str(rankPath),'r') as rf:
            dataOutput = list(map(int,rf.readline().strip().split(',')))
        return dataOutput

    def loadProbFile(self,rankPath):
        dataOutput = None
        with open(str(rankPath),'r') as rf:
            dataOutput = list(map(float,rf.readline().strip().split(',')))
        return dataOutput

    def __getitem__(self, idx):
        path = self.filesPath[idx]
        image = im.open(path)
        valenceLabel = None
        if self.typeExperiment == 'TERMS':
            label = torch.from_numpy(np.array(self.label[idx])).to(torch.float32)
        elif self.typeExperiment == 'EXP' or self.typeExperiment == 'BOTH':
            if self.exchangeLabel is not None:
                valenceLabel = torch.from_numpy(np.array(self.exchangeLabel[self.label[idx]])).to(torch.long)
            if self.typeExperiment == 'BOTH':
                valenceLabel = torch.from_numpy(np.array( [self.seconLabel[idx][0].astype(np.float32),self.seconLabel[idx][1].astype(np.float32)] )).to(torch.float32)
            label = torch.from_numpy(np.array(self.label[idx]).astype(np.uint8)).to(torch.long)
        elif self.typeExperiment == "RANK":
            label = torch.from_numpy(np.array(self.loadRankFile(self.label[idx])).astype(np.uint8)).to(torch.long)
        elif self.typeExperiment == "RANDOM":
            label = torch.from_numpy(np.array(np.random.randint(0,8))).to(torch.long)
        elif self.typeExperiment == "PROBS":
            label = torch.from_numpy(np.array(self.loadProbFile(self.label[idx])).astype(np.float32)).to(torch.float32)
        elif self.typeExperiment == 'PROBS_VA':
            label = [
                torch.from_numpy(np.array(self.loadProbFile(self.label[idx][0])).astype(np.float32)).to(torch.float32),
                torch.from_numpy(np.array([self.label[idx][1],self.label[idx][2]]))
            ]
        elif self.typeExperiment == 'PROBS_VAD':
            label = [
                torch.from_numpy(np.array(self.loadProbFile(self.label[idx][0])).astype(np.float32)).to(torch.float32),
                torch.from_numpy(np.array([self.label[idx][1],self.label[idx][2],self.label[idx][3]]))
            ]
        elif self.typeExperiment == "VAD":
            label = torch.from_numpy(
                np.array( 
                    [self.label[idx][0].astype(np.float32),self.label[idx][1].astype(np.float32),self.label[idx][2].astype(np.float32)] )).to(torch.float32)
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