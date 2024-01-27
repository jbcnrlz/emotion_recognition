import torch.utils.data as data, os, torch, numpy as np, sys, pandas as pd, random
from PIL import Image as im
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getDirectoriesInPath, getFilesInPath, printProgressBar

class WFD(data.Dataset):    
    def __init__(self, afectdata, transform,typeLabel='categ'):
        super(WFD,self).__init__()
        self.typeLabel = typeLabel
        self.emotionsIndex = ['Anger','Fear','Disgust','Sadness','Neutral','Happiness','Surprise','Other','Unknown']
        self.transform = transform        
        adCSV = pd.read_csv(os.path.join(afectdata,"All data.csv"))
        self.label = []
        self.emotionsAnnotated = []
        self.filesPath = []
        faces = getFilesInPath(os.path.join(afectdata,'WFD Stimuli'),imagesOnly=True)
        for idx, f in enumerate(faces):
            printProgressBar(idx,len(faces),length=50,prefix='Loading Faces...')
            fileName = f.split(os.path.sep)[-1]
            self.emotionsAnnotated.append(list(adCSV[adCSV['Image'] == fileName]['Emotion Category']))
            self.label.append(list(adCSV[adCSV['Image'] == fileName]['Most Selected Emotion'])[0])
            self.filesPath.append(f)
            
    def loadTermsFile(self,termsQuantity):
        return np.array(pd.read_csv('joinedWithDistance_%d.csv' % (termsQuantity)))[:,0]

    def __len__(self):
        return len(self.filesPath)

    def __getitem__(self, idx):
        path = self.filesPath[idx]
        image = im.open(path)
        if self.transform is not None:
            image = self.transform(image)

        label = self.label[idx] if self.typeLabel == 'categ' else self.emotionsIndex.index(self.label[idx])

        return image, label, self.filesPath[idx]