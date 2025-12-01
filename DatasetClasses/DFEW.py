import torch.utils.data as data, os, torch, numpy as np, sys, pandas as pd, random
from PIL import Image as im
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getDirectoriesInPath, getFilesInPath, printProgressBar
import threading
#from generateDenseOpticalFlow import runDenseOpFlow

class DFEW(data.Dataset):    
    def __init__(self, pathData, fold, train=False, transform=None):
        super(DFEW, self).__init__()
        self.transform = transform
        self.images = []
        self.label = []
        colsEmotion = ['1happy','2sad','3neutral','4angry','5surprise','6disgust','7fear']
        if fold is not None:
            csvFold = pd.read_csv(os.path.join(pathData, 'EmoLabel_DataSplit', 'test(single-labeled)' if not train else 'train(single-labeled)', f'set_{fold}.csv'))
            excelAnnotation = pd.read_excel(os.path.join(pathData, 'Annotation', 'annotation.xlsx'))
            for idx, c in enumerate(csvFold.iterrows()):
                printProgressBar(idx,len(csvFold),length=50,prefix='Loading Faces...')
                excelRow = excelAnnotation[excelAnnotation['order'] == c[0]]
                if not excelRow.empty:
                    faceImages = getFilesInPath(os.path.join(pathData, 'Clip', 'clip_224x224_16f',f'{c[1]["video_name"]:05d}'))
                    for f in faceImages:
                        self.images.append(f)
                        votesEmotion = []
                        for cEmo in colsEmotion:
                            votesEmotion.append(int(excelRow[cEmo].values[0]))
                        self.label.append([[i / sum(votesEmotion) for i in votesEmotion], c[1]['label']])
                                
    def __getitem__(self, idx):
        path = self.images[idx]
        image = im.open(path)
        if self.transform is not None:
            image = self.transform(image)
    
        return image, torch.Tensor(self.label[idx][0]), torch.from_numpy(np.array(self.label[idx][1])).to(torch.long), path
    
    def __len__(self):
        return len(self.images)
