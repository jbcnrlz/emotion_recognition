import argparse, os, sys, numpy as np, torch, pandas as pd
from torchvision import transforms
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getDirectoriesInPath, getFilesInPath
from DatasetClasses.AffectNet import AffectNet
from DatasetClasses.AffWild2 import AFF2Data
from sklearn.metrics import ndcg_score

def loadFileAffWild(pathData,typeData):
    returnData = {}
    filesRead = os.path.join(pathData,'annotations','RANK_Set',typeData)
    for f in getFilesInPath(filesRead):
        fileName = f.split(os.path.sep)[-1]
        returnData[fileName] = []
        with open(f,'r') as fLabel:
            fLabel.readline()
            for fl in fLabel:
                try:
                    returnData[fileName].append(list(map(int,fl.strip().split(','))))
                except:
                    returnData[fileName].append([])

    return returnData

def getRanks(vaLabel,vaDists):
    dists = torch.cdist(vaLabel,vaDists,p=2)
    return dists

def calculateNDCG(cOutput,datasetName,pathBase,bathSize):
    dataOutput = pd.read_csv(cOutput)
    data_transforms = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    if datasetName == 'affwild2':
        gallery = []
        probe = []
        csvParsed = np.array(dataOutput)
        dataLogs = csvParsed[:,:-1]
        filePath = csvParsed[:,-2]
        dataFromAffectNet = loadFileAffWild(pathBase,'Validation_Set')
        for idxLogs, dl in enumerate(dataLogs[:,:-1]):
            videoPath = filePath[idxLogs].split(os.path.sep)[-2] + '.txt'
            frameNumber = int(filePath[idxLogs].split(os.path.sep)[-1][:-4])
            if len(dataFromAffectNet[videoPath][frameNumber-1]) <= 0:
                continue
            gallery.append(dataFromAffectNet[videoPath][frameNumber-1])
            probe.append(list(dl.astype(np.float32)))
        gallery = np.array(gallery)
        probe = np.array(probe)
        print(ndcg_score(gallery,probe[:,:-1]))
        return None
    else:
        datasetVal = AffectNet(afectdata=os.path.join(pathBase,'val_set'),transform=data_transforms,typeExperiment='RANK',exchangeLabel=None)
    val_loader = torch.utils.data.DataLoader(datasetVal, batch_size=bathSize, shuffle=False)
    compare = [[],[]]
    for data in val_loader:
        _, labels, paths = data
        for idxPath, p in enumerate(paths):
            fileName = p.split(os.path.sep)[-1]
            for idx, dataCSV in enumerate(np.array(dataOutput)):
                if (len(dataCSV) == 10):
                    currPath = dataCSV[-2].split(os.path.sep)[-1]
                else:
                    currPath = dataCSV[-1].split(os.path.sep)[-1]
                if (currPath == fileName):
                    compare[0].append(labels[idxPath].numpy())
                    if len(dataCSV) < 10:
                        compare[1].append(dataCSV[:-1].astype(np.float32))
                    else:
                        compare[1].append(dataCSV[:-2].astype(np.float32))
                    break
    print(ndcg_score(np.array(compare[0]),np.array(compare[1])))

def calculateACC(csvFile,dataset,pathBase,batchSize,classesQuantity):
    dataOutput = pd.read_csv(csvFile)
    if dataset == 'affectnet':
        data_transforms = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
        datasetVal = AffectNet(afectdata=os.path.join(pathBase,'val_set'),transform=data_transforms,typeExperiment='EXP',loadLastLabel=False if classesQuantity == 8 else True)
        val_loader = torch.utils.data.DataLoader(datasetVal, batch_size=batchSize, shuffle=False)
        acc = [0,0]
        for data in val_loader:
            _, labels, paths = data
            for idxPath, p in enumerate(paths):
                fileName = p.split(os.path.sep)[-1]
                for idx, dataCSV in enumerate(np.array(dataOutput)):
                    if (len(dataCSV) == 10):
                        currPath = dataCSV[-2].split(os.path.sep)[-1]
                        rankSize = dataCSV[:-2].astype(np.float64)
                    else:
                        currPath = dataCSV[-2].split(os.path.sep)[-1]
                        rankSize = dataCSV[:-2].astype(np.float64)
                    if (currPath == fileName):
                        acc[int(labels[idxPath].numpy() == rankSize.argmax())] += 1
        print(acc)
    elif dataset == 'affwild2':
        #Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise
        emotionsOrder = [0,4,5,6,3,2,1,-1]
        acc = [0,0]
        for dataCSV in np.array(dataOutput):
            guessEmo = emotionsOrder[dataCSV[:-2].argmax()]
            acc[int(float(dataCSV[-1][1:-1])) == guessEmo] += 1

        print(acc)
        print(acc[1]/sum(acc))        

def rankCalculate(dataOutput,dataset,pathBase):
    dataOutput = pd.read_csv(dataOutput)
    if dataset == 'affwild2':
        emotionsOrder = np.array([0,4,5,6,3,2,1,-1])
        gallery = []
        probe = []
        csvParsed = np.array(dataOutput)
        dataLogs = csvParsed[:,:-1]
        if (csvParsed.shape[1] == 8):
            filePath = csvParsed[:,-1]
        else:
            filePath = csvParsed[:,-2]
        dataFromAffectNet = loadFileAffWild(pathBase,'Validation_Set')
        for idxLogs, dl in enumerate(dataLogs[:,:-1]):
            videoPath = filePath[idxLogs].split(os.path.sep)[-2] + '.txt'
            frameNumber = int(filePath[idxLogs].split(os.path.sep)[-1][:-4])
            if len(dataFromAffectNet[videoPath][frameNumber-1]) <= 0:
                continue
            gallery.append(dataFromAffectNet[videoPath][frameNumber-1])
            probe.append(list(dl.astype(np.float32)))
        gallery = np.array(gallery)
        probe = np.array(probe)
        print(np.sum(gallery[:,0] == emotionsOrder[(-probe).argsort()[:,0]]) / gallery.shape[0])
    else:
        data_transforms = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
        datasetVal = AffectNet(afectdata=os.path.join(pathBase,'val_set'),transform=data_transforms,typeExperiment='RANK',exchangeLabel=None)
        val_loader = torch.utils.data.DataLoader(datasetVal, batch_size=50, shuffle=False)
        compare = [[],[]]
        for data in val_loader:
            _, labels, paths = data
            for idxPath, p in enumerate(paths):
                fileName = p.split(os.path.sep)[-1]
                for idx, dataCSV in enumerate(np.array(dataOutput)):
                    if (len(dataCSV) == 10):
                        currPath = dataCSV[-2].split(os.path.sep)[-1]
                    else:
                        currPath = dataCSV[-1].split(os.path.sep)[-1]
                    if (currPath == fileName):
                        compare[0].append(labels[idxPath].numpy())
                        if len(dataCSV) < 10:
                            compare[1].append((-dataCSV[:-1].astype(np.float32)).argsort())
                        else:
                            compare[1].append((-dataCSV[:-2].astype(np.float32)).argsort())
                        break
        rank1 = [0,0]
        for i in range(len(compare[0])):
            rank1[int(compare[0][i][0] == compare[1][i][0])] += 1

        print(rank1)



def main():
    parser = argparse.ArgumentParser(description='Generate Emotion Ranks')
    parser.add_argument('--pathBase', help='Path for valence and arousal dataset', required=True)
    parser.add_argument('--batchSize', type=int, help='Size of the batch', required=True)
    parser.add_argument('--dataset', help='Size of the batch', required=True)
    parser.add_argument('--classifierOutput', help='Size of the batch', required=True)    
    parser.add_argument('--metricType', help='Size of the batch', required=True)    
    parser.add_argument('--classesQt', help='Size of the batch', required=False,type=int)  
    args = parser.parse_args()
    if args.metricType == 'ndcg':
        calculateNDCG(args.classifierOutput,args.dataset,args.pathBase,args.batchSize)
    elif args.metricType == 'acc':
        calculateACC(args.classifierOutput,args.dataset,args.pathBase,args.batchSize,args.classesQt)
    elif args.metricType == 'rank':
        rankCalculate(args.classifierOutput,args.dataset,args.pathBase)


if __name__ == '__main__':
    main()