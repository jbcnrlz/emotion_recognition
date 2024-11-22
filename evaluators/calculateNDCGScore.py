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
    if datasetName == 'affwild':
        gallery = []
        probe = []
        csvParsed = np.array(dataOutput)
        dataLogs = csvParsed[:,:-1]
        filePath = csvParsed[:,-1]
        dataFromAffectNet = loadFileAffWild(pathBase,'Validation_Set')
        for idxLogs, dl in enumerate(dataLogs):
            videoPath = filePath[idxLogs].split(os.path.sep)[-2] + '.txt'
            frameNumber = int(filePath[idxLogs].split(os.path.sep)[-1][:-4])
            if len(dataFromAffectNet[videoPath][frameNumber-1]) <= 0:
                continue
            gallery.append(dataFromAffectNet[videoPath][frameNumber-1])
            probe.append(list(dl.astype(np.float32)))
        print(ndcg_score(np.array(gallery),np.array(probe)))
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


if __name__ == '__main__':
    main()