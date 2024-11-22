import argparse, os, sys, numpy as np, torch
from torchvision import transforms
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from DatasetClasses.AffectNet import AffectNet
from DatasetClasses.AffWild2 import AFF2Data

def getRanks(vaLabel,vaDists):
    dists = torch.cdist(vaLabel,vaDists,p=2)
    return dists

def main():
    parser = argparse.ArgumentParser(description='Generate Emotion Ranks')
    parser.add_argument('--pathBase', help='Path for valence and arousal dataset', required=True)
    parser.add_argument('--batchSize', type=int, help='Size of the batch', required=True)
    parser.add_argument('--dataset', help='Size of the batch', required=True)
    args = parser.parse_args()
    data_transforms = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    if args.dataset == 'affwild':
        datasetVal = AFF2Data(args.pathBase,'Validation_Set',transform=data_transforms,type='VA')
        classesDist = torch.from_numpy(np.array([
            [0,0],
            [-0.51,0.59],
            [-0.6,0.35],
            [-0.64,0.6],
            [0.81,0.51],
            [-0.63,-0.27],
            [0.4,0.67]
        ])).type(torch.FloatTensor)
    else:
        datasetVal = AffectNet(afectdata=os.path.join(args.pathBase,'val_set'),transform=data_transforms,typeExperiment='BOTH',exchangeLabel=None)
        classesDist = torch.from_numpy(np.array([
            [0,0],
            [0.81,0.51],
            [-0.63,-0.27],
            [0.4,0.67],
            [-0.64,0.6],
            [-0.6,0.35],
            [-0.51,0.59],
            [-0.23,0.31]
        ])).type(torch.FloatTensor)
    val_loader = torch.utils.data.DataLoader(datasetVal, batch_size=args.batchSize, shuffle=False)
    folderDataset = os.path.join(args.pathBase,'val_set')
    markerFiles = {}
    for data in val_loader:
        if args.dataset == 'affwild':
            _, vaBatch, paths = data
        else:
            _, _, paths, vaBatch = data
        currRanking = getRanks(vaBatch,classesDist)        
        for idxP, p in enumerate(paths):
            if args.dataset == 'affwild':
                fileName = p.split(os.path.sep)[-2]
                frame = int(p.split(os.path.sep)[-1][:-4])                
                pathRank = os.path.join(args.pathBase,'annotations','Rank_Set','Validation_Set',fileName+'.txt')
                if pathRank not in markerFiles.keys():
                    markerFiles[pathRank] = []

                while (len(markerFiles[pathRank]) < frame):
                    markerFiles[pathRank].append('')
                
                orderLabels = currRanking[idxP].argsort()
                markerFiles[pathRank][frame-1] =  ','.join(list(map(str,orderLabels.numpy())))

            else:
                fileName = p.split(os.path.sep)[-1][:-4]
                pathRank = os.path.join(folderDataset,'annotations',fileName+'_rank.txt')
                with open(pathRank,'w') as rankAnn:
                    orderLabels = currRanking[idxP].argsort()
                    rankAnn.write( ','.join(list(map(str,orderLabels.numpy()))) )
            
    if args.dataset == 'affwild':
        for i in markerFiles:
            with open(i,'w') as mf:
                mf.write('Emotional Rank\n')
                for j in markerFiles[i]:
                    mf.write(j+'\n')
    print('oi')

if __name__ == '__main__':
    main()