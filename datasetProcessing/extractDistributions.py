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
    args = parser.parse_args()
    data_transforms = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    datasetVal = AffectNet(afectdata=os.path.join(args.pathBase,'val_set'),transform=data_transforms,typeExperiment='BOTH',exchangeLabel=None)
    classesDist = torch.from_numpy(np.array([
        [0,0,0,0],
        [0.81,0.21,0.51,0.26], #happy
        [-0.63,0.23,-0.27,0.34], #sad
        [0.4,0.3,0.67,0.27], #surprised
        [-0.64,0.2,0.6,0.32],#fear
        [-0.6,0.2,0.35,0.41],#disgust
        [-0.51,0.2,0.59,0.33],#angry
        [-0.23,0.39,0.31,0.33]#contempt
    ])).type(torch.FloatTensor)
    val_loader = torch.utils.data.DataLoader(datasetVal, batch_size=args.batchSize, shuffle=False)
    folderDataset = os.path.join(args.pathBase,'val_set')
    outputData = []
    for data in val_loader:
        _, _, paths, vaBatch = data
        outputData.append([])
        for v in vaBatch:
            outputData[-1].append(v.numpy()[0])
            outputData[-1].append(v.numpy()[1])
            for c in classesDist:
                outputData[-1].append(c.numpy()[0])
                outputData[-1].append(c.numpy()[1])
                outputData[-1].append(c.numpy()[2])
                outputData[-1].append(c.numpy()[3])
    
    with open('list.csv','w') as f:
        f.write("valence,arousal,neutral_v,neutral_a,happy_v,happy_a,sad_v,sad_a,surprised_v,surprised_a,fear_v,fear_a,disgust_v,disgust_a,angry_v,angry_a,contempt_v,contempt_a\n")
        for data in outputData:
            f.write(','.join(list(map(str,data)))+'\n')
    '''
    currRanking = getRanks(vaBatch,classesDist)
    for idxP, p in enumerate(paths):
        fileName = p.split(os.path.sep)[-1][:-4]
        pathRank = os.path.join(folderDataset,'annotations',fileName+'_rank.txt')
        with open(pathRank,'w') as rankAnn:
            orderLabels = currRanking[idxP].argsort()
            rankAnn.write( ','.join(list(map(str,orderLabels.numpy()))) )
    '''
if __name__ == '__main__':
    main()