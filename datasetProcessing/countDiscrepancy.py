import torch, numpy as np, os, sys, argparse, pandas as pd, shutil, math
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getDirectoriesInPath, getFilesInPath, printProgressBar

def getRanks(vaLabel,vaDists):
    dists = torch.cdist(vaLabel,vaDists,p=2)
    return dists

def loadEmotions(pathFile):
    returnData = []
    with open(pathFile) as f:
        f.readline()
        for l in f:
            returnData.append(list(map(float,l.split(','))))

    return returnData 

def main():
    parser = argparse.ArgumentParser(description='Fix anotation')
    parser.add_argument('--affectNetFolder', help='Path for the terms file', required=True)
    parser.add_argument('--affWildFolder', help='Path for resnet pretrained weights', required=True)
    args = parser.parse_args()

    classesDist = np.array([
        [0,0],
        [0.81,0.51],
        [-0.63,-0.27],
        [0.4,0.67],
        [-0.64,0.6],
        [-0.6,0.35],
        [-0.51,0.59],
        [-0.23,0.31]
    ])
    
    filesTrainAffectnet = getFilesInPath(args.affectNetFolder)
    subjectsaffect = [0,0]
    areadyWent = []
    for idx, f in enumerate(filesTrainAffectnet):
        printProgressBar(idx,len(filesTrainAffectnet),length=50,prefix='Calculating discrepancy affectnet')
        filePath = f.split(os.path.sep)[-1].split('_')[0]
        if filePath not in areadyWent:            
            valValue = np.load(os.path.join(args.affectNetFolder,'%d_val.npy' % (int(filePath)))).astype(np.float32)
            aroValue = np.load(os.path.join(args.affectNetFolder,'%d_aro.npy' % (int(filePath)))).astype(np.float32)
            valaro =  np.array([valValue, aroValue])
            distances = np.linalg.norm(classesDist - valaro, axis=1)
            annotatedEmotion = np.load(os.path.join(args.affectNetFolder,'%d_exp.npy' % (int(filePath))))
            subjectsaffect[(-distances).argmax() == annotatedEmotion.astype(np.uint16)] += 1
            areadyWent.append(filePath)

    print(subjectsaffect)
    
    classesDist = np.array([
        [0,0],#neutral
        [-0.51,0.59],#anger
        [-0.6,0.35],#disgust
        [-0.64,0.6],#fear
        [0.81,0.51],#happy
        [-0.63,-0.27],#sad
        [0.4,0.67],#surprise
        [-0.23,0.31] #contempt
    ])
    exprSet = getFilesInPath(os.path.join(args.affWildFolder,'EXPR_Set','Train_Set'),fullPath=False)
    exprSet += getFilesInPath(os.path.join(args.affWildFolder,'EXPR_Set','Validation_Set'),fullPath=False)
    vatSet = getFilesInPath(os.path.join(args.affWildFolder,'VA_Set','Train_Set'))
    vatSet += getFilesInPath(os.path.join(args.affWildFolder,'VA_Set','Validation_Set'))
    subjects = [0,0]
    for idx, f in enumerate(vatSet):
        printProgressBar(idx,len(vatSet),length=50,prefix='Calculating discrepancy affwild')
        filePath = f.split(os.path.sep)[-1]
        if filePath in exprSet:
            if os.path.exists(os.path.join(args.affWildFolder,'EXPR_Set','Train_Set',filePath)):
                emoFile = np.array(loadEmotions(os.path.join(args.affWildFolder,'EXPR_Set','Train_Set',filePath)))
            else:
                emoFile = np.array(loadEmotions(os.path.join(args.affWildFolder,'EXPR_Set','Validation_Set',filePath)))

            valaro = np.array(loadEmotions(f))
            for idxVa, va in enumerate(valaro):
                distances = np.linalg.norm(classesDist - va, axis=1)
                annotatedEmotion = int(emoFile[idxVa])
                annotatedEmotion = 7 if annotatedEmotion == -1 else annotatedEmotion
                subjects[(-distances).argmax() == annotatedEmotion] += 1

    print(subjects)
if __name__ == '__main__':
    main()