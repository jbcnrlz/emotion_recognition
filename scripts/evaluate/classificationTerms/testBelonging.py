import argparse, os, sys, numpy as np, pandas as pd, torch, math
from torchvision import transforms
from scipy.stats import norm
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import separatedSecondLevel, getSecondLevelFromVA
from DatasetClasses.AffectNet import AffectNet

def zScore(value,mean,std):
    return (value - mean) / std

def main():
    parser = argparse.ArgumentParser(description='Compare chisquared')
    parser.add_argument('--csvEmotions', help='Path for resnet pretrained weights', required=False,default=None)
    parser.add_argument('--datasetFiles', help='Weights', required=True)
    args = parser.parse_args()    
    va = np.array(pd.read_csv(args.csvEmotions))
    vaVals = va[:,1:-2]
    classesLabels = va[:,0]
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])
    datasetVal = AffectNet(afectdata=os.path.join(args.datasetFiles,'val_set'),transform=data_transforms)
    val_loader = torch.utils.data.DataLoader(datasetVal, batch_size=1, shuffle=False)
    possibleClasses = []
    for img,label,pathfile in val_loader:
        possibleClasses.append([pathfile,{}])
        label = label.numpy()
        biggerNegative = {}
        for i in range(vaVals.shape[0]):
            ppfResult = zScore(label[0,0],vaVals[i,0],vaVals[i,1])
            if ppfResult >= 0:
                possibleClasses[-1][-1][classesLabels[i]] = ppfResult
            else:
                if len(biggerNegative) > 0 and biggerNegative.items()[0] < ppfResult:
                    biggerNegative[classesLabels[i]] = ppfResult
                elif biggerNegative.items()[0] < ppfResult:
                    biggerNegative[classesLabels[i]] = ppfResult

        if len(possibleClasses[-1][-1]) > 0:
            possibleClasses[-1][-1] = sorted(possibleClasses[-1][-1].items(), key=lambda x: x[1])
        else:
            possibleClasses[-1][-1] = biggerNegative

    print('oi')
        

if __name__ == '__main__':
    main()