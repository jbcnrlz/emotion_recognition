import argparse, os, sys, shutil, numpy as np, torch
from torchvision import transforms
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getFilesInPath
from DatasetClasses.WFD import WFD

def main():
    parser = argparse.ArgumentParser(description='Generate NewFusedDataset')
    parser.add_argument('--pathBaseWFD', help='Path for datasets', required=True)
    parser.add_argument('--newDatasetPath', help='Path for datasets', required=True)
    args = parser.parse_args()
        
    data_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = WFD(args.pathBaseWFD,data_transforms)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=False)

    emotionsLabel = {'Anger' : 1,'Fear' : 3,'Disgust' : 2,'Sadness' : 5,'Neutral' : 0,'Happiness' : 4,'Surprise' : 6}

    print("Copying WFD")
    for _, currTargetBatch, fileAffectNet in train_loader:        
        for idxFile, fAff in enumerate(fileAffectNet):
            if currTargetBatch[idxFile] in emotionsLabel.keys():
                print("Copying file %s" % (fAff))
                filePatth = fAff.split(os.path.sep)[-1]
                shutil.copy(fAff,os.path.join(args.newDatasetPath,str(emotionsLabel[currTargetBatch[idxFile]]),filePatth))

    

if __name__ == '__main__':
    main()