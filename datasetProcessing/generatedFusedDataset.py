import argparse, cv2, sys, os, math, shutil, random, torch
from torchvision import transforms
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from helper.function import getFilesInPath, getDirectoriesInPath
from DatasetClasses.AffectNet import AffectNet

def readFile(pathFile):
    returnData = []
    with open(pathFile,'r') as pf:
        returnData = [f.strip() for f in pf]

    return list(map(int,returnData[1:]))

def main():
    parser = argparse.ArgumentParser(description='Generate New Dataset')
    parser.add_argument('--newDatasetPath', help='Path for videos', required=True)
    parser.add_argument('--rangeOfEmotions', help='Quantity of emotions', type=int, required=True)
    args = parser.parse_args()    

    pathForAnnAffWild = "/home/joaocardia/Affwild/annotations/EXPR_Set"
    frameSize = 30

    if os.path.exists(args.newDatasetPath):
        shutil.rmtree(args.newDatasetPath)

    os.makedirs(args.newDatasetPath)
    [os.makedirs(os.path.join(args.newDatasetPath,"%d" % (i) )) for i in range (args.rangeOfEmotions)]

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = AffectNet(afectdata=os.path.join("/home/joaocardia/AffectNet",'train_set'),transform=data_transforms,typeExperiment='EXP',exchangeLabel=None)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=False)

    print("Copying Affectnet")

    for _, currTargetBatch, fileAffectNet in train_loader:
        npCBatch = currTargetBatch.numpy()
        for idxFile, fAff in enumerate(fileAffectNet):
            if npCBatch[idxFile] < args.rangeOfEmotions:
                print("Copying file %s" % (fAff))
                fileNameAff = fAff.split(os.path.sep)[-1]
                shutil.copy(
                    fAff,
                    os.path.join(args.newDatasetPath,str(npCBatch[idxFile]),fileNameAff)
                )


    print("Copying AffWild")

    pathsForImages = getDirectoriesInPath(pathForAnnAffWild)
    for p in pathsForImages:        
        filesAnn = getFilesInPath(os.path.join(pathForAnnAffWild,p))
        for f in filesAnn:
            emotionsInFile = readFile(f)
            fileName = f.split(os.path.sep)[-1].split('.')[0]
            for fNumber in range(0,len(emotionsInFile),frameSize):
                secondImages = emotionsInFile[fNumber:] if fNumber + 30 >= len(emotionsInFile) else emotionsInFile[fNumber:fNumber+frameSize]
                if (sum(secondImages)) < 0:
                    continue
                chooseInterval = random.randint(0,len(secondImages)-1)
                while(secondImages[chooseInterval] < 0):
                    chooseInterval = random.randint(0,len(secondImages)-1)
                
                if os.path.exists(os.path.join("/home/joaocardia/Affwild/cropped_aligned",fileName,"%05d.jpg" % (fNumber + chooseInterval))):
                    print("Copying file %s" % (os.path.join("/home/joaocardia/Affwild/cropped_aligned",fileName,"%05d.jpg" % (fNumber + chooseInterval))))
                    shutil.copyfile(
                        os.path.join("/home/joaocardia/Affwild/cropped_aligned",fileName,"%05d.jpg" % (fNumber + chooseInterval)),
                        os.path.join(args.newDatasetPath,str(secondImages[chooseInterval]),"%s%05d.jpg" % (fileName,fNumber + chooseInterval))
                    )

if __name__ == '__main__':
    main()