import argparse, torch, os, sys, numpy as np
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getFilesInPath

def getExpressionForImage(imageNumber,imagePaths):
    for idx, path in enumerate(imagePaths):
        fileName = path.split(os.path.sep)[-1]
        if f"{imageNumber[:-4]}_exp.npy" == fileName:
            return int(np.load(path).item())
    return None

def saveToCSV(preds,pathCSV):
    emotions = ["label","fileImage","Expression"]
    with open(pathCSV,'w') as pcsv:
        pcsv.write('%s\n' % (';'.join([emotions[f] for f in range(len(preds[0]))])))
        for idx, p in enumerate(preds):
            pcsv.write(f'{";".join(p).strip()}\n')

def train():
    parser = argparse.ArgumentParser(description='Extract features from resnet emotion')
    parser.add_argument('--labelFile', help='Weights', required=True)
    parser.add_argument('--output', help='Weights', required=True)
    parser.add_argument('--annotationFolder', help='Weights', required=True)
    args = parser.parse_args()

    annsFiles = getFilesInPath(args.annotationFolder)

    expressions = ["neutral","happy","sad","surprised","fear","disgust","angry","contempt"]

    outputFile = []
    with open(args.labelFile,'r') as fil:
        for f in fil:
            dadosImage = f.strip().split(' - ')
            expressionImage = expressions[getExpressionForImage(dadosImage[1],annsFiles)]
            dadosImage.append(expressionImage)
            outputFile.append(dadosImage)

    saveToCSV(outputFile,args.output)

if __name__ == '__main__':
    train()