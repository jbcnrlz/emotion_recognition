import argparse, os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getFilesInPath
from networks.extractFeaturesBayesian import saveToCSV

def main():
    parser = argparse.ArgumentParser(description='Extract features from resnet emotion')
    parser.add_argument('--emotionFile', help='Weights', required=True)
    parser.add_argument('--affectnetRoot', help='Weights', required=True)
    args = parser.parse_args()

    files = []
    with open(args.emotionFile, 'r') as pcsv:
        for p in pcsv:
            dataFileName = p.strip().split(' - ')[-1]
            dataFileName = dataFileName.split(os.path.sep)[-1]
            files.append(dataFileName[:-4])  # Remove .jpg or .png extension

    pathFiles = os.path.join(args.affectnetRoot, 'annotations')
    filesInPath = getFilesInPath(pathFiles)
    fileToUtilize = []
    for f in filesInPath:        
        if '_prob_rank' not in f:
            continue
        currFileName = f.split(os.path.sep)[-1]
        for filCurr in files:
            if f"{filCurr}_prob_rank.txt" == currFileName:
                fileToUtilize.append(f)                

    outputCSVData = []
    for file in fileToUtilize:
        dataFromFile = None
        with open(file, 'r') as f:            
            for line in f:
                line = line.strip()
                dataFromFile = list(map(float, line.split(',')))
        outputCSVData.append(dataFromFile)
    saveToCSV(outputCSVData, fileToUtilize, 'emotion_predictions_gt.csv')


if __name__ == '__main__':
    main()