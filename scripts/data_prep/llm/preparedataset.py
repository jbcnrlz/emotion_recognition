import numpy as np, os, sys, argparse, pandas as pd 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getFilesInPath
import warnings
warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description='Generate Emotion Ranks')
    parser.add_argument('--pathCSV', help='Path for valence and arousal dataset', required=True)
    parser.add_argument('--folderWithImages', help='Path for valence and arousal dataset', required=True)
    parser.add_argument('--outputCSV', help='Path for valence and arousal dataset', required=True)
    args = parser.parse_args()

    csvFile = pd.read_csv(args.pathCSV)
    images = getFilesInPath(args.folderWithImages)
    newDataFrame = pd.DataFrame(columns=csvFile.columns)
    for i in images:
        fileName = r"\\" + i.split(os.path.sep)[-1]
        fData = csvFile[csvFile['file'].str.contains(fileName)]
        newDataFrame = pd.concat([newDataFrame, fData])

    newDataFrame.to_csv(args.outputCSV, index=False)

if __name__ == "__main__":
    main()