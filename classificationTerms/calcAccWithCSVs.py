import argparse, pandas as pd, numpy as np, os, sys
from black import out
from scipy.spatial.distance import euclidean
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getFilesInPath

def openFileForTerms(pathFile):
    outputFeatures = []
    with open(pathFile,'r') as pf:
        for f in pf:
            outputFeatures.append(f.rstrip('\n'))

    return outputFeatures

def main():
    parser = argparse.ArgumentParser(description='Generate CSV')
    parser.add_argument('--datasetClass', help='Path for the terms file', required=True)
    parser.add_argument('--estimatedClass', help='Path for the terms file', required=True)
    args = parser.parse_args()
    
    dataset = openFileForTerms(args.datasetClass)
    estimated = openFileForTerms(args.estimatedClass)

    result = [0,0]

    for i in range(len(dataset)):
        print("%s %s" % (dataset[i],estimated[i]))
        result[int(dataset[i] in estimated[i])] += 1

    print(result)
    print("General Accuracy = %f" % (result[1] / len(dataset)))
    

if __name__ == '__main__':
    main()