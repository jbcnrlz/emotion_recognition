import argparse, os, sys
from scipy.spatial.distance import euclidean
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

def openFileForTerms(pathFile):
    outputFeatures = []
    with open(pathFile,'r') as pf:
        for f in pf:
            outputFeatures.append(f.rstrip('\n').split(','))

    return outputFeatures

def main():
    parser = argparse.ArgumentParser(description='Generate CSV')
    parser.add_argument('--datasetClass', help='Path for the terms file', required=True)
    parser.add_argument('--estimatedClass', help='Path for the terms file', required=True)
    args = parser.parse_args()
    
    dataset = openFileForTerms(args.datasetClass)[1:]
    estimated = openFileForTerms(args.estimatedClass)[1:]

    result = [0,0]

    for i in range(len(dataset)):
        fileToCompapreDataset = dataset[i][1].split(os.path.sep)[-1]
        for j in range(len(estimated)):
            fileToCompapreEstimated = estimated[j][1].split(os.path.sep)[-1]
            if fileToCompapreDataset == fileToCompapreEstimated:
                if (estimated[j][0] == dataset[i][0]):
                    print(estimated[j][0] + " " + dataset[i][0])
                result[int(estimated[j][0] == dataset[i][0])] += 1
                break

    print(result)
    print("General Accuracy = %f" % (result[1] / len(dataset)))
    

if __name__ == '__main__':
    main()