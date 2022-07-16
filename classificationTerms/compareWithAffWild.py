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
    parser.add_argument('--datasetFolder', help='Path for the terms file', required=True)
    parser.add_argument('--estimatedClass', help='Path for the terms file', required=True)
    args = parser.parse_args()
        
    estimated = openFileForTerms(args.estimatedClass)[1:]

    currFile = ''
    comparisonFile = None
    result = [0,0]
    for idx, e in enumerate(estimated):
        print("%d from %d" % (idx,len(estimated)))
        fileWithFrames = e[-1].split(os.path.sep)[-2]
        if os.path.exists(os.path.join(args.datasetFolder,"%s_terms.txt" % (fileWithFrames))) and (fileWithFrames != currFile):
            comparisonFile = openFileForTerms(os.path.join(args.datasetFolder,"%s_terms.txt" % (fileWithFrames)))[1:]
            currFile = fileWithFrames
        elif not os.path.exists(os.path.join(args.datasetFolder,"%s_terms.txt" % (fileWithFrames))):
            continue

        frameNumber = int(e[-1].split(os.path.sep)[-1][:-4])
        result[int(e[0] == comparisonFile[frameNumber-1][0])] += 1

    print(result)
    print("General Accuracy = %f" % (result[1] / len(estimated)))
    

if __name__ == '__main__':
    main()