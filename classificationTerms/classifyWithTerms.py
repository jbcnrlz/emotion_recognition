import argparse, pandas as pd, numpy as np, os, sys
from scipy.spatial.distance import euclidean
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getFilesInPath

def outputClassificationFile(pathToFile,result,blockSize):
    with open(pathToFile,'w') as ptf:
        ptf.write(str(blockSize) + '\n')
        for r in result:
            ptf.write(r + '\n')

def main():
    parser = argparse.ArgumentParser(description='Classify subjects')
    parser.add_argument('--termsCSV', help='Path for the terms file', required=True)
    parser.add_argument('--classifyFiles', help='Path for CSVs with Valence and arousal', nargs="+", required=True)
    parser.add_argument('--sizeBlock', help="Size of the chunks for mean", default=2000, type=int)
    args = parser.parse_args()
    tFiles = np.array(pd.read_csv(args.termsCSV))
    classesLabel = tFiles[:,0]
    vaValues = tFiles[:,[1,3]].astype(np.float32)

    filesToProcess = []
    for c in args.classifyFiles:
        if os.path.isfile(c):
            filesToProcess.append(c)
        else:
            filesToProcess += getFilesInPath(c)

    for c in filesToProcess:
        classFiles = np.array(pd.read_csv(c))
        averages = []
        if args.sizeBlock == 1:
            vas = classFiles[:,[0,1]]
            files = classFiles[:,2]
            distances = np.zeros((len(vas),len(vaValues)))
            for idxAv, i in enumerate(vas):
                for idxVa, j in enumerate(vaValues):
                    print("Calculating distance from sample %d to Class %s" % (idxAv,classesLabel[idxVa]))
                    distances[idxAv,idxVa] = euclidean(i,j)
            
            classificationValue = classesLabel[distances.argsort()[:,0]]
            pathToResult = c[:-4] + '_terms' + c[-4:]
            outputClassificationFile(pathToResult,classificationValue,args.sizeBlock)

        else:
            for i in range(0,len(classFiles),args.sizeBlock):
                endingStep = i + args.sizeBlock if (i + args.sizeBlock) < len(classFiles) else -1
                if len(classFiles[i:endingStep]) > 0:
                    for axisClean in [0,1]:
                        valsToCalculate = classFiles[i:endingStep][classFiles[i:endingStep,axisClean] >= -1]
                        valsToCalculate = classFiles[i:endingStep][classFiles[i:endingStep,axisClean] <= 1]
                    averages.append((np.mean(valsToCalculate[:,0]),np.mean(valsToCalculate[:,1])))

            averages = np.array(averages)

            distances = np.zeros((len(averages),len(vaValues)))
            for idxAv, i in enumerate(averages):
                for idxVa, j in enumerate(vaValues):
                    distances[idxAv,idxVa] = euclidean(i,j)
            
            classificationValue = classesLabel[distances.argsort()[:,0]]
            pathToResult = c[:-4] + '_terms' + c[-4:]
            outputClassificationFile(pathToResult,classificationValue,args.sizeBlock)

if __name__ == '__main__':
    main()