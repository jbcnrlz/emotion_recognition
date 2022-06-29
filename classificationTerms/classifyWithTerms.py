import argparse, pandas as pd, numpy as np, os, sys
from scipy.spatial.distance import euclidean
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getFilesInPath

def outputClassificationFile(pathToFile,result,blockSize):    
    with open(pathToFile,'w') as ptf:
        ptf.write(str(blockSize) + '\n')
        if (type(result) is dict):
            for r in result:
                for i in range(len(result[r])):
                    ptf.write('%s,%d,%s\n' % (result[r][i],i,r))
        else:        
            for r in result:
                ptf.write(r + '\n')

def sortVideoFrames(features):
    featuresOrdered = {}
    for f in features:
        splittedPath = f[-1].split(os.path.sep)
        if "/" in splittedPath[0]:
            splittedPath = splittedPath[0].split('/') + splittedPath[1:]
        subjectVideo = splittedPath[splittedPath.index('afew-va') + 1]
        if subjectVideo not in featuresOrdered.keys():
            featuresOrdered[subjectVideo] = []
        
        frameNo = int(splittedPath[-1][:-4])
        initialFrame = 0
        while(len(featuresOrdered[subjectVideo]) <= frameNo):
            if (initialFrame >= len(featuresOrdered[subjectVideo])):
                featuresOrdered[subjectVideo].append([-10,-10])

            initialFrame += 1

        featuresOrdered[subjectVideo][frameNo] = [f[0],f[1]]

    return featuresOrdered


def main():
    parser = argparse.ArgumentParser(description='Classify subjects')
    parser.add_argument('--termsCSV', help='Path for the terms file', required=True)
    parser.add_argument('--classifyFiles', help='Path for CSVs with Valence and arousal', nargs="+", required=True)
    parser.add_argument('--sizeBlock', help="Size of the chunks for mean", default=2000, type=int)
    parser.add_argument('--forceBlock', help="Force block algoritm", default=0, type=bool)
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
        if args.sizeBlock == 1 and args.forceBlock == False:
            vas = classFiles[:,[0,1]]
            #files = classFiles[:,2]
            distances = np.zeros((len(vas),len(vaValues)))
            for idxAv, i in enumerate(vas):
                for idxVa, j in enumerate(vaValues):
                    print("Calculating distance from sample %d to Class %s" % (idxAv,classesLabel[idxVa]))
                    distances[idxAv,idxVa] = euclidean(i,j)
            
            classificationValue = classesLabel[distances.argsort()[:,0]]
            pathToResult = c[:-4] + '_terms' + c[-4:]
            outputClassificationFile(pathToResult,classificationValue,args.sizeBlock)

        else:
            orderedFeatures = sortVideoFrames(classFiles)
            averagedVA = {}
            for v in orderedFeatures:
                averagedVA[v] = []
                for i in range(0,len(orderedFeatures[v]),args.sizeBlock):
                    endingStep = i + args.sizeBlock if (i + args.sizeBlock) < len(classFiles) else -1
                    if len(orderedFeatures[v][i:endingStep]) > 0:
                        valSum = sum([vval[0] for vval in orderedFeatures[v][i:endingStep]])
                        aroSum = sum([vval[1] for vval in orderedFeatures[v][i:endingStep]])
                        averagedVA[v].append([valSum / len(orderedFeatures[v][i:endingStep]),aroSum / len(orderedFeatures[v][i:endingStep])])
                    
            classificationValue = {}
            for v in averagedVA:
                distances = np.zeros((len(averagedVA[v]),len(vaValues)))
                for idxAv, i in enumerate(averagedVA[v]):
                    for idxVa, j in enumerate(vaValues):
                        distances[idxAv,idxVa] = euclidean(i,j)
            
                classificationValue[v] = classesLabel[distances.argsort()[:,0]]
            pathToResult = '%s_%s_terms_blocksize_%d_%s' % (c[:-4],args.termsCSV,args.sizeBlock,c[-4:])
            outputClassificationFile(pathToResult,classificationValue,args.sizeBlock)

if __name__ == '__main__':
    main()