import os, sys, argparse, pandas as pd, numpy as np
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getFilesInPath, getDirectoriesInPath, getFeatureFromText

def normalize(rangesLabel,labelsToJoin):
    totalPerCluster = [0] * len(labelsToJoin)
    outputImage = []
    for r in rangesLabel:
        outputImage.append([0] * len(labelsToJoin))
        for qt in rangesLabel[r]:
            if qt == 'total':
                continue

            for idx, l in enumerate(labelsToJoin):
                if qt in l:
                    outputImage[-1][idx] += rangesLabel[r][qt]
                    totalPerCluster[idx] += rangesLabel[r][qt]

    
    for i in range(len(outputImage)):
        for j in range(len(outputImage[i])):
            outputImage[i][j] = outputImage[i][j] / totalPerCluster[j]

    return outputImage

def main():
    parser = argparse.ArgumentParser(description='Evaluate folders')
    parser.add_argument('--pathBase', help='Path for cluster', required=True)
    parser.add_argument('--pathAffWild', help='Path for cluster', required=True)
    parser.add_argument('--typeAnnotation', help='Path for cluster', required=True)    
    parser.add_argument('--csvEmotions', help='Path for resnet pretrained weights', required=True)
    parser.add_argument('--normalize', help='Should normalize?', required=False, default=None)
    args = parser.parse_args()
    #use valence and arousal to understand the cluster (plotting to understand)
    #images = getFilesInPath(args.pathAffWild)
    emotions = np.array(pd.read_csv(args.csvEmotions))
    clusters = getDirectoriesInPath(args.pathBase)
    clustersEval = {}
    for c in clusters: 
        if c not in clustersEval.keys():
            clustersEval[c] = {}
        imagesFromClusters = getFilesInPath(os.path.join(args.pathBase,c))
        clustersEval[c]['total'] = len(imagesFromClusters)
        for i in imagesFromClusters:
            filePath = int(i.split(os.path.sep)[-1][:-4])
            if (args.typeAnnotation == 'original'):
                express = int(np.load(os.path.join(args.pathAffWild,'annotations','%d_exp.npy' % (filePath))))
                currEmotion = emotions[express][0]
                if currEmotion not in clustersEval[c].keys():
                    clustersEval[c][currEmotion] = 0
                clustersEval[c][currEmotion] += 1

            elif (args.typeAnnotation == 'hajer'):
                currEmotion = getFeatureFromText(os.path.join(args.pathAffWild,'relabel','%d_relabel.txt' % (filePath)))
                if currEmotion is None:
                    continue
                currEmotion = currEmotion.split(' ')[0]
                if currEmotion not in clustersEval[c].keys():
                    clustersEval[c][currEmotion] = 0
                clustersEval[c][currEmotion] += 1

    if args.normalize is not None:
        print(normalize(clustersEval,[['neutral'],['sadness','happy','contempt','surprise','fear','anger','disgust']]))
    else:
        print(clustersEval)


if __name__ == '__main__':
    main()