import os, sys, argparse, pandas as pd, numpy as np, pyperclip
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

def getEmotion(filePath):
    with open(filePath,'r') as fp:
        for f in fp:
            return int(float(f.strip()))

def main():
    parser = argparse.ArgumentParser(description='Evaluate folders')
    parser.add_argument('--pathBase', help='Path for cluster', required=True)
    parser.add_argument('--pathDataset', help='Path for cluster', required=True)
    parser.add_argument('--typeAnnotation', help='Path for cluster', required=True)    
    parser.add_argument('--csvEmotions', help='Path for resnet pretrained weights', required=True)
    parser.add_argument('--normalize', help='Should normalize?', required=False, default=None)
    parser.add_argument('--matlabPrint', help='Should normalize?', required=False, default=None)
    parser.add_argument('--yLimit', help='Should normalize?', required=False, default=None)
    parser.add_argument('--dataset', help='Should normalize?', required=False, default='affwild')
    parser.add_argument('--copyClipboard', help='Should normalize?', required=False, type=bool, default=True)
    args = parser.parse_args()
    #use valence and arousal to understand the cluster (plotting to understand)
    #images = getFilesInPath(args.pathAffWild)
    if args.dataset == 'wfd':
        datasetItems = pd.read_csv(args.pathDataset)
    emotions = np.array(pd.read_csv(args.csvEmotions))
    clusters = getDirectoriesInPath(args.pathBase)
    clustersEval = {}
    for c in clusters: 
        if c not in clustersEval.keys():
            clustersEval[c] = {}
        imagesFromClusters = getFilesInPath(os.path.join(args.pathBase,c))
        clustersEval[c]['total'] = len(imagesFromClusters)
        for i in imagesFromClusters:
            if args.dataset == 'affwild':
                filePath = int(i.split(os.path.sep)[-1][:-4])
            else:
                filePath = i.split(os.path.sep)[-1]                
            if (args.typeAnnotation == 'original'):
                if args.dataset == 'affwild':
                    express = int(np.load(os.path.join(args.pathDataset,'annotations','%d_exp.npy' % (filePath))))
                elif args.dataset == 'wfd':
                    express = list(datasetItems[datasetItems['Image'] == filePath]['Most Selected Emotion'])[0]
                else:
                    folder = filePath.split('_')[0]
                    subject = filePath.split('_')[1]
                    express = getEmotion(os.path.join(args.pathDataset,'Emotion',folder,subject,filePath[:-4] + '_emotion.txt'))
                if args.dataset == 'wfd':
                    currEmotion = express
                else:
                    currEmotion = emotions[express][0]
                if currEmotion not in clustersEval[c].keys():
                    clustersEval[c][currEmotion] = 0
                clustersEval[c][currEmotion] += 1

            elif (args.typeAnnotation == 'hajer'):
                currEmotion = getFeatureFromText(os.path.join(args.pathDataset,'relabel','%d_relabel.txt' % (filePath)))
                if currEmotion is None:
                    continue
                currEmotion = currEmotion.split(' ')[0]
                if currEmotion not in clustersEval[c].keys():
                    clustersEval[c][currEmotion] = 0
                clustersEval[c][currEmotion] += 1

    if args.matlabPrint is not None:
        divisionGraphs = len(clustersEval) // 2 if len(clustersEval) % 2 == 0 else (len(clustersEval) // 2) + 1
        outputGraph = "tiledlayout(%d,2);\n" % (divisionGraphs)
        for k in clustersEval:
            dataPlot = clustersEval[k]
            dataPlot.pop('total')
            outputGraph += 'a%d = [' % (int(k))
            keysForGraph = list(dataPlot.keys())
            keysForGraph.sort()
            dataForPlot = [str(dataPlot[d]) for d in keysForGraph]
            outputGraph += ' '.join(dataForPlot) + '];\n'
            outputGraph += 'b%d = {' % (int(k))
            dataForPlot = ["'%s'" % str(d) for d in keysForGraph]
            outputGraph += ' '.join(dataForPlot) + '};\n'
            outputGraph += 'nexttile;\n'
            outputGraph += 'bar(a%d);\n' %(int(k))
            outputGraph += 'xticklabels(b%d);\n' %(int(k))
            outputGraph += 'title(\'Cluster %d\');\n' %(int(k))
            if args.yLimit is not None:
                outputGraph += 'axis([0 inf 0 %d]);\n' % (int(args.yLimit))
                outputGraph += "axis 'auto x';\n"

        if args.copyClipboard:
            pyperclip.copy(outputGraph)
            print("Plot went to clipboard - paste it on MATLAB")
        else:
            print(outputGraph)
            
    elif args.normalize is not None:
        print(normalize(clustersEval,[['neutral'],['sadness','happy','contempt','surprise','fear','anger','disgust']]))
    else:
        print(clustersEval)


if __name__ == '__main__':
    main()