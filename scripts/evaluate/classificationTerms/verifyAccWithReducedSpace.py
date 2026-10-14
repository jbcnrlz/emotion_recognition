import argparse, os, sys, numpy as np, pandas as pd
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getFilesInPath, getFeatureFromText

def main():
    parser = argparse.ArgumentParser(description='Verify Accuracy')
    parser.add_argument('--csvClassification', help='Path for resnet pretrained weights', required=True)
    parser.add_argument('--pathFile', help='Weights', required=True)
    parser.add_argument('--annotationDir', help='Path for valence and arousal dataset', required=True, nargs='+')    
    args = parser.parse_args()

    expressions = np.array(pd.read_csv(args.csvClassification))

    files = getFilesInPath(os.path.join(args.pathFile,'images'))
    for ad in args.annotationDir:
        perClass = {}
        results = [0,0]
        for f in files:
            imgName = int(f.split(os.path.sep)[-1][:-4])
            label = getFeatureFromText(os.path.join(args.pathFile,ad,'%d_relabel.txt' % (imgName)))
            for e in expressions:
                probeName = e[-1].split(os.path.sep)[-1]
                probeName = probeName.split('/')[-1]
                probeName = int(probeName[:-4])
                if imgName == probeName:
                    if (label == e[0]):
                        if e[0] not in perClass.keys():
                            perClass[e[0]] = 0
                        perClass[e[0]] += 1
                    results[int(label == e[0])] += 1
                    break

        print("Results from %s ==> Correct answers %d (%.2f%%), Wrong answers %d (%.2f%%)" % (ad,results[1],(results[1] / sum(results)) * 100,results[0],(results[0] / sum(results))*100))
        print(perClass)



if __name__ == '__main__':
    main()