import os, sys, argparse, numpy as np
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import loadNeighFiles
from sklearn.naive_bayes import GaussianNB


def main():
    parser = argparse.ArgumentParser(description='Generate GMM')
    parser.add_argument('--neighsFile', help='Path for the terms file', required=True)
    args = parser.parse_args()

    fileWithNeighs = loadNeighFiles(args.neighsFile)
    results = [0,0]
    wResults = {}
    for f in fileWithNeighs:
        tset = np.array(fileWithNeighs[f]['neighbours'])
        tsetx = tset[:,:-1]
        tsety = tset[:,-1]
        clf = GaussianNB()
        clf.fit(tsetx,tsety)
        pred = clf.predict(np.array([fileWithNeighs[f]['va']]))
        results[int(fileWithNeighs[f]['label'] == pred[0])] += 1
        if not fileWithNeighs[f]['label'] == pred[0]:
            wResults[f] = fileWithNeighs[f]
    print(results)

if __name__ == '__main__':
    main()