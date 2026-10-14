import argparse, os, sys, numpy as np, pandas as pd, matplotlib.pyplot as plt, shutil
import warnings
from importlib.resources import path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

warnings.filterwarnings("ignore")

def test():
    parser = argparse.ArgumentParser(description='Extract VA with RESNETEMOTION')
    parser.add_argument('--pathCSV', help='Weights', required=True, nargs='+')
    parser.add_argument('--labels', help='Weights', required=True, nargs='+')
    parser.add_argument('--ranks', help='Weights', required=True, type=int)
    args = parser.parse_args()    
    for idxFile, pathCSV in enumerate(args.pathCSV):
        considered = 0
        csvFiles = pd.read_csv(pathCSV)
        logits = np.array(csvFiles)[:,[0,1,2,3,4,5,6]].astype(np.float64)
        labels = np.array(csvFiles)[:,[-1]].astype(np.uint8).flatten()
        ranks = np.array([0 for i in range(args.ranks)])
        
        for idx, l in enumerate(logits):
            l = -l
            if (len(np.where(l.argsort() == labels[idx])[0]) == 0):
                continue
            curRank = np.where(l.argsort() == labels[idx])[0][0]
            if curRank < args.ranks:
                ranks[curRank:] += 1
            considered += 1

        ranks = ranks / considered
        print(ranks)        
        plt.plot(ranks,label=args.labels[idxFile])        
    plt.legend()
    plt.show()

if __name__ == '__main__':
    test()