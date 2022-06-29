import argparse, pandas as pd, numpy as np, os, sys
from black import out
from scipy.spatial.distance import euclidean
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getFilesInPath

def main():
    parser = argparse.ArgumentParser(description='Generate CSV')
    parser.add_argument('--basePath', help='Path for the terms file', required=True)
    parser.add_argument('--lastClass', help='Path for the terms file', required=True, type=int)
    parser.add_argument('--outputFile', help='Path for the terms file', required=True)
    args = parser.parse_args()
    
    outputData = []
    files = ['_val','_aro']
    for cidx in range(args.lastClass+1):
        sample = []
        for f in files:
            fullFilePath = os.path.join(args.basePath,'%d%s.npy' % (cidx,f))
            if os.path.exists(fullFilePath):
                value = np.load(fullFilePath).astype(np.float32)
                sample.append(value)

        if (len(sample) > 0):
            sample.append(fullFilePath)
            outputData.append(sample)

    with open(args.outputFile,'w') as of:
        of.write("valence,arousal,file\n")
        for o in outputData:
            of.write("%f,%f,%s\n" % (o[0],o[1],o[2]))

if __name__ == '__main__':
    main()