import argparse, pandas as pd, numpy as np, sys, os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getFilesInPath

def outputTXTAnnotation(filePath,term):
    with open(filePath,'w') as fp:
        fp.write(term)

def main():
    parser = argparse.ArgumentParser(description='Format Terms to AffectNet')
    parser.add_argument('--pathBase', help='Path for data', required=True)
    parser.add_argument('--annotationPath', help='Path for data', required=True)
    parser.add_argument('--annotationFile', help='Path for annotation file', required=False)
    parser.add_argument('--lastClass', help='Path for the terms file', required=True, type=int)
    args = parser.parse_args()

    baseFile = np.array(pd.read_csv(args.annotationFile))
    terms = baseFile[:,0]
    filesToTerms = baseFile[:,1]
    for cidx in range(args.lastClass+1):
        if os.path.exists(os.path.join(args.pathBase,'%d.jpg' % (cidx))):
            print("Doing file %s" %(os.path.join(args.pathBase,'%d.jpg' % (cidx))))
            currFileIndex = np.where(filesToTerms == os.path.join(args.pathBase,'%d.jpg' % (cidx)).replace('images','annotations'))[0][0]
            term = terms[currFileIndex]
            fileName = '%d_terms.txt' % (cidx)
            outputTXTAnnotation(os.path.join(args.annotationPath,fileName),term)

if __name__ == '__main__':
    main()