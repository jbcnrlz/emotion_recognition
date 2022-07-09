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
    args = parser.parse_args()

    baseFile = np.array(pd.read_csv(args.annotationFile))
    terms = baseFile[:,0]
    filesToTerms = baseFile[:,1]
    files = getFilesInPath(args.pathBase,imagesOnly=True)
    for f in files:
        currFileIndex = np.where(filesToTerms == f.replace('images','annotations'))[0][0]
        term = terms[currFileIndex]
        fileName = f.split(os.path.sep)[-1][:-4]
        outputTXTAnnotation(os.path.join(args.annotationPath,"%s_term.txt" % (fileName)),term)


if __name__ == '__main__':
    main()