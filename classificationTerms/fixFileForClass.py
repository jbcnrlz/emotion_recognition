import os, sys, numpy as np, argparse, pandas as pd
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getFilesInPath

def main():
    parser = argparse.ArgumentParser(description='Join files to video')
    parser.add_argument('--pathBase', help='Path for videos', required=True)
    args = parser.parse_args()

    allFiles = getFilesInPath(args.pathBase)
    outputFiles = []
    for a in allFiles:
        fileName = a.split(os.path.sep)[-1][:-4]
        fileOpened = np.array(pd.read_csv(a))
        for idxV, f in enumerate(fileOpened):
            fileForCSV = os.path.join(fileName,"%05d.jpg" % (idxV + 1) )
            outputFiles.append([f[0],f[1],fileForCSV])

    with open('affwild_mimamo_va.csv','w') as amv:
        amv.write("valence,arousal,file\n")
        for of in outputFiles:
            amv.write("%f,%f,%s\n" % (of[0],of[1],of[2]))
    
if __name__ == '__main__':
    main()