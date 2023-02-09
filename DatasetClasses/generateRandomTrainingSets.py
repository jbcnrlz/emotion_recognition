import os, sys, argparse, random, shutil, math, numpy as np
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getDirectoriesInPath, getFilesInPath

def main():
    parser = argparse.ArgumentParser(description='Generate Random Sampling')
    parser.add_argument('--pathBase', help='Path for video', required=True)
    parser.add_argument('--sizeSampling', help='Path for video', required=True,type=int)
    parser.add_argument('--ignoreEmotion', help='Path for video', required=False,default=None,nargs='+',type=int)
    args = parser.parse_args()

    filePathSep = args.pathBase.split(os.path.sep)
    filesForPath = getFilesInPath(args.pathBase)    
    eachFolder = [[] for i in range(math.ceil(len(filesForPath) / args.sizeSampling))]
    for f in filesForPath:
        if args.ignoreEmotion is not None:
            fileName = f.split(os.path.sep)[-1][:-4]
            expType = np.load(os.path.join(os.path.sep.join(filePathSep[:-1]),'annotations','%d_exp.npy' % (int(fileName)))).astype(np.uint8)
            if expType in args.ignoreEmotion:
                continue

        currFold = random.randint(0,len(eachFolder) - 1)
        while(len(eachFolder[currFold]) >= args.sizeSampling):
            currFold = random.randint(0,len(eachFolder) - 1)

        eachFolder[currFold].append(f)

    for idxF, cFolder in enumerate(eachFolder):
        currFolder = os.path.join(os.path.sep.join(filePathSep[:-1]),filePathSep[-1]+'_%d' % (idxF))
        if not os.path.exists(currFolder):
            os.makedirs(currFolder)
        for c in cFolder:
            shutil.copyfile(c,os.path.join(currFolder,c.split(os.path.sep)[-1]))
    
if __name__ == '__main__':
    main()