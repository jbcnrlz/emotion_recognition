import sys, os, argparse, json, cv2, numpy as np
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getFilesInPath, getDirectoriesInPath

def segmentFace():
    parser = argparse.ArgumentParser(description='Segment face')
    parser.add_argument('--pathBase', help='Path for video', required=True)    
    args = parser.parse_args()
    dirs = getDirectoriesInPath(args.pathBase)
    for d in dirs:
        print("Doing folder - %s" % (d))
        subjectNumber = d.split(os.path.sep)[-1]
        subjectAnn = json.load(open(os.path.join(args.pathBase,d,subjectNumber+'.json')))
        faceFiles = getFilesInPath(os.path.join(args.pathBase,d),imagesOnly=True)
        for f in faceFiles:
            if not os.path.exists(os.path.join(args.pathBase,d,'faces')):
                os.makedirs(os.path.join(args.pathBase,d,'faces'))
            frameNumber = f.split(os.path.sep)[-1][:-4]
            landmarksForFrace = np.array(subjectAnn['frames'][frameNumber]['landmarks'])
            mins = landmarksForFrace.min(axis=0).astype(np.int16)
            mins[mins < 0] = 0
            maxs = landmarksForFrace.max(axis=0).astype(np.int16)
            maxs[maxs < 0] = 0
            w = maxs[0] - mins[0]
            h = maxs[1] - mins[1]
            imOpened = cv2.imread(f)
            cropped = imOpened[mins[1]:mins[1] + h,mins[0]:mins[0] + w,:]
            fileImageName = f.split(os.path.sep)[-1]
            cv2.imwrite(os.path.join(args.pathBase,d,'faces',fileImageName),cropped)            


if __name__ == '__main__':
    segmentFace()