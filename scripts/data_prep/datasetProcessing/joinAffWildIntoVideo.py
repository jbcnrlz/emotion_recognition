import os, sys, cv2, numpy as np, argparse
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getFilesInPath

def main():
    parser = argparse.ArgumentParser(description='Join files to video')
    parser.add_argument('--pathBase', help='Path for videos', required=True)
    parser.add_argument('--annotationFile', help='Path for videos', required=True)
    args = parser.parse_args()

    for annFile in getFilesInPath(args.annotationFile):
        fileName = annFile.split(os.path.sep)[-1][:-10]
        print('Doing %s'%(fileName))
        frameN = 1
        foundOne = 0
        videoFile = []
        while True:
            fileFrame = os.path.join(args.pathBase,'cropped_aligned',fileName,"%05d.jpg" % (frameN))
            if os.path.exists(fileFrame):
                foundOne = 1
                videoFile.append(fileFrame)                
            elif foundOne == 1:
                break
            frameN += 1

        out = cv2.VideoWriter(os.path.join(args.pathBase,'cropped_aligned',fileName + '.avi'),cv2.VideoWriter_fourcc('M','J','P','G'), 20, cv2.imread(videoFile[0]).shape[:-1])
        for vf in videoFile:
            videoFrame = cv2.imread(vf)
            #out.write(cv2.resize(videoFrame,(224,224)))
            out.write(videoFrame)
        out.release()
        

if __name__ == '__main__':
    main()