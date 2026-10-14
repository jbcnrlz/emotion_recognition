import argparse, os, sys, cv2, math
import enum
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getFilesInPath

def returnAnnFiles(pathFile):
    blockSize = 0
    outputValues = []
    with open(pathFile,'r') as pf:
        for p in pf:
            if blockSize <= 0:
                blockSize = int(p.strip())
            else:
                outputValues.append(p.strip())

    return blockSize, outputValues

def writeClassToVideo(videoPath,labels,blockSize,outputFilePath):
    vcap = cv2.VideoCapture(videoPath)

    codecVideo = int(vcap.get(cv2.CAP_PROP_FOURCC))
    fps = math.floor(vcap.get(cv2.CAP_PROP_FPS))
    size = (int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(outputFilePath,codecVideo, fps, size)

    sucs = True
    currBlock = 0
    currLabel = None
    while sucs:
        sucs, imgv = vcap.read()
        if not sucs:
            break

        if currLabel is None or currBlock == blockSize:
            currLabel = [l.pop(0) for l in labels]
            currBlock = 0

        for idxCl, cl in enumerate(currLabel):
            cv2.putText(imgv,cl,(10,(100 * (idxCl + 1))),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0))
        out.write(imgv)
        currBlock += 1
        
def main():
    parser = argparse.ArgumentParser(description='Write on video')
    parser.add_argument('--videoFolders', help='Folders with the video files', nargs="+", required=True)
    parser.add_argument('--annotationFolders', help='Folders with the annotated files', nargs="+", required=True)
    args = parser.parse_args()

    videosFiles = []
    annFiles = []
    for v in args.videoFolders:
        videosFiles += getFilesInPath(v)

    for an in args.annotationFolders:
        annFiles += getFilesInPath(an)


    for v in videosFiles:
        videoFileName = v.split(os.path.sep)[-1]
        videoFileName = videoFileName.split('.')[0]
        labelFiles = []
        bs = 0
        for a in annFiles:
            fileName = a.split(os.path.sep)[-1]
            fileName = fileName.split('_')[0]
            if fileName == videoFileName:
                bs, lbls = returnAnnFiles(a)
                labelFiles.append(lbls)

        writeClassToVideo(v,labelFiles,bs,os.path.join('result',v.split(os.path.sep)[-1]))

if __name__ == '__main__':
    main()