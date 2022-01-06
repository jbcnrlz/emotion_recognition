import os, sys, cv2, numpy as np
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getFilesInPath

def main():
    arqvs = getFilesInPath("D:/PycharmProjects/emotion_recognition/clipe1")
    maxImSize = {}
    for a in arqvs:
        roiFile = int(a.split(os.path.sep)[-1].split('_')[1])
        frame = int(a.split(os.path.sep)[-1].split('_')[-1].split('.')[0])
        if (roiFile not in maxImSize.keys()) or (maxImSize[roiFile] < frame):
            maxImSize[roiFile] = frame

    '''
    imgs = [[],[]]
    for a in arqvs:
        roiFile = int(a.split(os.path.sep)[-1].split('_')[1])
        if roiFile < len(imgs):
            imFile = cv2.imread(a)
            nImFile = cv2.resize(imFile,(80,100))
            imgs[roiFile].append(nImFile)
    '''
    for rN in maxImSize:    
        out = cv2.VideoWriter('clip1_%d.avi' % (rN),cv2.VideoWriter_fourcc('M','J','P','G'), 20, (80,100))
        for i in range(maxImSize[rN]):
            faceImage = "D:/PycharmProjects/emotion_recognition/clipe1/roi_%d_frame_%d.jpg" % (rN,i)
            if os.path.exists(faceImage):
                imFile = cv2.imread(faceImage)
            else:
                print(faceImage + " Not found")
                imFile = np.zeros((100,80,3), np.uint8)
            nImFile = cv2.resize(imFile,(80,100))
            out.write(nImFile)

        out.release()

if __name__ == '__main__':
    main()