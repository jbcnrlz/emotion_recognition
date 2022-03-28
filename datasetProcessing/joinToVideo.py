import os, sys, cv2, numpy as np, argparse
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getFilesInPath

def main():
    parser = argparse.ArgumentParser(description='Join files to video')
    parser.add_argument('--pathBase', nargs="+",help='Path for videos', required=True)
    args = parser.parse_args()

    for pb in args.pathBase:
        arqvs = getFilesInPath(pb,imagesOnly=True)
        maxImSize = {}
        biggerV = 0
        for a in arqvs:
            roiFile = int(a.split(os.path.sep)[-1].split('_')[1])
            frame = int(a.split(os.path.sep)[-1].split('_')[-1].split('.')[0])
            if (roiFile not in maxImSize.keys()) or (maxImSize[roiFile] < frame):
                maxImSize[roiFile] = frame

            if frame > biggerV:
                biggerV = frame

        for rN in maxImSize:    
            out = cv2.VideoWriter(os.path.join(pb,'clip1_%d.avi') % (rN),cv2.VideoWriter_fourcc('M','J','P','G'), 20, (80,100))
            lastFound = None
            for i in range(biggerV):
                faceImage = os.path.join(pb,"roi_%d_frame_%d.jpg") % (rN,i)
                if os.path.exists(faceImage):
                    imFile = cv2.imread(faceImage)
                    lastFound = faceImage
                elif lastFound is None:
                    print(faceImage + " Not found")
                    imFile = np.zeros((100,80,3), np.uint8)
                else:
                    imFile = cv2.imread(lastFound)
                nImFile = cv2.resize(imFile,(80,100))
                out.write(nImFile)

            out.release()

if __name__ == '__main__':
    main()