import argparse, cv2, sys, os, math

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from helper.function import getFilesInPath

def main():
    parser = argparse.ArgumentParser(description='Separate videos')
    parser.add_argument('--pathBase', help='Path for videos', required=True)
    parser.add_argument('--sizeClip', help='Size of the chunk to be utilized (in minutes)', type=int, required=True)
    args = parser.parse_args()    
    
    clipFolder = os.path.join(args.pathBase,'clips')
    if not os.path.exists(clipFolder):
        os.makedirs(clipFolder)


    videos = getFilesInPath(args.pathBase)
    for v in videos:
        vCap = cv2.VideoCapture(v)
        codecVideo = int(vCap.get(cv2.CAP_PROP_FOURCC))
        fps = math.floor(vCap.get(cv2.CAP_PROP_FPS))
        size = (int(vCap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(vCap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        sizeClipFrames = fps * args.sizeClip * 60
        clipNumber = 1
        fileName = v.split(os.path.sep)[-1].split('.')
        out = cv2.VideoWriter(os.path.join(args.pathBase,'clips',fileName[0] + '_' + str(clipNumber) + '.' + fileName[1]),codecVideo, fps, size)
        sizeClip = 0
        print("Starting clip 1")
        while (vCap.isOpened()):
            ret, frame = vCap.read()
            if ret==True and (sizeClip < sizeClipFrames):
                out.write(frame)
                sizeClip += 1
            elif (sizeClip >= sizeClipFrames):
                print("Wrapped clip %d" % (clipNumber))
                sizeClip = 0
                clipNumber += 1
                out = cv2.VideoWriter(os.path.join(args.pathBase,'clips',fileName[0] + '_' + str(clipNumber) + '.' + fileName[1]),codecVideo, fps, size)
                print("Starting clip %d" % (clipNumber))
            else:
                break

        print("oe")


if __name__ == '__main__':
    main()