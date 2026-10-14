import cv2, face_alignment, argparse, os, sys, math
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getFilesInPath

def cropVideo(video,frames,fileName='teste_%d.mp4'):
    codecVideo = int(video.get(cv2.CAP_PROP_FOURCC))
    fps = math.floor(video.get(cv2.CAP_PROP_FPS))
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(fileName,codecVideo, fps, size)
    for fr in frames:
        out.write(fr)

def outputFramesNumber(frames,pathFile):
    with open(pathFile,'w') as pf:
        for f in frames:
            pf.write(str(f)+'\n')

def main():
    parser = argparse.ArgumentParser(description='Separate videos')
    parser.add_argument('--pathBase', help='Path for video', required=True)
    parser.add_argument('--outputPathVideo', help='Path for video', required=True)
    args = parser.parse_args()
    videosFiles = getFilesInPath(args.pathBase)        

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    for v in videosFiles:
        print("Looking for interactions ins %s video" % (v))
        framesOutput = []
        framesNumber = []
        cFrame = 0
        vcap = cv2.VideoCapture(v)
        fileName = v.split(os.path.sep)[-1]
        pathForInterVideo = os.path.join(args.outputPathVideo,fileName)
        pathforTXTFile = os.path.join(args.outputPathVideo,fileName[:-3]+'txt')
        sucs = True
        while sucs:
            print("Current frame: %d" % (cFrame))
            sucs, imgv = vcap.read()
            if not sucs:
                break

            faces = fa.face_detector.detect_from_image(imgv)
            if len(faces) > 1:
                framesOutput.append(imgv)
                framesNumber.append(cFrame)

            cFrame += 1

        if len(framesOutput) > 0:
            cropVideo(vcap,framesOutput,pathForInterVideo)
            outputFramesNumber(framesNumber,pathforTXTFile)



if __name__ == '__main__':
    main()