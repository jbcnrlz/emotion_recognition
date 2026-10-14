import argparse, os, sys, numpy as np, face_alignment, cv2
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getDirectoriesInPath, getFilesInPath

def test():
    parser = argparse.ArgumentParser(description='Extract faces from video')
    parser.add_argument('--pathBase', help='Path for valence and arousal dataset', required=True)
    args = parser.parse_args()
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    dirs = getDirectoriesInPath(args.pathBase)
    for d in dirs:
        folderName = d.split(os.path.sep)[-1]
        if folderName != 'youtube_videos_temp':
            videoFiles = getFilesInPath(os.path.join(args.pathBase,d))
            for v in videoFiles:
                fileVideoName = v.split(os.path.sep)[-1].split('.')[0]
                dirForFaces = os.path.join(args.pathBase,d,fileVideoName)
                if not os.path.exists(dirForFaces):
                    os.makedirs(os.path.join(args.pathBase,d,fileVideoName))
                else:
                    continue

                frameNumber = 0
                vcap = cv2.VideoCapture(v)
                sucs = 1
                print("Extracting faces from video " + fileVideoName)
                while sucs:
                    sucs, imgv = vcap.read()
                    if not sucs:
                        break

                    faces = fa.face_detector.detect_from_image(imgv)        
                    landmarks = fa.get_landmarks(imgv)
                    if landmarks:
                        landmarks = np.array(landmarks)
                        for fnum, b in enumerate(faces):
                            b = list(map(int,b))
                            fImage = imgv[b[1]:b[3],b[0]:b[2]]
                            cv2.imwrite(os.path.join(dirForFaces,"%d.jpg" % (frameNumber)),fImage)
                            break

                    frameNumber += 1

if __name__ == '__main__':
    test()