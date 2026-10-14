import argparse, os, sys, face_alignment, cv2
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getFilesInPath

def main():
    parser = argparse.ArgumentParser(description='Evaluate t-SNE for clusters')
    parser.add_argument('--folderImages', help='Path for the terms file', required=True)
    args = parser.parse_args()

    folderName = args.folderImages.split(os.path.sep)[-1] + '_segmented'
    if not os.path.exists(folderName):
        os.makedirs(folderName)

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    faces = getFilesInPath(args.folderImages)
    for f in faces:
        faceImage = cv2.imread(f)
        facesSeg = fa.face_detector.detect_from_image(faceImage)
        if facesSeg:
            fileName = f.split(os.path.sep)[-1]
            coordinatesFace = list(map(int,facesSeg[0]))
            segmentedFace = faceImage[coordinatesFace[1]:coordinatesFace[3],coordinatesFace[0]:coordinatesFace[2]]
            if len(segmentedFace) > 0 and len(segmentedFace[0]) > 0:
                cv2.imwrite(os.path.join(folderName,fileName),segmentedFace)



if __name__ == '__main__':
    main()