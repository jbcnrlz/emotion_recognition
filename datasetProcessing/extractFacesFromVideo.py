import cv2, shutil, os, face_alignment
import numpy as np

def findROI(rois,areas):
    returnData = []
    for a in areas:
        for rn, r in enumerate(rois):
            dist = np.linalg.norm(a - r)
            if (dist < 20) and (dist < 20):
                returnData.append(rn)
                break

    return returnData

def isNewROI(currRoi,foundFaces):
    returnData = []
    for a in foundFaces:
        colision = 0
        for b in currRoi:
            dist = np.linalg.norm(a - b)
            if (dist < 20) and (dist < 20):
                colision = 1

        if not colision:
            returnData.append(a)

    return returnData


def main():
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

    if os.path.exists('clipe1'):
        shutil.rmtree('clipe1')

    os.makedirs('clipe1')
    sucs = 1
    vcap = cv2.VideoCapture("teste_0.mp4")
    frame_number=0
    rois = []
    while sucs:
        sucs, imgv = vcap.read()
        if not sucs:
            break

        faces = fa.face_detector.detect_from_image(imgv)        
        landmarks = fa.get_landmarks(imgv)
        if landmarks:
            landmarks = np.array(landmarks)
            newROIs = isNewROI(rois, landmarks[:,33])
            for nt in newROIs:
                rois.append(nt)

            roins = findROI(rois, landmarks[:,33])   

            for idx, r in enumerate(roins):
                rois[r] = landmarks[idx,33]

            for fnum, b in enumerate(faces):
                if roins[fnum] > 1:
                    continue
                print("Extraindo face %d" % (roins[fnum]))

                b = list(map(int,b))
                fImage = imgv[b[1]:b[3],b[0]:b[2]]
                cv2.imwrite(os.path.join('clipe1',"roi_" + str(roins[fnum]) + "_frame_"+str(frame_number)+".jpg"),fImage)
        frame_number+=1



if __name__ == '__main__':
    main()