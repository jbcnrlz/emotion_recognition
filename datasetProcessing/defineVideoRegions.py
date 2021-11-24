import cv2, face_alignment, numpy as np, math, matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter

def generateListFromData(originalList,maxFrame):
    output = []
    for i in range(maxFrame):
        pass

def cropVideo(video,centerMoments,momentSize=2):
    codecVideo = int(video.get(cv2.CAP_PROP_FOURCC))
    fps = math.floor(video.get(cv2.CAP_PROP_FPS))
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    momentSize = (momentSize * 60) * fps
    currPoint = 0
    cFrame = 0
    startingFrame = centerMoments[currPoint] - int(momentSize / 2)
    endingFrame = centerMoments[currPoint] + int(momentSize / 2)
    outputClip = []
    saveFrame = 0
    while (video.isOpened()):
        ret, frame = video.read()
        if not ret:
            break
        
        if cFrame == endingFrame:
            saveFrame = 0
            out = cv2.VideoWriter('teste_%d.mp4' % (currPoint),codecVideo, fps, size)
            for oc in outputClip:
                out.write(oc)
            outputClip = []
            currPoint += 1
            if currPoint >= len(centerMoments):
                break
            startingFrame = centerMoments[currPoint] - int(momentSize / 2)
            endingFrame = centerMoments[currPoint] + int(momentSize / 2)

        if cFrame == startingFrame:
            saveFrame = 1

        if saveFrame:
            outputClip.append(frame)

        cFrame += 1

def extractMoments(windowSize=500,momentSize=2400):

    csvMarkup = pd.read_csv('ouput.csv')
    output = {}
    for j in range(csvMarkup['subject'].max()+1):
        output[j] = {}
        for i in range(csvMarkup['frame'].max()):
            output[j][i] = 0
    
    for idx, d in csvMarkup.iterrows():
        output[int(d['subject'])][int(d['frame'])] = d['distance']

    pData = []

    for d in range(2):

        pData.append(savgol_filter(list(output[d].values()),51,3))
        pData[-1] = pData[-1] / max(pData[-1])

        #scal = MinMaxScaler()
        #scal.fit(pData[-1])
        #pData[-1] = scal.transform(pData[-1])

    incWindow = int(momentSize / 2)
    i = incWindow
    pointsChange = []
    while i < ((len(pData[0]) - windowSize) - incWindow):
        if ((pData[0][i] > pData[1][i]) and (pData[0][i+windowSize] < pData[1][i+windowSize])):
            while (pData[0][i] > pData[1][i]):
                i += 1

            pointsChange.append(i)
            i += incWindow
        elif ((pData[0][i] < pData[1][i]) and (pData[0][i+windowSize] > pData[1][i+windowSize])):
            while (pData[0][i] < pData[1][i]):
                i += 1

            pointsChange.append(i)
            i += incWindow

        i += 1

    #plt.plot(pData[0])
    #plt.plot(pData[1])

    #for xc in pointsChange:
    #    plt.axvline(x=xc,color='r',linestyle='--')

    #plt.show()

    return pointsChange, pData

def findAngle(p1,p2):
    if (p2[0] == p1[0]):
        return 'undefined'
    return math.degrees(np.arctan((p2[1] - p1[1]) / (p2[0] - p1[0])))

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

def generateROIList(videoFeed,thres=7):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    rois = []
    measures = []

    with open('ouput.csv','w') as mfl:
        mfl.write('frame,subject,distance,angle\n')
        while (videoFeed.isOpened()):
            ret, frame = videoFeed.read()
            if not ret:
                break
            pred = fa.get_landmarks(frame)
            if pred:
                measures.append({})
                pred = np.array(pred)
                newROIs = isNewROI(rois, pred[:,33])
                for nt in newROIs:
                    rois.append(nt)
                roins = findROI(rois, pred[:,33])            
                for idx, r in enumerate(roins):
                    rois[r] = pred[idx,33]
                    measures[-1][r] = {'points' : pred[idx,[51,62,66,57]]}                    
                    if 'measures' in measures[-1].keys():
                        measures[-1]['measures'][r] = [euclidean(pred[idx,51],pred[idx,57]) if euclidean(pred[idx,51],pred[idx,57]) > thres else 0,euclidean(pred[idx,62],pred[idx,66]),findAngle(pred[idx,57],pred[idx,51])]
                    else:
                        measures[-1]['measures'] = {r : [euclidean(pred[idx,51],pred[idx,57]) if euclidean(pred[idx,51],pred[idx,57]) > thres else 0,euclidean(pred[idx,62],pred[idx,66]),findAngle(pred[idx,57],pred[idx,51])]}
                    mfl.write('%d,%d,%f,%f\n' % (len(measures),r,measures[-1]['measures'][r][0],measures[-1]['measures'][r][1]))

            else:
                measures.append([])
        
if __name__ == '__main__':
    vCap = cv2.VideoCapture("D:/data_sep_videos/2GRAV1qua1.mp4")
    #generateROIList(vCap)
    a, b = extractMoments(momentSize=12000)
    cropVideo(vCap,a,momentSize=5)
    print('oi')
    '''
    vCap = cv2.VideoCapture("D:/data_sep_videos/clips/2GRAV1qua1_5.mp4")
    a = generateROIList(vCap,2)
    xvalues = []
    for frame in a:
        if len(frame) > 0:
            keys = frame['measures'].keys()
            if 1 in keys:
                xvalues.append(frame['measures'][1][0])   
    '''
