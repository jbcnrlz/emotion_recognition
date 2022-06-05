import numpy as np, math, argparse, os, sys, subprocess
from scipy.spatial.distance import euclidean
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getFilesInPath

def cropVideo(video,centerMoments,momentSize=2,fps=20,fileName='teste_%d.mp4'):
    momentSize = momentSize * 60
    for idx, c in enumerate(centerMoments):
        secondClip = int(c / fps)
        startingFrame = secondClip - int(momentSize / 2)
        endingFrame = secondClip + int(momentSize / 2)
        p = subprocess.call(["ffmpeg",
                                "-y",
                                "-i", video,
                                "-ss", str(startingFrame),
                                "-c:v", "libx264", "-preset", "superfast",
                                "-f", "mp4",
                                "-c:a", "aac",
                                "-to", str(endingFrame),
                                '-strict', '-2',
                                fileName % (idx)],
                            )

    print('oi')
    pass

def extractMoments(windowSize=500,momentSize=2400,fileName='output.csv'):

    csvMarkup = pd.read_csv(fileName)
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

    return pointsChange, pData

def main():
    parser = argparse.ArgumentParser(description='Separate videos')
    parser.add_argument('--pathBase', help='Path for video', required=True)    
    args = parser.parse_args()
    for v in getFilesInPath(args.pathBase):
        if v[-3:] == 'csv':
            continue
        fileNameCSV = v.split('.')[0] + '.csv'
        if not os.path.exists(fileNameCSV):
            continue
        clipName = v.split('.')[0] + '_extracted_clip_%d.mp4'
        a, b = extractMoments(momentSize=12000,fileName=fileNameCSV)
        print('opa')
        cropVideo(v,a,5,fileName=clipName)


if __name__ == '__main__':
    main()
