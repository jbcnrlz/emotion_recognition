import matplotlib.pyplot as plt, cv2
import pandas as pd
from scipy.signal import savgol_filter
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from torchvision import transforms
from matplotlib.figure import Figure
from scipy.spatial.distance import euclidean
def calcDeriv(subject1):
    return np.gradient(subject1)

def avgSignal(signal,window,step):
    outAvgSig = []
    outStdSig = []
    currStep = 0
    while currStep < len(signal):
        if (currStep+window >= len(signal)):
            outAvgSig.append(np.mean(signal[currStep:-1]))
            outStdSig.append(np.std(signal[currStep:-1]))
        else:
            outAvgSig.append(np.mean(signal[currStep:currStep+window]))
            outStdSig.append(np.std(signal[currStep:currStep+window]))
        currStep += step

    return outAvgSig, outStdSig

def genDataFromCSV(csvs):
    dataCsv = []
    for i, c in enumerate(csvs):
        dataCsv.append(pd.read_csv(c))
    return (savgol_filter(list(dataCsv[0]['valence']),51,3),savgol_filter(list(dataCsv[0]['arousal']),51,3)), (savgol_filter(list(dataCsv[1]['valence']),51,3),savgol_filter(list(dataCsv[1]['arousal']),51,3))

def genGraph(csvs):
    fig, axs = plt.subplots(2,2)
    dataCsv = []
    a = MinMaxScaler(feature_range=(-1,1))
    for i, c in enumerate(csvs):
        dataCsv.append(pd.read_csv(c))
        axs[0][0].plot(savgol_filter(list(dataCsv[-1]['valence']),51,3))
        axs[0][1].plot(avgSignal(list(dataCsv[-1]['valence']),10,5))
        axs[1][0].plot(savgol_filter(list(dataCsv[-1]['arousal']),51,3))
        axs[1][1].plot(avgSignal(list(dataCsv[-1]['arousal']),10,5))

    dydx = calcDeriv(savgol_filter(list(dataCsv[0]['valence']),51,3))
    dydx = [0]+list(a.fit_transform(dydx.reshape(-1,1)).flatten())
    #axs[0].plot(dydx)
    dydx = calcDeriv(savgol_filter(list(dataCsv[1]['valence']),51,3))
    dydx = [0]+list(a.fit_transform(dydx.reshape(-1,1)).flatten())
    #axs[0].plot(dydx)
    dydx = calcDeriv(savgol_filter(list(dataCsv[0]['arousal']),51,3))
    dydx = [0]+list(a.fit_transform(dydx.reshape(-1,1)).flatten())
    #axs[1].plot(dydx)
    plt.show()
    return (savgol_filter(list(dataCsv[0]['valence']),51,3),savgol_filter(list(dataCsv[0]['arousal']),51,3)), (savgol_filter(list(dataCsv[1]['valence']),51,3),savgol_filter(list(dataCsv[1]['arousal']),51,3))

def outputWithAnnotation(videoPath,outputVideo,points):
    vcap = cv2.VideoCapture(videoPath)
    fourcc = cv2.VideoWriter_fourcc('F', 'M', 'P', '4')
    width  = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(outputVideo, fourcc, 20.0, (width, height))
    drawRect = 0
    cFrame = 0
    while (vcap.isOpened()):
        sucs, imgv = vcap.read()
        if sucs:
            if drawRect or cFrame in points:
                cv2.rectangle(imgv,(0,0),(width,height),(255,0,0),2)
                if drawRect == 0:
                    drawRect = 100
                else:
                    drawRect -= 1
            out.write(imgv)

            cFrame += 1
        else:
            break

    vcap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    windowSize = 10
    va1, va2 = genDataFromCSV(['oc_clip1_0.csv','oc_clip1_1.csv'])
    avgV1, stdV1 = avgSignal(va1[0][:min(len(va1[0]),len(va2[0]))],windowSize,windowSize)
    avgV2, stdV2 = avgSignal(va2[0][:min(len(va1[0]),len(va2[0]))],windowSize,windowSize)

    maxAvgStd = np.max([np.mean(stdV1),np.mean(stdV2)])
    dists = np.absolute(np.array(avgV1) - np.array(avgV2))
    indexes = (-dists).argsort()

    quantitySizes = 1000
    pointsChoosed = []
    for i in indexes:
        if (np.max([stdV1[i],stdV2[i]]) >= maxAvgStd):
            startingFrame = (i*windowSize)
            pointsChoosed.append(startingFrame)
            endingFrame = (startingFrame + windowSize)
            startingFrame = startingFrame/20
            endingFrame = endingFrame/20
            print("Between %d and %d frames" % (startingFrame,endingFrame))            
            quantitySizes -= 1
        
        if quantitySizes == 0:
            break
    
    outputWithAnnotation("D:/PycharmProjects/emotion_recognition/selectedClips/10-60-1280x720.mp4","out.avi",pointsChoosed)

    '''
    fig, axs = plt.subplots(2,1)
    axs[0].plot(va1[0][:min(len(va1[0]),len(va2[0]))])
    axs[0].plot(va2[0][:min(len(va1[0]),len(va2[0]))])    
    axs[1].plot(stdV1)
    axs[1].plot(stdV2)
    
    for c in pointsChoosed:
        axs[0].plot([c,c],[-0.5,0.5])
        axs[1].plot([c,c],[-0.5,0.5])
    
    plt.show()
    '''
    '''
    a = MinMaxScaler(feature_range=(-1,1))
    dydx = calcDeriv(savgol_filter(list(va1[0]),51,3),savgol_filter(list(va2[0]),51,3))
    dydx = [0]+list(a.fit_transform(dydx.reshape(-1,1)).flatten())

    dydx2 = calcDeriv(savgol_filter(list(va1[1]),51,3),savgol_filter(list(va2[1]),51,3))
    dydx2 = [0]+list(a.fit_transform(dydx2.reshape(-1,1)).flatten())    

    fig, ax = plt.subplots(2,1)
    canvas = FigureCanvas(fig)    
    #ax = fig.gca()
    ax[0].set_yticks((-1,1))
    ax[1].set_yticks((-1,1))
    width, height = fig.get_size_inches() * fig.get_dpi()
    fourcc = cv2.VideoWriter_fourcc('F', 'M', 'P', '4')
    out = cv2.VideoWriter('out.avi', fourcc, 20.0, (int(width), int(height)*2))
    vcap = cv2.VideoCapture("teste_0.mp4")
    idxAndamento = 0
    while (vcap.isOpened()):
        sucs, imgv = vcap.read()
        if sucs:

            ax[0].set(xlabel='',ylabel='Valence')
            ax[1].set(xlabel='Frame',ylabel='Arousal')
            ax[0].plot(va1[0])
            ax[0].plot(va2[0])
            ax[0].plot(dydx)
            ax[0].plot([idxAndamento,idxAndamento],[-1,1])

            ax[1].plot(va1[1])
            ax[1].plot(va2[1])
            ax[1].plot(dydx2)
            ax[1].plot([idxAndamento,idxAndamento],[-1,1])

            canvas.draw()
            im = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
            nconc = np.concatenate((imgv, im))
            out.write(nconc)
            ax[0].clear()
            ax[1].clear()
            idxAndamento += 1            
            #cv2.imwrite('teste.jpg',im)
        else:
            break

    vcap.release()
    out.release()
    cv2.destroyAllWindows()

    #opCSV1 = pd.read_csv('clip1_0.csv')
    #opCSV2 = pd.read_csv('clip1_1.csv')
    #plt.plot(savgol_filter(list(opCSV1['valence']),51,3))
    #plt.plot(savgol_filter(list(opCSV2['valence']),51,3))
    #dydx = [0] + list(calcDeriv(savgol_filter(list(opCSV1['valence']),51,3),savgol_filter(list(opCSV2['valence']),51,3)))
    #plt.plot(dydx)
    #plt.show()
    '''