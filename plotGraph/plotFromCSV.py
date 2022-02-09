import matplotlib.pyplot as plt, cv2
import pandas as pd
from scipy.signal import savgol_filter
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from torchvision import transforms
from matplotlib.figure import Figure

def calcDeriv(subject1,subject2=None):
    if subject2 is None:
        subject2 = list(range(len(subject1)))

    return np.gradient(subject1)

def genGraph(csvs):
    fig, axs = plt.subplots(2,1)
    dataCsv = []
    a = MinMaxScaler(feature_range=(-1,1))
    for i, c in enumerate(csvs):
        dataCsv.append(pd.read_csv(c))
        axs[0].plot(savgol_filter(list(dataCsv[-1]['valence']),51,3))
        axs[1].plot(savgol_filter(list(dataCsv[-1]['arousal']),51,3))
    
    dydx = calcDeriv(savgol_filter(list(dataCsv[0]['valence']),51,3))
    dydx = [0]+list(a.fit_transform(dydx.reshape(-1,1)).flatten())
    axs[0].plot(dydx)
    dydx = calcDeriv(savgol_filter(list(dataCsv[1]['valence']),51,3))
    dydx = [0]+list(a.fit_transform(dydx.reshape(-1,1)).flatten())
    axs[0].plot(dydx)
    dydx = calcDeriv(savgol_filter(list(dataCsv[0]['arousal']),51,3),savgol_filter(list(dataCsv[1]['arousal']),51,3))
    dydx = [0]+list(a.fit_transform(dydx.reshape(-1,1)).flatten())
    axs[1].plot(dydx)
    plt.show()
    return (savgol_filter(list(dataCsv[0]['valence']),51,3),savgol_filter(list(dataCsv[0]['arousal']),51,3)), (savgol_filter(list(dataCsv[1]['valence']),51,3),savgol_filter(list(dataCsv[1]['arousal']),51,3))

if __name__ == "__main__":
    va1, va2 = genGraph(['clip1_0.csv','clip1_1.csv'])

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