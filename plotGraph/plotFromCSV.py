import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def calcDeriv(number,x=None):
    if x is None:
        x = list(range(len(number)))

    return np.gradient(number,x)

def genGraph(csvs):
    fig, axs = plt.subplots(2,1)
    dataCsv = []
    a = MinMaxScaler(feature_range=(-1,1))
    for i, c in enumerate(csvs):
        dataCsv.append(pd.read_csv(c))
        axs[0].plot(savgol_filter(list(dataCsv[-1]['valence']),51,3))
        axs[1].plot(savgol_filter(list(dataCsv[-1]['arousal']),51,3))
    
    dydx = calcDeriv(savgol_filter(list(dataCsv[0]['valence']),51,3),savgol_filter(list(dataCsv[1]['valence']),51,3))
    dydx = [0]+list(a.fit_transform(dydx.reshape(-1,1)).flatten())
    axs[0].plot(dydx)
    dydx = calcDeriv(savgol_filter(list(dataCsv[0]['arousal']),51,3),savgol_filter(list(dataCsv[1]['arousal']),51,3))
    dydx = [0]+list(a.fit_transform(dydx.reshape(-1,1)).flatten())
    axs[1].plot(dydx)
    plt.show()

if __name__ == "__main__":
    genGraph(['clip1_0.csv','clip1_1.csv'])
    #opCSV1 = pd.read_csv('clip1_0.csv')
    #opCSV2 = pd.read_csv('clip1_1.csv')    
    #plt.plot(savgol_filter(list(opCSV1['valence']),51,3))
    #plt.plot(savgol_filter(list(opCSV2['valence']),51,3))
    #dydx = [0] + list(calcDeriv(savgol_filter(list(opCSV1['valence']),51,3),savgol_filter(list(opCSV2['valence']),51,3)))
    #plt.plot(dydx)
    #plt.show()