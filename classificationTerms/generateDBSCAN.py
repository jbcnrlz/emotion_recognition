import argparse, pandas as pd, numpy as np, os, sys
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import saveCSV
from generateTermsWithDistance import bhattacharyya_gaussian_distance

def generateDBSCAN():
    parser = argparse.ArgumentParser(description='Generate DBSCAN')
    parser.add_argument('--termsCSV', help='Path for the terms file', required=True)
    args = parser.parse_args()
    tFiles = np.array(pd.read_csv(args.termsCSV))
    classesLabel = tFiles[:,0]
    vaValues = tFiles[:,[1,3,2,4]].astype(np.float32)

    klMatrix = np.zeros((len(classesLabel),len(classesLabel)))
    for i in range(len(classesLabel)):
        for j in range(len(classesLabel)):
            print("Bhattacharyya distance from %s and %s" % (classesLabel[i],classesLabel[j]))            
            covPt1 = np.zeros((2,2))
            covPt1[0,0] = vaValues[i][2]
            covPt1[1,1] = vaValues[i][3]
            covPt2 = np.zeros((2,2))
            covPt2[0,0] = vaValues[j][2]
            covPt2[1,1] = vaValues[j][3]
            klMatrix[i,j] = bhattacharyya_gaussian_distance(vaValues[i,0:2],covPt1,vaValues[j,0:2],covPt2)


    estimator = DBSCAN(eps=3, min_samples=90,metric='precomputed')

    a = estimator.fit_predict(klMatrix)
    #estimator.predict(vaValues[:,[0,1]])

    labelsJoined = [''] * args.components
    for idx, cluster in enumerate(a):
        if labelsJoined[cluster] == '':
            labelsJoined[cluster] = classesLabel[idx]
        else:
            labelsJoined[cluster] += ' + %s' % (classesLabel[idx])

    #outputTXT(a,classesLabel,'clustering.csv')
    saveCSV('joinedWithKMEANS.csv',labelsJoined,np.concatenate((estimator.cluster_centers_,estimator.cluster_centers_),axis=1))
    
    
if __name__ == '__main__':
    generateDBSCAN()