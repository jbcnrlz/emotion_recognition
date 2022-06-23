import argparse, pandas as pd, numpy as np, os, sys
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import saveCSV

def generateKMEANS():
    parser = argparse.ArgumentParser(description='Generate GMM')
    parser.add_argument('--termsCSV', help='Path for the terms file', required=True)
    parser.add_argument('--components', help='Quantity of components', required=True, type=int)
    args = parser.parse_args()
    tFiles = np.array(pd.read_csv(args.termsCSV))
    classesLabel = tFiles[:,0]
    vaValues = tFiles[:,[1,3,2,4]].astype(np.float32)
    estimator = KMeans(            
        n_clusters=args.components,
        random_state=0           
    )

    estimator.fit(vaValues[:,[0,1]])
    a = estimator.predict(vaValues[:,[0,1]])

    labelsJoined = [''] * args.components
    for idx, cluster in enumerate(a):
        if labelsJoined[cluster] == '':
            labelsJoined[cluster] = classesLabel[idx]
        else:
            labelsJoined[cluster] += ' + %s' % (classesLabel[idx])

    #outputTXT(a,classesLabel,'clustering.csv')
    saveCSV('joinedWithKMEANS.csv',labelsJoined,np.concatenate((estimator.cluster_centers_,estimator.cluster_centers_),axis=1))
    
    
if __name__ == '__main__':
    generateKMEANS()