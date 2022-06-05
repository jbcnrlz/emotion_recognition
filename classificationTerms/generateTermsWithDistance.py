import argparse, pandas as pd, numpy as np, os, sys, itertools
from sklearn.mixture import BayesianGaussianMixture
from scipy.stats import norm, multivariate_normal

def klDivergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def outputTXT(clusters,classes,file):
    with open(file,'w') as fterms:
        fterms.write("Term,cluster\n")
        for idxC, c in enumerate(classes):
            fterms.write("%s,%d\n" % (c,clusters[idxC]))

def joinDistributions():
    parser = argparse.ArgumentParser(description='Generate GMM')
    parser.add_argument('--termsCSV', help='Path for the terms file', required=True)
    args = parser.parse_args()
    tFiles = np.array(pd.read_csv(args.termsCSV))
    classesLabel = tFiles[:,0]
    vaValues = tFiles[:,[1,3,2,4]].astype(np.float32)

    klMatrix = np.zeros((len(classesLabel),len(classesLabel)))
    for i in range(len(classesLabel)):
        for j in range(len(classesLabel)):
            print("KL de %s para %s" % (classesLabel[i],classesLabel[j]))
            x = np.arange(-1, 1, 0.01)
            x = np.array(list(itertools.product(x,x)))
            #x = np.stack((x,x)).T
            covPt1 = np.zeros((2,2))
            covPt1[0,0] = vaValues[i][2]
            covPt1[1,1] = vaValues[i][3]
            pt1 = multivariate_normal.pdf(x,mean=vaValues[i,0:2],cov=covPt1)
            covPt2 = np.zeros((2,2))
            covPt2[0,0] = vaValues[j][2]
            covPt2[1,1] = vaValues[j][3]
            pt2 = multivariate_normal.pdf(x,mean=vaValues[j,0:2],cov=covPt2)
            klMatrix[i,j] = klDivergence(pt1,pt2)
    
    print('oi')

    #outputTXT(a,classesLabel,'clustering.csv')

if __name__ == '__main__':
    joinDistributions()