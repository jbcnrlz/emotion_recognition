import argparse, pandas as pd, numpy as np, os, sys, itertools
import enum
from sklearn.mixture import BayesianGaussianMixture
from scipy.stats import norm, multivariate_normal
from scipy.special import kl_div
from generateGMM import getDistribution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from helper.function import saveCSV

def joinProbability(simProbs,originalNames,originalValues,limitJoin=0):

    outputName = []
    outputProb = []
    originalDists = np.copy(simProbs)
    simProbs[simProbs == 0] = 1000
    while True:
        joinedProb = np.where(simProbs == simProbs.min())
        toBeJoined = originalNames[joinedProb[1][0]]
        mainClass = originalNames[joinedProb[0][0]]
        if (toBeJoined.count('+') >  limitJoin) or (mainClass.count('+') >  limitJoin):
            simProbs[joinedProb[1][0]][joinedProb[0][0]] = simProbs[joinedProb[0][0]][joinedProb[1][0]] = 1000
        else:
            break

        if (simProbs.min() == 1000):
            return joinProbability(originalDists,originalNames,originalValues,limitJoin+1)

    joinedProb = (originalValues[joinedProb[1][0]] + originalValues[joinedProb[0][0]]) / 2
    outputName.append(toBeJoined + ' + ' + mainClass)
    outputProb.append(joinedProb)
    for idxN, name in enumerate(originalNames):
        if name != toBeJoined and name != mainClass:
            outputName.append(name)
            outputProb.append(originalValues[idxN])

    return np.array(outputProb), outputName

def bhattacharyya_distance(distribution1,distribution2):
    """ Estimate Bhattacharyya Distance (between General Distributions)
    
    Args:
        distribution1: a sample distribution 1
        distribution2: a sample distribution 2
    
    Returns:
        Bhattacharyya distance
    """
    sq = 0
    for i in range(len(distribution1)):
        sq  += np.sqrt(distribution1[i]*distribution2[i])
    
    return -np.log(sq)

def bhattacharyya_gaussian_distance(mean1,cov1,mean2,cov2):
    """ Estimate Bhattacharyya Distance (between Gaussian Distributions)
    
    Args:
        distribution1: a sample gaussian distribution 1
        distribution2: a sample gaussian distribution 2
    
    Returns:
        Bhattacharyya distance
    """

    cov = (1 / 2) * (cov1 + cov2)

    T1 = (1 / 8) * (
        np.sqrt((mean1 - mean2) * np.linalg.inv(cov) * (mean1 - mean2).T)[0][0]
    )
    T2 = (1 / 2) * np.log(
        np.linalg.det(cov) / np.sqrt(np.linalg.det(cov1) * np.linalg.det(cov2))
    )

    return T1 + T2

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
    parser.add_argument('--quantityClusters', help='Path for the terms file', required=True, type=int)
    args = parser.parse_args()
    tFiles = np.array(pd.read_csv(args.termsCSV))
    classesLabel = tFiles[:,0]
    vaValues = tFiles[:,[1,3,2,4]].astype(np.float32)
    while len(vaValues) > args.quantityClusters:
        print("%d clusters - still %d to go" % (len(vaValues),len(vaValues) - args.quantityClusters))
        dists = []
        for i in range(len(vaValues)):
            dists += [getDistribution(vaValues[i,[0,1]],vaValues[i,[2,3]])]
        dists = np.array(dists)        
        
        klMatrix = np.zeros((len(classesLabel),len(classesLabel)))
        for i in range(len(classesLabel)):
            for j in range(len(classesLabel)):
                klMatrix[i,j] = bhattacharyya_gaussian_distance(vaValues[i,0:2],np.cov(dists[i].T),vaValues[j,0:2],np.cov(dists[j].T))
                #klMatrix[i,j] = klDivergence(pt1,pt2)
                
        vaValues, classesLabel = joinProbability(klMatrix,classesLabel,vaValues)

    print('oi')
    saveCSV('joinedWithDistance_%d.csv' % (args.quantityClusters),classesLabel,vaValues[:,[0,2,1,3]])

    #outputTXT(a,classesLabel,'clustering.csv')

if __name__ == '__main__':
    joinDistributions()