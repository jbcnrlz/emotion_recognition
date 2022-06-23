import argparse, pandas as pd, numpy as np, os, sys, itertools
from sklearn.mixture import BayesianGaussianMixture
from scipy.stats import norm, multivariate_normal
from scipy.special import kl_div
from generateGMM import getDistribution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from helper.function import saveCSV

def joinProbability(simProbs,pDist,probsNames):
    newJoined = []
    alreadyWent = []
    names = []
    for i in range(simProbs.shape[0]):
        if probsNames[i] not in alreadyWent:
            alreadyWent.append(probsNames[i])
            currProb = simProbs[i]
            indexesJoin = currProb.argsort()
            pToJoin = 1
            while ((probsNames[indexesJoin[pToJoin]] in alreadyWent) or (probsNames[indexesJoin[pToJoin]] == probsNames[i])):
                pToJoin += 1
                if pToJoin >= len(probsNames):
                    break
            
            if pToJoin < len(probsNames):
                newJoined.append((pDist[i] + pDist[indexesJoin[pToJoin]]) / 2)                
                alreadyWent.append(probsNames[indexesJoin[pToJoin]])
                names.append("%s + %s" % (probsNames[i],probsNames[indexesJoin[pToJoin]]))

    return np.array(newJoined), names

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
    args = parser.parse_args()
    tFiles = np.array(pd.read_csv(args.termsCSV))
    classesLabel = tFiles[:,0]
    vaValues = tFiles[:,[1,3,2,4]].astype(np.float32)    
    for k in range(2):
        '''
        dists = []
        for i in range(len(classesLabel)):
            dists += [getDistribution(vaValues[i,[0,1]],vaValues[i,[2,3]])]

        dists = np.array(dists)
        '''
        klMatrix = np.zeros((len(classesLabel),len(classesLabel)))
        for i in range(len(classesLabel)):
            for j in range(len(classesLabel)):
                #print("KL de %s para %s" % (classesLabel[i],classesLabel[j]))
                #klMatrix[i,j] = bhattacharyya_distance(dists[i],dists[j])
                
                covPt1 = np.zeros((2,2))
                covPt1[0,0] = vaValues[i][2]
                covPt1[1,1] = vaValues[i][3]
                pt1 = multivariate_normal(mean=vaValues[i,0:2],cov=covPt1)
                #pt1 = np.array([pt1.rvs() for i in range(4000)])
                covPt2 = np.zeros((2,2))
                covPt2[0,0] = vaValues[j][2]
                covPt2[1,1] = vaValues[j][3]
                pt2 = multivariate_normal(mean=vaValues[j,0:2],cov=covPt2)
                #pt2 = np.array([pt2.rvs() for i in range(4000)])            
                klMatrix[i,j] = bhattacharyya_gaussian_distance(vaValues[i,0:2],covPt1,vaValues[j,0:2],covPt2)
                #klMatrix[i,j] = klDivergence(pt1,pt2)
                
        vaValues, classesLabel = joinProbability(klMatrix,vaValues,classesLabel)
    print('oi')
    saveCSV('joinedWithDistance.csv',classesLabel,vaValues[:,[0,2,1,3]])

    #outputTXT(a,classesLabel,'clustering.csv')

if __name__ == '__main__':
    joinDistributions()