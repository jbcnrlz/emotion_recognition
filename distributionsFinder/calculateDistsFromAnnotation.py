import os, sys, argparse, pandas as pd, numpy as np, shutil
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getFilesInPath
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

def getAnnotationsFiles(dirGet,lastClassNumber):
    datasetFolder = dirGet.split(os.path.sep)[:-1]
    datasetFolder = os.path.join(os.path.sep.join(datasetFolder),'images')
    fullData = []
    filePath = []
    for i in range(0,lastClassNumber+1):
        if os.path.exists(os.path.join(datasetFolder,'%d.jpg' % (i))):
            dataAnnotation = []
            for fileType in ['%d_val.npy','%d_aro.npy']:
                valueForData = float(np.load(os.path.join(dirGet,fileType % (i))))
                dataAnnotation.append(valueForData)
                
            filePath.append(os.path.join(datasetFolder,'%d.jpg' % (i)))
            fullData.append(dataAnnotation)

    fullData = np.array(fullData)
    return filePath, fullData

def main():
    parser = argparse.ArgumentParser(description='Generate GMM')
    parser.add_argument('--pathAnnotationTrain', help='Path for the terms file', required=True)
    parser.add_argument('--pathAnnotationTest', help='Path for the terms file', required=True)
    parser.add_argument('--lastClassNumber', help='Path for the terms file', required=True, type=int)
    parser.add_argument('--clusterFolder', help='Path for the terms file', required=True)
    args = parser.parse_args()

    if os.path.exists(args.clusterFolder):
        shutil.rmtree(args.clusterFolder)

    os.makedirs(args.clusterFolder)

    _, fullData = getAnnotationsFiles(args.pathAnnotationTrain,args.lastClassNumber)

    estimator = GaussianMixture(
        n_components=8,
        covariance_type='spherical',
        random_state=0
    )
    estimator = estimator.fit(fullData)

    filePath, fullData = getAnnotationsFiles(args.pathAnnotationTest,args.lastClassNumber)

    clusters = estimator.predict(fullData)
    biggerCluster = max(clusters)
    for i in range(biggerCluster+1):
        os.makedirs(os.path.join(args.clusterFolder,str(i)))    

    for idxC, fp in enumerate(filePath):
        fileName = fp.split(os.path.sep)[-1]
        shutil.copy(fp,os.path.join(args.clusterFolder,str(clusters[idxC]),fileName))
    print(clusters)


if __name__ == '__main__':
    main()