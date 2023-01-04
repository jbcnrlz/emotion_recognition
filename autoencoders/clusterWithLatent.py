import argparse, numpy as np, pandas as pd, shutil, os
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

def main():
    parser = argparse.ArgumentParser(description='Generate GMM for latent')
    parser.add_argument('--latentCSV', help='Path for the terms file', required=True)
    parser.add_argument('--components', help='Quantity of components', required=True, type=int)
    parser.add_argument('--type', help='Quantity of components', required=False, default='predict')
    parser.add_argument('--dirCluster', help='Quantity of components', required=False, default='predict')    
    args = parser.parse_args()
    tFiles = np.array(pd.read_csv(args.latentCSV))
    classesLabel = tFiles[:,-1]
    vaValues = tFiles[:,:-1].astype(np.float32)
    estimator = GaussianMixture(
        n_components=args.components,
        covariance_type='spherical',
        random_state=0
    )

    if os.path.exists(args.dirCluster):
        shutil.rmtree(args.dirCluster)
    os.makedirs(args.dirCluster)

    if (args.type == 'predict'):
        clusters = estimator.fit_predict(vaValues)
        for imgN, c in enumerate(clusters):
            if ( not os.path.exists(os.path.join(args.dirCluster,str(c)))):
                os.makedirs(os.path.join(args.dirCluster,str(c)))

            fileName = classesLabel[imgN].split(os.path.sep)[-1]
            shutil.copy(classesLabel[imgN],os.path.join(args.dirCluster,str(c),fileName))
    else:

        clusters = estimator.fit(vaValues)
        probsForClasses = clusters.predict_proba(vaValues)
        clustersProbability = (-probsForClasses).argsort()

        for imgN, probsC in enumerate(clustersProbability):
            for c in probsC:
                if probsForClasses[imgN][c] > 0.1e-5:
                    if ( not os.path.exists(os.path.join(args.dirCluster,str(c)))):
                        os.makedirs(os.path.join(args.dirCluster,str(c)))

                    fileName = classesLabel[imgN].split(os.path.sep)[-1]
                    shutil.copy(classesLabel[imgN],os.path.join(args.dirCluster,str(c),fileName))
                else:
                    break

if __name__ == '__main__':
    main()

