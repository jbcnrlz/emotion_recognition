import argparse, numpy as np, pandas as pd, shutil, os
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

def main():
    parser = argparse.ArgumentParser(description='Generate GMM for latent')
    parser.add_argument('--latentCSV', help='Path for the terms file', required=True)
    parser.add_argument('--components', help='Quantity of components', required=True, type=int)
    args = parser.parse_args()
    tFiles = np.array(pd.read_csv(args.latentCSV))
    classesLabel = tFiles[:,-1]
    vaValues = tFiles[:,:-1].astype(np.float32)
    estimator = GaussianMixture(
        n_components=args.components,
        covariance_type='spherical',
        random_state=0
    )
    clusters = estimator.fit_predict(vaValues)

    if os.path.exists('clusterLatent'):
        shutil.rmtree('clusterLatent')
    os.makedirs('clusterLatent')

    for imgN, c in enumerate(clusters):
        if ( not os.path.exists(os.path.join('clusterLatent',str(c)))):
            os.makedirs(os.path.join('clusterLatent',str(c)))

        fileName = classesLabel[imgN].split(os.path.sep)[-1]
        shutil.copy(classesLabel[imgN],os.path.join('clusterLatent',str(c),fileName))

if __name__ == '__main__':
    main()

