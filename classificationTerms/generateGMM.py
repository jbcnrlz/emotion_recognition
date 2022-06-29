import argparse, pandas as pd, numpy as np, os, sys
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import saveCSV

def outputTXT(clusters,classes,file):
    with open(file,'w') as fterms:
        fterms.write("Term,cluster\n")
        for idxC, c in enumerate(classes):
            fterms.write("%s,%d\n" % (c,clusters[idxC]))

def getDistribution(mean,std,size=2000):
    covPt1 = np.zeros((2,2))
    covPt1[0,0] = std[0]
    covPt1[1,1] = std[1]
    rng = multivariate_normal(mean=mean,cov=covPt1)
    return np.array([rng.rvs() for i in range(size)])


def generateGMM():
    parser = argparse.ArgumentParser(description='Generate GMM')
    parser.add_argument('--termsCSV', help='Path for the terms file', required=True)
    parser.add_argument('--components', help='Quantity of components', required=True, type=int)
    args = parser.parse_args()
    tFiles = np.array(pd.read_csv(args.termsCSV))
    classesLabel = tFiles[:,0]
    vaValues = tFiles[:,[1,3,2,4]].astype(np.float32)
    dists = []
    for i in range(151):
        dists += [getDistribution(vaValues[i,[0,1]],vaValues[i,[2,3]])]
    dists = np.array(dists)
    dists = dists.reshape((dists.shape[0] * dists.shape[1],2))
    estimator = GaussianMixture(            
        n_components=args.components,
        covariance_type='full',
        random_state=0
        #weight_concentration_prior_type="dirichlet_distribution",
        #reg_covar=0,
        #init_params="random",
        #max_iter=1500,
        #mean_precision_prior=0.8,            
    )

    estimator.fit(vaValues[:,[0,1]])
    a = estimator.predict(vaValues[:,[0,1]])

    labelsJoined = [''] * args.components
    for idx, cluster in enumerate(a):
        if labelsJoined[cluster] == '':
            labelsJoined[cluster] = classesLabel[idx]
        else:
            labelsJoined[cluster] += ' + %s' % (classesLabel[idx])

    saveCSV('joinedWithGMM.csv',labelsJoined,np.concatenate((estimator.means_,estimator.means_),axis=1))
    
    
if __name__ == '__main__':
    generateGMM()

'''
# Parameters of the dataset
random_state, n_components, n_features = 2, 3, 2
colors = np.array(["#0072B2", "#F0E442", "#D55E00"])

covars = np.array(
    [[[0.7, 0.0], [0.0, 0.1]], [[0.5, 0.0], [0.0, 0.1]], [[0.5, 0.0], [0.0, 0.1]]]
)
samples = np.array([200, 500, 200])
means = np.array([[0.0, -0.70], [0.0, 0.0], [0.0, 0.70]])

# mean_precision_prior= 0.8 to minimize the influence of the prior
estimators = [
    (
        "Finite mixture with a Dirichlet distribution\nprior and " r"$\gamma_0=$",
        BayesianGaussianMixture(
            weight_concentration_prior_type="dirichlet_distribution",
            n_components=2 * n_components,
            reg_covar=0,
            init_params="random",
            max_iter=1500,
            mean_precision_prior=0.8,
            random_state=random_state,
        ),
        [0.001, 1, 1000],
    ),
    (
        "Infinite mixture with a Dirichlet process\n prior and" r"$\gamma_0=$",
        BayesianGaussianMixture(
            weight_concentration_prior_type="dirichlet_process",
            n_components=2 * n_components,
            reg_covar=0,
            init_params="random",
            max_iter=1500,
            mean_precision_prior=0.8,
            random_state=random_state,
        ),
        [1, 1000, 100000],
    ),
]

# Generate data
rng = np.random.RandomState(random_state)
X = np.vstack(
    [
        rng.multivariate_normal(means[j], covars[j], samples[j])
        for j in range(n_components)
    ]
)
y = np.concatenate([np.full(samples[j], j, dtype=int) for j in range(n_components)])

# Plot results in two different figures
for title, estimator, concentrations_prior in estimators:
    plt.figure(figsize=(4.7 * 3, 8))
    plt.subplots_adjust(
        bottom=0.04, top=0.90, hspace=0.05, wspace=0.05, left=0.03, right=0.99
    )

    gs = gridspec.GridSpec(3, len(concentrations_prior))
    for k, concentration in enumerate(concentrations_prior):
        estimator.weight_concentration_prior = concentration
        estimator.fit(X)
        plot_results(
            plt.subplot(gs[0:2, k]),
            plt.subplot(gs[2, k]),
            estimator,
            X,
            y,
            r"%s$%.1e$" % (title, concentration),
            plot_title=k == 0,
        )

plt.show()
'''