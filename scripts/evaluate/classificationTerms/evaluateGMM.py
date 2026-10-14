import numpy as np, itertools, pandas as pd
from scipy import linalg
import matplotlib.pyplot as plt, matplotlib as mpl
from sklearn import mixture

from generateGMM import getDistribution

# Number of samples per component
n_samples = 500

# Generate random sample, two components
tFiles = np.array(pd.read_csv('hajer_categ.csv'))
classesLabel = tFiles[:,0]
vaValues = tFiles[:,[1,3,2,4]].astype(np.float32)
dists = []
for i in range(151):
    dists += [getDistribution(vaValues[i,[0,1]],vaValues[i,[2,3]])]
dists = np.array(dists)
X = dists.reshape((dists.shape[0] * dists.shape[1],2))

lowest_bic = np.infty
bic = []
n_components_range = [5,24,48]
cv_types = ["spherical", "tied", "diag", "full"]
for cv_type in cv_types:
    for n_components in n_components_range:
        print("Type %s - Component number %d" % (cv_type,n_components))
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(
            n_components=n_components, covariance_type=cv_type
        )
        gmm.fit(X)
        bic.append(gmm.bic(X))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm

bic = np.array(bic)
color_iter = itertools.cycle(["navy", "turquoise", "cornflowerblue", "darkorange"])
clf = best_gmm
bars = []

# Plot the BIC scores
plt.figure(figsize=(8, 6))
spl = plt.subplot(1, 1, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + 0.2 * (i - 2)
    bars.append(
        plt.bar(
            xpos,
            bic[i * len(n_components_range) : (i + 1) * len(n_components_range)],
            width=0.2,
            color=color,
        )
    )
plt.xticks(n_components_range)
plt.ylim([bic.min() * 1.01 - 0.01 * bic.max(), bic.max()])
plt.title("BIC score per model")
xpos = (
    np.mod(bic.argmin(), len(n_components_range))
    + 0.65
    + 0.2 * np.floor(bic.argmin() / len(n_components_range))
)
plt.text(xpos, bic.min() * 0.97 + 0.03 * bic.max(), "*", fontsize=14)
spl.set_xlabel("Number of components")
spl.legend([b[0] for b in bars], cv_types)
plt.show()