import numpy as np, pandas as pd, os, sys, argparse
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getFilesInPath
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from calculateResults.distroResults import ranked_probability_score, kl_divergence, multiclass_brier_score
from scipy.spatial.distance import pdist, squareform

def calculateDistances(points):
    return squareform(pdist(points, metric='euclidean'))

def getGMM():
    classesDist = np.array([
        [0,0,0,0],
        [0.81,0.21,0.51,0.26], #happy
        [-0.63,0.23,-0.27,0.34], #sad
        [0.4,0.3,0.67,0.27], #surprised
        [-0.64,0.2,0.6,0.32],#fear
        [-0.6,0.2,0.35,0.41],#disgust
        [-0.51,0.2,0.59,0.33],#angry
        [-0.23,0.39,0.31,0.33],#contempt
        [0.65,0.29,-0.33,0.36],#leisurely-protected-relaxed
        [0.15,0.41,-0.18,0.30],#aloof-consoled-humble-modest-nonchalant-reserved-reverent-sheltered-solemn
        [0.74,0.22,-0.13,0.32],#secure
        [0.79,0.25,-0.01,0.46],#untroubled
        [0.19,0.57,-0.4,0.21]#quiet
    ])
   
    emotions = {"neutral" : [],"happy" : [] ,"sad" : [],"surprised" : [],"fear" : [],"disgust":[],"angry":[],"contempt": [], "serene" : [], "contemplative" : [], "secure" : [], "untroubled" : [], "quiet" : []}
    idx = -1
    covm = []
    means = []
    for k in emotions:
        idx += 1
        emotions[k] = [[classesDist[idx][0],classesDist[idx][2]], [[classesDist[idx][1]**2,0],[0,classesDist[idx][3]**2]]]
        covm.append([[classesDist[idx][1]**2,0],[0,classesDist[idx][3]**2]])
        means.append([classesDist[idx][0],classesDist[idx][2]])

    X = []
    labels = []
    for i, (emotion, (mean, cov)) in enumerate(emotions.items()):
        samples = np.random.multivariate_normal(mean, cov, 1000)
        X.append(samples)
        labels.extend([i] * len(samples))

    X = np.vstack(X)
    n_components = len(emotions)  # Número de estados emocionais
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(X)
    gmm.covariances_ = np.array(covm)
    gmm.means_ = np.array(means)
    return gmm

def main():
    parser = argparse.ArgumentParser(description='Generate Emotion Ranks')
    parser.add_argument('--pathBase', help='Path for valence and arousal dataset', required=True)
    args = parser.parse_args()

    values = None

    images = getFilesInPath(os.path.join(args.pathBase,'images'))
    for i in images:
        fileName = i.split(os.path.sep)[-1][:-4]
        val = np.load(os.path.join(args.pathBase,'annotations',f"{fileName}_val.npy")).astype(np.float64)
        aro = np.load(os.path.join(args.pathBase,'annotations',f"{fileName}_aro.npy")).astype(np.float64)

        values = np.array([val, aro]) if values is None else np.vstack((values, np.array([val, aro])))


    kdtree = KDTree(values)
    gmm = getGMM()
    results = {'kl' : [], 'brier' : [], 'rps' : []}
    for v in values:      
        query_point = v
        _, indices = kdtree.query(v, k=7)

        distros = gmm.predict_proba(values[indices])
        results['kl'].append(kl_divergence(distros[1:4,:], distros[4:,:]))
        results["brier"].append(multiclass_brier_score(distros[1:4,:], distros[4:,:]))
        results["rps"].append(ranked_probability_score(distros[1:4,:], distros[4:,:]))
        
    results["brier"] = f'{np.mean(results["brier"])} ± {np.std(results["brier"])}'
    results["kl"] = f'{np.mean(results["kl"])} ± {np.std(results["kl"])}'
    results["rps"] = f'{np.mean(results["rps"])} ± {np.std(results["rps"])}'
    print("Closer points")
    print(f'KL Divergence: {results["kl"]}')
    print(f'Brier Score: {results["brier"]}')
    print(f'RPS: {results["rps"]}')

    getDists = calculateDistances(values)
    results = {'kl' : [], 'brier' : [], 'rps' : []}
    for i in range(len((-getDists).argsort()[:,:6])):
        farther = values[(-getDists[i]).argsort()[:6]]
        distros = gmm.predict_proba(farther)
        results['kl'].append(kl_divergence(distros[:3,:], distros[3:,:]))
        results["brier"].append(multiclass_brier_score(distros[:3,:], distros[3:,:]))
        results["rps"].append(ranked_probability_score(distros[:3,:], distros[3:,:]))
    
    results["brier"] = f'{np.mean(results["brier"])} ± {np.std(results["brier"])}'
    results["kl"] = f'{np.mean(results["kl"])} ± {np.std(results["kl"])}'
    results["rps"] = f'{np.mean(results["rps"])} ± {np.std(results["rps"])}'

    print("Farther points")
    print(f'KL Divergence: {results["kl"]}')
    print(f'Brier Score: {results["brier"]}')
    print(f'RPS: {results["rps"]}')

if __name__ == "__main__":
    main()