import matplotlib.pyplot as plt, argparse, pandas as pd, os, math
import numpy as np
from matplotlib.ticker import NullFormatter

from sklearn import manifold

def getEmotions(pathDataset,files):
    returnEmotionsData = []
    for f in files:
        filePath = int(f.split(os.path.sep)[-1][:-4])
        express = int(np.load(os.path.join(pathDataset,'annotations','%d_exp.npy' % (filePath))))
        returnEmotionsData.append(express)

    return returnEmotionsData

def main():
    parser = argparse.ArgumentParser(description='Evaluate t-SNE for clusters')
    parser.add_argument('--latentCSV', help='Path for the terms file', required=True, nargs='+')
    parser.add_argument('--components', help='Quantity of components', required=True, type=int)
    parser.add_argument('--pathDataset', help='Path for cluster', required=True)
    parser.add_argument('--csvEmotions', help='Path for resnet pretrained weights', required=True)
    args = parser.parse_args()
    fig = plt.figure()

    ncols = 1 if len(args.latentCSV) == 1 else 2
    nrows = math.ceil(len(args.latentCSV) / 2)
    emotions = np.array(pd.read_csv(args.csvEmotions))
    colorsForPlot = np.array(['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:grey','tab:olive','tab:cyan'])
    for idx, latentFile in enumerate(args.latentCSV):
        tFiles = np.array(pd.read_csv(latentFile))
        classesLabel = tFiles[:,-1]
        emotionsAnnotated = getEmotions(args.pathDataset,classesLabel)
        features = tFiles[:,:-1].astype(np.float32)
        tsne = manifold.TSNE(
                n_components=args.components,
                init="random",
                random_state=0,
                perplexity=100,
                n_iter=300,
        )
        Y = tsne.fit_transform(features)
        ax = fig.add_subplot(nrows,ncols,idx+1)
        ax.scatter(Y[:,0],Y[:,1],c=colorsForPlot[emotionsAnnotated])
        ax.title.set_text(latentFile)
    plt.show()
    '''
    for idxPlot, data in enumerate(Y):
        ax.scatter(x, y, c=color, s=scale, label=color,alpha=0.3, edgecolors='none')
    express = int(np.load(os.path.join(args.pathDataset,'annotations','%d_exp.npy' % (filePath))))
    '''
    print("ok")

if __name__ == '__main__':
    main()