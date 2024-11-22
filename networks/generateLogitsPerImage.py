import argparse, torch, os, sys, numpy as np, pandas as pd, matplotlib.pyplot as plt, shutil
import warnings
from importlib.resources import path
from torchvision import transforms
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from PIL import Image
from matplotlib.patches import Ellipse

warnings.filterwarnings("ignore")

def test():
    parser = argparse.ArgumentParser(description='Extract VA with RESNETEMOTION')
    parser.add_argument('--pathCSV', help='Weights', required=True)
    parser.add_argument('--folderLogits', help='Weights', required=True)
    args = parser.parse_args()
    csvFiles = pd.read_csv(args.pathCSV)
    logits = np.array(csvFiles)[:,:-2]
    paths = np.array(csvFiles)[:,-2:-1]
    labels = np.array(csvFiles)[:,-1]
    emotionsLabels = [ 'neutral', 'happy', 'sad', 'surprise','fear', 'disgust', 'anger', 'contempt']

    classesDist = np.array(
        [
            [0,0.01,0,0.01],   [0.81,0.21,0.51,0.26],  [-0.63,0.23,-0.27,0.34],[0.4,0.3,0.67,0.27], #neutral, happy, sad, surprise
            [-0.64,0.2,0.6,0.32],[-0.6,0.2,0.35,0.41],[-0.43,0.29,0.67,0.27]  #fear, disgust, anger, contempt
        ], dtype = np.float32)

    imageSaved = []

    if os.path.exists(args.folderLogits):
        shutil.rmtree(args.folderLogits)

    os.makedirs(args.folderLogits)
    results = [0,0]
    for idx, l in enumerate(logits):        
        if labels[idx] < len(emotionsLabels):
            results[int(l.argmax() == labels[idx])] += 1
            ax1 = plt.subplot2grid((2,2),(0,0),rowspan=2)
            ax2 = plt.subplot2grid((2,2),(0,1),rowspan=1)
            ax3 = plt.subplot2grid((2,2),(1,1),rowspan=1)
            ax1.bar(emotionsLabels,l)        
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30, ha='right')
            imPlot = plt.imread(paths[idx][0])
            imName = paths[idx][0].split(os.path.sep)[-1].split('.')[0]
            valValue = np.load(os.path.join(os.path.sep.join(paths[idx][0].split(os.path.sep)[:-2]),'annotations','%d_val.npy' % (int(imName))))
            aroValue = np.load(os.path.join(os.path.sep.join(paths[idx][0].split(os.path.sep)[:-2]),'annotations','%d_aro.npy' % (int(imName))))
            ax2.imshow(imPlot)
            ax2.text(150,200,emotionsLabels[labels[idx]],bbox={'facecolor': 'red' if l.argmax() != labels[idx] else 'green', 'alpha': 1, 'pad': 10}) 
            ax3.scatter(classesDist[:,0],classesDist[:,2])
            for idxEm, d in enumerate(classesDist):
                el = Ellipse((d[0],d[2]),width=d[1],height=d[3])
                ax3.add_patch(el)
                ax3.annotate(emotionsLabels[idxEm], (d[0], d[2]))
            ax3.scatter(valValue.astype(np.float64),aroValue.astype(np.float64),c='tab:red')
            imageSaved.append(args.folderLogits + '/%s_%s_logits.png'%('wrong' if l.argmax() != labels[idx] else 'right',imName))
            plt.savefig(imageSaved[-1])
            plt.close()    


if __name__ == '__main__':
    test()