import argparse, torch, os, sys, numpy as np, pandas as pd, matplotlib.pyplot as plt
from importlib.resources import path
from torchvision import transforms
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from PIL import Image

def test():
    parser = argparse.ArgumentParser(description='Extract VA with RESNETEMOTION')
    parser.add_argument('--pathCSV', help='Weights', required=True)
    args = parser.parse_args()
    csvFiles = pd.read_csv(args.pathCSV)
    logits = np.array(csvFiles)[:,:-2]
    paths = np.array(csvFiles)[:,-2:-1]
    labels = np.array(csvFiles)[:,-1]
    emotionsLabels = [ 'neutral', 'happy', 'sad', 'surprise','fear', 'disgust', 'anger', 'contempt' ]
    imageSaved = []
    for idx, l in enumerate(logits):
        fig, ax = plt.subplots(1,2)        
        ax[0].bar(emotionsLabels,l)        
        ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=30, ha='right')
        imPlot = plt.imread(paths[idx][0])
        imName = paths[idx][0].split(os.path.sep)[-1].split('.')[0]
        ax[1].imshow(imPlot)
        ax[1].text(150,200,emotionsLabels[labels[idx]],bbox={'facecolor': 'red' if l.argmax() != labels[idx] else 'green', 'alpha': 1, 'pad': 10})        
        imageSaved.append('imagesLogits/%s_%s_logits.png'%('wrong' if l.argmax() != labels[idx] else 'right',imName))
        plt.savefig(imageSaved[-1])
        plt.close()    


if __name__ == '__main__':
    test()