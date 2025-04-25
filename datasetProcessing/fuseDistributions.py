
import numpy as np, argparse, torch, os, sys, pandas as pd
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from DatasetClasses.AffectNet import AffectNet
from scipy.stats import multivariate_normal
from scipy.integrate import dblquad

def min_pdf(x, y, dist1, dist2):
    point = np.array([x, y])
    return min(dist1.pdf(point), dist2.pdf(point))

def main():
    parser = argparse.ArgumentParser(description='Fuse distributions')
    parser.add_argument('--pathBase', help='Path for valence and arousal dataset', required=True)
    parser.add_argument('--batchSize', type=int, help='Size of the batch', required=True)
    args = parser.parse_args()
    data_transforms = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    datasetVal = AffectNet(afectdata=os.path.join(args.pathBase,'val_set'),transform=data_transforms,typeExperiment='BOTH',exchangeLabel=None)
    fullCSV = pd.read_csv("hajer_categ.CSV")
    classesDist = fullCSV[['valence mean','valence std','arousal mean','arousal std']].to_numpy()

    emotions = {}
    for c in fullCSV['class']:
        emotions[c] = []
    
    idx = -1
    covm = []
    means = []
    distribuitions = {}
    for k in emotions:
        idx += 1
        means.append([classesDist[idx][0],classesDist[idx][2]])
        covm.append([[classesDist[idx][1]**2,0],[0,classesDist[idx][3]**2]])
        distribuitions[k] = multivariate_normal(mean=np.array([classesDist[idx][0],classesDist[idx][2]]), cov=np.array([[classesDist[idx][1]**2,0],[0,classesDist[idx][3]**2]]))

    x_lower, x_upper = -1,1
    y_lower, y_upper = -1,1

    distsInter = {}
    i = 0
    newDistros = []
    for idx1, d in enumerate(distribuitions):
        distsInter[d] = {}
        for idx2, d2 in enumerate(distribuitions):
            if d != d2:
                print(i)
                intersection_area, _ = dblquad( lambda x, y : min(distribuitions[d].pdf(np.array([x,y])),distribuitions[d2].pdf(np.array([x,y]))), x_lower, x_upper, y_lower,y_upper)
                ttArea1, _ = dblquad( lambda y, x: distribuitions[d].pdf([x, y]) , x_lower, x_upper, y_lower,y_upper)
                ttArea2, _ = dblquad( lambda y, x: distribuitions[d2].pdf([x, y]), x_lower, x_upper, y_lower,y_upper)
                intersection_percentage = (intersection_area / min(ttArea1, ttArea2 ))
                if (intersection_percentage >= 0.80):
                    newDistros.append([[d,d2],[means[idx1],means[idx2]],[covm[idx1],covm[idx2]]])
                    print('oi')
                    break
                distsInter[d][d2] = intersection_percentage
                i += 1

    print(distsInter)               
        

def filter():
    import pandas as pd

    # Carregar o arquivo CSV enviado pelo usuário
    file_path = "hajer_categ.CSV"
    df = pd.read_csv(file_path)

    # Exibir as primeiras linhas para entender a estrutura
    df.head()

    # Filtrar as músicas onde "valence mean" e "arousal mean" são negativos
    df_filtered = df[(df["valence mean"] > 0) & (df["arousal mean"] < 0)]

    # Exibir as primeiras linhas do resultado
    print(df_filtered.head())
    df_filtered.drop(columns=['dominance mean','dominance std'],axis=1, inplace=True)
    df_filtered.to_csv("filtered.csv")   


if __name__ == '__main__':
    filter()