'''
Para estimar o valor de dominância (dominance) com base nos valores de valência (valence) e excitação (arousal) usando os dados do arquivo fornecido, podemos seguir uma abordagem baseada em distribuições gaussianas multivariadas. Aqui está uma solução em Python:
'''
import numpy as np, pandas as pd, argparse
from scipy.stats import multivariate_normal
import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getFilesInPath, printProgressBar

def estimate_dominance(valence, arousal, mean, cov):
    """
    Estima o valor de dominância dado valence e arousal usando uma distribuição Gaussiana multivariada.
    
    Parâmetros:
    - valence: valor de valência observado
    - arousal: valor de excitação observado
    - mean: vetor de médias [valence_mean, arousal_mean, dominance_mean]
    - cov: matriz de covariância 3x3
    
    Retorna:
    - estimated_dominance: valor estimado de dominância
    - confidence: variância da estimativa
    """
    # Particionar a média e covariância
    mean_va = mean[:2]  # média de valence e arousal
    mean_d = mean[2]    # média de dominance
    
    cov_va = cov[:2, :2]  # covariância entre valence e arousal
    cov_va_d = cov[:2, 2] # covariância entre valence/arousal e dominance
    cov_d_va = cov[2, :2] # transposta de cov_va_d
    var_d = cov[2, 2]     # variância de dominance
    
    # Calcular a média condicional
    inv_cov_va = np.linalg.inv(cov_va)
    mean_d_given_va = mean_d + cov_d_va @ inv_cov_va @ (np.array([valence, arousal]) - mean_va)
    
    # Calcular a variância condicional
    var_d_given_va = var_d - cov_d_va @ inv_cov_va @ cov_va_d
    
    return mean_d_given_va, var_d_given_va

def main():
    parser = argparse.ArgumentParser(description='Generate dominance values from valence and arousal')
    parser.add_argument('--pathFile', help='Which model to use', required=True)
    parser.add_argument('--emotionFile', help='Which model to use', required=True)
    args = parser.parse_args()

    # Carregar os dados
    data = pd.read_csv(args.emotionFile)

    # Calcular a média global e covariância global de todas as emoções básicas
    global_mean = data[['valence mean', 'arousal mean', 'dominance mean']].mean().values
    global_cov = np.cov(data[['valence mean', 'arousal mean', 'dominance mean']].values.T)


    files = getFilesInPath(args.pathFile,imagesOnly=True)
    datasetDir = os.path.sep.join(args.pathFile.split(os.path.sep)[:-1])
    for idx, f in enumerate(files):
        printProgressBar(idx + 1, len(files), prefix = 'Processing:', suffix = 'Complete', length = 50)
        filename = f.split(os.path.sep)[-1][:-4]
        valValue = np.load(os.path.join(datasetDir,'annotations',f'{filename}_val.npy'))
        aroValue = np.load(os.path.join(datasetDir,'annotations',f'{filename}_aro.npy'))
        estimated_dom, _ = estimate_dominance(valValue.astype(np.float64), aroValue.astype(np.float64), global_mean, global_cov)
        np.save(os.path.join(datasetDir,'annotations',f'{filename}_dom.npy'), estimated_dom)

if __name__ == "__main__":
    main()

