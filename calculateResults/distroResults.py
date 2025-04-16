import numpy as np, pandas as pd, os, sys, argparse
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

def ranked_probability_score(y_true, y_pred):
    """
    Calcula o Ranked Probability Score (RPS) para problemas multiclasse ordinais.

    Args:
        y_true (np.array): Rótulos reais em formato one-hot ou probabilidades (shape [n_samples, n_classes]).
        y_pred (np.array): Probabilidades previstas (shape [n_samples, n_classes]).

    Returns:
        float: Média do RPS sobre todas as amostras.
    """
    n_samples, n_classes = y_true.shape
    
    # Converte one-hot para probabilidades se necessário
    if np.any(y_true.sum(axis=1) == 1) and np.all(np.isin(y_true, [0, 1])):
        y_true = y_true.astype(float)
    
    # Calcula as probabilidades acumuladas
    cum_y_true = np.cumsum(y_true, axis=1)
    cum_y_pred = np.cumsum(y_pred, axis=1)
    
    # Calcula o RPS para cada amostra
    rps = np.mean(np.sum((cum_y_true - cum_y_pred) ** 2, axis=1) / (n_classes - 1))
    
    return rps


def kl_divergence(y_true, y_pred, epsilon=1e-10):
    """
    Calcula a Divergência Kullback-Leibler (KL) entre distribuições verdadeiras e previstas.
    
    Args:
        y_true (np.array): Rótulos reais (shape [n_samples, n_classes]).
        y_pred (np.array): Probabilidades previstas (shape [n_samples, n_classes]).
        epsilon (float): Pequeno valor para evitar log(0).
    
    Returns:
        float: Média da KL-divergência sobre todas as amostras.
    """
    # Garante que as distribuições são válidas (somatório = 1)
    y_true = np.clip(y_true, epsilon, 1)  # Evita zeros
    y_true = y_true / np.sum(y_true, axis=1, keepdims=True)
    y_pred = np.clip(y_pred, epsilon, 1)
    y_pred = y_pred / np.sum(y_pred, axis=1, keepdims=True)
    
    # Calcula KL-divergência para cada amostra
    kl = np.sum(y_true * np.log(y_true / y_pred), axis=1)
    return np.mean(kl)

def multiclass_brier_score(y_true, y_pred):
    """
    Calcula o Brier Score para problemas multiclasse.
    
    Args:
        y_true (np.array): Rótulos reais (shape [n_samples, n_classes]).
        y_pred (np.array): Probabilidades previstas (shape [n_samples, n_classes]).
    
    Returns:
        float: Brier Score (quanto menor, melhor).
    """
    return np.mean(np.sum((y_true - y_pred) ** 2, axis=1))

def loadAnnotation(path):
    returnData = None
    with open(path,'r') as f:
        for line in f.readlines():
            returnData = [float(x) for x in line.split(',')]

    return returnData

def main():
    parser = argparse.ArgumentParser(description='Generate Emotion Ranks')
    parser.add_argument('--pathBase', help='Path for valence and arousal dataset', required=True)
    parser.add_argument('--csvFiles', help='Size of the batch', required=True)
    args = parser.parse_args()


    estimated = pd.read_csv(args.csvFiles)
    dados = np.array(estimated)[:,:-1].astype(np.float64)
    arquivos = np.array(estimated)[:,-1]
    gtCol = []
    for idx, d in enumerate(dados):
        fileName = arquivos[idx].split(os.path.sep)[-1][:-4]
        gt = loadAnnotation(os.path.join(args.pathBase,'annotations',f"{fileName}_prob_rank.txt"))
        gtCol.append(gt)

    gtCol = np.array(gtCol)
    print(f'KL Divergence: {kl_divergence(gtCol, dados)}')
    print(f'Brier Score: {multiclass_brier_score(gtCol, dados)}')
    print(f'RPS: {ranked_probability_score(gtCol, dados)}')

if __name__ == "__main__":
    main()