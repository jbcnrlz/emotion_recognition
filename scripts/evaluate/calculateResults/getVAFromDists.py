import numpy as np, os, sys, argparse, pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error as rmse
from PIL import Image  # Para carregar imagens
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getFilesInPath

class GMM_ReverseEstimator:
    def __init__(self, means, covariances, weights=None):
        """
        Inicializa o estimador com os parâmetros do GMM.
        
        Parâmetros:
        means : array-like (n_components, 2)
            Médias de cada componente no formato [valence, arousal]
        covariances : array-like (n_components, 2, 2)
            Matrizes de covariância de cada componente
        weights : array-like (n_components,), opcional
            Pesos de cada componente no GMM (padrão: uniforme)
        """
        self.means = np.array(means)
        self.covariances = np.array(covariances)
        self.n_components = len(means)
        self.weights = np.ones(self.n_components)/self.n_components if weights is None else np.array(weights)
        
        # Pré-computa as distribuições normais
        self.distributions = [
            multivariate_normal(mean=means[i], cov=covariances[i])
            for i in range(self.n_components)
        ]
    
    def negative_log_likelihood(self, x, target_probs):
        """
        Calcula a log-verossimilhança negativa para otimização.
        
        Parâmetros:
        x : array-like (2,)
            Valores [valence, arousal] atuais
        target_probs : array-like (n_components,)
            Probabilidades alvo para cada componente
            
        Retorna:
        float: Valor negativo da log-verossimilhança ponderada
        """
        valence, arousal = x
        point = np.array([valence, arousal])
        
        # Calcula a densidade de probabilidade para cada componente
        probs = np.array([dist.pdf(point) for dist in self.distributions])
        
        # Pondera pelas probabilidades alvo e pesos do GMM
        weighted_probs = target_probs * self.weights * probs
        log_likelihood = np.log(weighted_probs.sum())
        
        # Retorna o negativo para minimização
        return -log_likelihood
    
    def estimate(self, target_probs, initial_guess=None):
        """
        Estima valence e arousal que melhor explicam as probabilidades alvo.
        
        Parâmetros:
        target_probs : array-like (n_components,)
            Probabilidades alvo para cada componente
        initial_guess : array-like (2,), opcional
            Palpite inicial para [valence, arousal] (padrão: média ponderada)
            
        Retorna:
        dict: Resultado com valores estimados e informações da otimização
        """
        # Usa a média ponderada como palpite inicial se não for fornecido
        if initial_guess is None:
            initial_guess = np.sum(target_probs[:, np.newaxis] * self.means, axis=0)
        
        # Otimização para minimizar a log-verossimilhança negativa
        result = minimize(
            fun=self.negative_log_likelihood,
            x0=initial_guess,
            args=(target_probs),
            method='L-BFGS-B',
            bounds=[(-1, 1), (-1, 1)]  # Valence e arousal tipicamente entre -1 e 1
        )
        return np.array([result.x[0],result.x[1]])
        '''
        return {
            'valence': result.x[0],
            'arousal': result.x[1],
            'success': result.success,
            'message': result.message,
            'log_likelihood': -result.fun
        }
        '''

def getCovVariance():
    classesDist = np.array([
        [0,0.01,0,0.01],
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
    avgs = np.zeros((13,2))
    covs = np.zeros((13,2,2))
    for idx, k in enumerate(classesDist):
        covs[idx] = np.array([[classesDist[idx][1]**2,0],[0,classesDist[idx][3]**2]])
        avgs[idx] = np.array([classesDist[idx][0],classesDist[idx][2]])

    return avgs, covs

def estimate_valence_arousal(probabilities):
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
   
    means = []
    for c in classesDist:
        means.append([c[0],c[2]])

    probabilities = np.asarray(probabilities)
    means = np.asarray(means)
    
    # Verifica se as probabilidades somam ~1
    if not np.isclose(np.sum(probabilities), 1.0, atol=1e-3):
        raise ValueError("As probabilidades devem somar aproximadamente 1.")
    
    # Média ponderada das médias (estimativa de valence e arousal)
    weighted_mean = np.sum(probabilities[:, np.newaxis] * means, axis=0)
    valence_est, arousal_est = weighted_mean
    
    return np.array([valence_est, arousal_est])

def getGTProbs(pathFile):
    probs = None
    with open(pathFile,'r') as f:
        for fi in f:
            probs = np.array(list(map(float,fi.split(','))))
    return probs

def plot_valence_arousal(files, vas, estimatedGT, estimatedModel):
    for idx, f in enumerate(files):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))        
        # Valence (V) e Arousal (A) reais
        ax1.scatter(vas[idx, 0], vas[idx, 1], c='blue', label='Real (V, A)', alpha=0.6, marker='o')    
        # Valence e Arousal estimados (GT)
        ax1.scatter(estimatedGT[idx, 0], estimatedGT[idx, 1], c='green', label='Estimated (GT)', alpha=0.6, marker='s')        
        # Valence e Arousal estimados (Modelo)
        ax1.scatter(estimatedModel[idx, 0], estimatedModel[idx, 1], c='red', label='Estimated (Model)', alpha=0.6, marker='^')

        ax1.set_xlim(-1, 1)
        ax1.set_ylim(-1, 1)

        ax1.set_xlabel('Valence')
        ax1.set_ylabel('Arousal')
        ax1.set_title('Valence vs Arousal for Images')
        ax1.legend()
        ax1.grid(True)
        ax1.set_aspect('equal', 'box')
    
        # Subgráfico 3: Imagem da face
        if isinstance(f, str):  # Se for um caminho de arquivo
            img = Image.open(f)
        else:  # Se for um array NumPy (imagem já carregada)
            img = f

        ax2.imshow(img)
        ax2.axis('off')  # Remove os eixos
        ax2.set_title('Face Image')
        plt.tight_layout()
        # Salva o gráfico
        plt.savefig(os.path.join('estimatedVSGT', f"{os.path.basename(f)[:-4]}.png"))
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='Generate Emotion Ranks')
    parser.add_argument('--pathBase', help='Path for valence and arousal dataset', required=True)
    parser.add_argument('--csvEmotions', help='Path for valence and arousal dataset', required=True)
    args = parser.parse_args()

    values = None
    vas = None
    vasEstimated = None
    csv = np.array(pd.read_csv(args.csvEmotions))
    files = csv[:,-1]
    emotions = csv[:,:-1].astype(np.float64)

    images = getFilesInPath(os.path.join(args.pathBase,'images'))
    for i in images:
        fileName = i.split(os.path.sep)[-1][:-4]
        probs = getGTProbs(os.path.join(args.pathBase,'annotations',f"{fileName}_prob_rank.txt"))
        values = probs if values is None else np.vstack((values, probs))

        val = np.load(os.path.join(args.pathBase,'annotations',f"{fileName}_val.npy")).astype(np.float64)
        aro = np.load(os.path.join(args.pathBase,'annotations',f"{fileName}_aro.npy")).astype(np.float64)

        vas = np.array([val, aro]) if vas is None else np.vstack((vas, np.array([val, aro])))

        for idx, f in enumerate(files):
            if fileName in f.split(os.path.sep)[-1][:-4]:
                vasEstimated = emotions[idx] if vasEstimated is None else np.vstack((vasEstimated, emotions[idx]))
                break

    avgsM,covM = getCovVariance()
    estims = GMM_ReverseEstimator(avgsM,covM)

    estimatedGT = None
    estimatedGTVS = None
    estimatedModel = None
    for idx, v in enumerate(values):
        estimatedGT = estimate_valence_arousal(v) if estimatedGT is None else np.vstack((estimatedGT, estimate_valence_arousal(v)))
        estimatedModel = estimate_valence_arousal(vasEstimated[idx]) if estimatedModel is None else np.vstack((estimatedModel, estimate_valence_arousal(vasEstimated[idx])))
        estimatedGTVS = estims.estimate(vasEstimated[idx]) if estimatedGTVS is None else np.vstack((estimatedGTVS,estims.estimate(vasEstimated[idx])))

    print("RMSE: ", rmse(vas, estimatedGT))
    print("RMSE: ", rmse(vas, estimatedModel))
    print("RMSE: ", rmse(vas, estimatedGTVS))

    plot_valence_arousal(files, vas, estimatedGT, estimatedModel)


if __name__ == "__main__":
    main()