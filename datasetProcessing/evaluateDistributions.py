import numpy as np, argparse, torch, os, sys
import pandas as pd
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
from PIL import Image 
from scipy.special import logsumexp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from DatasetClasses.AffectNet import AffectNet, OPTAffectNet


def normal_pdf(x, mu, sigma):
    """
    Calcula o valor da PDF para um ponto x, dada a média mu e o desvio-padrão sigma.
    f(x) = (1 / (sigma * sqrt(2*pi))) * exp(-0.5 * ((x - mu) / sigma)^2)
    """
    # Evita divisão por zero
    if sigma == 0:
        return np.inf if x == mu else 0.0

    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(exponent)


def normal_pdf_stable(x, mu, sigma):
    """
    Versão estável da PDF normal que evita underflow
    """
    epsilon = 1e-8
    sigma = max(sigma, epsilon)
    
    # Para valores muito distantes, retorna probabilidade muito baixa
    z = (x - mu) / sigma
    if np.abs(z) > 10:  # Mais de 10 desvios padrão
        return epsilon
    
    exponent = -0.5 * z ** 2
    normalization = 1.0 / (sigma * np.sqrt(2 * np.pi))
    
    return normalization * np.exp(exponent)


def calculate_likelihood_probabilities(vaBatch, classesDist, use_log_space=True):
    """
    Calcula probabilidades usando likelihood de forma numericamente estável
    
    Args:
        vaBatch: Tensor com valores VAD [batch_size, 3]
        classesDist: Distribuições das classes [n_classes, 6] - [mu_v, std_v, mu_a, std_a, mu_d, std_d]
        use_log_space: Se True, usa espaço logarítmico para evitar underflow
    
    Returns:
        Probabilidades normalizadas [batch_size, n_classes]
    """
    batch_size = vaBatch.shape[0]
    n_classes = classesDist.shape[0]
    
    if use_log_space:
        # Cálculo em espaço logarítmico (mais estável)
        log_probs = np.zeros((batch_size, n_classes))
        
        for c in range(n_classes):
            mu_v, std_v, mu_a, std_a, mu_d, std_d = classesDist[c]
            
            # Evitar std zero adicionando epsilon
            epsilon = 1e-8
            std_v = max(std_v, epsilon)
            std_a = max(std_a, epsilon)
            std_d = max(std_d, epsilon)
            
            # PDF normal em espaço log
            log_pdf_v = -0.5 * ((vaBatch[:, 0] - mu_v) / std_v) ** 2 - np.log(std_v * np.sqrt(2 * np.pi))
            log_pdf_a = -0.5 * ((vaBatch[:, 1] - mu_a) / std_a) ** 2 - np.log(std_a * np.sqrt(2 * np.pi))
            log_pdf_d = -0.5 * ((vaBatch[:, 2] - mu_d) / std_d) ** 2 - np.log(std_d * np.sqrt(2 * np.pi))
            
            # Soma das log-probabilidades (equivalente a multiplicação em espaço normal)
            log_probs[:, c] = log_pdf_v + log_pdf_a + log_pdf_d
        
        # Normalizar usando logsumexp para evitar underflow/overflow
        log_probs_normalized = log_probs - logsumexp(log_probs, axis=1, keepdims=True)
        
        # Converter de volta para probabilidades
        probs = np.exp(log_probs_normalized)
        
    else:
        # Versão em espaço normal (menos estável, mas mais direta)
        probs = np.ones((batch_size, n_classes))
        
        for c in range(n_classes):
            mu_v, std_v, mu_a, std_a, mu_d, std_d = classesDist[c]
            
            # Evitar std zero
            epsilon = 1e-8
            std_v = max(std_v, epsilon)
            std_a = max(std_a, epsilon)
            std_d = max(std_d, epsilon)
            
            # Calcular probabilidades
            prob_v = normal_pdf_stable(vaBatch[:, 0], mu_v, std_v)
            prob_a = normal_pdf_stable(vaBatch[:, 1], mu_a, std_a)
            prob_d = normal_pdf_stable(vaBatch[:, 2], mu_d, std_d)
            
            probs[:, c] = prob_v * prob_a * prob_d
        
        # Normalizar as probabilidades
        probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-8)
    
    return probs


def calculate_likelihood_vectorized(vaBatch, classesDist):
    """
    Versão vetorizada para melhor performance
    """
    batch_size = vaBatch.shape[0]
    n_classes = classesDist.shape[0]
    
    # Expandir dimensões para broadcasting
    va_expanded = vaBatch[:, np.newaxis, :]  # [batch_size, 1, 3]
    means = classesDist[:, [0, 2, 4]]  # [n_classes, 3]
    stds = classesDist[:, [1, 3, 5]]  # [n_classes, 3]
    
    # Evitar std zero
    stds = np.maximum(stds, 1e-8)
    
    # Calcular log-probabilidades de forma vetorizada
    z = (va_expanded - means) / stds  # [batch_size, n_classes, 3]
    log_probs = -0.5 * z ** 2 - np.log(stds * np.sqrt(2 * np.pi))  # [batch_size, n_classes, 3]
    
    # Somar log-probs ao longo das dimensões VAD
    log_probs_sum = np.sum(log_probs, axis=2)  # [batch_size, n_classes]
    
    # Normalizar
    log_probs_normalized = log_probs_sum - logsumexp(log_probs_sum, axis=1, keepdims=True)
    probs = np.exp(log_probs_normalized)
    
    return probs


def validate_probabilities(probs):
    """
    Valida se as probabilidades estão corretas
    """
    print("Validação das Probabilidades:")
    print(f"Shape: {probs.shape}")
    print(f"Soma por linha (deve ser ~1.0): Min={probs.sum(axis=1).min():.6f}, Max={probs.sum(axis=1).max():.6f}")
    print(f"Valores NaN: {np.isnan(probs).sum()}")
    print(f"Valores infinitos: {np.isinf(probs).sum()}")
    print(f"Range das probabilidades: Min={probs.min():.6f}, Max={probs.max():.6f}")
    
    # Verificar se alguma probabilidade é zero para todas as classes
    zero_probs = (probs == 0).all(axis=1).sum()
    print(f"Linhas com todas probabilidades zero: {zero_probs}")


# Lista de emoções corrigida (14 emoções na ordem correta do GMM)
# Ordem: 13 do CSV + 1 (neutral) adicionado no final
#CORRECT_EMOTIONS = ["happy", "contempt", "elated", "hopeful", "surprised", "proud", "loved", "angry", "astonished", "disgusted", "fearful", "sad", "fatigued", "neutral"]
CORRECT_EMOTIONS = ["neutral", "happy", "sad", "surprised", "fear", "disgust", "angry", "contempt"]
AFFECTNET_EMOTIONS = ["neutral", "happy", "sad", "surprised", "fear", "disgust", "angry", "contempt"]


def outputCSV(probs, vads, path, emos, outputFile):
    new_data = []
    
    # Processar cada linha
    for index, row in enumerate(probs):
        # Projetar distribuição de emoções        
        
        # Criar nova linha com a distribuição projetada
        newRow = {}
        for c in range(len(CORRECT_EMOTIONS)):
            newRow[CORRECT_EMOTIONS[c]] = row[c]

        newRow['valence'] = vads[index][0]
        newRow['arousal'] = vads[index][1]
        newRow['dominance'] = vads[index][2]
        newRow['emotion'] = emos[index]
        newRow['path'] = path[index]

        new_data.append(newRow)
    
    # Criar novo DataFrame
    new_df = pd.DataFrame(new_data)
    
    # Salvar novo CSV
    new_df.to_csv(outputFile, index=False)
    
    print(f"Arquivo salvo com sucesso: {outputFile}")
    print(f"Total de registros processados: {len(new_df)}")
    
    return new_df


def saveRankFile(probs,paths,extensionNameFile=""):
    for idx, p in enumerate(probs):
        splitedPath = paths[idx].split(os.path.sep)
        fileName = splitedPath[-1].split(".")[0]
        annotationsFolder = os.path.join(os.path.sep.join(splitedPath[0:-2]),'annotations')
        with open(os.path.join(annotationsFolder,f"{fileName}_prob_rank{extensionNameFile}.txt"),'w') as f:
            joinedProbs = ','.join([str(x) for x in p])
            f.write(joinedProbs+'\n')


def main():
    parser = argparse.ArgumentParser(description='Generate Emotion Ranks')
    parser.add_argument('--pathBase', help='Path for valence and arousal dataset', required=True)
    parser.add_argument('--batchSize', type=int, help='Size of the batch', required=True)
    parser.add_argument('--distroFile', help='Size of the batch', required=True)
    parser.add_argument('--typeEstimation', help='Type estimation',default="GMM", required=True)
    parser.add_argument('--use_log_space', action='store_true', help='Use log space for stability')
    parser.add_argument('--vectorized', action='store_true', help='Use vectorized computation')
    parser.add_argument('--validate_probs', action='store_true', help='Validate probability outputs')
    parser.add_argument('--extensionForAnnotations', default="", help='Extension for annotation files')
    args = parser.parse_args()

    classesDist = pd.read_csv(args.distroFile).drop(columns=['class']).to_numpy()
    # Adiciona o componente 'neutral' no final (13 + 1 = 14)
    classesDist = np.vstack( (classesDist, np.array([[0,0.1,0,0.1,0,0.1]])) )

    data_transforms = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    datasetVal = AffectNet(afectdata=os.path.join(args.pathBase,'train_set'),transform=data_transforms,typeExperiment='VAD_ADJUSTED_EXP',exchangeLabel=None)

    if args.typeEstimation == "GMM":
        idx = -1
        covm = []
        means = []
        for k in classesDist:
            idx += 1
            covm.append([
                [k[1]**2,0,0],
                [0,k[3]**2,0],
                [0,0,k[5]**2]
            ])
            means.append([k[0],k[2],k[4]])

        X = []
        labels = []
        for i in range(len(means)):
            samples = np.random.multivariate_normal(means[i], covm[i], 1000)
            X.append(samples)
            labels.extend([i] * len(samples))

        X = np.vstack(X)
        n_components = len(means) # Número de estados emocionais (14)
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        gmm.fit(X)
        gmm.covariances_ = np.array(covm)
        gmm.means_ = np.array(means)

    val_loader = torch.utils.data.DataLoader(datasetVal, batch_size=args.batchSize, shuffle=False)
    outputData = None
    probs = None
    vas = None
    pts = None
    lbls = []
    
    print(f"Usando método: {args.typeEstimation}")
    if args.typeEstimation == "LIKELIHOOD":
        print(f"Log space: {args.use_log_space}")
        print(f"Vectorized: {args.vectorized}")
    
    for data in val_loader:
        #_, labels, paths, vaBatch = data
        _, vaBatch, paths = data
        emotionsLabel = vaBatch[:,-1].numpy().astype(np.uint8)
        vaBatch = vaBatch[:,:-1]
        
        if args.typeEstimation == "GMM":
            probs = np.concatenate((probs,gmm.predict_proba(vaBatch.numpy()))) if probs is not None else gmm.predict_proba(vaBatch.numpy())
        
        elif args.typeEstimation == "LIKELIHOOD":
            if args.vectorized:
                # Usar versão vetorizada
                batch_probs = calculate_likelihood_vectorized(vaBatch.numpy(), classesDist)
            else:
                # Usar a nova implementação
                batch_probs = calculate_likelihood_probabilities(
                    vaBatch.numpy(), 
                    classesDist, 
                    use_log_space=args.use_log_space
                )
            
            probs = np.concatenate((probs, batch_probs)) if probs is not None else batch_probs
        
        #lbls = np.concatenate((lbls,labels.numpy())) if lbls is not None else labels.numpy()
        outputData = np.concatenate((outputData,vaBatch.numpy())) if outputData is not None else vaBatch.numpy()
        vas = np.concatenate((vas,vaBatch.numpy())) if vas is not None else vaBatch.numpy()
        pts = np.concatenate((pts,paths)) if pts is not None else paths
        lbls = lbls + [AFFECTNET_EMOTIONS[int(x)] if int(x) != 255 else 'affwild' for x in emotionsLabel.tolist()]

    # Validar probabilidades se solicitado
    if args.validate_probs and probs is not None:
        validate_probabilities(probs)

    saveRankFile(probs,pts,args.extensionForAnnotations)
    outputCSV(probs, vas, pts, lbls, 'val_set_emotion_distribution.csv')

    print("Processamento concluído com sucesso!")


if __name__ == '__main__':
    main()