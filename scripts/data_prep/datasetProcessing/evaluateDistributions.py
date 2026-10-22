import numpy as np, argparse, torch, os, sys, pandas as pd
from tqdm import tqdm
from torchvision import transforms
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from DatasetClasses.AffectNet import AffectNet

AFFECTNET_EMOTIONS = ["neutral", "happy", "sad", "surprised", "fear", "disgust", "angry", "contempt"]

EMPIRICAL_CORRELATIONS = {
    'neutral':   {'VA': 0.05},
    'happy':     {'VA': 0.15},
    'sad':       {'VA': 0.75},
    'surprised': {'VA': 0.35},
    'fear':      {'VA': 0.65}, 
    'disgust':   {'VA': 0.25}, 
    'angry':     {'VA': 0.55},
    'contempt':  {'VA': 0.25}
}

def calculate_empirical_priors(data_loader, affectnet_emotion_names):
    print("[*] Calculando Priors empíricos diretamente do dataset...")
    num_classes = len(affectnet_emotion_names)
    counts = np.zeros(num_classes)
    total_valid = 0
    
    for _, vaBatch, _ in data_loader:
        emotions = vaBatch[:, -1].numpy().astype(int)
        for emo in emotions:
            if emo != 255 and emo < num_classes:
                counts[emo] += 1
                total_valid += 1
                
    if total_valid == 0:
        print("[!] Aviso: Nenhuma label válida encontrada. Usando distribuição uniforme.")
        priors_array = np.ones(num_classes) / num_classes
    else:
        priors_array = counts / total_valid
        
    priors_array = np.maximum(priors_array, 1e-6)
    priors_array = priors_array / np.sum(priors_array) 
    
    priors_dict = {affectnet_emotion_names[i]: priors_array[i] for i in range(num_classes)}
    
    print("--- Priors Empíricos Calculados ---")
    for emo_name, p in priors_dict.items():
        print(f"Classe {emo_name}: {p*100:.2f}%")
    print("-----------------------------------")
    
    return priors_dict

def build_full_covariance_matrix(std_v, std_a, emotion_name):
    eps = 1e-4
    std_v, std_a = max(std_v, eps), max(std_a, eps)
    corrs = EMPIRICAL_CORRELATIONS.get(emotion_name, {'VA': 0.0})
    rho_va = corrs['VA']
    cov_va = rho_va * std_v * std_a
    
    cov_matrix = np.array([[std_v**2, cov_va], [cov_va, std_a**2]], dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    eigenvalues = np.maximum(eigenvalues, 1e-6)
    cov_matrix_valid = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    return (cov_matrix_valid + cov_matrix_valid.T) / 2.0

def calculate_multivariate_likelihood(vaBatch_2d, means, cov_matrices, emotion_names, priors_dict, temperature=4.0):
    batch_size = vaBatch_2d.shape[0]
    n_classes = len(means)
    probs = np.zeros((batch_size, n_classes))
    neutral_idx = emotion_names.index('neutral') if 'neutral' in emotion_names else -1
    
    log_posteriors = np.full((batch_size, n_classes), -np.inf)
    activations = np.zeros((batch_size, n_classes))
    
    for c in range(n_classes):
        if c == neutral_idx: continue
        emo_name = emotion_names[c]
        prior = priors_dict.get(emo_name, 1e-6)
        try:
            rv = multivariate_normal(mean=means[c], cov=cov_matrices[c], allow_singular=True)
            max_logpdf = rv.logpdf(means[c])
            current_logpdf = rv.logpdf(vaBatch_2d)
            activations[:, c] = np.exp((current_logpdf - max_logpdf) / temperature)
            log_posteriors[:, c] = (current_logpdf + np.log(prior)) / temperature
        except Exception as e:
            print(f"Aviso: Erro na classe {c} ({emo_name}): {e}")
            
    max_activation = np.clip(np.max(activations, axis=1), 0.0, 1.0)
    residual_prob = 1.0 - max_activation
    
    valid_logs = np.where(log_posteriors != -np.inf, log_posteriors, -1e10)
    max_log = np.max(valid_logs, axis=1, keepdims=True)
    unnormalized_posteriors = np.exp(valid_logs - max_log)
    
    if neutral_idx != -1: unnormalized_posteriors[:, neutral_idx] = 0.0
    sum_posteriors = np.sum(unnormalized_posteriors, axis=1, keepdims=True)
    sum_posteriors[sum_posteriors == 0] = 1e-10
    
    for c in range(n_classes):
        if c == neutral_idx: probs[:, c] = residual_prob
        else: probs[:, c] = max_activation * (unnormalized_posteriors[:, c] / sum_posteriors.flatten())
            
    return probs / np.sum(probs, axis=1, keepdims=True)

# === NOVA FUNÇÃO DE SIMILARIDADE ANGULAR E INTENSIDADE ===
def calculate_angular_similarity(vaBatch_2d, means, emotion_names, sigma_neutral=0.15):
    """
    Abordagem baseada em Coordenadas Polares:
    - O Neutro é determinado pela 'Intensidade' (r = distância da origem).
    - As outras emoções são distribuídas pela Similaridade de Cosseno (direção angular).
    """
    batch_size = vaBatch_2d.shape[0]
    n_classes = len(means)
    probs = np.zeros((batch_size, n_classes))
    neutral_idx = emotion_names.index('neutral') if 'neutral' in emotion_names else -1
    
    # 1. Calcula a Intensidade (r)
    r = np.linalg.norm(vaBatch_2d, axis=1)
    
    # 2. Neutro baseado puramente na intensidade r (Decaimento Gaussiano invertido)
    prob_neutral = np.exp(- (r**2) / (2 * sigma_neutral**2))
    
    active_weights = np.zeros((batch_size, n_classes))
    
    for c in range(n_classes):
        if c == neutral_idx:
            probs[:, c] = prob_neutral
            continue
            
        mu = means[c]
        norm_mu = np.linalg.norm(mu)
        if norm_mu == 0: continue
            
        # 3. Similaridade de Cosseno (direção)
        safe_r = np.where(r == 0, 1e-10, r)
        dot_product = np.dot(vaBatch_2d, mu)
        cos_sim = dot_product / (safe_r * norm_mu)
        
        # Filtra direções opostas (ângulo > 90 graus resulta em 0 peso)
        active_weights[:, c] = np.maximum(0, cos_sim)
        
    # 4. Distribuição da Probabilidade Restante
    sum_weights = np.sum(active_weights, axis=1, keepdims=True)
    sum_weights[sum_weights == 0] = 1e-10
    
    residual = 1.0 - prob_neutral
    
    for c in range(n_classes):
        if c != neutral_idx:
            probs[:, c] = residual * (active_weights[:, c] / sum_weights.flatten())
            
    return probs / np.sum(probs, axis=1, keepdims=True)

def validate_probabilities(probs):
    print("\n--- Validação das Probabilidades ---")
    row_sums = probs.sum(axis=1)
    print(f"Soma por linha (deve ser ~1.0): Min={row_sums.min():.6f}, Max={row_sums.max():.6f}")
    print("-" * 36)

def outputCSV(probs, vads_full, path, emos, emotion_names, outputFile):
    new_data = []
    for index, row in enumerate(probs):
        newRow = {emotion_names[c]: row[c] for c in range(len(emotion_names))}
        newRow.update({'valence': vads_full[index][0], 'arousal': vads_full[index][1], 'dominance': vads_full[index][2], 'emotion': emos[index], 'path': path[index]})
        new_data.append(newRow)
    pd.DataFrame(new_data).to_csv(outputFile, index=False)
    print(f"Arquivo CSV salvo: {outputFile}")

def saveRankFile(probs, paths, extensionNameFile=""):
    for idx, p in enumerate(probs):
        splitedPath = paths[idx].split(os.path.sep)
        fileName = splitedPath[-1].split(".")[0]
        annotationsFolder = os.path.join(os.path.sep.join(splitedPath[0:-2]), 'annotations')
        os.makedirs(annotationsFolder, exist_ok=True)
        with open(os.path.join(annotationsFolder, f"{fileName}_prob_rank{extensionNameFile}.txt"), 'w') as f:
            f.write(','.join([f"{x:.6f}" for x in p]) + '\n')

def main():
    parser = argparse.ArgumentParser(description='Generate Emotion Ranks')
    parser.add_argument('--pathBase', help='Path for valence and arousal dataset', required=True)
    parser.add_argument('--batchSize', type=int, help='Size of the batch', required=True)
    parser.add_argument('--distroFile', help='Emotion distributions CSV file', required=True)
    # ANGULAR adicionado as opções!
    parser.add_argument('--typeEstimation', choices=['GMM', 'LIKELIHOOD', 'ANGULAR'], default="ANGULAR")
    parser.add_argument('--validate_probs', action='store_true', help='Validate probability outputs')
    parser.add_argument('--extensionForAnnotations', default="", help='Extension for annotation files')
    parser.add_argument('--temperature', type=float, default=4.0, help='Temperature para LIKELIHOOD')
    parser.add_argument('--sigmaNeutral', type=float, default=0.15, help='Raio de intensidade para o Neutro na estimativa ANGULAR')
    args = parser.parse_args()

    df_distro = pd.read_csv(args.distroFile)
    emotion_names_csv = df_distro['class'].tolist()
    classesDist = df_distro.drop(columns=['class']).to_numpy()
    
    if 'neutral' not in emotion_names_csv:
        classesDist = np.vstack((classesDist, np.array([[0.1, 0.05, 0.1, 0.05, 0.1, 0.05]]))) 
        emotion_names_csv.append('neutral')

    means, cov_matrices = [], []
    for idx, row in enumerate(classesDist):
        mu_v, std_v, mu_a, std_a = row[:4] 
        means.append([mu_v, mu_a])
        cov_matrices.append(build_full_covariance_matrix(std_v, std_a, emotion_names_csv[idx]))
        
    means, cov_matrices = np.array(means), np.array(cov_matrices)

    data_transforms = transforms.Compose([
        transforms.Resize((256, 256)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    for s in ["val_set", "train_set"]:
        datasetVal = AffectNet(afectdata=os.path.join(args.pathBase, s), transform=data_transforms, typeExperiment='VAD_EXP', exchangeLabel=None)
        val_loader = torch.utils.data.DataLoader(datasetVal, batch_size=args.batchSize, shuffle=False)

        if args.typeEstimation == "GMM":
            X, labels = [], []
            for i in range(len(means)):
                samples = np.random.multivariate_normal(means[i], cov_matrices[i], 1000)
                X.append(samples)
                labels.extend([i] * len(samples))
            gmm = GaussianMixture(n_components=len(means), covariance_type='full', random_state=42)
            gmm.fit(np.vstack(X))
            gmm.covariances_ = cov_matrices
            gmm.means_ = means

        priors_dict = calculate_empirical_priors(val_loader, AFFECTNET_EMOTIONS)
        
        outputData, probs, vas, pts, lbls = None, None, None, None, []
        print(f"\nIniciando inferência ({s}) usando método: {args.typeEstimation}")
        
        for data in tqdm(val_loader, desc='Generating Emotion Ranks'):
            _, vaBatch_raw, paths = data
            emotionsLabel = vaBatch_raw[:, -1].numpy().astype(np.uint8)
            vaBatch_full = vaBatch_raw[:, :-1].numpy() 
            vaBatch_2d = vaBatch_raw[:, :2].numpy()
            
            if args.typeEstimation == "GMM":
                batch_probs = gmm.predict_proba(vaBatch_2d)
            elif args.typeEstimation == "LIKELIHOOD":
                batch_probs = calculate_multivariate_likelihood(vaBatch_2d, means, cov_matrices, emotion_names_csv, priors_dict, args.temperature)
            elif args.typeEstimation == "ANGULAR":
                batch_probs = calculate_angular_similarity(vaBatch_2d, means, emotion_names_csv, args.sigmaNeutral)
                
            probs = np.concatenate((probs, batch_probs)) if probs is not None else batch_probs
            vas = np.concatenate((vas, vaBatch_full)) if vas is not None else vaBatch_full
            pts = np.concatenate((pts, paths)) if pts is not None else paths
            lbls.extend([AFFECTNET_EMOTIONS[int(x)] if int(x) != 255 else 'affwild' for x in emotionsLabel.tolist()])

        if args.validate_probs: validate_probabilities(probs)

        saveRankFile(probs, pts, args.extensionForAnnotations)
        outputCSV(probs, vas, pts, lbls, emotion_names_csv, f'{s}_emotion_distribution_{args.typeEstimation}.csv')

if __name__ == '__main__':
    main()