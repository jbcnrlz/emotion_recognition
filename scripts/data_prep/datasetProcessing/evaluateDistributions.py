import numpy as np, argparse, torch, os, sys, pandas as pd
from tqdm import tqdm
from torchvision import transforms
from sklearn.mixture import GaussianMixture
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
while PROJECT_ROOT != os.path.dirname(PROJECT_ROOT):
    if os.path.exists(os.path.join(PROJECT_ROOT, "DatasetClasses")) or os.path.exists(os.path.join(PROJECT_ROOT, "pyproject.toml")):
        break
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from DatasetClasses.AffectNet import AffectNet

# Essa lista serve APENAS para traduzir o label numérico original do AffectNet (Ground Truth)
# Ela NÃO dita mais a ordem das colunas no CSV final.
AFFECTNET_EMOTIONS = ["neutral", "happy", "sad", "surprised", "fear", "disgust", "angry", "contempt"]

# Correlações empíricas (3D VAD). Buscadas por chave (nome), então a ordem não importa.
EMPIRICAL_CORRELATIONS = {
    'neutral':   {'VD': 0.10, 'AD': 0.10, 'VA': 0.05},
    'happy':     {'VD': 0.75, 'AD': 0.25, 'VA': 0.15},
    'sad':       {'VD': 0.65, 'AD': 0.55, 'VA': 0.75},
    'surprised': {'VD': 0.05, 'AD': 0.65, 'VA': 0.35},
    'fear':      {'VD': -0.25, 'AD': 0.85, 'VA': 0.65}, 
    'disgust':   {'VD': -0.55, 'AD': 0.35, 'VA': 0.25}, 
    'angry':     {'VD': -0.45, 'AD': 0.75, 'VA': 0.55},
    'contempt':  {'VD': 0.35, 'AD': 0.45, 'VA': 0.25}
}

def calculate_empirical_priors(data_loader, affectnet_emotion_names):
    """
    Varre o dataset e retorna um DICIONÁRIO mapeando o NOME da emoção 
    à sua probabilidade a priori, garantindo que a ordem não importe.
    """
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
    
    # Mapeia o nome da emoção para a probabilidade
    priors_dict = {affectnet_emotion_names[i]: priors_array[i] for i in range(num_classes)}
    
    print("--- Priors Empíricos Calculados ---")
    for emo_name, p in priors_dict.items():
        print(f"Classe {emo_name}: {p*100:.2f}%")
    print("-----------------------------------")
    
    return priors_dict

def build_full_covariance_matrix_3d(std_v, std_a, std_d, emotion_name):
    """
    Constrói a matriz de covariância 3x3 incluindo as correlações VAD (Covariância Completa)
    e garante matematicamente que ela seja Positiva Semidefinida (PSD).
    """
    eps = 1e-4
    std_v, std_a, std_d = max(std_v, eps), max(std_a, eps), max(std_d, eps)
    
    corrs = EMPIRICAL_CORRELATIONS.get(emotion_name, {'VD': 0.0, 'AD': 0.0, 'VA': 0.0})
    rho_va, rho_vd, rho_ad = corrs.get('VA', 0.0), corrs.get('VD', 0.0), corrs.get('AD', 0.0)
    
    cov_va = rho_va * std_v * std_a
    cov_vd = rho_vd * std_v * std_d
    cov_ad = rho_ad * std_a * std_d
    
    cov_matrix = np.array([
        [std_v**2, cov_va,   cov_vd],
        [cov_va,   std_a**2, cov_ad],
        [cov_vd,   cov_ad,   std_d**2]
    ], dtype=np.float64)
    
    # Projeção para PSD (Evita Erro de Matriz Singular)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    eigenvalues = np.maximum(eigenvalues, 1e-6)
    cov_matrix_valid = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    cov_matrix_valid = (cov_matrix_valid + cov_matrix_valid.T) / 2.0
    
    return cov_matrix_valid

def build_full_covariance_matrix_2d(std_v, std_a, emotion_name):
    """
    Constrói a matriz de covariância 2x2 para Valence e Arousal.
    """
    eps = 1e-4
    std_v, std_a = max(std_v, eps), max(std_a, eps)
    corrs = EMPIRICAL_CORRELATIONS.get(emotion_name, {'VA': 0.0})
    rho_va = corrs.get('VA', 0.0)
    cov_va = rho_va * std_v * std_a
    
    cov_matrix = np.array([[std_v**2, cov_va], [cov_va, std_a**2]], dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    eigenvalues = np.maximum(eigenvalues, 1e-6)
    cov_matrix_valid = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    return (cov_matrix_valid + cov_matrix_valid.T) / 2.0

def build_full_covariance_matrix(std_v, std_a, std_d_or_emotion, emotion_name=None):
    """
    Função de compatibilidade:
    - Se chamada com 3 argumentos: build_full_covariance_matrix(std_v, std_a, emotion_name) -> 2D
    - Se chamada com 4 argumentos: build_full_covariance_matrix(std_v, std_a, std_d, emotion_name) -> 3D
    """
    if emotion_name is None:
        return build_full_covariance_matrix_2d(std_v, std_a, std_d_or_emotion)
    else:
        return build_full_covariance_matrix_3d(std_v, std_a, std_d_or_emotion, emotion_name)

def calculate_multivariate_likelihood_old(vaBatch, means, cov_matrices, emotion_names, priors_dict):
    """
    [FORMA ANTIGA / ORIGINAL - 3D VAD]:
    Calcula as probabilidades combinando a Distribuição Normal Multivariada 3D
    com as Priors empíricas mapeadas por nome através do Teorema de Bayes direto.
    """
    batch_size = vaBatch.shape[0]
    n_classes = len(means)
    log_probs = np.zeros((batch_size, n_classes))
    
    for c in range(n_classes):
        emo_name = emotion_names[c]
        
        # Busca a prior exata pelo nome. Se for uma emoção "fundida" que não existe
        # no AffectNet original, assume um valor ínfimo para não enviesar.
        prior = priors_dict.get(emo_name, 1e-6)
        
        try:
            rv = multivariate_normal(mean=means[c], cov=cov_matrices[c], allow_singular=True)
            # TEOREMA DE BAYES COMPLETO
            log_probs[:, c] = rv.logpdf(vaBatch) + np.log(prior)
        except Exception as e:
            print(f"Aviso: Matriz singular ou erro na classe {c} ({emo_name}): {e}")
            log_probs[:, c] = -1e10 
            
    # Normalização via logsumexp
    log_probs_normalized = log_probs - logsumexp(log_probs, axis=1, keepdims=True)
    probs = np.exp(log_probs_normalized)
    
    return probs

def calculate_multivariate_likelihood_2d(vaBatch_2d, means, cov_matrices, emotion_names, priors_dict, temperature=4.0):
    """
    [FORMA 2D - VA COM TEMPERATURA E RESÍDUO NEUTRO]:
    Calcula as probabilidades em 2D com fator de temperatura e
    cálculo de resíduo neutro baseado na ativação máxima das outras emoções.
    """
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

def calculate_multivariate_likelihood(vaBatch, means, cov_matrices, emotion_names, priors_dict, temperature=None):
    """
    Wrapper unificado de verossimilhança multivariada:
    - Se vaBatch tem 3 dimensões (VAD) ou temperature é None, utiliza a forma antiga (3D).
    - Se vaBatch tem 2 dimensões (VA) e temperature é fornecida, utiliza a forma 2D com temperatura.
    """
    if vaBatch.shape[1] >= 3 or temperature is None:
        return calculate_multivariate_likelihood_old(vaBatch, means, cov_matrices, emotion_names, priors_dict)
    else:
        return calculate_multivariate_likelihood_2d(vaBatch, means, cov_matrices, emotion_names, priors_dict, temperature=temperature)

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
    print(f"Shape: {probs.shape}")
    row_sums = probs.sum(axis=1)
    print(f"Soma por linha (deve ser ~1.0): Min={row_sums.min():.6f}, Max={row_sums.max():.6f}")
    print(f"Valores NaN/Inf: {np.isnan(probs).sum()} / {np.isinf(probs).sum()}")
    print("-" * 36)

def outputCSV(probs, vads_full, path, emos, emotion_names, outputFile):
    new_data = []
    for index, row in enumerate(probs):
        newRow = {emotion_names[c]: row[c] for c in range(len(emotion_names))}
        newRow.update({
            'valence': vads_full[index][0],
            'arousal': vads_full[index][1],
            'dominance': vads_full[index][2],
            'emotion': emos[index],
            'path': path[index]
        })
        new_data.append(newRow)
    pd.DataFrame(new_data).to_csv(outputFile, index=False)
    print(f"Arquivo CSV salvo com sucesso: {outputFile}")

def saveRankFile(probs, paths, extensionNameFile=""):
    for idx, p in enumerate(probs):
        splitedPath = paths[idx].split(os.path.sep)
        fileName = splitedPath[-1].split(".")[0]
        annotationsFolder = os.path.join(os.path.sep.join(splitedPath[0:-2]), 'annotations')
        os.makedirs(annotationsFolder, exist_ok=True)
        
        if extensionNameFile:
            suffix = f"_{extensionNameFile}" if not extensionNameFile.startswith('_') else extensionNameFile
        else:
            suffix = ""
            
        with open(os.path.join(annotationsFolder, f"{fileName}_prob_rank{suffix}.txt"), 'w') as f:
            f.write(','.join([f"{x:.6f}" for x in p]) + '\n')

def main():
    parser = argparse.ArgumentParser(description='Generate Emotion Ranks')
    parser.add_argument('--pathBase', help='Path for valence and arousal dataset', required=True)
    parser.add_argument('--batchSize', type=int, help='Size of the batch', required=True)
    parser.add_argument('--distroFile', help='Emotion distributions CSV file', required=True)
    parser.add_argument('--typeEstimation', 
                        choices=['GMM', 'LIKELIHOOD', 'LIKELIHOOD_OLD', 'LIKELIHOOD_ORIGINAL', 'LIKELIHOOD_3D', 'LIKELIHOOD_2D', 'ANGULAR'], 
                        default="LIKELIHOOD",
                        help="Método de estimativa: LIKELIHOOD_OLD / LIKELIHOOD_3D (forma antiga Bayesiana VAD 3D), LIKELIHOOD / LIKELIHOOD_2D (2D com temperatura e resíduo), GMM ou ANGULAR")
    parser.add_argument('--oldLikelihood', action='store_true', 
                        help='Habilita a forma antiga de estimativa com likelihood (3D VAD Bayesiana original)')
    parser.add_argument('--likelihoodVersion', choices=['old', 'new', '3d', '2d'], default=None,
                        help="Versão de likelihood: 'old'/'3d' para a forma antiga, 'new'/'2d' para a forma 2D com temperatura")
    parser.add_argument('--validate_probs', action='store_true', help='Validate probability outputs')
    parser.add_argument('--extensionForAnnotations', default="", help='Extension for annotation files')
    parser.add_argument('--temperature', type=float, default=4.0, help='Temperature para LIKELIHOOD (método 2D)')
    parser.add_argument('--sigmaNeutral', type=float, default=0.15, help='Raio de intensidade para o Neutro na estimativa ANGULAR')
    parser.add_argument('--splits', nargs='+', default=["val_set", "train_set"], help='Splits do dataset a processar (padrão: val_set train_set)')
    args = parser.parse_args()

    # Identifica se deve usar a forma antiga
    is_old_likelihood = (
        args.oldLikelihood or
        (args.likelihoodVersion in ['old', '3d']) or
        (args.typeEstimation in ['LIKELIHOOD_OLD', 'LIKELIHOOD_ORIGINAL', 'LIKELIHOOD_3D', 'LIKELIHOOD_VAD'])
    )

    # 1. Carregar as distribuições e extrair A ORDEM EXATA das emoções
    df_distro = pd.read_csv(args.distroFile)
    emotion_names_csv = df_distro['class'].tolist()
    classesDist = df_distro.drop(columns=['class']).to_numpy()
    
    if 'neutral' not in emotion_names_csv:
        classesDist = np.vstack((classesDist, np.array([[0.1, 0.05, 0.1, 0.05, 0.1, 0.05]]))) 
        emotion_names_csv.append('neutral')

    print(f"[*] Ordem das emoções detectada no arquivo: {emotion_names_csv}")

    # 2. Construir Médias e Matrizes (2D e 3D)
    means_2d, cov_matrices_2d = [], []
    means_3d, cov_matrices_3d = [], []
    
    for idx, row in enumerate(classesDist):
        emotion_name = emotion_names_csv[idx]
        mu_v, std_v, mu_a, std_a = row[:4] 
        means_2d.append([mu_v, mu_a])
        cov_matrices_2d.append(build_full_covariance_matrix_2d(std_v, std_a, emotion_name))

        if len(row) >= 6:
            mu_d, std_d = row[4], row[5]
        else:
            mu_d, std_d = 0.0, 0.25
        means_3d.append([mu_v, mu_a, mu_d])
        cov_matrices_3d.append(build_full_covariance_matrix_3d(std_v, std_a, std_d, emotion_name))
        
    means_2d, cov_matrices_2d = np.array(means_2d), np.array(cov_matrices_2d)
    means_3d, cov_matrices_3d = np.array(means_3d), np.array(cov_matrices_3d)

    # Transformações e DataLoader
    data_transforms = transforms.Compose([
        transforms.Resize((256, 256)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    for s in args.splits:
        set_dir = os.path.join(args.pathBase, s)
        if not os.path.exists(set_dir):
            print(f"[*] Split '{s}' não encontrado em '{args.pathBase}'. Pulando...")
            continue

        datasetVal = AffectNet(afectdata=set_dir, transform=data_transforms, typeExperiment='VAD_EXP', exchangeLabel=None)
        val_loader = torch.utils.data.DataLoader(datasetVal, batch_size=args.batchSize, shuffle=False)

        if args.typeEstimation == "GMM":
            means_gmm = means_3d if is_old_likelihood else means_2d
            covs_gmm = cov_matrices_3d if is_old_likelihood else cov_matrices_2d

            X, labels = [], []
            for i in range(len(means_gmm)):
                samples = np.random.multivariate_normal(means_gmm[i], covs_gmm[i], 1000)
                X.append(samples)
                labels.extend([i] * len(samples))
            gmm = GaussianMixture(n_components=len(means_gmm), covariance_type='full', random_state=42)
            gmm.fit(np.vstack(X))
            gmm.covariances_ = covs_gmm
            gmm.means_ = means_gmm

        priors_dict = calculate_empirical_priors(val_loader, AFFECTNET_EMOTIONS)
        
        outputData, probs, vas, pts, lbls = None, None, None, None, []
        
        if is_old_likelihood:
            method_desc = "LIKELIHOOD_OLD (3D VAD Bayesiana Original com Priors Empíricos)"
            est_tag = "LIKELIHOOD_OLD"
        elif args.typeEstimation == "GMM":
            method_desc = "GMM (Gaussian Mixture Model)"
            est_tag = "GMM"
        elif args.typeEstimation == "ANGULAR":
            method_desc = "ANGULAR (Similaridade de Cosseno + Decaimento Radial)"
            est_tag = "ANGULAR"
        else:
            method_desc = f"LIKELIHOOD (2D VA com Temperature={args.temperature} e Resíduo Neutro)"
            est_tag = "LIKELIHOOD"

        print(f"\nIniciando inferência ({s}) usando método: {method_desc}")
        
        for data in tqdm(val_loader, desc=f'Generating Emotion Ranks [{s}]'):
            _, vaBatch_raw, paths = data
            emotionsLabel = vaBatch_raw[:, -1].numpy().astype(np.uint8)
            vaBatch_full = vaBatch_raw[:, :-1].numpy() 
            vaBatch_2d = vaBatch_raw[:, :2].numpy()
            vaBatch_3d = vaBatch_full
            
            if is_old_likelihood:
                batch_probs = calculate_multivariate_likelihood_old(
                    vaBatch_3d, means_3d, cov_matrices_3d, emotion_names_csv, priors_dict
                )
            elif args.typeEstimation == "GMM":
                batch_probs = gmm.predict_proba(vaBatch_3d if is_old_likelihood else vaBatch_2d)
            elif args.typeEstimation == "ANGULAR":
                batch_probs = calculate_angular_similarity(vaBatch_2d, means_2d, emotion_names_csv, args.sigmaNeutral)
            else: # LIKELIHOOD (2D)
                batch_probs = calculate_multivariate_likelihood_2d(
                    vaBatch_2d, means_2d, cov_matrices_2d, emotion_names_csv, priors_dict, args.temperature
                )
                
            probs = np.concatenate((probs, batch_probs)) if probs is not None else batch_probs
            vas = np.concatenate((vas, vaBatch_full)) if vas is not None else vaBatch_full
            pts = np.concatenate((pts, paths)) if pts is not None else paths
            lbls.extend([AFFECTNET_EMOTIONS[int(x)] if int(x) != 255 and int(x) < len(AFFECTNET_EMOTIONS) else 'affwild' for x in emotionsLabel.tolist()])

        if args.validate_probs: validate_probabilities(probs)

        saveRankFile(probs, pts, args.extensionForAnnotations)
        outputCSV(probs, vas, pts, lbls, emotion_names_csv, f'{s}_emotion_distribution_{est_tag}.csv')

if __name__ == '__main__':
    main()
