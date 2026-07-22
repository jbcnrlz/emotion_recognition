import numpy as np, argparse, torch, os, sys, pandas as pd
from tqdm import tqdm
from torchvision import transforms
from sklearn.mixture import GaussianMixture
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from DatasetClasses.AffectNet import AffectNet

# Essa lista serve APENAS para traduzir o label numérico original do AffectNet (Ground Truth)
# Ela NÃO dita mais a ordem das colunas no CSV final.
AFFECTNET_EMOTIONS = ["neutral", "happy", "sad", "surprised", "fear", "disgust", "angry", "contempt"]

# Correlações empíricas. Buscadas por chave (nome), então a ordem não importa.
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

def build_full_covariance_matrix(std_v, std_a, std_d, emotion_name):
    """
    Constrói a matriz de covariância 3x3 incluindo as correlações (Covariância Completa)
    e garante matematicamente que ela seja Positiva Semidefinida (PSD).
    """
    eps = 1e-4
    std_v, std_a, std_d = max(std_v, eps), max(std_a, eps), max(std_d, eps)
    
    corrs = EMPIRICAL_CORRELATIONS.get(emotion_name, {'VD': 0.0, 'AD': 0.0, 'VA': 0.0})
    rho_va, rho_vd, rho_ad = corrs['VA'], corrs['VD'], corrs['AD']
    
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

def calculate_multivariate_likelihood(vaBatch, means, cov_matrices, emotion_names, priors_dict):
    """
    Calcula as probabilidades combinando a Distribuição Normal Multivariada 
    com as Priors empíricas mapeadas por nome.
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
            
    # Normalização
    log_probs_normalized = log_probs - logsumexp(log_probs, axis=1, keepdims=True)
    probs = np.exp(log_probs_normalized)
    
    return probs

def validate_probabilities(probs):
    print("\n--- Validação das Probabilidades ---")
    print(f"Shape: {probs.shape}")
    row_sums = probs.sum(axis=1)
    print(f"Soma por linha (deve ser ~1.0): Min={row_sums.min():.6f}, Max={row_sums.max():.6f}")
    print(f"Valores NaN/Inf: {np.isnan(probs).sum()} / {np.isinf(probs).sum()}")
    print("-" * 36)

# NOVO: A função agora recebe "emotion_names" (a ordem exata lida do seu arquivo)
def outputCSV(probs, vads, path, emos, emotion_names, outputFile):
    new_data = []
    for index, row in enumerate(probs):
        # Mapeia dinamicamente usando a ordem real lida do distroFile
        newRow = {emotion_names[c]: row[c] for c in range(len(emotion_names))}
        newRow.update({
            'valence': vads[index][0],
            'arousal': vads[index][1],
            'dominance': vads[index][2],
            'emotion': emos[index], # Rótulo original Ground Truth
            'path': path[index]
        })
        new_data.append(newRow)
    
    new_df = pd.DataFrame(new_data)
    new_df.to_csv(outputFile, index=False)
    print(f"Arquivo CSV salvo com sucesso: {outputFile}")

def saveRankFile(probs, paths, extensionNameFile=""):
    for idx, p in enumerate(probs):
        splitedPath = paths[idx].split(os.path.sep)
        fileName = splitedPath[-1].split(".")[0]
        annotationsFolder = os.path.join(os.path.sep.join(splitedPath[0:-2]), 'annotations')
        
        os.makedirs(annotationsFolder, exist_ok=True)
        
        with open(os.path.join(annotationsFolder, f"{fileName}_prob_rank_{extensionNameFile}.txt"), 'w') as f:
            f.write(','.join([f"{x:.6f}" for x in p]) + '\n')

def main():
    parser = argparse.ArgumentParser(description='Generate Emotion Ranks')
    parser.add_argument('--pathBase', help='Path for valence and arousal dataset', required=True)
    parser.add_argument('--batchSize', type=int, help='Size of the batch', required=True)
    parser.add_argument('--distroFile', help='Emotion distributions CSV file', required=True)
    parser.add_argument('--typeEstimation', choices=['GMM', 'LIKELIHOOD'], default="LIKELIHOOD")
    parser.add_argument('--validate_probs', action='store_true', help='Validate probability outputs')
    parser.add_argument('--extensionForAnnotations', default="", help='Extension for annotation files')
    args = parser.parse_args()

    # 1. Carregar as distribuições e extrair A ORDEM EXATA das emoções
    df_distro = pd.read_csv(args.distroFile)
    emotion_names_csv = df_distro['class'].tolist()
    classesDist = df_distro.drop(columns=['class']).to_numpy()
    
    if 'neutral' not in emotion_names_csv:
        classesDist = np.vstack((classesDist, np.array([[0.1, 0.05, 0.1, 0.05, 0.1, 0.05]]))) 
        emotion_names_csv.append('neutral')

    print(f"[*] Ordem das emoções detectada no arquivo: {emotion_names_csv}")

    # 2. Construir Médias e Matrizes
    means = []
    cov_matrices = []
    for idx, row in enumerate(classesDist):
        mu_v, std_v, mu_a, std_a, mu_d, std_d = row
        means.append([mu_v, mu_a, mu_d])
        emotion_name = emotion_names_csv[idx]
        cov_matrix = build_full_covariance_matrix(std_v, std_a, std_d, emotion_name)
        cov_matrices.append(cov_matrix)
        
    means = np.array(means)
    cov_matrices = np.array(cov_matrices)

    # Transformações e DataLoader
    data_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    for s in ["val_set"]:
        datasetVal = AffectNet(afectdata=os.path.join(args.pathBase, s), transform=data_transforms, typeExperiment='VAD_EXP', exchangeLabel=None)
        val_loader = torch.utils.data.DataLoader(datasetVal, batch_size=args.batchSize, shuffle=False)

        if args.typeEstimation == "GMM":
            X, labels = [], []
            for i in range(len(means)):
                samples = np.random.multivariate_normal(means[i], cov_matrices[i], 1000)
                X.append(samples)
                labels.extend([i] * len(samples))

            X = np.vstack(X)
            gmm = GaussianMixture(n_components=len(means), covariance_type='full', random_state=42)
            gmm.fit(X)
            gmm.covariances_ = cov_matrices
            gmm.means_ = means

        priors_dict = calculate_empirical_priors(val_loader, AFFECTNET_EMOTIONS)
        outputData, probs, vas, pts, lbls = None, None, None, None, []
        print(f"Iniciando inferência usando método: {args.typeEstimation} com Priors Empíricos")
        
        for data in tqdm(val_loader, desc='Generating Emotion Ranks'):
            _, vaBatch, paths = data
            emotionsLabel = vaBatch[:, -1].numpy().astype(np.uint8)
            vaBatch = vaBatch[:, :-1].numpy() 
            
            if args.typeEstimation == "GMM":
                batch_probs = gmm.predict_proba(vaBatch)
            elif args.typeEstimation == "LIKELIHOOD":
                # 2. Passamos a ordem do CSV (emotion_names_csv) e o Dicionário de Priors (priors_dict)
                batch_probs = calculate_multivariate_likelihood(
                    vaBatch, means, cov_matrices, emotion_names_csv, priors_dict
                )
                
            probs = np.concatenate((probs, batch_probs)) if probs is not None else batch_probs
            vas = np.concatenate((vas, vaBatch)) if vas is not None else vaBatch
            pts = np.concatenate((pts, paths)) if pts is not None else paths
            lbls.extend([AFFECTNET_EMOTIONS[int(x)] if int(x) != 255 else 'affwild' for x in emotionsLabel.tolist()])

        if args.validate_probs and probs is not None:
            validate_probabilities(probs)

        saveRankFile(probs,pts,args.extensionForAnnotations)
        outputCSV(probs, vas, pts, lbls, emotion_names_csv, f'{s}_emotion_distribution.csv')

if __name__ == '__main__':
    main()