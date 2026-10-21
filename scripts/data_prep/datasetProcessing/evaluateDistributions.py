import numpy as np, argparse, torch, os, sys, pandas as pd
from tqdm import tqdm
from torchvision import transforms
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
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
    
    cov_matrix = np.array([
        [std_v**2, cov_va],
        [cov_va,   std_a**2]
    ], dtype=np.float64)
    
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    eigenvalues = np.maximum(eigenvalues, 1e-6)
    cov_matrix_valid = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    cov_matrix_valid = (cov_matrix_valid + cov_matrix_valid.T) / 2.0
    
    return cov_matrix_valid

def calculate_multivariate_likelihood(vaBatch_2d, means, cov_matrices, emotion_names, priors_dict, temperature=4.0):
    batch_size = vaBatch_2d.shape[0]
    n_classes = len(means)
    probs = np.zeros((batch_size, n_classes))
    
    neutral_idx = emotion_names.index('neutral') if 'neutral' in emotion_names else -1
    
    log_posteriors = np.full((batch_size, n_classes), -np.inf)
    activations = np.zeros((batch_size, n_classes))
    
    for c in range(n_classes):
        if c == neutral_idx:
            continue
            
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
            
    max_activation = np.max(activations, axis=1)
    max_activation = np.clip(max_activation, 0.0, 1.0)
    residual_prob = 1.0 - max_activation
    
    valid_logs = np.where(log_posteriors != -np.inf, log_posteriors, -1e10)
    max_log = np.max(valid_logs, axis=1, keepdims=True)
    
    unnormalized_posteriors = np.exp(valid_logs - max_log)
    
    if neutral_idx != -1:
        unnormalized_posteriors[:, neutral_idx] = 0.0
        
    sum_posteriors = np.sum(unnormalized_posteriors, axis=1, keepdims=True)
    sum_posteriors[sum_posteriors == 0] = 1e-10
    
    for c in range(n_classes):
        if c == neutral_idx:
            probs[:, c] = residual_prob
        else:
            probs[:, c] = max_activation * (unnormalized_posteriors[:, c] / sum_posteriors.flatten())
            
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    
    return probs

def test_temperatures(val_loader, means, cov_matrices, emotion_names, priors_dict, temperatures=[1.0, 2.0, 4.0, 8.0, 10.0]):
    print("\n--- Testando Valores de Temperature Scaling ---")
    data_iter = iter(val_loader)
    _, vaBatch_raw, _ = next(data_iter)
    vaBatch_2d = vaBatch_raw[:, :2].numpy()
    
    neutral_idx = emotion_names.index('neutral') if 'neutral' in emotion_names else -1
    if neutral_idx == -1:
        return

    print(f"{'Temperatura':<12} | {'Média Prob. Neutro':<20} | {'% Neutro Dominante':<25}")
    print("-" * 65)
    for t in temperatures:
        probs = calculate_multivariate_likelihood(vaBatch_2d, means, cov_matrices, emotion_names, priors_dict, temperature=t)
        mean_neutral_prob = np.mean(probs[:, neutral_idx])
        predictions = np.argmax(probs, axis=1)
        neutral_pred_pct = np.mean(predictions == neutral_idx) * 100
        print(f"{t:<12.2f} | {mean_neutral_prob:<20.4f} | {neutral_pred_pct:.2f}%")
    print("-" * 65)

def plot_2d_predictions_with_ellipses(val_loader, means, cov_matrices, emotion_names, priors_dict, temperature=4.0, n_std=2.0):
    print(f"\n[*] Gerando gráfico 2D com Elipses de Covariância (temperature={temperature})...")
    data_iter = iter(val_loader)
    _, vaBatch_raw, _ = next(data_iter)
    vaBatch_2d = vaBatch_raw[:, :2].numpy()
    
    probs = calculate_multivariate_likelihood(vaBatch_2d, means, cov_matrices, emotion_names, priors_dict, temperature=temperature)
    predictions = np.argmax(probs, axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap('tab10')
    
    for c in range(len(emotion_names)):
        idx = (predictions == c)
        if np.any(idx):
            ax.scatter(vaBatch_2d[idx, 0], vaBatch_2d[idx, 1], color=cmap(c), label=f"Pred: {emotion_names[c]}", alpha=0.6, edgecolors='w', s=40)
            
    neutral_idx = emotion_names.index('neutral') if 'neutral' in emotion_names else -1
    
    for c in range(len(emotion_names)):
        if c == neutral_idx: continue
        mean_x, mean_y = means[c]
        cov = cov_matrices[c]
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
        width, height = 2 * n_std * np.sqrt(eigvals)
        
        ellip = Ellipse(xy=(mean_x, mean_y), width=width, height=height, angle=angle, edgecolor=cmap(c), facecolor=cmap(c), alpha=0.15, linewidth=2)
        ax.add_patch(ellip)
        ellip_edge = Ellipse(xy=(mean_x, mean_y), width=width, height=height, angle=angle, edgecolor=cmap(c), facecolor='none', alpha=0.8, linewidth=1.5, linestyle='--')
        ax.add_patch(ellip_edge)
        
        ax.scatter(mean_x, mean_y, color=cmap(c), marker='X', s=150, edgecolor='black')
        ax.text(mean_x + 0.03, mean_y + 0.03, emotion_names[c], fontsize=10, weight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.axvline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_title(fr"Classificação VA com Elipses de Covariância ({n_std}$\sigma$) | Temperatura = {temperature}")
    ax.set_xlabel("Valência")
    ax.set_ylabel("Arousal")
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"va_ellipses_temp_{temperature}.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_neutral_heatmap_temperature(means, cov_matrices, emotion_names, priors_dict, temperature=4.0, resolution=200):
    print(f"\n[*] Gerando Heatmap do Neutro no espaço VA (temperature={temperature})...")
    neutral_idx = emotion_names.index('neutral') if 'neutral' in emotion_names else -1
    if neutral_idx == -1: return

    v_vals = np.linspace(-1.2, 1.2, resolution)
    a_vals = np.linspace(-1.2, 1.2, resolution)
    V, A = np.meshgrid(v_vals, a_vals)
    grid_points = np.c_[V.ravel(), A.ravel()]
    
    probs = calculate_multivariate_likelihood(grid_points, means, cov_matrices, emotion_names, priors_dict, temperature=temperature)
    neutral_probs = probs[:, neutral_idx].reshape(V.shape)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(V, A, neutral_probs, levels=50, cmap='magma', alpha=0.9)
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Probabilidade do Neutro', rotation=270, labelpad=20)
    
    for c in range(len(emotion_names)):
        if c == neutral_idx: continue
        mean_x, mean_y = means[c]
        ax.scatter(mean_x, mean_y, color='cyan', marker='X', s=100, edgecolor='black', zorder=5)
        ax.text(mean_x + 0.03, mean_y + 0.03, emotion_names[c], fontsize=9, color='white', weight='bold', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1))

    ax.axhline(0, color='white', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(0, color='white', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_title(f"Mapa de Calor: Probabilidade Residual do Neutro (Temperatura = {temperature})")
    ax.set_xlabel("Valência")
    ax.set_ylabel("Arousal")
    
    plt.tight_layout()
    plt.savefig(f"heatmap_neutral_temp_{temperature}.png", dpi=300, bbox_inches='tight')
    plt.close()

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
    
    new_df = pd.DataFrame(new_data)
    new_df.to_csv(outputFile, index=False)
    print(f"Arquivo CSV salvo com sucesso: {outputFile}")

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
    parser.add_argument('--typeEstimation', choices=['GMM', 'LIKELIHOOD'], default="LIKELIHOOD")
    parser.add_argument('--validate_probs', action='store_true', help='Validate probability outputs')
    parser.add_argument('--extensionForAnnotations', default="", help='Extension for annotation files')
    parser.add_argument('--temperature', type=float, default=4.0, help='Temperature Scaling factor para a Gaussiana (padrão: 4.0)')
    args = parser.parse_args()

    df_distro = pd.read_csv(args.distroFile)
    emotion_names_csv = df_distro['class'].tolist()
    classesDist = df_distro.drop(columns=['class']).to_numpy()
    
    if 'neutral' not in emotion_names_csv:
        classesDist = np.vstack((classesDist, np.array([[0.1, 0.05, 0.1, 0.05, 0.1, 0.05]]))) 
        emotion_names_csv.append('neutral')

    print(f"[*] Ordem das emoções detectada no arquivo: {emotion_names_csv}")

    means = []
    cov_matrices = []
    for idx, row in enumerate(classesDist):
        mu_v, std_v, mu_a, std_a = row[:4] 
        means.append([mu_v, mu_a])
        emotion_name = emotion_names_csv[idx]
        
        cov_matrix = build_full_covariance_matrix(std_v, std_a, emotion_name)
        cov_matrices.append(cov_matrix)
        
    means = np.array(means)
    cov_matrices = np.array(cov_matrices)

    data_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
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

            X = np.vstack(X)
            gmm = GaussianMixture(n_components=len(means), covariance_type='full', random_state=42)
            gmm.fit(X)
            gmm.covariances_ = cov_matrices
            gmm.means_ = means

        priors_dict = calculate_empirical_priors(val_loader, AFFECTNET_EMOTIONS)
        
        # Bloco de Validação e Teste (roda no dataset de validação)
        if args.typeEstimation == "LIKELIHOOD" and s == "val_set":
            test_temperatures(val_loader, means, cov_matrices, emotion_names_csv, priors_dict)
            plot_2d_predictions_with_ellipses(val_loader, means, cov_matrices, emotion_names_csv, priors_dict, temperature=args.temperature)
            plot_neutral_heatmap_temperature(means, cov_matrices, emotion_names_csv, priors_dict, temperature=args.temperature)

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
                batch_probs = calculate_multivariate_likelihood(
                    vaBatch_2d, means, cov_matrices, emotion_names_csv, priors_dict, temperature=args.temperature
                )
                
            probs = np.concatenate((probs, batch_probs)) if probs is not None else batch_probs
            vas = np.concatenate((vas, vaBatch_full)) if vas is not None else vaBatch_full
            pts = np.concatenate((pts, paths)) if pts is not None else paths
            lbls.extend([AFFECTNET_EMOTIONS[int(x)] if int(x) != 255 else 'affwild' for x in emotionsLabel.tolist()])

        if args.validate_probs and probs is not None:
            validate_probabilities(probs)

        saveRankFile(probs, pts, args.extensionForAnnotations)
        outputCSV(probs, vas, pts, lbls, emotion_names_csv, f'{s}_emotion_distribution.csv')

if __name__ == '__main__':
    main()