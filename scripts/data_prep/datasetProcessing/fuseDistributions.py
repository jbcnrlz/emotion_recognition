import numpy as np
import pandas as pd
import argparse
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# Emoções base que queremos preservar nos nomes
UNIVERSAL_EMOTIONS = {'happy', 'sad', 'surprised', 'contempt', 'fearful', 'angry', 'disgusted', 'neutral'}

def bhattacharyya_distance(mu1, var1, mu2, var2):
    """
    Calcula a distância analítica de Bhattacharyya entre duas distribuições Gaussianas
    com matriz de covariância diagonal.
    """
    var_pool = (var1 + var2) / 2.0
    term1 = (1/8) * np.sum((mu1 - mu2)**2 / var_pool)
    term2 = (1/2) * np.sum(np.log(var_pool / np.sqrt(var1 * var2)))
    return term1 + term2

def fuse_emotions_hac(emotions_data, n_clusters=14):
    """
    Realiza a fusão de emoções usando Clusterização Hierárquica e Moment Matching Analítico.
    """
    classes = emotions_data['class'].values
    
    # Extrair médias e variâncias (std^2) garantindo estabilidade numérica
    means = emotions_data[['valence mean', 'arousal mean', 'dominance mean']].values
    stds = emotions_data[['valence std', 'arousal std', 'dominance std']].values
    stds = np.clip(stds, 1e-6, None) # Evita divisão por zero
    vars_ = stds ** 2
    
    n_samples = len(classes)
    
    # 1. Construir matriz de distância condensada para o HAC
    dist_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            dist = bhattacharyya_distance(means[i], vars_[i], means[j], vars_[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
            
    condensed_dist = squareform(dist_matrix)
    
    # 2. Aplicar Clusterização Hierárquica (Critério Ward ou Complete limitam o espalhamento)
    Z = linkage(condensed_dist, method='complete')
    
    # 3. Cortar o dendrograma para obter o número exato de clusters desejados
    labels = fcluster(Z, t=n_clusters, criterion='maxclust')
    
    fused_emotions = []
    
    # 4. Fazer o Moment Matching para cada cluster formado
    for cluster_id in np.unique(labels):
        idx = np.where(labels == cluster_id)[0]
        
        cluster_classes = classes[idx]
        cluster_means = means[idx]
        cluster_vars = vars_[idx]
        
        # --- A. Lógica de Nomenclatura ---
        universals_in_cluster = [c for c in cluster_classes if c in UNIVERSAL_EMOTIONS]
        
        if universals_in_cluster:
            # Se houver emoção universal, ela domina o nome. Se houver mais de uma, junta.
            new_name = " + ".join(universals_in_cluster)
        else:
            # Pega até os 2 primeiros nomes para não ficar gigante
            new_name = " + ".join(cluster_classes[:2])
            if len(cluster_classes) > 2:
                 new_name += " (etc)"
                 
        # --- B. Moment Matching (Cálculo Analítico) ---
        N = len(idx)
        # Nova média é a média das médias
        new_mean = np.sum(cluster_means, axis=0) / N
        
        # Nova variância leva em conta a variância de cada um MAIS o distanciamento da nova média
        new_var = np.zeros(3)
        for i in range(N):
            new_var += cluster_vars[i] + (cluster_means[i] - new_mean)**2
        new_var = new_var / N
        
        new_std = np.sqrt(new_var)
        
        fused_emotions.append({
            'class': new_name,
            'valence mean': new_mean[0],
            'valence std': new_std[0],
            'arousal mean': new_mean[1],
            'arousal std': new_std[1],
            'dominance mean': new_mean[2],
            'dominance std': new_std[2]
        })
        
    return pd.DataFrame(fused_emotions)

def main():
    parser = argparse.ArgumentParser(description='Fuse emotion distributions analytically via HAC')
    parser.add_argument('--distFile', help='Path to input CSV', required=True)    
    parser.add_argument('--outputDistFile', help='Path to output CSV', required=True)
    parser.add_argument('--clusters', help='Number of final emotions (default: 14)', type=int, default=14)
    args = parser.parse_args()
    
    print("Carregando dados...")
    emotions_data = pd.read_csv(args.distFile)
    
    # Tratamento básico
    cols = ['valence mean', 'valence std', 'arousal mean', 'arousal std', 'dominance mean', 'dominance std']
    for col in cols:
        emotions_data[col] = pd.to_numeric(emotions_data[col], errors='coerce')
    emotions_data = emotions_data.dropna(subset=['valence mean', 'arousal mean', 'dominance mean'])
    
    print(f"Iniciando clusterização para {args.clusters} emoções macro...")
    # Uma única chamada resolve o problema inteiro de uma vez
    fused_df = fuse_emotions_hac(emotions_data, n_clusters=args.clusters)
    
    fused_df.to_csv(f"{args.outputDistFile}.csv", index=False)
    print(f"Fusão completa! Salvo em {args.outputDistFile}.csv")

if __name__ == "__main__":
    main()