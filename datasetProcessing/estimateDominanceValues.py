import numpy as np
import pandas as pd
import argparse
import os
import sys
from scipy.stats import multivariate_normal

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# NOTA PARA O ARTIGO: Documente de onde estas correlações empíricas vieram.
# Se foram extraídas de literatura prévia ou de um dataset de validação, cite a fonte.
EMPIRICAL_CORRELATIONS = {
    'neutral': {'VD': 0.10, 'AD': 0.10, 'VA': 0.05},
    'happy': {'VD': 0.75, 'AD': 0.25, 'VA': 0.15},
    'contempt': {'VD': 0.35, 'AD': 0.45, 'VA': 0.25},
    'surprised': {'VD': 0.05, 'AD': 0.65, 'VA': 0.35},
    'angry': {'VD': -0.45, 'AD': 0.75, 'VA': 0.55},
    'disgusted': {'VD': -0.55, 'AD': 0.35, 'VA': 0.25},
    'fearful': {'VD': -0.25, 'AD': 0.85, 'VA': 0.65},
    'sad': {'VD': 0.65, 'AD': 0.55, 'VA': 0.75}
}

def estimate_dominance_regression(valence_obs, arousal_obs, emotion_stats):
    """
    Estima dominance usando regressão linear múltipla de forma vetorizada.
    Suporta tanto valores escalares quanto arrays do NumPy (processamento em lote).
    """
    μ_V, μ_A, μ_D = emotion_stats['means']['V'], emotion_stats['means']['A'], emotion_stats['means']['D']
    σ_V, σ_A, σ_D = emotion_stats['stds']['V'], emotion_stats['stds']['A'], emotion_stats['stds']['D']
    ρ_VD, ρ_AD, ρ_VA = emotion_stats['correlations']['VD'], emotion_stats['correlations']['AD'], emotion_stats['correlations']['VA']
    
    # Previne que a correlação chegue a 1.0 ou -1.0 (evita divisão por zero/singularidade)
    ρ_VA_safe = np.clip(ρ_VA, -0.99, 0.99)
    denominator = 1 - ρ_VA_safe**2 
    
    # Fórmulas analíticas para os coeficientes beta
    β1 = (σ_D / max(σ_V, 1e-6)) * ((ρ_VD - ρ_VA_safe * ρ_AD) / denominator)
    β2 = (σ_D / max(σ_A, 1e-6)) * ((ρ_AD - ρ_VA_safe * ρ_VD) / denominator)
    β0 = μ_D - β1 * μ_V - β2 * μ_A
    
    return β0 + β1 * valence_obs + β2 * arousal_obs

def prepare_emotion_data(data):
    """
    Prepara os dados, calculando as matrizes de covariância necessárias para o 
    cálculo de likelihood Multivariado (Respondendo ao ponto levantado pelo Revisor 4).
    """
    emotion_data = {}
    neutral_in_data = 'neutral' in data['class'].values
    
    for _, row in data.iterrows():
        emotion = row['class']
        
        # Busca a correlação ou assume independência (0.0) para emoções fundidas
        corrs = EMPIRICAL_CORRELATIONS.get(emotion, {'VD': 0.0, 'AD': 0.0, 'VA': 0.0})
        
        # Construção da matriz de covariância 2D [Valence, Arousal]
        # Cov(V, A) = ρ_VA * σ_V * σ_A
        cov_VA = corrs['VA'] * row['valence std'] * row['arousal std']
        cov_matrix_2d = np.array([
            [max(row['valence std']**2, 1e-6), cov_VA],
            [cov_VA, max(row['arousal std']**2, 1e-6)]
        ])
        
        emotion_data[emotion] = {
            'means': {'V': row['valence mean'], 'A': row['arousal mean'], 'D': row['dominance mean']},
            'stds': {'V': row['valence std'], 'A': row['arousal std'], 'D': row['dominance std']},
            'correlations': corrs,
            'cov_matrix_2d': cov_matrix_2d
        }
        
    # Fallback para caso 'neutral' não esteja no arquivo CSV
    if not neutral_in_data:
        corrs_neutral = EMPIRICAL_CORRELATIONS['neutral']
        cov_VA_neutral = corrs_neutral['VA'] * 0.01 * 0.01
        emotion_data['neutral'] = {
            'means': {'V': 0.1, 'A': 0.1, 'D': 0.1},
            'stds': {'V': 0.01, 'A': 0.01, 'D': 0.01},
            'correlations': corrs_neutral,
            'cov_matrix_2d': np.array([[0.0001, cov_VA_neutral], [cov_VA_neutral, 0.0001]])
        }
        print("✅ Emoção 'neutral' adicionada com valores padrão.")
        
    return emotion_data

def estimate_dominance_batch(valence_arousal, emotion_data, emotion_names):
    """
    Cálculo 100% vetorizado para máxima eficiência.
    valence_arousal: array NumPy de shape (N, 2)
    """
    n_points = valence_arousal.shape[0]
    n_emotions = len(emotion_names)
    
    likelihoods = np.zeros((n_points, n_emotions))
    d_estimates = np.zeros((n_points, n_emotions))
    
    # Extrair colunas inteiras para cálculos vetorizados
    v_obs = valence_arousal[:, 0]
    a_obs = valence_arousal[:, 1]
    
    for i, emotion in enumerate(emotion_names):
        stats = emotion_data[emotion]
        
        # 1. Likelihood usando Distribuição Normal Multivariada
        mean_va = np.array([stats['means']['V'], stats['means']['A']])
        try:
            rv = multivariate_normal(mean=mean_va, cov=stats['cov_matrix_2d'], allow_singular=True)
            likelihoods[:, i] = rv.pdf(valence_arousal)
        except np.linalg.LinAlgError:
            # Proteção contra matrizes corrompidas/singulares
            likelihoods[:, i] = 1e-10
            
        # 2. Estimativas de Regressão (calcula para todas as imagens de uma vez)
        d_estimates[:, i] = estimate_dominance_regression(v_obs, a_obs, stats)
        
    # 3. Ponderação Bayesiana (Posteriors)
    sum_likelihoods = np.sum(likelihoods, axis=1, keepdims=True) + 1e-12
    posteriors = likelihoods / sum_likelihoods
    
    # 4. Combinação (Média ponderada pelo mapa de posteriors)
    final_dominance = np.sum(posteriors * d_estimates, axis=1)
    
    # 5. Restringe a saída final ao intervalo realista [-1, 1]
    return np.clip(final_dominance, -1.0, 1.0)

def main():
    parser = argparse.ArgumentParser(description='Generate dominance values from valence and arousal')
    parser.add_argument('--pathFile', help='Path to directory containing .npy files', required=True)
    parser.add_argument('--emotionFile', help='Path to Emotion distributions CSV file', required=True)
    args = parser.parse_args()

    # Carregar estatísticas base
    data = pd.read_csv(args.emotionFile)
    
    # Preparar dicionário computacional
    emotion_data = prepare_emotion_data(data)
    emotion_names = list(emotion_data.keys())
    
    print(f"🎭 Emoções carregadas: {', '.join(emotion_names)}")
    print(f"📊 Total de emoções: {len(emotion_names)}")
    
    # Mapeamento de diretórios
    files = [f for f in os.listdir(args.pathFile) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    dataset_dir = os.path.dirname(args.pathFile.rstrip(os.sep))
    annotations_dir = os.path.join(dataset_dir, 'annotations')
    
    if not os.path.exists(annotations_dir):
        print(f"⚠️ Diretório de anotações não encontrado: {annotations_dir}")
        print("Criando diretório...")
        os.makedirs(annotations_dir, exist_ok=True)
    
    print(f"📁 Processando {len(files)} imagens de {args.pathFile}")
    
    # Processamento em lote
    batch_size = 5000 # Aumentado pois agora é vetorizado e muito mais leve
    
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i + batch_size]
        
        valence_values = []
        arousal_values = []
        valid_files = []
        
        # Leitura dos arquivos NPY do batch
        for filename in batch_files:
            base_name = os.path.splitext(filename)[0]
            val_path = os.path.join(annotations_dir, f'{base_name}_val.npy')
            aro_path = os.path.join(annotations_dir, f'{base_name}_aro.npy')
            
            if os.path.exists(val_path) and os.path.exists(aro_path):
                try:
                    val_value = np.load(val_path)
                    aro_value = np.load(aro_path)
                    
                    # Extrai o valor caso o npy tenha salvo como array de 1 elemento
                    if isinstance(val_value, np.ndarray): val_value = val_value.item()
                    if isinstance(aro_value, np.ndarray): aro_value = aro_value.item()
                        
                    valence_values.append(val_value)
                    arousal_values.append(aro_value)
                    valid_files.append(base_name)
                except Exception as e:
                    print(f"Erro ao ler {base_name}: {e}")
                    continue
        
        if not valid_files:
            continue
            
        # Converter para array 2D: shape (N, 2)
        va_pairs = np.column_stack((valence_values, arousal_values)).astype(np.float32)
        
        # Mágica Vetorizada acontece aqui
        dominance_estimates = estimate_dominance_batch(va_pairs, emotion_data, emotion_names)
        
        # Salvar resultados
        for j, base_name in enumerate(valid_files):
            np.save(os.path.join(annotations_dir, f'{base_name}_dom.npy'), 
                   dominance_estimates[j].astype(np.float32))
            
        print(f"✅ Batch {i//batch_size + 1} processado: {len(valid_files)} anotações salvas.")
        
        if i == 0 and len(dominance_estimates) > 0:
            print(f"📈 Estatísticas do dominance no primeiro batch:")
            print(f"   Mínimo: {np.min(dominance_estimates):.3f}")
            print(f"   Máximo: {np.max(dominance_estimates):.3f}")
            print(f"   Média:  {np.mean(dominance_estimates):.3f}")
            print(f"   Desvio: {np.std(dominance_estimates):.3f}")

if __name__ == "__main__":
    main()