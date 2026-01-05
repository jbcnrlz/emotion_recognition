import numpy as np
import pandas as pd
import argparse
import os
import sys
from scipy.linalg import inv
from scipy.stats import norm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# Correlações otimizadas baseadas na análise dos dados
OPTIMIZED_CORRELATIONS = {
    'neutral': {'VD': 0.10, 'AD': 0.10, 'VA': 0.05},  # Baixas correlações para emoção neutra
    'happy': {'VD': 0.75, 'AD': 0.25, 'VA': 0.15},
    'contempt': {'VD': 0.35, 'AD': 0.45, 'VA': 0.25},
    'surprised': {'VD': 0.05, 'AD': 0.65, 'VA': 0.35},
    'angry': {'VD': -0.45, 'AD': 0.75, 'VA': 0.55},
    'disgusted': {'VD': -0.55, 'AD': 0.35, 'VA': 0.25},
    'fearful': {'VD': -0.25, 'AD': 0.85, 'VA': 0.65},
    'sad': {'VD': 0.65, 'AD': 0.55, 'VA': 0.75}
}

def estimate_dominance_regression(emotion, valence_obs, arousal_obs, emotion_stats):
    """
    Estima dominance usando regressão linear multivariada com correlações VAD
    """
    # Extrair estatísticas da emoção
    μ_V, μ_A, μ_D = emotion_stats['means']['V'], emotion_stats['means']['A'], emotion_stats['means']['D']
    σ_V, σ_A, σ_D = emotion_stats['stds']['V'], emotion_stats['stds']['A'], emotion_stats['stds']['D']
    ρ_VD, ρ_AD, ρ_VA = emotion_stats['correlations']['VD'], emotion_stats['correlations']['AD'], emotion_stats['correlations']['VA']
    
    # Calcular coeficientes de regressão
    denominator = (1 - ρ_VA**2 + 1e-10)  # Evitar divisão por zero
    
    β1 = (σ_D / σ_V) * ((ρ_VD - ρ_VA * ρ_AD) / denominator)
    β2 = (σ_D / σ_A) * ((ρ_AD - ρ_VA * ρ_VD) / denominator)
    β0 = μ_D - β1 * μ_V - β2 * μ_A
    
    # Estimar dominance
    dominance_estimated = β0 + β1 * valence_obs + β2 * arousal_obs
    
    # Garantir que está entre -1 e 1
    dominance_estimated = np.clip(dominance_estimated, -1.0, 1.0)
    
    return dominance_estimated

def prepare_emotion_data(data):
    """
    Prepara os dados das emoções no formato necessário para a nova abordagem
    Inclui a emoção neutral com valores padrão se não estiver no CSV
    """
    emotion_data = {}
    
    # Verificar se 'neutral' está nos dados, se não, adicionar
    neutral_in_data = 'neutral' in data['class'].values
    
    for _, row in data.iterrows():
        emotion = row['class']
        emotion_data[emotion] = {
            'means': {
                'V': row['valence mean'],
                'A': row['arousal mean'], 
                'D': row['dominance mean']
            },
            'stds': {
                'V': row['valence std'],
                'A': row['arousal std'],
                'D': row['dominance std']
            },
            'correlations': OPTIMIZED_CORRELATIONS.get(emotion, {'VD': 0.0, 'AD': 0.0, 'VA': 0.0})
        }
    
    # Adicionar neutral se não estiver presente nos dados
    if not neutral_in_data:
        emotion_data['neutral'] = {
            'means': {
                'V': 0.1,
                'A': 0.1, 
                'D': 0.1
            },
            'stds': {
                'V': 0.01,
                'A': 0.01,
                'D': 0.01
            },
            'correlations': OPTIMIZED_CORRELATIONS['neutral']
        }
        print("✅ Emoção 'neutral' adicionada com valores padrão")
    
    return emotion_data

def estimate_dominance_batch(valence_arousal, emotion_data, emotion_names):
    """
    Estima dominância em lote usando a abordagem baseada em correlações VAD
    """
    n_points = valence_arousal.shape[0]
    n_emotions = len(emotion_names)
    
    # Inicializar arrays para resultados
    dominance_estimates = np.zeros(n_points)
    
    for k in range(n_points):
        v_new, a_new = valence_arousal[k]
        
        # Calcular likelihood para cada emoção
        likelihoods = np.zeros(n_emotions)
        for i, emotion in enumerate(emotion_names):
            stats = emotion_data[emotion]
            v_mean, v_std = stats['means']['V'], stats['stds']['V']
            a_mean, a_std = stats['means']['A'], stats['stds']['A']
            
            # Probabilidade de observar v_new e a_new na distribuição da emoção
            prob_v = norm.pdf(v_new, loc=v_mean, scale=v_std)
            prob_a = norm.pdf(a_new, loc=a_mean, scale=a_std)
            likelihoods[i] = prob_v * prob_a
        
        # Usar probabilidades iguais como prior
        priors = np.ones(n_emotions) / n_emotions
        
        # Calcular probabilidades posteriores
        posteriors = likelihoods * priors
        posteriors /= np.sum(posteriors)  # Normalizar
        
        # Para cada emoção, estimar dominance baseado nos valores observados
        d_estimates_per_emotion = np.zeros(n_emotions)
        
        for i, emotion in enumerate(emotion_names):
            stats = emotion_data[emotion]
            d_estimates_per_emotion[i] = estimate_dominance_regression(
                emotion, v_new, a_new, stats
            )
        
        # Combinação final: média ponderada das estimativas por emoção
        dominance_estimates[k] = np.sum(posteriors * d_estimates_per_emotion)
        
        # Garantir que está entre -1 e 1
        dominance_estimates[k] = np.clip(dominance_estimates[k], -1.0, 1.0)
    
    return dominance_estimates

def main():
    parser = argparse.ArgumentParser(description='Generate dominance values from valence and arousal')
    parser.add_argument('--pathFile', help='Path to files', required=True)
    parser.add_argument('--emotionFile', help='Emotion data file', required=True)
    args = parser.parse_args()

    # Carregar os dados das emoções
    data = pd.read_csv(args.emotionFile)
    
    # Preparar dados no novo formato (inclui neutral)
    emotion_data = prepare_emotion_data(data)
    emotion_names = list(emotion_data.keys())
    
    print(f"🎭 Emoções carregadas: {', '.join(emotion_names)}")
    print(f"📊 Total de emoções: {len(emotion_names)}")
    
    # Obter lista de arquivos
    files = [f for f in os.listdir(args.pathFile) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    dataset_dir = os.path.dirname(args.pathFile.rstrip(os.sep))
    annotations_dir = os.path.join(dataset_dir, 'annotations')
    
    print(f"📁 Processando {len(files)} arquivos de {args.pathFile}")
    
    # Processar em lote para reduzir I/O
    batch_size = 1000
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i + batch_size]
        
        # Carregar todos os dados do batch
        valence_values = []
        arousal_values = []
        valid_files = []
        
        for filename in batch_files:
            base_name = os.path.splitext(filename)[0]
            val_path = os.path.join(annotations_dir, f'{base_name}_val.npy')
            aro_path = os.path.join(annotations_dir, f'{base_name}_aro.npy')
            
            if os.path.exists(val_path) and os.path.exists(aro_path):
                try:
                    val_value = np.load(val_path)
                    aro_value = np.load(aro_path)
                    valence_values.append(val_value)
                    arousal_values.append(aro_value)
                    valid_files.append(base_name)
                except:
                    continue
        
        if not valid_files:
            continue
            
        # Converter para arrays numpy
        valence_array = np.array(valence_values, dtype=np.float32)
        arousal_array = np.array(arousal_values, dtype=np.float32)
        va_pairs = np.column_stack((valence_array, arousal_array))
        
        # Processar em lote usando a nova abordagem
        dominance_estimates = estimate_dominance_batch(va_pairs, emotion_data, emotion_names)
        
        # Salvar resultados
        for j, base_name in enumerate(valid_files):
            np.save(os.path.join(annotations_dir, f'{base_name}_dom.npy'), 
                   dominance_estimates[j].astype(np.float32))
            
        print(f"✅ Batch {i//batch_size + 1} processado: {len(valid_files)} arquivos")
        
        # Mostrar estatísticas do primeiro batch para debug
        if i == 0 and len(dominance_estimates) > 0:
            print(f"📈 Estatísticas do dominance no primeiro batch:")
            print(f"   Mínimo: {np.min(dominance_estimates):.3f}")
            print(f"   Máximo: {np.max(dominance_estimates):.3f}")
            print(f"   Média: {np.mean(dominance_estimates):.3f}")
            print(f"   Desvio padrão: {np.std(dominance_estimates):.3f}")

if __name__ == "__main__":
    main()