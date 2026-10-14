import numpy as np
import pandas as pd
import argparse
import os
import sys
import logging
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# ==========================================
# Configuração do Logger e Pasta de Logs
# ==========================================
LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

log_filename = os.path.join(LOGS_DIR, f"pipeline_dominance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

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
    Estima dominance usando regressão ancorada com limite elástico (Bounded Perturbation).
    """
    μ_V, μ_A, μ_D = emotion_stats['means']['V'], emotion_stats['means']['A'], emotion_stats['means']['D']
    σ_V, σ_A, σ_D = emotion_stats['stds']['V'], emotion_stats['stds']['A'], emotion_stats['stds']['D']
    ρ_VD, ρ_AD, ρ_VA = emotion_stats['correlations']['VD'], emotion_stats['correlations']['AD'], emotion_stats['correlations']['VA']
    
    ρ_VA_safe = np.clip(ρ_VA, -0.99, 0.99)
    denominator = 1 - ρ_VA_safe**2 
    
    β1 = (σ_D / max(σ_V, 1e-6)) * ((ρ_VD - ρ_VA_safe * ρ_AD) / denominator)
    β2 = (σ_D / max(σ_A, 1e-6)) * ((ρ_AD - ρ_VA_safe * ρ_VD) / denominator)
    
    linear_perturbation = β1 * (valence_obs - μ_V) + β2 * (arousal_obs - μ_A)
    max_deviation = 1.5 * σ_D
    bounded_perturbation = max_deviation * np.tanh(linear_perturbation / max(max_deviation, 1e-6))
    
    return μ_D + bounded_perturbation

def prepare_emotion_data(data):
    emotion_data = {}
    neutral_in_data = 'neutral' in data['class'].values
    
    for _, row in data.iterrows():
        emotion = row['class']
        corrs = EMPIRICAL_CORRELATIONS.get(emotion, {'VD': 0.0, 'AD': 0.0, 'VA': 0.0})
        emotion_data[emotion] = {
            'means': {'V': row['valence mean'], 'A': row['arousal mean'], 'D': row['dominance mean']},
            'stds': {'V': row['valence std'], 'A': row['arousal std'], 'D': row['dominance std']},
            'correlations': corrs
        }
        
    if not neutral_in_data:
        corrs_neutral = EMPIRICAL_CORRELATIONS['neutral']
        emotion_data['neutral'] = {
            'means': {'V': 0.1, 'A': 0.1, 'D': 0.1},
            'stds': {'V': 0.01, 'A': 0.01, 'D': 0.01},
            'correlations': corrs_neutral
        }
        logging.info("Emoção 'neutral' adicionada com valores padrão.")
        
    return emotion_data

def estimate_dominance_with_prior(valence_arousal, true_emotions_batch, emotion_data, emotion_names):
    n_points = valence_arousal.shape[0]
    final_dominance = np.zeros(n_points)
    v_obs = valence_arousal[:, 0]
    a_obs = valence_arousal[:, 1]
    
    all_estimates = np.zeros((n_points, len(emotion_names)))
    for i, emotion in enumerate(emotion_names):
        stats = emotion_data[emotion]
        all_estimates[:, i] = estimate_dominance_regression(v_obs, a_obs, stats)
        
    name_to_idx = {name: i for i, name in enumerate(emotion_names)}
    idx_neutral = name_to_idx.get('neutral', 0) 
    
    for j in range(n_points):
        emotion_label = true_emotions_batch[j]
        if emotion_label in name_to_idx:
            idx = name_to_idx[emotion_label]
            final_dominance[j] = all_estimates[j, idx]
        else:
            final_dominance[j] = all_estimates[j, idx_neutral]
            
    return np.clip(final_dominance, -1.0, 1.0)

def main():
    logging.info("=== Iniciando Pipeline Completo de Geração de Dominância ===")
    
    parser = argparse.ArgumentParser(description='Generate dominance anchored on original emotions')
    parser.add_argument('--pathFile', help='Path to directory containing images and annotations', required=True)
    parser.add_argument('--emotionFile', help='Path to Emotion distributions CSV file', required=True)
    args = parser.parse_args()

    try:
        data = pd.read_csv(args.emotionFile)
        logging.info(f"Arquivo CSV de distribuições carregado com sucesso: {args.emotionFile}")
    except Exception as e:
        logging.error(f"Falha ao carregar o arquivo CSV de emoções: {e}")
        sys.exit(1)

    emotion_data = prepare_emotion_data(data)
    emotion_names = list(emotion_data.keys())
    logging.info(f"Emoções de Referência Mapeadas: {', '.join(emotion_names)}")
    
    files = [f for f in os.listdir(args.pathFile) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    dataset_dir = os.path.dirname(args.pathFile.rstrip(os.sep))
    annotations_dir = os.path.join(dataset_dir, 'annotations')
    
    if not os.path.exists(annotations_dir):
        logging.warning(f"Diretório de anotações não encontrado: {annotations_dir}. Criando novo diretório...")
        os.makedirs(annotations_dir, exist_ok=True)
    
    logging.info(f"Preparando para processar {len(files)} imagens do diretório: {args.pathFile}")
    
    batch_size = 5000 
    id_to_emotion = {
        0: 'neutral', 1: 'happy', 2: 'sad', 3: 'surprised', 
        4: 'fearful', 5: 'disgusted', 6: 'angry', 7: 'contempt'
    }
    
    arquivos_ignorados = 0
    anotacoes_geradas = 0

    for i in range(0, len(files), batch_size):
        batch_files = files[i:i + batch_size]
        
        # Guardaremos também o filename original completo para o log
        valence_values, arousal_values, true_emotions_batch, true_emotions_id_batch, valid_files, full_filenames = [], [], [], [], [], []
        
        for filename in batch_files:
            base_name = os.path.splitext(filename)[0]
            val_path = os.path.join(annotations_dir, f'{base_name}_val.npy')
            aro_path = os.path.join(annotations_dir, f'{base_name}_aro.npy')
            exp_path = os.path.join(annotations_dir, f'{base_name}_exp.npy') 
            
            if os.path.exists(val_path) and os.path.exists(aro_path) and os.path.exists(exp_path):
                try:
                    val_value = np.load(val_path)
                    aro_value = np.load(aro_path)
                    exp_id = int(np.load(exp_path))
                    
                    if isinstance(val_value, np.ndarray): val_value = val_value.item()
                    if isinstance(aro_value, np.ndarray): aro_value = aro_value.item()
                        
                    valence_values.append(val_value)
                    arousal_values.append(aro_value)
                    true_emotions_batch.append(id_to_emotion.get(exp_id, 'neutral'))
                    true_emotions_id_batch.append(exp_id)
                    valid_files.append(base_name)
                    full_filenames.append(filename) # <--- NOVO: Guarda o nome completo (ex: 3440.jpg)
                except Exception as e:
                    logging.error(f"Erro de corrupção ao ler arquivos de {base_name}: {e}")
                    arquivos_ignorados += 1
                    continue
            else:
                arquivos_ignorados += 1
        
        if not valid_files: 
            continue
            
        va_pairs = np.column_stack((valence_values, arousal_values)).astype(np.float32)
        dominance_estimates = estimate_dominance_with_prior(va_pairs, true_emotions_batch, emotion_data, emotion_names)

        # =====================================================================
        # SALVAMENTO E LOG INDIVIDUAL PARA CADA ARQUIVO DO LOTE
        # =====================================================================
        for j, base_name in enumerate(valid_files):
            em_real = true_emotions_batch[j]
            em_id = true_emotions_id_batch[j]
            d_calc = dominance_estimates[j]
            orig_filename = full_filenames[j] # <--- NOVO: Recupera o nome completo
            
            # ATUALIZADO: Log agora exibe o nome de arquivo completo (ex: 294.jpg em vez de apenas 294)
            logging.info(f"Arquivo: {orig_filename} | ID: {em_id} | Emoção Lida: {em_real.upper()} | Dominância Calculada: {d_calc:.3f}")
            
            try:
                np.save(os.path.join(annotations_dir, f'{base_name}_dom.npy'), d_calc.astype(np.float32))
                anotacoes_geradas += 1
            except Exception as e:
                logging.error(f"Falha ao salvar a dominância de {base_name}: {e}")
        # =====================================================================
                
        # Registra o resumo do lote no log
        lote_atual = i // batch_size + 1
        logging.info(f"--- Lote {lote_atual} processado. Arquivos salvos: {len(valid_files)} ---")
        if len(dominance_estimates) > 0:
            logging.info(f"   [Estatísticas do Lote {lote_atual}] Mínimo: {np.min(dominance_estimates):.3f} | Máximo: {np.max(dominance_estimates):.3f} | Média: {np.mean(dominance_estimates):.3f}")

    logging.info("=== Processamento Concluído ===")
    logging.info(f"Total de arquivos gerados (_dom.npy): {anotacoes_geradas}")
    if arquivos_ignorados > 0:
        logging.warning(f"Total de arquivos ignorados (falta de dados base ou erros de leitura): {arquivos_ignorados}")
    logging.info(f"Log finalizado e salvo em: {os.path.abspath(log_filename)}")

if __name__ == "__main__":
    main()