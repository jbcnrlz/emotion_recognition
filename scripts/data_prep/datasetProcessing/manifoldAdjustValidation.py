import numpy as np
import pandas as pd
import os
import argparse
import logging
from datetime import datetime
from scipy.stats import multivariate_normal

# ==========================================
# Configuração do Logger e Pasta de Logs
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

log_filename = os.path.join(LOGS_DIR, f"vad_adjustment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s', # Formato mais limpo focado na mensagem
    handlers=[
        logging.FileHandler(log_filename, mode='w', encoding='utf-8'),
        logging.StreamHandler() # Opcional: imprime na tela também
    ]
)

# Correlações empíricas base
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

def build_3d_covariance(row, corrs):
    """Constrói uma Matriz de Covariância 3D [V, A, D] baseada nos desvios e correlações."""
    s_v = max(float(str(row['valence std']).replace(',', '.')), 1e-4)
    s_a = max(float(str(row['arousal std']).replace(',', '.')), 1e-4)
    s_d = max(float(str(row['dominance std']).replace(',', '.')), 1e-4)
    
    c_va = np.clip(corrs['VA'], -0.99, 0.99) * s_v * s_a
    c_vd = np.clip(corrs['VD'], -0.99, 0.99) * s_v * s_d
    c_ad = np.clip(corrs['AD'], -0.99, 0.99) * s_a * s_d
    
    cov = np.array([
        [s_v**2, c_va, c_vd],
        [c_va, s_a**2, c_ad],
        [c_vd, c_ad, s_d**2]
    ], dtype=np.float64)
    
    min_eig = np.min(np.real(np.linalg.eigvals(cov)))
    if min_eig < 0:
        cov -= 10 * min_eig * np.eye(3)
        
    return cov

def prepare_distributions(df):
    """Prepara os objetos da Gaussiana Multivariada para cada classe."""
    dists = {}
    
    if 'neutral' not in df['class'].str.lower().values:
        df = pd.concat([df, pd.DataFrame([{
            'class': 'neutral', 'valence mean': 0.0, 'valence std': 0.1,
            'arousal mean': 0.0, 'arousal std': 0.1, 'dominance mean': 0.0, 'dominance std': 0.1
        }])], ignore_index=True)
        
    for _, row in df.iterrows():
        emo = row['class'].lower()
        corrs = EMPIRICAL_CORRELATIONS.get(emo, {'VD': 0.0, 'AD': 0.0, 'VA': 0.0})
        
        mean_vec = np.array([
            float(str(row['valence mean']).replace(',', '.')), 
            float(str(row['arousal mean']).replace(',', '.')), 
            float(str(row['dominance mean']).replace(',', '.'))
        ], dtype=np.float64)
        
        cov_mat = build_3d_covariance(row, corrs)
        
        try:
            rv = multivariate_normal(mean=mean_vec, cov=cov_mat, allow_singular=True)
            dists[emo] = {'mean': mean_vec, 'rv': rv}
        except Exception as e:
            logging.error(f"Erro ao criar distribuição para {emo}: {e}")
            
    return dists

def predict_emotion(vad_point, dists, emotion_names):
    """Calcula a verossimilhança do ponto em todas as classes e retorna a classe vencedora."""
    likelihoods = [dists[emo]['rv'].pdf(vad_point) for emo in emotion_names]
    return emotion_names[np.argmax(likelihoods)]

def pull_to_boundary(vad_orig, target_emo, dists, emotion_names, steps=50, margin=0.05):
    """Desloca o ponto gradualmente em direção ao centroide da emoção alvo."""
    target_emo = target_emo.lower()
    
    if predict_emotion(vad_orig, dists, emotion_names) == target_emo:
        return vad_orig, False

    target_mean = dists[target_emo]['mean']
    
    for alpha in np.linspace(0, 1, steps):
        vad_curr = vad_orig + alpha * (target_mean - vad_orig)
        if predict_emotion(vad_curr, dists, emotion_names) == target_emo:
            safe_alpha = min(1.0, alpha + margin)
            vad_safe = vad_orig + safe_alpha * (target_mean - vad_orig)
            return np.clip(vad_safe, -1.0, 1.0), True
            
    return np.clip(target_mean, -1.0, 1.0), True

def main():
    logging.info("="*50)
    logging.info("INICIANDO PIPELINE DE AJUSTE VETORIAL VAD")
    logging.info("="*50)

    parser = argparse.ArgumentParser(description="Puxa pontos VAD para dentro da fronteira da classe real.")
    parser.add_argument('--pathFile', required=True, help='Path para o diretório base contendo imagens/anotações')
    parser.add_argument('--emotionFile', required=True, help='Path para o CSV de distribuições')
    args = parser.parse_args()

    try:
        df_distros = pd.read_csv(args.emotionFile)
    except Exception as e:
        logging.error(f"Erro fatal ao carregar o CSV: {e}")
        return

    dists = prepare_distributions(df_distros)
    emotion_names = list(dists.keys())

    dataset_dir = os.path.dirname(args.pathFile.rstrip(os.sep))
    annotations_dir = os.path.join(dataset_dir, 'annotations')
    
    if not os.path.exists(annotations_dir):
        logging.error(f"Diretório de anotações não encontrado: {annotations_dir}")
        return
        
    files = [f for f in os.listdir(annotations_dir) if f.endswith('_val.npy')]
    
    id_to_emotion = {
        0: 'neutral', 1: 'happy', 2: 'sad', 3: 'surprised', 
        4: 'fearful', 5: 'disgusted', 6: 'angry', 7: 'contempt'
    }

    ajustados = 0
    ignorados = 0
    
    logging.info(f"Arquivos detectados para análise: {len(files)}")

    for val_file in files:
        base_name = val_file.replace('_val.npy', '')

        if base_name == '5171':
            print('opa')
        
        val_path = os.path.join(annotations_dir, f'{base_name}_val.npy')
        aro_path = os.path.join(annotations_dir, f'{base_name}_aro.npy')
        dom_path = os.path.join(annotations_dir, f'{base_name}_dom.npy')
        exp_path = os.path.join(annotations_dir, f'{base_name}_exp.npy')

        if not (os.path.exists(aro_path) and os.path.exists(dom_path) and os.path.exists(exp_path)):
            ignorados += 1
            continue
            
        try:
            v_raw = np.load(val_path, allow_pickle=True)
            a_raw = np.load(aro_path, allow_pickle=True)
            d_raw = np.load(dom_path, allow_pickle=True)
            
            v = float(v_raw.item() if isinstance(v_raw, np.ndarray) else v_raw)
            a = float(a_raw.item() if isinstance(a_raw, np.ndarray) else a_raw)
            d = float(d_raw.item() if isinstance(d_raw, np.ndarray) else d_raw)
            
            exp_id = int(np.load(exp_path))
            target_emo = id_to_emotion.get(exp_id, 'neutral')
            vad_orig = np.array([v, a, d], dtype=np.float64) 
            
            vad_novo, foi_ajustado = pull_to_boundary(vad_orig, target_emo, dists, emotion_names)
            
            if foi_ajustado:
                # GRAVA O LOG COM O ANTES E DEPOIS
                logging.info(
                    f"Arquivo: {base_name} | Alvo: {target_emo.upper():<9} | "
                    f"Orig: V={vad_orig[0]:+.3f}, A={vad_orig[1]:+.3f}, D={vad_orig[2]:+.3f}  ->  "
                    f"Novo: V={vad_novo[0]:+.3f}, A={vad_novo[1]:+.3f}, D={vad_novo[2]:+.3f}"
                )
                
                # Salva os novos arquivos
                np.save(val_path, vad_novo[0].astype(np.float32))
                np.save(aro_path, vad_novo[1].astype(np.float32))
                np.save(dom_path, vad_novo[2].astype(np.float32))
                ajustados += 1
                
        except Exception as e:
            ignorados += 1

    logging.info("="*50)
    logging.info("RESUMO FINAL DO AJUSTE")
    logging.info("="*50)
    logging.info(f"Total lido:        {len(files)}")
    logging.info(f"Ajustados:         {ajustados}")
    logging.info(f"Coerentes/Intactos:{len(files) - ajustados - ignorados}")
    logging.info(f"Ignorados (Erro):  {ignorados}")
    logging.info(f"Log salvo em:      {log_filename}")

if __name__ == "__main__":
    main()