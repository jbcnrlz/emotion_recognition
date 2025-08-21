import numpy as np
import pandas as pd
import argparse
import os
import sys
from scipy.linalg import inv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

def estimate_dominance_batch(valence_arousal, mean, cov, inv_cov_va, cov_d_va, mean_d):
    """
    Estima dominância em lote para múltiplos pares valence/arousal.
    Versão vetorizada para melhor performance.
    """
    # Calcular a média condicional para todos os pontos
    diff = valence_arousal - mean[:2]
    mean_d_given_va = mean_d + np.dot(cov_d_va, np.dot(inv_cov_va, diff.T)).T
    
    return mean_d_given_va

def main():
    parser = argparse.ArgumentParser(description='Generate dominance values from valence and arousal')
    parser.add_argument('--pathFile', help='Path to files', required=True)
    parser.add_argument('--emotionFile', help='Emotion data file', required=True)
    args = parser.parse_args()

    # Carregar os dados
    data = pd.read_csv(args.emotionFile)

    # Calcular estatísticas globais uma única vez
    global_mean = data[['valence mean', 'arousal mean', 'dominance mean']].mean().values
    global_cov = np.cov(data[['valence mean', 'arousal mean', 'dominance mean']].values.T, rowvar=False)

    # Pré-calcular componentes da matriz uma única vez
    mean_va = global_mean[:2]
    mean_d = global_mean[2]
    cov_va = global_cov[:2, :2]
    cov_va_d = global_cov[:2, 2]
    cov_d_va = global_cov[2, :2]
    
    # Pré-calcular a inversa da matriz de covariância (mais custosa)
    inv_cov_va = inv(cov_va)

    # Obter lista de arquivos
    files = [f for f in os.listdir(args.pathFile) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    dataset_dir = os.path.dirname(args.pathFile.rstrip(os.sep))
    annotations_dir = os.path.join(dataset_dir, 'annotations')
    
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
        
        # Processar em lote
        dominance_estimates = estimate_dominance_batch(
            va_pairs, global_mean, global_cov, inv_cov_va, cov_d_va, mean_d
        )
        
        # Salvar resultados
        for j, base_name in enumerate(valid_files):
            np.save(os.path.join(annotations_dir, f'{base_name}_dom.npy'), 
                   dominance_estimates[j].astype(np.float32))

if __name__ == "__main__":
    main()