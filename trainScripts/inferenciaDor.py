import torch, pandas as pd, numpy as np, os, sys
from scipy.signal import savgol_filter
import argparse
import os
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from networks.EmotionResnetVA import PainIndexCrossAttention

def infer_pain_frame_by_frame(
    features_csv, 
    protocol_txt, 
    model_weights, 
    output_csv, 
    window_size=30, 
    sg_window=11, 
    sg_poly=3,
    batch_size=128
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando inferência no dispositivo: {device}")

    # 1. Carregamento do Modelo
    model = PainIndexCrossAttention(window_size=window_size).to(device)
    
    # Carrega os pesos salvos
    checkpoint = torch.load(model_weights, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Pesos do modelo carregados com sucesso de: {model_weights}")

    # 2. Leitura e Cruzamento dos Dados (Usando o fix do rsplit)
    print("Processando protocolo e features...")
    protocol_data = []
    with open(protocol_txt, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(' ', 2)
            if len(parts) == 3:
                protocol_data.append(parts)
                
    prot_df = pd.DataFrame(protocol_data, columns=['file_path', 'label', 'video_id'])
    prot_df['label'] = prot_df['label'].astype(int)
    prot_df['video_id'] = prot_df['video_id'].astype(int)
    
    feat_df = pd.read_csv(features_csv)
    
    prot_df['filename'] = prot_df['file_path'].apply(lambda x: os.path.basename(x.replace('\\', '/')))
    feat_df['filename'] = feat_df['file'].apply(lambda x: os.path.basename(x.replace('\\', '/')))
    
    df = pd.merge(feat_df, prot_df, on='filename', how='inner')
    
    emotions = ['happy', 'contempt', 'surprised', 'angry', 'disgusted', 'fearful', 'sad', 'neutral']
    grouped = df.groupby('video_id')
    
    all_results = []

    # 3. Processamento por Vídeo (Com Padding Causal)
    with torch.no_grad():
        for video_id, group_df in tqdm(grouped, desc="Inferindo por vídeo"):
            group_df = group_df.sort_values('filename').reset_index(drop=True)
            
            # Aplica Savitzky-Golay
            for col in emotions:
                w_len = min(sg_window, len(group_df))
                w_len = w_len if w_len % 2 != 0 else w_len - 1
                if w_len > sg_poly:
                    group_df[col] = savgol_filter(group_df[col].values, window_length=w_len, polyorder=sg_poly)
            
            video_features = group_df[emotions].values
            num_frames = len(video_features)
            
            # Padding: Repete o primeiro frame (window_size - 1) vezes no início
            # Isso garante que o frame 0 tenha uma janela simulada de T frames para o modelo processar
            pad_width = window_size - 1
            padded_features = np.vstack([np.tile(video_features[0], (pad_width, 1)), video_features])
            
            windows_x = []
            
            # Fatiamento das janelas para TODOS os frames originais
            for i in range(num_frames):
                window_x = padded_features[i : i + window_size]
                windows_x.append(window_x)
                
            windows_tensor = torch.tensor(np.array(windows_x), dtype=torch.float32)
            
            # Processamento em lotes (Batches) para não sobrecarregar a memória e ser rápido
            video_predictions = []
            for i in range(0, num_frames, batch_size):
                batch_x = windows_tensor[i : i + batch_size].to(device)
                
                # Desempacota a predição (ignora os pesos de atenção por enquanto)
                preds, _ = model(batch_x)
                
                video_predictions.extend(preds.cpu().numpy().flatten())
                
            # Armazena os resultados deste vídeo
            group_df['predicted_pain'] = video_predictions
            all_results.append(group_df[['file_path', 'filename', 'video_id', 'label', 'predicted_pain']])

    # 4. Salvamento dos Resultados
    final_results_df = pd.concat(all_results, ignore_index=True)
    
    # Arredonda a predição para evitar muitas casas decimais
    final_results_df['predicted_pain'] = final_results_df['predicted_pain'].round(4)
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    final_results_df.to_csv(output_csv, index=False)
    print(f"\nInferência concluída! Resultados salvos em: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferência Quadro a Quadro do Índice de Dor")
    
    parser.add_argument("--features", type=str, required=True, 
                        help="Caminho para o CSV contendo as features unificadas.")
    parser.add_argument("--protocol_txt", type=str, required=True, 
                        help="Caminho para o arquivo TXT (treino ou validação) para analisar.")
    parser.add_argument("--weights", type=str, required=True, 
                        help="Caminho para os pesos do modelo treinado (.pth).")
    parser.add_argument("--output", type=str, default="./results/frame_predictions.csv", 
                        help="Caminho para salvar o CSV com os resultados.")
    
    parser.add_argument("--window_size", type=int, default=30, 
                        help="Tamanho da janela temporal causal (default: 30).")
    parser.add_argument("--batch_size", type=int, default=128, 
                        help="Tamanho do batch para inferência (default: 128).")
    
    args = parser.parse_args()
    
    infer_pain_frame_by_frame(
        features_csv=args.features,
        protocol_txt=args.protocol_txt,
        model_weights=args.weights,
        output_csv=args.output,
        window_size=args.window_size,
        batch_size=args.batch_size
    )