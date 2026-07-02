import os
import numpy as np
import pandas as pd
import argparse

def process_protocol(base_path, protocol_name):
    print(f"--- Processando Protocolo: {protocol_name} ---")
    
    images_dir = os.path.join(base_path, protocol_name, 'images')
    annotations_dir = os.path.join(base_path, protocol_name, 'annotations')
    
    # Verifica se as pastas existem antes de prosseguir
    if not os.path.exists(images_dir) or not os.path.exists(annotations_dir):
        print(f"[Aviso] Pastas para {protocol_name} não encontradas em {base_path}. Pulando...")
        return

    # Mapeamento do .npy (0-7)
    idx_to_emotion = {
        0: 'neutral', 1: 'happy', 2: 'sad', 3: 'surprised', 
        4: 'fear', 5: 'disgust', 6: 'angry', 7: 'contempt'
    }

    # Ordem das colunas no .txt (conforme sua descrição)
    dist_columns = ['happy', 'contempt', 'surprised', 'angry', 'disgust', 'fear', 'sad', 'neutral']
    
    data_list = []
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for img_name in image_files:
        basename = os.path.splitext(img_name)[0]
        img_path = os.path.abspath(os.path.join(images_dir, img_name))
        
        exp_file = os.path.join(annotations_dir, f"{basename}_exp.npy")
        prob_file = os.path.join(annotations_dir, f"{basename}_prob_rank_universal.txt")
        
        if os.path.exists(exp_file) and os.path.exists(prob_file):
            try:
                # Carregar Rótulo
                emotion_idx = int(np.load(exp_file))
                emotion_label = idx_to_emotion.get(emotion_idx, "unknown")
                
                # Carregar Distribuição
                with open(prob_file, 'r') as f:
                    content = f.read().strip().replace(',', ' ')
                    probs = [float(x) for x in content.split()]
                
                # Montar linha
                row = {dist_columns[i]: probs[i] for i in range(len(dist_columns))}
                row['emotion'] = emotion_label
                row['path'] = img_path
                data_list.append(row)
            except Exception as e:
                print(f"Erro no arquivo {basename}: {e}")

    if data_list:
        df = pd.DataFrame(data_list)
        # Reordenar colunas: Distribuição -> Emoção -> Caminho
        final_cols = dist_columns + ['emotion', 'path']
        df = df[final_cols]
        
        output_name = f"{protocol_name}_dataset.csv"
        df.to_csv(output_name, index=False)
        print(f"Sucesso: {output_name} gerado com {len(df)} linhas.\n")
    else:
        print(f"Nenhum par imagem/anotação válido encontrado para {protocol_name}.\n")

def main():
    # Configuração do Argparse
    parser = argparse.ArgumentParser(description="Gera CSVs de distribuição de emoções a partir de um dataset estruturado.")
    
    # Adiciona o argumento do caminho do dataset
    parser.add_argument(
        '--path', 
        type=str, 
        required=True, 
        help="Caminho para a pasta raiz que contém 'train_set' e 'val_set'."
    )

    args = parser.parse_args()

    # Executa o processamento para cada pasta
    process_protocol(args.path, 'train_set')
    process_protocol(args.path, 'val_set')

if __name__ == "__main__":
    main()