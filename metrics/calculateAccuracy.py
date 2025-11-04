import argparse, os, sys, numpy as np, math, random, re, pandas as pd
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Finetune resnet')    
    parser.add_argument('--csvFileComparison', help='Size of the batch', required=True)
    args = parser.parse_args()

    # Carregar o CSV
    df = pd.read_csv(args.csvFileComparison)

    # Colunas das emoções
    emotion_columns = ['happy', 'contempt', 'surprised', 'angry', 'disgusted', 'fearful', 'sad','neutral']
    emotion_labels = {1: 'happy', 7: 'contempt', 3: 'surprised', 6: 'angry', 
                    5: 'disgusted', 4: 'fearful', 2: 'sad',0:'neutral'}

    correct_predictions = 0
    total_predictions = 0
    missing_files = 0

    for index, row in df.iterrows():
        # Extrair o caminho do arquivo de imagem
        image_path = row['file']
        annotation_path = str(image_path).replace('images', 'annotations').replace('.jpg', '_exp.npy')
        
        try:
            true_emotion = np.load(annotation_path)
            emotion_probs = row[emotion_columns].values
            predicted_emotion_idx = np.argmax(emotion_probs)
            if emotion_columns[predicted_emotion_idx] == emotion_labels[int(true_emotion)]:
                correct_predictions += 1
            
            total_predictions += 1
            if index % 100 == 0:
                print(f"Processed {index} files...")
                
        except FileNotFoundError:
            print(f"Arquivo de anotação não encontrado: {annotation_path}")
            missing_files += 1
        except Exception as e:
            print(f"Erro ao processar {annotation_path}: {e}")
            missing_files += 1

    # Calcular acurácia
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        print(f"\nResultados:")
        print(f"Total de arquivos processados: {total_predictions}")
        print(f"Previsões corretas: {correct_predictions}")
        print(f"Acurácia: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Arquivos de anotação não encontrados: {missing_files}")
    else:
        print("Nenhum arquivo foi processado com sucesso.")


if __name__ == '__main__':
    main()