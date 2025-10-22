import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import os, sys
import pandas as pd
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from networks.EmotionResnetVA import ResNet50WithAttentionGMM

def load_model(checkpoint_path, num_classes=14, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Carrega o modelo com os pesos pré-treinados"""
    model = ResNet50WithAttentionGMM(num_classes=num_classes,bottleneck='none',bayesianHeadType='VAD')
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Modelo carregado de {checkpoint_path}")
    else:
        print(f"Checkpoint não encontrado em {checkpoint_path}")
    
    model.to(device)
    model.eval()
    return model

def extract_faces_from_video(video_path, output_dir, frame_interval=10):
    """Extrai frames de um vídeo e salva como imagens"""
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            # Converter BGR para RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Salvar frame
            output_path = os.path.join(output_dir, f"frame_{saved_count:06d}.jpg")
            cv2.imwrite(output_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            saved_count += 1
            
        frame_count += 1
    
    cap.release()
    print(f"Extraídos {saved_count} frames de {video_path}")
    return saved_count

def preprocess_image(image_path, transform=None):
    """Pré-processa uma imagem para o modelo"""
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def process_videos_folder(videos_folder, output_base_dir, checkpoint_path, frame_interval=10):
    """Processa todos os vídeos em uma pasta e subpastas"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(checkpoint_path, device=device)
    
    # Encontrar todos os arquivos de vídeo recursivamente
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(videos_folder, '**', ext), recursive=True))
    
    print(f"Encontrados {len(video_files)} vídeos para processar (incluindo subpastas)")
    
    all_results = []
    
    for video_path in video_files:
        # Manter a estrutura de subpastas no output
        relative_path = os.path.relpath(video_path, videos_folder)
        video_relative_dir = os.path.dirname(relative_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Criar estrutura de diretórios mantendo as subpastas
        frames_dir = os.path.join(output_base_dir, 'frames', video_relative_dir, video_name)
        print(f"Processando vídeo: {video_path}")
        
        # Extrair frames
        num_frames = extract_faces_from_video(video_path, frames_dir, frame_interval)
        
        # Processar cada frame
        frame_files = sorted(glob.glob(os.path.join(frames_dir, '*.jpg')))
        
        for frame_file in frame_files:
            try:
                # Pré-processar imagem
                input_tensor = preprocess_image(frame_file).to(device)
                
                # Fazer predição
                with torch.no_grad():
                    probs, distributions, va = model(input_tensor)
                    probabilities = torch.softmax(probs, dim=1)[0].cpu().numpy()
                
                # Adicionar resultado
                result = list(probabilities) + [frame_file]
                all_results.append(result)
                
            except Exception as e:
                print(f"Erro ao processar {frame_file}: {e}")
    
    return all_results

def save_results_to_csv(results, output_csv_path):
    """Salva os resultados em um arquivo CSV com o mesmo layout"""
    emotion_columns = [
        "happy","contempt","elated","hopeful","surprised",'proud','loved','angry','astonished',
        'disgusted','fearful','sad','fatigued','neutral','file'
    ]
    
    df = pd.DataFrame(results, columns=emotion_columns)
    df.to_csv(output_csv_path, index=False)
    print(f"Resultados salvos em {output_csv_path}")
    return df

# Configurações principais
if __name__ == "__main__":
    # Configurar paths
    VIDEOS_FOLDER = "D:\\PycharmProjects\\emotion_recognition\\output_faces\\t1"  # Alterar para seu path
    OUTPUT_BASE_DIR = "outputFramesVideo"
    CHECKPOINT_PATH = "expNovo/RESNETATT_best_val_loss.pth.tar"  # Alterar para seu path
    OUTPUT_CSV_PATH = "resnet50att_oneatt_bceloss_results.csv"
    
    # Criar diretório de saída
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    # Processar vídeos
    print("Iniciando processamento de vídeos...")
    results = process_videos_folder(
        videos_folder=VIDEOS_FOLDER,
        output_base_dir=OUTPUT_BASE_DIR,
        checkpoint_path=CHECKPOINT_PATH,
        frame_interval=1  # Extrair 1 frame a cada 10 frames
    )
    
    # Salvar resultados
    if results:
        df = save_results_to_csv(results, OUTPUT_CSV_PATH)
        print(f"Processamento concluído! {len(results)} frames processados.")
        print("\nPrimeiras 5 linhas do CSV:")
        print(df.head())
    else:
        print("Nenhum resultado foi gerado.")