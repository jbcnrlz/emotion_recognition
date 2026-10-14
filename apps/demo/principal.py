import cv2
import torch
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from torchvision import transforms
import time

# Adicionar o diretório do script ao path para importar os modelos
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# Importar os modelos (ajuste conforme sua estrutura)
try:
    from networks.EmotionResnetVA import ResnetWithBayesianHead, ResnetWithBayesianGMMHead, ResNet50WithAttentionGMM
except ImportError:
    print("Não foi possível importar os modelos. Verifique a estrutura do projeto.")
    sys.exit(1)

# Mapeamento de emoções para valence e arousal (valores hipotéticos - ajuste conforme seu modelo)
emotion_to_va = {
    "happy": (0.8, 0.6),
    "contempt": (-0.3, 0.4),
    "elated": (0.9, 0.7),
    "surprised": (0.4, 0.8),
    "love": (0.9, 0.3),
    "protected": (0.5, -0.2),
    "astonished": (0.3, 0.9),
    "disgusted": (-0.7, 0.5),
    "angry": (-0.8, 0.7),
    "fearfull": (-0.6, 0.9),
    "sad": (-0.8, -0.3),
    "neutral": (0.0, 0.0)
}

# Cores para cada emoção no gráfico de barras
emotion_colors = {
    "happy": "#FFD700",       # Amarelo dourado
    "contempt": "#A9A9A9",    # Cinza escuro
    "elated": "#FF69B4",      # Rosa quente
    "surprised": "#00BFFF",   # Azul claro
    "love": "#FF1493",        # Rosa profundo
    "protected": "#228B22",   # Verde floresta
    "astonished": "#4B0082",  # Índigo
    "disgusted": "#8B4513",   # Marrom
    "angry": "#FF4500",       # Vermelho laranja
    "fearfull": "#800080",    # Roxo
    "sad": "#1E90FF",         # Azul dodger
    "neutral": "#D3D3D3"      # Cinza claro
}

def list_available_cameras(max_to_test=10):
    """Lista todas as câmeras disponíveis no sistema"""
    available_cameras = []
    
    print("Procurando câmeras disponíveis...")
    for i in range(max_to_test):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available_cameras.append(i)
                print(f"Câmera {i} encontrada - Resolução: {frame.shape[1]}x{frame.shape[0]}")
            cap.release()
        else:
            print(f"Câmera {i} não disponível")
    
    return available_cameras

def getEmotionLabel(faceImage, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    faceImage = data_transforms(faceImage).to(device).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        outputs = model(faceImage)[0]
        outputs = torch.nn.Softmax(dim=1)(outputs)
    
    emotions = ["happy", "contempt", "elated", "surprised", "love", "protected", 
                "astonished", "disgusted", "angry", "fearfull", "sad", "neutral"]
    
    _, predicted = torch.max(outputs.data, 1)
    return emotions[predicted], outputs.data.cpu().numpy()[0]

def create_valence_arousal_plot(emotion_probs, current_va=None):
    """Cria um gráfico do plano valence-arousal com as emoções plotadas"""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Configurar o gráfico
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Valence (Prazer)')
    ax.set_ylabel('Arousal (Excitação)')
    ax.set_title('Plano Valence-Arousal')
    
    # Plotar as emoções baseadas nas probabilidades
    for i, emotion in enumerate(emotion_to_va.keys()):
        va = emotion_to_va[emotion]
        prob = emotion_probs[i]
        size = 50 + (prob * 500)  # Tamanho do marcador baseado na probabilidade
        alpha = 0.5 + (prob * 0.5)  # Transparência baseada na probabilidade
        ax.scatter(va[0], va[1], s=size, alpha=alpha, 
                  c=emotion_colors[emotion], label=emotion, edgecolors='black')
    
    # Plotar o ponto atual (se fornecido)
    if current_va is not None:
        ax.scatter(current_va[0], current_va[1], s=200, c='red', 
                  marker='X', edgecolors='black', label='Atual')
    
    # Adicionar quadrantes
    ax.text(0.5, 0.5, 'Feliz\nExcitado', ha='center', va='center', 
            fontsize=10, alpha=0.7, bbox=dict(facecolor='yellow', alpha=0.2))
    ax.text(-0.5, 0.5, 'Triste\nZangado', ha='center', va='center', 
            fontsize=10, alpha=0.7, bbox=dict(facecolor='red', alpha=0.2))
    ax.text(-0.5, -0.5, 'Calmo\nDescontente', ha='center', va='center', 
            fontsize=10, alpha=0.7, bbox=dict(facecolor='blue', alpha=0.2))
    ax.text(0.5, -0.5, 'Relaxado\nContentamento', ha='center', va='center', 
            fontsize=10, alpha=0.7, bbox=dict(facecolor='green', alpha=0.2))
    
    return fig

def create_emotion_bar_chart(emotion_probs, emotions):
    """Cria um gráfico de barras para as probabilidades das emoções"""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Criar barras coloridas
    colors = [emotion_colors[emotion] for emotion in emotions]
    bars = ax.bar(emotions, emotion_probs, color=colors, alpha=0.7)
    
    # Adicionar valores nas barras
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{emotion_probs[i]:.2f}', ha='center', va='bottom', rotation=0, fontsize=8)
    
    ax.set_ylabel('Probabilidade')
    ax.set_title('Distribuição de Emoções')
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig

def fig_to_array(fig):
    """Converte uma figura matplotlib para um array numpy para exibição no OpenCV"""
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    img_arr = np.asarray(buf)
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2BGR)
    plt.close(fig)
    return img_arr

def create_dashboard(camera_frame, bar_img, va_img, emotion_text, va_text, camera_text):
    """Cria um painel com a imagem da câmera e os gráficos lado a lado"""
    # Redimensionar a imagem da câmera para um tamanho padrão
    camera_height, camera_width = camera_frame.shape[:2]
    target_height = 480
    scale_factor = target_height / camera_height
    new_width = int(camera_width * scale_factor)
    resized_camera = cv2.resize(camera_frame, (new_width, target_height))
    
    # Criar um fundo preto para o painel
    dashboard_height = max(target_height, bar_img.shape[0] + va_img.shape[0] + 50)
    dashboard_width = new_width + max(bar_img.shape[1], va_img.shape[1]) + 20
    
    dashboard = np.zeros((dashboard_height, dashboard_width, 3), dtype=np.uint8)
    
    # Posicionar a imagem da câmera
    dashboard[0:target_height, 0:new_width] = resized_camera
    
    # Posicionar o gráfico de barras
    y_offset = 10
    dashboard[y_offset:y_offset+bar_img.shape[0], new_width+10:new_width+10+bar_img.shape[1]] = bar_img
    
    # Posicionar o gráfico valence-arousal
    y_offset += bar_img.shape[0] + 10
    dashboard[y_offset:y_offset+va_img.shape[0], new_width+10:new_width+10+va_img.shape[1]] = va_img
    
    # Adicionar informações de texto
    cv2.putText(dashboard, emotion_text, (10, target_height + 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(dashboard, va_text, (10, target_height + 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(dashboard, camera_text, (10, target_height + 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return dashboard

def main():
    parser = argparse.ArgumentParser(description='Demo for emotion recognition')
    parser.add_argument('--model', help='Which model to use', required=True)
    parser.add_argument('--wts', help='Weights to do FER', required=True)
    parser.add_argument('--resnetSize', help='Weights to do FER', required=False, type=int, default=18)
    parser.add_argument('--camera', help='Camera index to use (default: 0)', type=int, default=None)
    args = parser.parse_args()

    # Listar câmeras disponíveis se nenhuma foi especificada
    if args.camera is None:
        available_cameras = list_available_cameras()
        
        if not available_cameras:
            print("Nenhuma câmera encontrada. Verifique as conexões.")
            return
        
        # Selecionar a primeira câmera disponível por padrão
        camera_index = available_cameras[0]
        print(f"Usando câmera {camera_index} (primeira disponível)")
        
        # Perguntar se o usuário quer selecionar outra câmera
        if len(available_cameras) > 1:
            response = input(f"Deseja selecionar outra câmera? Disponível: {available_cameras} (s/N): ")
            if response.lower() == 's':
                try:
                    camera_index = int(input(f"Digite o índice da câmera {available_cameras}: "))
                    if camera_index not in available_cameras:
                        print(f"Câmera {camera_index} não disponível. Usando câmera {available_cameras[0]}")
                        camera_index = available_cameras[0]
                except ValueError:
                    print("Entrada inválida. Usando câmera padrão.")
                    camera_index = available_cameras[0]
    else:
        camera_index = args.camera
        # Verificar se a câmera selecionada está disponível
        cap_test = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap_test.isOpened():
            print(f"Câmera {camera_index} não está disponível.")
            available_cameras = list_available_cameras()
            if available_cameras:
                camera_index = available_cameras[0]
                print(f"Usando câmera {camera_index} em vez disso.")
            else:
                print("Nenhuma câmera disponível. Saindo.")
                return
        cap_test.release()

    # Carregar modelo de detecção de faces
    model_file = 'faceDetection/res10_300x300_ssd_iter_140000_fp16.caffemodel'
    config_file = 'faceDetection/deploy.prototxt'
    
    if not os.path.exists(model_file) or not os.path.exists(config_file):
        print("Arquivos de detecção facial não encontrados. Verifique os caminhos.")
        return
    
    net = cv2.dnn.readNetFromCaffe(config_file, model_file)

    # Carregar modelo de emoções
    model = None
    if args.model == "resnetBayes":
        model = ResnetWithBayesianHead(13, resnetModel=args.resnetSize)
    elif args.model == "resnetBayesGMM":
        model = ResnetWithBayesianGMMHead(classes=13, resnetModel=args.resnetSize)
    elif args.model == "resnetAttentionGMM":        
        model = ResNet50WithAttentionGMM(num_classes=12, bottleneck='none', bayesianHeadType='VAD')
    
    try:
        checkpoint = torch.load(args.wts)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.to("cuda" if torch.cuda.is_available() else "cpu")
    except Exception as e:
        print(f"Erro ao carregar os pesos do modelo: {e}")
        return

    # Inicializar a webcam com a câmera selecionada
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print(f"Não foi possível abrir a câmera {camera_index}")
        return
    
    # Configurar resolução (opcional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print(f"Câmera {camera_index} aberta com sucesso!")
    print("Pressione 'q' para sair")
    print("Pressione 'c' para listar câmeras disponíveis e trocar")
    
    # Variáveis para suavização
    emotion_history = []
    history_length = 5
    emotions_list = ["happy", "contempt", "elated", "surprised", "love", "protected", 
                     "astonished", "disgusted", "angry", "fearfull", "sad", "neutral"]
    
    # Criar janela com layout personalizado
    cv2.namedWindow('Reconhecimento de Emoções', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Reconhecimento de Emoções', 1200, 700)
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Não foi possível receber o frame. Tentando reconectar...")
            # Tentar reconectar à câmera
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                print("Falha ao reconectar. Saindo...")
                break
            continue

        # Espelhar o frame para uma experiência mais natural
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Detectar faces
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, 
                                    (300, 300), [104, 117, 123])
        net.setInput(blob)
        detections = net.forward()

        emotion_probs = None
        dominant_emotion = "Nenhuma face detectada"
        valence, arousal = 0, 0
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.7:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x2, y2) = box.astype("int")
                
                # Garantir que as coordenadas estão dentro dos limites do frame
                x, y = max(0, x), max(0, y)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Desenhar retângulo ao redor da face
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                
                # Extrair ROI (Região de Interesse) - a face
                roi = frame[y:y2, x:x2]
                
                if roi.size == 0:
                    continue
                
                # Reconhecer emoção
                emoLabel, probs = getEmotionLabel(roi, model)
                emotion_probs = probs
                dominant_emotion = emoLabel
                
                # Suavizar as probabilidades
                emotion_history.append(probs)
                if len(emotion_history) > history_length:
                    emotion_history.pop(0)
                
                smoothed_probs = np.mean(emotion_history, axis=0) if emotion_history else probs
                
                # Calcular valence e arousal médios baseado nas probabilidades
                valence = 0
                arousal = 0
                for j, emotion in enumerate(emotions_list):
                    va = emotion_to_va[emotion]
                    valence += smoothed_probs[j] * va[0]
                    arousal += smoothed_probs[j] * va[1]
                
                current_va = (valence, arousal)
                
                # Adicionar texto com a emoção dominante
                label_text = f"Emoção: {dominant_emotion}"
                cv2.putText(frame, label_text, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Criar visualizações
        bar_img = np.zeros((300, 600, 3), dtype=np.uint8)
        va_img = np.zeros((400, 400, 3), dtype=np.uint8)
        
        if emotion_probs is not None:
            # Gráfico de barras
            bar_fig = create_emotion_bar_chart(smoothed_probs, emotions_list)
            bar_img = fig_to_array(bar_fig)
            
            # Gráfico valence-arousal
            va_fig = create_valence_arousal_plot(smoothed_probs, current_va)
            va_img = fig_to_array(va_fig)
        
        # Criar texto para informações
        emotion_text = f"Emoção Dominante: {dominant_emotion}"
        va_text = f"Valence: {valence:.2f}, Arousal: {arousal:.2f}"
        camera_text = f"Câmera: {camera_index}"
        
        # Criar dashboard com layout separado
        dashboard = create_dashboard(frame, bar_img, va_img, emotion_text, va_text, camera_text)
        
        # Mostrar dashboard
        cv2.imshow('Reconhecimento de Emoções', dashboard)
        
        # Processar teclas de controle
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Listar câmeras disponíveis e permitir trocar
            print("Listando câmeras disponíveis...")
            available_cameras = list_available_cameras()
            
            if len(available_cameras) > 1:
                try:
                    new_camera = int(input(f"Digite o índice da nova câmera {available_cameras}: "))
                    if new_camera in available_cameras:
                        # Fechar câmera atual e abrir a nova
                        cap.release()
                        camera_index = new_camera
                        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
                        if not cap.isOpened():
                            print(f"Falha ao abrir câmera {camera_index}. Voltando para a anterior.")
                            camera_index = available_cameras[0]
                            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
                        else:
                            print(f"Troca para câmera {camera_index} bem-sucedida!")
                    else:
                        print("Câmera não disponível. Mantendo a atual.")
                except ValueError:
                    print("Entrada inválida. Mantendo câmera atual.")
            else:
                print("Apenas uma câmera disponível.")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()