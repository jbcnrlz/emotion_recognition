import cv2, torch, argparse
import torch, os, sys, numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from networks.EmotionResnetVA import ResnetWithBayesianHead, ResnetWithBayesianGMMHead
from matplotlib.backends.backend_agg import FigureCanvasAgg

def getEmotionLabel(faceImage,model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    faceImage = data_transforms(faceImage).to(device).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        outputs = model(faceImage)[0]
        outputs = torch.nn.Softmax(dim=1)(outputs)
    # Aqui você pode mapear os índices de previsão para rótulos de emoção
    emotions = ["neutral","happy","sad","surprised","fear","disgust","angry","contempt","serene","contemplative","secure","untroubled","quiet"]
    _, predicted = torch.max(outputs.data, 1)
    return emotions[predicted], outputs.data.cpu().numpy()[0]


def main():
    parser = argparse.ArgumentParser(description='Demo for emotion recognition')
    parser.add_argument('--model', help='Which model to use', required=True)
    parser.add_argument('--wts', help='Weights to do FER', required=True)
    args = parser.parse_args()

    model_file = 'faceDetection/res10_300x300_ssd_iter_140000_fp16.caffemodel'
    config_file = 'faceDetection/deploy.prototxt'
    net = cv2.dnn.readNetFromCaffe(config_file, model_file)

    model = None
    if args.model == "resnetBayes":
        model = ResnetWithBayesianHead(13,resnetModel=50)
    elif args.model == "resnetBayesGMM":
        model = ResnetWithBayesianGMMHead(classes=13,resnetModel=18)
    checkpoint = torch.load(args.wts)
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    model.to("cuda")

    fig, ax = plt.subplots(figsize=(3, 2))
    ax.set_title("Emotion distribution")
    ax.set_xlabel("Emotions")
    ax.set_ylabel("Probabilities")
    canvas = FigureCanvasAgg(fig)

    # Inicializa a webcam (0 é a câmera padrão)
    cap = cv2.VideoCapture( 2, cv2.CAP_DSHOW )
    
    # Verifica se a webcam foi aberta corretamente
    if not cap.isOpened():
        print("Não foi possível abrir a webcam")
        return
    
    print("Webcam aberta com sucesso!")
    print("Pressione 'q' para sair")
    
    while True:
        # Captura frame por frame
        ret, frame = cap.read()
        
        # Se o frame foi lido corretamente
        if not ret:
            print("Não foi possível receber o frame. Saindo...")
            break

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), [104, 117, 123])
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Filtro de confiança
            if confidence > 0.7:  # Ajuste este valor conforme necessário
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x2, y2) = box.astype("int")

                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                roi = frame[y:y2, x:x2]
                emoLabel, ots = getEmotionLabel(roi,model)

                posicao = (x, y2+30)
                fonte = cv2.FONT_HERSHEY_SIMPLEX
                escala = 1.5  # Tamanho da fonte
                cor = (0, 255, 0)  # Verde em BGR
                espessura = 2
                linetype = cv2.LINE_AA  # Anti-aliasing para texto mais suave
                cv2.putText(frame, emoLabel, posicao, fonte, escala, cor, espessura, linetype)
                emts = ["neutral","happy","sad","surprised","fear","disgust","angry","contempt","serene","contemplative","secure","untroubled","quiet"]
                ax.clear()                
                ax.bar(emts,ots,alpha=0.5, color='blue')
                ax.set_title("Emotion distribution")
                ax.set_xlabel("Emotions")
                ax.set_ylabel("Probabilities")                
                #ax.set_xticklabels(emts, rotation=45)
                canvas.draw()
                buf = canvas.buffer_rgba()
                hist_img = np.asarray(buf)
                hist_img = cv2.cvtColor(hist_img, cv2.COLOR_RGBA2BGR)
                hist_img = cv2.resize(hist_img, (200, 150))
                hist_x = x + x2 + 10
                hist_y = y
                if hist_x + hist_img.shape[1] > frame.shape[1]:
                    hist_x = x - hist_img.shape[1] - 10
                if hist_y + hist_img.shape[0] > frame.shape[0]:
                    hist_y = frame.shape[0] - hist_img.shape[0]

                frame[hist_y:hist_y+hist_img.shape[0], hist_x:hist_x+hist_img.shape[1]] = hist_img

        # Mostra o frame em uma janela
        cv2.imshow('Webcam', frame)
        
        # Sai do loop quando 'q' for pressionado
        if cv2.waitKey(1) == ord('q'):
            break
        
    
    # Libera a webcam e fecha todas as janelas
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()