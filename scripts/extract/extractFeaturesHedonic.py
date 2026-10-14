import argparse, torch, os, sys, tqdm, numpy as np
from torchvision import transforms
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from DatasetClasses.HedonicDataset import VideoFramesDataset
from torch import nn
from networks.EmotionResnetVA import ResNet50WithCrossAttention,ResnetWithBayesianHead, ResnetWithBayesianGMMHead, ResNet50WithAttentionGMM, ResNet50WithAttentionLikelihood, ResNet50WithAttentionLikelihoodNoVA, Glint360kResNetWithCrossAttention
from torch.utils.data import DataLoader

def saveToCSV(preds, files, pathCSV, emoHeaders=None):
    # Se num_classes não for fornecido, tenta inferir do shape de preds
    if preds is not None and len(preds) > 0:
        num_classes = preds.shape[1] if hasattr(preds, 'shape') and len(preds.shape) > 1 else len(preds[0])
    else:
        num_classes = 0

    # Lista de emoções para os cabeçalhos - agora dinâmica
    emotion_headers = []
    
    # Adiciona cabeçalhos para as classes de predição
    if emoHeaders is None:
        for i in range(num_classes):
            emotion_headers.append(f'class_{i:03d}')
    else:
        emotion_headers = list(emoHeaders)
    
    # Adiciona cabeçalhos para VAD (se existir)
    
    # Escreve o cabeçalho no arquivo CSV
    with open(pathCSV, 'w') as pcsv:
        pcsv.write('%s,file\n' % (','.join(emotion_headers)))
        
        # Escreve os dados
        for idx, p in enumerate(preds):
            # Escreve as predições de classe
            if hasattr(p, '__len__'):
                for fp in p:
                    pcsv.write(f'{fp},')
            else:
                pcsv.write(f'{p},')
            
            # Escreve o nome do arquivo
            pcsv.write(f'"{files[idx]}"\n')


def main():
    parser = argparse.ArgumentParser(description='Extract features from resnet emotion')
    parser.add_argument('--weights', help='Weights', required=True)
    parser.add_argument('--pathBase', help='Path for valence and arousal dataset', required=True)
    parser.add_argument('--batch', type=int, help='Size of the batch', required=True)
    parser.add_argument('--output', default=None, help='File to save csv', required=True)
    parser.add_argument('--emotionModel', help='Model for feature extraction', required=False, default="resnetBayesGMM")
    parser.add_argument('--classQuantity', type=int, help='Number of emotion classes', required=True)
    parser.add_argument('--emotion_labels', type=str, default=None,
                       help='Custom emotion labels (comma-separated)')

    args = parser.parse_args()


    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
    ])

    emotion_labels = None
    if args.emotion_labels:
        emotion_labels = [label.strip() for label in args.emotion_labels.split(',')]
        print(f"Custom emotion labels: {emotion_labels}")

    checkpoint = torch.load(args.weights)

    model = None
    if args.emotionModel == "resnetBayesGMM":
        model = ResnetWithBayesianGMMHead(classes=args.classQuantity, resnetModel=args.resnetInnerModel)
    elif args.emotionModel == "resnetBayes":
        model = ResnetWithBayesianHead(args.classQuantity, resnetModel=args.resnetInnerModel)
    elif args.emotionModel == "resnetAttentionGMM":
        model = ResNet50WithAttentionGMM(num_classes=args.classQuantity, bottleneck='none', bayesianHeadType='VAD')
    elif args.emotionModel == "resnetAttentionLikelihood":
        model = ResNet50WithAttentionLikelihood(num_classes=args.classQuantity, bottleneck='none', bayesianHeadType='VAD')
    elif args.emotionModel == "simpleNetwork":
        model = ResNet50WithAttentionLikelihoodNoVA(num_classes=args.classQuantity, bottleneck='none')
    elif args.emotionModel == 'simpleNetworkCrossAtt':
        # ATUALIZADO: Instanciação da classe com o novo nome
        model = ResNet50WithCrossAttention(num_classes=args.classQuantity,bottleneck='none',num_sectors=8)
    elif args.emotionModel == 'glint360ksimpleNetworkCrossAtt':
        model = Glint360kResNetWithCrossAttention(num_classes=args.classQuantity, pretrained_path=None, num_sectors=7)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.weights)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.to(device)

    # Instanciando passando APENAS o root_dir
    dataset = VideoFramesDataset(
        root_dir=args.pathBase, 
        transform=transform
    )

    print(f"Total de frames carregados: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=False)
    model.eval()

    soft = nn.Softmax(dim=1)    
    # Processar batch por batch
    paths = []
    predictions = None
    with torch.no_grad():
        # ADICIONADO: tqdm envolvendo o dataloader
        for images, _, pathFile in tqdm.tqdm(dataloader, desc="Extraindo features", unit="batch"):
            paths.extend(pathFile)
            # Forward pass normal (sem gradientes)
            if (not isinstance(model, ResNet50WithAttentionLikelihoodNoVA) and not isinstance(model, ResNet50WithCrossAttention) and not isinstance(model, Glint360kResNetWithCrossAttention)):
                outputs, _, vad = model(images.to(device))
            else:
                outputs = model(images.to(device))
                vad = None
            outputs = soft(outputs)
            prediction = outputs.cpu().numpy()
            predictions = prediction if predictions is None else np.concatenate((prediction , predictions))    

    saveToCSV(predictions, paths, args.output, emoHeaders=emotion_labels)


if __name__ == '__main__':
    main()