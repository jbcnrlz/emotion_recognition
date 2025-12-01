import argparse, torch, os, sys, numpy as np, math
from torchvision import transforms
from torchvision import models
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from DatasetClasses.DFEW import DFEW
from torch import nn
from networks.EmotionResnetVA import ResnetWithBayesianHead, ResnetWithBayesianGMMHead, ResNet50WithAttentionGMM, ResNet50WithAttentionLikelihood
from helper.function import visualizeAttentionMaps, printProgressBar

# Versão alternativa que detecta automaticamente a ordem das colunas
def alignEmotionProbabilitiesAuto(prob_matrix_8classes, prob_matrix_7classes, seven_classes_columns=None):
    """
    Versão que detecta automaticamente a ordem das colunas na matriz de 7 classes
    """
    
    # Ordem original das 8 classes
    original_8classes_order = ['happy', 'contempt', 'surprised', 'angry', 'disgusted', 'fearful', 'sad', 'neutral']
    
    # Ordem desejada para as 7 classes
    target_7classes_order = ['Happy', 'Sad', 'Neutral', 'Angry', 'Surprise', 'Disgust', 'Fear']
    
    # Mapeamento
    emotion_mapping_8_to_7 = {
        'happy': 'Happy',
        'contempt': 'Angry',
        'surprised': 'Surprise', 
        'angry': 'Angry',
        'disgusted': 'Disgust',
        'fearful': 'Fear',
        'sad': 'Sad',
        'neutral': 'Neutral'
    }
    
    # Se a ordem das 7 classes não for fornecida, assumir que já está na ordem desejada
    if seven_classes_columns is None:
        seven_classes_columns = target_7classes_order
    
    # Verificar se todas as classes esperadas estão presentes
    missing_classes = set(target_7classes_order) - set(seven_classes_columns)
    if missing_classes:
        raise ValueError(f"Classes faltando na matriz de 7 classes: {missing_classes}")
    
    # Criar matriz alinhada para 8 classes -> 7 classes
    aligned_8classes = np.zeros((prob_matrix_8classes.shape[0], 7))
    
    for i, emotion_8 in enumerate(original_8classes_order):
        emotion_7 = emotion_mapping_8_to_7[emotion_8]
        target_idx = target_7classes_order.index(emotion_7)
        aligned_8classes[:, target_idx] += prob_matrix_8classes[:, i]
    
    # Normalizar
    row_sums = aligned_8classes.sum(axis=1, keepdims=True)
    aligned_8classes = np.divide(aligned_8classes, row_sums, 
                                out=np.zeros_like(aligned_8classes), 
                                where=row_sums!=0)
    
    # Reordenar a matriz de 7 classes para a ordem desejada
    reorder_indices = [seven_classes_columns.index(cls) for cls in target_7classes_order]
    aligned_7classes = prob_matrix_7classes[:, reorder_indices]
    
    mapping_info = {
        'original_8_classes': original_8classes_order,
        'current_7_classes': seven_classes_columns,
        'target_7_classes': target_7classes_order,
        'reorder_indices': reorder_indices,
        'mapping': emotion_mapping_8_to_7
    }
    
    return aligned_8classes, aligned_7classes, mapping_info


def saveToCSV(preds,files,pathCSV):
    emotions = ['Happy', 'Sad', 'Neutral', 'Angry', 'Surprise', 'Disgust', 'Fear']
    with open(pathCSV,'w') as pcsv:
        pcsv.write('%s,file\n' % (','.join([emotions[f] for f in range(len(emotions))])))
        for idx, p in enumerate(preds):
            for fp in p:
                pcsv.write(f'{fp},')
            pcsv.write(f"{files[idx]}\n")

def train():
    parser = argparse.ArgumentParser(description='Extract features from resnet emotion')
    parser.add_argument('--weights', help='Weights', required=True)
    parser.add_argument('--pathBase', help='Path for valence and arousal dataset', required=True)
    parser.add_argument('--batch', type=int, help='Size of the batch', required=True)
    parser.add_argument('--output', default=None, help='File to save csv', required=True)
    parser.add_argument('--dataset', help='Dataset for feature extractoin', required=False, default="OMG")
    parser.add_argument('--resnetInnerModel', help='Model for feature extraction', required=False,type=int, default=18)
    parser.add_argument('--emotionModel', help='Model for feature extraction', required=False, default="resnetBayesGMM")
    parser.add_argument('--classQuantity', help='Model for feature extraction', required=False, type=int, default="14")
    args = parser.parse_args()

    checkpoint = torch.load(args.weights)

    model = None
    if args.emotionModel == "resnetBayesGMM":
        model = ResnetWithBayesianGMMHead(classes=args.classQuantity,resnetModel=args.resnetInnerModel)
    elif args.emotionModel == "resnetBayes":
        model = ResnetWithBayesianHead(args.classQuantity,resnetModel=args.resnetInnerModel)
    elif args.emotionModel == "resnetAttentionGMM":
        model = ResNet50WithAttentionGMM(num_classes=args.classQuantity,bottleneck='none',bayesianHeadType='VAD')
    elif args.emotionModel == "resnetAttentionLikelihood":
        model = ResNet50WithAttentionLikelihood(num_classes=args.classQuantity,bottleneck='none',bayesianHeadType='VAD')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.weights)
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    model.to(device)
    print("Model loaded")
    print(model)

    data_transforms = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
    for i in range(1,6):
        print(f"Processing fold {i}")
        dataset = DFEW(pathData=args.pathBase,fold=i,transform=data_transforms)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=False)
        model.eval()
        pathFile = []
        predictions = None
        labelsGrouped = None

        soft =  nn.Softmax(dim=1)
        with torch.no_grad():
            for idx, data in enumerate(val_loader):
                printProgressBar(idx,math.ceil(len(dataset.images) / args.batch),length=50,prefix='Loading Faces...')
                images, distrosProb, _, pathsForFiles = data
                outputs, _, _ = model(images.to(device))
                outputs = soft(outputs)

                prediction = outputs.cpu().detach().numpy()
                predictions = prediction if predictions is None else np.concatenate((prediction,predictions))

                currLabel = distrosProb.cpu().detach().numpy()
                labelsGrouped = currLabel if labelsGrouped is None else np.concatenate((currLabel,labelsGrouped))
                
                pathFile = list(pathsForFiles) + pathFile
        predictions, labelsGrouped, _ = alignEmotionProbabilitiesAuto(predictions,labelsGrouped)
        saveToCSV(predictions,pathFile,f"fold_{i}_" + args.output)
        saveToCSV(labelsGrouped,pathFile,f"fold_{i}_" + args.output[:-4]+"_labels.csv")

if __name__ == '__main__':
    train()