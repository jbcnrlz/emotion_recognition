import argparse, torch, os, sys, numpy as np, math
from torchvision import transforms
from torchvision import models
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from DatasetClasses.AffectNet import AffectNet
from torch import nn
from networks.EmotionResnetVA import ResnetWithBayesianHead, ResnetWithBayesianGMMHead, ResNet50WithAttentionGMM, ResNet50WithAttentionLikelihood, ResNet50WithAttentionLikelihoodNoVA
from helper.function import visualizeAttentionMaps, printProgressBar

def saveToCSV(preds,files,pathCSV,vad=None,emoLabels=None,mapping=None):
    emotions = [
        "happy","contempt","elated","hopeful","surprised",'proud','loved',
        'angry','astonished','disgusted','fearful','sad','fatigued','neutral','valence','arousal','dominance'
    ]
    with open(pathCSV,'w') as pcsv:
        pcsv.write('%s,file\n' % (','.join([emotions[f] for f in range(len(emotions))])))
        for idx, p in enumerate(preds):
            for fp in p:
                pcsv.write(f'{fp},')
            if vad is not None:
                for val in vad[idx]:
                    pcsv.write(f'{val},')
            if emoLabels is not None:
                if mapping is not None:
                    pcsv.write(f'{mapping[emoLabels[idx]]},')
                else:
                    pcsv.write(f'{emoLabels[idx]},')
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
    elif args.emotionModel == "simpleNetwork":
        model = ResNet50WithAttentionLikelihoodNoVA(num_classes=args.classQuantity,bottleneck='none')

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
    
    dataset = AffectNet(afectdata=args.pathBase,transform=data_transforms,typeExperiment='UNIVERSAL_VAD_EXP')
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=False)
    model.eval()
    pathFile = []
    predictions = None
    labelsGrouped = None

    labelTrue = None
    labelPreds = None

    vadPreds = None
    vadTrue = None
    soft =  nn.Softmax(dim=1)
    with torch.no_grad():
        for idxBtc, data in enumerate(val_loader):
            printProgressBar(idxBtc,math.ceil(len(dataset) / args.batch),length=50,prefix='Loading Faces...')
            images, (labels, vadLabels, emotion), pathsForFiles = data            
            if not isinstance(model, ResNet50WithAttentionLikelihoodNoVA):                
                outputs, _, vad = model(images.to(device))
            else:
                outputs = model(images.to(device))
            outputs = soft(outputs)

            _, pred = torch.max(outputs.data,1)
            labelPreds = pred.cpu().detach().numpy() if labelPreds is None else np.concatenate((pred.cpu().detach().numpy(),labelPreds))
            labelTrue = emotion.cpu().detach().numpy() if labelTrue is None else np.concatenate((emotion.cpu().detach().numpy(),labelTrue))

            prediction = outputs.cpu().detach().numpy()
            predictions = prediction if predictions is None else np.concatenate((prediction,predictions))

            if not isinstance(model, ResNet50WithAttentionLikelihoodNoVA):
                vp = vad.cpu().detach().numpy()
                vadPreds = vp if vadPreds is None else np.concatenate((vp,vadPreds))

            currLabel = labels.cpu().detach().numpy()
            labelsGrouped = currLabel if labelsGrouped is None else np.concatenate((currLabel,labelsGrouped))

            if not isinstance(model, ResNet50WithAttentionLikelihoodNoVA):
                vadC = vadLabels.cpu().detach().numpy()
                vadTrue = vadC if vadTrue is None else np.concatenate((vadC,vadTrue))
            
            pathFile = list(pathsForFiles) + pathFile
            '''
            if args.emotionModel == "resnetAttentionGMM":
                for idx, am in enumerate(images):
                    visualizeAttentionMaps(images[idx],model.attention_maps,image_name=f"attention_map_{args.emotionModel}_{args.resnetInnerModel}_{pathsForFiles[idx].split(os.path.sep)[-1]}.png")
            '''
    saveToCSV(predictions,pathFile,args.output,emoLabels=labelPreds,mapping=[1,7,3,6,5,4,2,0])
    saveToCSV(labelsGrouped,pathFile,args.output[:-4]+"_labels.csv",emoLabels=labelTrue)

if __name__ == '__main__':
    train()