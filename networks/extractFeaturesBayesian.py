import argparse, torch, os, sys, numpy as np
from torchvision import transforms
from torchvision import models
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from DatasetClasses.AffectNet import AffectNet
from torch import nn
from networks.EmotionResnetVA import ResnetWithBayesianHead, ResnetWithBayesianGMMHead, ResNet50WithAttentionGMM
from helper.function import visualizeAttentionMaps

def saveToCSV(preds,files,pathCSV):
    emotions = ["neutral","happy","sad","surprised","fear","disgust","angry","contempt","serene","contemplative","secure","untroubled","quiet"]
    with open(pathCSV,'w') as pcsv:
        pcsv.write('%s,file\n' % (','.join([emotions[f] for f in range(len(preds[0]))])))
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
    args = parser.parse_args()

    checkpoint = torch.load(args.weights)

    model = None
    if args.emotionModel == "resnetBayesGMM":
        model = ResnetWithBayesianGMMHead(classes=13,resnetModel=args.resnetInnerModel)
    elif args.emotionModel == "resnetBayes":
        model = ResnetWithBayesianHead(13,resnetModel=args.resnetInnerModel)
    elif args.emotionModel == "resnetAttentionGMM":
        model = ResNet50WithAttentionGMM(num_classes=13,bottleneck='none')
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
    
    dataset = AffectNet(afectdata=args.pathBase,transform=data_transforms,typeExperiment='EXP')
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=False)
    model.eval()
    pathFile = []
    predictions = None
    soft =  nn.Softmax(dim=1)
    with torch.no_grad():
        for idxBtc, data in enumerate(val_loader):
            print("Extraction Batch %d" % (idxBtc))
            images, labels, pathsForFiles = data
            outputs = model(images.to(device))[0]
            outputs = soft(outputs)

            prediction = outputs.cpu().detach().numpy()
            predictions = prediction if predictions is None else np.concatenate((prediction,predictions))

            pathFile = list(pathsForFiles) + pathFile
            '''
            if args.emotionModel == "resnetAttentionGMM":
                for idx, am in enumerate(images):
                    visualizeAttentionMaps(images[idx],model.attention_maps,image_name=f"attention_map_{args.emotionModel}_{args.resnetInnerModel}_{pathsForFiles[idx].split(os.path.sep)[-1]}.png")
            '''
    saveToCSV(predictions,pathFile,args.output)

if __name__ == '__main__':
    train()