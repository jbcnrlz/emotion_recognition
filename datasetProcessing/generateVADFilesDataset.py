import argparse, torch, os, sys, numpy as np, math, random, re
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from DatasetClasses.AffectNet import AffectNet
from helper.function import saveStatePytorch, printProgressBar, overlay_attention_maps
from networks.EmotionResnetVA import ResnetWithBayesianGMMHead, ResNet50WithAttentionGMM
from torch import nn, optim
import torch.distributions as dist, random
from torch.nn import functional as F
from loss.FocalLoss import FocalLoss


def train():
    parser = argparse.ArgumentParser(description='Finetune resnet')
    parser.add_argument('--pathBase', help='Path for valence and arousal dataset', required=True)
    parser.add_argument('--batchSize', type=int, help='Size of the batch', required=True)
    parser.add_argument('--output', default=None, help='Folder to save weights', required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labelsAffectNet = ['neutral','happy','sad','surprise','fear','disgust','anger','contempt','none','uncertain']
    data_transforms = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    print("Loading trainig set")
    #dataset = AffectNet(afectdata=os.path.join(args.pathBase,'train_set'),transform=data_transforms,typeExperiment='PROBS_VAD')
    #train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize, shuffle=True)
    datasetVal = AffectNet(afectdata=os.path.join(args.pathBase,'val_set'),transform=data_transforms,typeExperiment='PROBS_VAD_EXP')
    val_loader = torch.utils.data.DataLoader(datasetVal, batch_size=args.batchSize, shuffle=False)

    loaders = [val_loader]
    outputValues = []
    for l in loaders:
        iteration = 0
        for idx, (_, currTargetBatch, paths) in enumerate(l):
            printProgressBar(iteration, len(datasetVal.filesPath), prefix = 'Progress:', suffix = 'Complete', length = 50)
            vaBatch = currTargetBatch[1]
            outputValues = outputValues + [ [v[0],v[1],v[2],paths[idx],labelsAffectNet[int(currTargetBatch[2][idx])]] for idx, v in enumerate(vaBatch.cpu().numpy())]
            iteration+=1

    with open(args.output,'w') as f:
        f.write("valence,arousal,dominance,class,path\n")
        for o in outputValues:        
            f.write(f"{o[0]},{o[1]},{o[2]},{o[4]},{o[3]}\n")

if __name__ == '__main__':
    train()