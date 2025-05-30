import argparse, torch, os, sys, numpy as np, math
from torchvision import transforms
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from loss.CenterLoss import CenterLoss
from DatasetClasses.AffectNet import AffectNet
from helper.function import saveStatePytorch, printProgressBar, loadNeighFiles
from DatasetClasses.JoinedDataset import JoinedDataset
from torch import nn, optim

import torch

def mmd_loss(x, y, kernel='rbf', sigma=None):
    """
    Calcula a MMD entre duas amostras x e y usando um kernel.
    
    Args:
        x (torch.Tensor): Amostras da distribuição P (shape: [batch_size, features]).
        y (torch.Tensor): Amostras da distribuição Q (shape: [batch_size, features]).
        kernel (str): Tipo de kernel ('rbf' ou 'linear').
        sigma (float): Largura do kernel RBF. Se None, usa a mediana das distâncias.
    
    Returns:
        torch.Tensor: Valor da MMD.
    """
    batch_size = x.size(0)
    
    # Kernel RBF
    if kernel == 'rbf':
        if sigma is None:
            # Calcula sigma usando a mediana das distâncias (heurística comum)
            xx = torch.cdist(x, x, p=2)
            yy = torch.cdist(y, y, p=2)
            xy = torch.cdist(x, y, p=2)
            sigma = (torch.median(xx) + torch.median(yy) + torch.median(xy)) / 3
        
        xx = torch.exp(-torch.cdist(x, x, p=2) / (2 * sigma**2))
        yy = torch.exp(-torch.cdist(y, y, p=2) / (2 * sigma**2))
        xy = torch.exp(-torch.cdist(x, y, p=2) / (2 * sigma**2))
    
    # Kernel linear
    elif kernel == 'linear':
        xx = torch.mm(x, x.t())
        yy = torch.mm(y, y.t())
        xy = torch.mm(x, y.t())
    
    else:
        raise ValueError("Kernel não suportado. Use 'rbf' ou 'linear'.")
    
    # MMD² = E[k(x, x')] + E[k(y, y')] - 2E[k(x, y)]
    mmd = xx.mean() + yy.mean() - 2 * xy.mean()
    return mmd

def train():
    parser = argparse.ArgumentParser(description='Finetune resnet')
    parser.add_argument('--pathBase', help='Path for valence and arousal dataset', required=True)
    parser.add_argument('--batchSize', type=int, help='Size of the batch', required=True)
    parser.add_argument('--epochs', type=int,help='Epochs to be run', required=True)
    parser.add_argument('--output', default=None, help='Folder to save weights', required=True)
    parser.add_argument('--learningRate', help='Learning Rate', required=False, default=0.01, type=float)
    parser.add_argument('--tensorboardname', help='Learning Rate', required=False, default='RESNETPROB')
    parser.add_argument('--optimizer', help='Optimizer', required=False, default="sgd")
    parser.add_argument('--freeze', help='Freeze weights', required=False, type=int, default=0)
    parser.add_argument('--numberOfClasses', help='Freeze weights', required=False, type=int, default=13)
    parser.add_argument('--trainDataset', help='File with neighbours', required=False,default="affectnet")
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    writer = SummaryWriter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model -- Using " + str(device))
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, args.numberOfClasses,bias=False)
    if args.freeze:
        print("Freezing weights")
        for param in model.parameters():
            param.requires_grad = bool(args.freeze)
    model.to(device)    
    print("Model loaded")
    print(model)
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ]),
        'test' : transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ]
    )}
    print("Loading trainig set")
    if args.trainDataset == 'affectnet':
        dataset = AffectNet(afectdata=os.path.join(args.pathBase,'train_set'),transform=data_transforms['train'],typeExperiment='PROBS',exchangeLabel=None)        
    elif args.trainDataset == 'joineddataset':
        dataset = JoinedDataset(args.pathBase,transform=data_transforms['train'])

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize, shuffle=True)

    datasetVal = AffectNet(afectdata=os.path.join(args.pathBase,'val_set'),transform=data_transforms['test'],typeExperiment='PROBS',exchangeLabel=None,loadLastLabel=(args.trainDataset != 'joineddataset'))
    val_loader = torch.utils.data.DataLoader(datasetVal, batch_size=args.batchSize, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=args.learningRate)

    scheduler = optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.1)


    criterion = mmd_loss
    os.system('cls' if os.name == 'nt' else 'clear')
    print("Started traning")
    print('Training Phase =================================================================== BTL  BVL BAC')
    bestForFold = bestForFoldTLoss = 500000
    bestRankForFold = -1
    alpha = 0.1
    nFile = None     

    for ep in range(args.epochs):
        ibl = ibr = ibtl = ' '
        model.train()
        lossAcc = []
        totalImages = 0
        iteration = 0
        for currBatch, currTargetBatch, pathfile in train_loader:
            logitsClass = np.zeros((currBatch.shape[0],1)).flatten()
            printProgressBar(iteration,math.ceil(len(dataset.filesPath)/args.batchSize),length=50,prefix='Procesing face - training')
            totalImages += currBatch.shape[0]
            currTargetBatch, currBatch = currTargetBatch.to(device), currBatch.to(device)
            classification = model(currBatch)
            loss = criterion(classification, currTargetBatch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lossAcc.append(loss.item())
            iteration += 1

        lossAvg = sum(lossAcc) / len(lossAcc)
        writer.add_scalar('RESNETAtt/Loss/train', lossAvg, ep)
        scheduler.step()
        model.eval()
        total = 0
        correct = 0
        loss_val = []
        iteration = 0
        with torch.no_grad():
            for data in val_loader:
                printProgressBar(iteration,math.ceil(len(datasetVal.filesPath)/args.batchSize),length=50,prefix='Procesing face - testing')
                images, labels, _ = data
                classification = model(images.to(device))
                labels = labels.to(device)

                loss = criterion(classification, labels)

                loss_val.append(loss)
                total += labels.size(0)
                iteration += 1

        cResult = correct / total        
        tLoss = sum(loss_val) / len(loss_val)
        writer.add_scalar('RESNETAtt/Loss/val', tLoss, ep)
        writer.add_scalar('RESNETAtt/Acc', cResult, ep)
        state_dict = model.state_dict()
        opt_dict = optimizer.state_dict()

        if bestForFoldTLoss > tLoss:
            ibtl = 'X'
            fName = '%s_best_val_loss.pth.tar' % ('RESNETATT')
            fName = os.path.join(args.output, fName)
            saveStatePytorch(fName, state_dict, opt_dict, ep + 1)
            bestForFoldTLoss = tLoss

        if bestForFold > lossAvg:
            ibl = 'X'
            fName = '%s_best_loss.pth.tar' % ('RESNETATT')
            fName = os.path.join(args.output, fName)
            saveStatePytorch(fName, state_dict, opt_dict, ep + 1)
            bestForFold = lossAvg

        if bestRankForFold < cResult:
            ibr = 'X'
            fName = '%s_best_ccc.pth.tar' % ('RESNETATT')
            fName = os.path.join(args.output, fName)
            saveStatePytorch(fName, state_dict, opt_dict, ep + 1)
            bestRankForFold = cResult

        print('[EPOCH %03d] T. Loss %.5f V. Loss %.5f [%c] [%c]' % (ep, lossAvg, tLoss, ibl,ibtl))

if __name__ == '__main__':
    train()