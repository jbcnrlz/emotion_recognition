import argparse, torch, os, sys, numpy as np, math, random
from torchvision import transforms
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from loss.CenterLoss import CenterLoss
from loss.LTRLosses import approxNDCGLoss, rankNet, neuralNDCG
from DatasetClasses.AffectNet import AffectNet
from helper.function import saveStatePytorch, printProgressBar, loadNeighFiles
from networks.ResnetEmotionHead import ResnetEmotionHeadClassifierAttention, ResnetEmotionHeadDANImplementation
from DatasetClasses.AffWild2 import AFF2Data
from DatasetClasses.JoinedDataset import JoinedDataset
from torch import nn, optim
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm

def getRanks(vaLabel,vaDists):
    dists = torch.cdist(vaLabel,vaDists,p=2)
    return dists



def train():
    parser = argparse.ArgumentParser(description='Finetune resnet')
    parser.add_argument('--pathBase', help='Path for valence and arousal dataset', required=True)
    parser.add_argument('--batchSize', type=int, help='Size of the batch', required=True)
    parser.add_argument('--epochs', type=int,help='Epochs to be run', required=True)
    parser.add_argument('--output', default=None, help='Folder to save weights', required=True)
    parser.add_argument('--learningRate', help='Learning Rate', required=False, default=0.01, type=float)
    parser.add_argument('--tensorboardname', help='Learning Rate', required=False, default='DANVA')
    parser.add_argument('--optimizer', help='Optimizer', required=False, default="sgd")
    parser.add_argument('--freeze', help='Freeze weights', required=False, type=int, default=0)
    parser.add_argument('--numberOfClasses', help='Freeze weights', required=False, type=int, default=0)
    parser.add_argument('--additiveLoss', help='Adding additive Loss', required=False,default=None)
    parser.add_argument('--neighsFiles', help='File with neighbours', required=False,default=None)
    parser.add_argument('--trainDataset', help='File with neighbours', required=False,default="affectnet")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classesDist = torch.from_numpy(np.array([
        [0,0],
        [0.81,0.51],
        [-0.63,-0.27],
        [0.4,0.67],
        [-0.64,0.6],
        [-0.6,0.35],
        [-0.51,0.59],
        [-0.23,0.31]
    ])).type(torch.FloatTensor).to(device)
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    writer = SummaryWriter()    
    print("Loading model -- Using " + str(device))
    model = models.resnet18(pretrained=False)
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
    ])}
    print("Loading trainig set")
    dataset = AffectNet(afectdata=os.path.join(args.pathBase,'train_set'),transform=data_transforms['train'],typeExperiment='BOTH',exchangeLabel=None)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize, shuffle=True)

    datasetVal = AffectNet(afectdata=os.path.join(args.pathBase,'val_set'),transform=data_transforms['test'],typeExperiment='BOTH',exchangeLabel=None)
    val_loader = torch.utils.data.DataLoader(datasetVal, batch_size=args.batchSize, shuffle=False)

    if args.additiveLoss == 'centerloss':
        print("Using CenterLoss with ADAM")
        cLoss = CenterLoss(args.numberOfClasses,512).to(device)
        params = list(model.parameters()) + list(cLoss.parameters())
        optimizer = optim.Adam(params,lr=args.learningRate)
    else:
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(),lr = args.learningRate, momentum = 0.9)
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.learningRate)

    scheduler = optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.1)
    criterion = nn.CrossEntropyLoss().to(device)

    #similarityScore = nn.CosineEmbeddingLoss().to(device)
    multiLabelLoss = nn.MultiLabelSoftMarginLoss().to(device)

    os.system('cls' if os.name == 'nt' else 'clear')
    print("Started traning")
    print('Training Phase =================================================================== BTL  BVL BAC')
    bestForFold = bestForFoldTLoss = 500000
    bestRankForFold = -1
    alpha = 0.1
    wtsCEL = 0.6
    wtsMLL = 0.4
    for ep in range(args.epochs):
        ibl = ibr = ibtl = ' '
        model.train()
        lossAcc = []
        ceLossHist = []
        rankNetLossHist = []
        totalImages = 0
        iteration = 0
        for currBatch, currTargetBatch, pathfile, vaBatch in train_loader:
            printProgressBar(iteration,math.ceil(len(dataset.filesPath)/args.batchSize),length=50,prefix='Procesing face - training')
            totalImages += currBatch.shape[0]
            currTargetBatch, currBatch = currTargetBatch.to(device), currBatch.to(device)
            classification = model(currBatch)

            #loss = criterion(classification, currTargetBatch) + similarityScore(features,featuresSimilar,torch.ones(features.shape[0]).to(device))
            #loss = criterion(classification, currTargetBatch)
            #loss = (wtsCEL * criterion(classification, currTargetBatch)) + (wtsMLL * multiLabelLoss(classification, labelsToCompare[currTargetBatch]))
            #loss = criterion(classification, currTargetBatch)
            currRanking = getRanks(vaBatch.to(device),classesDist) 
            lrank = rankNet(classification,currRanking) 
            clValue = criterion(classification, currTargetBatch)
            loss = lrank * 0.4 + 0.6 * clValue
            #loss= clValue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lossAcc.append(loss.item())
            ceLossHist.append(clValue.item())
            rankNetLossHist.append(lrank.item())
            iteration += 1

        lossAvg = sum(lossAcc) / len(lossAcc)
        ceLossHist = sum(ceLossHist) / len(ceLossHist)
        rankNetLossHist = sum(rankNetLossHist) / len(rankNetLossHist)
        writer.add_scalar('RESNETAtt/Loss/train', lossAvg, ep)
        writer.add_scalar('RESNETAtt/CELoss/train', ceLossHist,ep)
        writer.add_scalar('RESNETAtt/Ranknet/train', rankNetLossHist,ep)
        scheduler.step()
        model.eval()
        total = 0
        correct = 0
        loss_val = []
        ceLossHist = []
        rankNetLossHist = []
        iteration = 0
        with torch.no_grad():
            for data in val_loader:
                printProgressBar(iteration,math.ceil(len(datasetVal.filesPath)/args.batchSize),length=50,prefix='Procesing face - testing')
                images, labels, _, vaBatch = data
                classification = model(images.to(device))
                _, predicted = torch.max(classification.data, 1)
                labels = labels.to(device)

                #loss = criterion(classification, labels)
                #loss = (wtsCEL * criterion(classification, labels)) + (wtsMLL * multiLabelLoss(classification, labelsToCompare[labels]))
                #loss = criterion(classification, labels)                    
                currRanking = getRanks(vaBatch.to(device),classesDist) 
                lrank = rankNet(classification,currRanking) 
                clValue = criterion(classification, labels)
                loss = lrank * 0.4 + 0.6 * clValue
                #loss = clValue

                loss_val.append(loss)
                ceLossHist.append(clValue.item())
                rankNetLossHist.append(lrank.item())
                total += labels.size(0)
                correct += (predicted == labels.to(device)).sum().item()
                iteration += 1

        cResult = correct / total
        tLoss = sum(loss_val) / len(loss_val)
        ceLossHist = sum(ceLossHist) / len(ceLossHist)
        rankNetLossHist = sum(rankNetLossHist) / len(rankNetLossHist)
        writer.add_scalar('RESNETAtt/Loss/val', tLoss, ep)
        writer.add_scalar('RESNETAtt/CELoss/val', ceLossHist,ep)
        writer.add_scalar('RESNETAtt/Ranknet/val', rankNetLossHist,ep)
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

        print('[EPOCH %03d] T. Loss %.5f V. Loss %.5f V. Score %.5f [%c] [%c] [%c]' % (ep, lossAvg, tLoss, cResult, ibl,ibtl,ibr))

if __name__ == '__main__':
    train()