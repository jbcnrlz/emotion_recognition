import argparse, torch, os, sys, numpy as np, math, random
from torchvision import transforms
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

def getLogits(classSimilarity):
    returnLogits = np.zeros((len(classSimilarity),len(classSimilarity)))
    for currClass, c in enumerate(classSimilarity):
        if len(c[0]) == 1 and c[0][0] == currClass:
            returnLogits[currClass][:] = 0.03
            returnLogits[currClass][0] = 0.82            
        else:
            validLabels = [currClass] + c[0]
            returnLogits[currClass][:] = 0.1 / len(validLabels)
            returnLogits[currClass][currClass] = 0.6
            returnLogits[currClass][c[0]] = 0.3 / len(c[0])                    

    return returnLogits.astype(np.float64)



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
    classesDist = [
            [[0],[1,2,3,4,5,6]],   
            [[3],[2,4,5,6]],  
            [[2],[1,3]],
            [[1],[2,4,5,6,]],
            [[5,6,],[1,3]],
            [[4,6],[1,3]],
            [[4,5],[1,3]],
            #[[2,4,5,6],[1,3]],
        ]
    labelsToCompare = torch.from_numpy(getLogits(classesDist)).type(torch.FloatTensor).to(device)
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    writer = SummaryWriter()    
    print("Loading model -- Using " + str(device))
    model = ResnetEmotionHeadClassifierAttention(classes=args.numberOfClasses, resnetModel='resnet18')
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
    if args.trainDataset == 'affectnet':
        dataset = AffectNet(afectdata=os.path.join(args.pathBase,'train_set'),transform=data_transforms['train'],typeExperiment='EXP',exchangeLabel=None,loadLastLabel=(args.trainDataset != 'joineddataset'))
    elif args.trainDataset == 'joineddataset':
        dataset = JoinedDataset(args.pathBase,transform=data_transforms['train'])

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize, shuffle=True)

    datasetVal = AffectNet(afectdata=os.path.join(args.pathBase,'val_set'),transform=data_transforms['test'],typeExperiment='EXP',exchangeLabel=None,loadLastLabel=(args.trainDataset != 'joineddataset'))
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
        totalImages = 0
        iteration = 0
        for currBatch, currTargetBatch, pathfile in train_loader:
            #samplesFortesting = dataset.sample(classes=list(range(args.numberOfClasses)),exclude=pathfile).to(device)
            #similarIndexes = [classesDist[cClose][0][random.randint(0,len(classesDist[cClose][0])-1)] for cClose in currTargetBatch]
            printProgressBar(iteration,math.ceil(len(dataset.filesPath)/args.batchSize),length=50,prefix='Procesing face - training')
            totalImages += currBatch.shape[0]
            currTargetBatch, currBatch = currTargetBatch.to(device), currBatch.to(device)            

            features, classification, _ = model(currBatch)            
            #featuresSimilar, _, _ = model(samplesFortesting[similarIndexes])

            #similar, nonSimilar = getSamples(features,currTargetBatch,classesDist)

            if args.additiveLoss is not None:
                cLossV = alpha * cLoss(features,currTargetBatch)
                ceLossV = criterion(classification, currTargetBatch)
                loss = cLossV + ceLossV
            else:
                #loss = criterion(classification, currTargetBatch) + similarityScore(features,featuresSimilar,torch.ones(features.shape[0]).to(device))
                #loss = criterion(classification, currTargetBatch)
                #loss = (wtsCEL * criterion(classification, currTargetBatch)) + (wtsMLL * multiLabelLoss(classification, labelsToCompare[currTargetBatch]))
                #loss = criterion(classification, currTargetBatch)
                loss = neuralNDCG(classification,labelsToCompare[currTargetBatch])

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
                features, classification, _ = model(images.to(device))
                _, predicted = torch.max(classification.data, 1)
                labels = labels.to(device)

                if args.additiveLoss is not None:
                    cLossV = alpha * cLoss(features,labels)
                    ceLossV = criterion(classification, labels)
                    loss = cLossV + ceLossV
                else:
                    #loss = criterion(classification, labels)
                    #loss = (wtsCEL * criterion(classification, labels)) + (wtsMLL * multiLabelLoss(classification, labelsToCompare[labels]))
                    #loss = criterion(classification, labels)
                    loss = neuralNDCG(classification,labelsToCompare[labels])

                loss_val.append(loss)
                total += labels.size(0)
                correct += (predicted == labels.to(device)).sum().item()
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

        print('[EPOCH %03d] T. Loss %.5f V. Loss %.5f V. Score %.5f [%c] [%c] [%c]' % (ep, lossAvg, tLoss, cResult, ibl,ibtl,ibr))

if __name__ == '__main__':
    train()