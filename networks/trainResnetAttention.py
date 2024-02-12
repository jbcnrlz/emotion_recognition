import argparse, torch, os, sys, numpy as np, math
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from loss.CenterLoss import CenterLoss
from DatasetClasses.AffectNet import AffectNet
from helper.function import saveStatePytorch, printProgressBar, loadNeighFiles
from networks.ResnetEmotionHead import ResnetEmotionHeadClassifierAttention, ResnetEmotionHeadDANImplementation
from DatasetClasses.AffWild2 import AFF2Data
from DatasetClasses.JoinedDataset import JoinedDataset
from torch import nn, optim
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm

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

    classesDist = np.array(
        [
            [0,0.0001],   [0.605,0.21],  [-0.522,0.15],[0.605,0.21], #neutral, happy, sad, surprise
            [-0.522,0.15],[-0.522,0.15 ],[-0.522,0.15],[-0.522,0.15]  #fear, disgust, anger, contempt
        ], dtype = np.float32
    )

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    writer = SummaryWriter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model -- Using " + str(device))
    model = ResnetEmotionHeadClassifierAttention(classes=args.numberOfClasses, resnetModel='resnet18')
    #model = ResnetEmotionHeadDANImplementation(classes=args.numberOfClasses)
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
        dataset = AffectNet(afectdata=os.path.join(args.pathBase,'train_set'),transform=data_transforms['train'],typeExperiment='EXP',exchangeLabel=None)        
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
    os.system('cls' if os.name == 'nt' else 'clear')
    print("Started traning")
    print('Training Phase =================================================================== BTL  BVL BAC')
    bestForFold = bestForFoldTLoss = 500000
    bestRankForFold = -1
    alpha = 0.1
    nFile = None
    if args.neighsFiles is not None:
        nFile = loadNeighFiles(args.neighsFiles)        

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
            features, classification, _ = model(currBatch)
            if nFile is not None:
                for idx, p in enumerate(pathfile):
                    expected = classification.clone().detach().cpu()
                    gnb = LogisticRegression(C=10,penalty='l1',solver='saga',multi_class='multinomial',max_iter=100000)
                    cFileName = p.split(os.path.sep)[-1]
                    cNs = np.array(nFile[cFileName]['neighbours'])
                    if (len(np.unique(cNs[:,-1])) > 1):
                        #cNs[:,-1][cNs[:,-1] > 1] = 1
                        if np.all(cNs[:,0] == cNs[0,0]) and np.all(cNs[:,1] == cNs[0,1]):
                            logitsClass[idx] = int(cNs[:,-1][0])
                        else:
                            for idxN in range(2):
                                cNs[cNs[:,idxN] == 0,idxN] = 1e-10
                            gnb.fit(cNs[:,:-1],cNs[:,-1])
                            logitsClass[idx] = int(gnb.predict(np.array([nFile[cFileName]['va']]))[0])
                    else:
                        logitsClass[idx] = currTargetBatch[idx].clone().detach().cpu()

                labelTr = torch.tensor(logitsClass,dtype=torch.long).to(device)

            if nFile is not None:
                expected = classification.clone().detach().cpu()
                fResult = []
                for valCalc in range(expected.shape[0]):
                    fResult.append(norm.pdf(expected[valCalc],loc=classesDist[labelTr[valCalc]][0],scale=classesDist[labelTr[valCalc]][1]))
                classification = classification * torch.tensor(np.array(fResult,dtype=np.float32)).to(device)


            if args.additiveLoss is not None:
                cLossV = alpha * cLoss(features,currTargetBatch)
                ceLossV = criterion(classification, currTargetBatch)
                loss = cLossV + ceLossV
            else:
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
                features, classification, _ = model(images.to(device))
                _, predicted = torch.max(classification.data, 1)
                labels = labels.to(device)

                if args.additiveLoss is not None:
                    cLossV = alpha * cLoss(features,labels)
                    ceLossV = criterion(classification, labels)
                    loss = cLossV + ceLossV
                else:
                    loss = criterion(classification, labels)

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