import argparse, torch, os, sys, numpy as np, math
from torchvision import transforms
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from DatasetClasses.AffectNet import AffectNet
from helper.function import saveStatePytorch, printProgressBar
from networks.ResnetEmotionHead import ResnetEmotionHeadClassifierAttention
from DatasetClasses.AffWild2 import AFF2Data
from torch import nn, optim

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
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model")
    model = ResnetEmotionHeadClassifierAttention(classes=8, resnetModel='resnet18')
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
            transforms.RandomApply([
                transforms.RandomAffine(20, scale=(0.8, 1), translate=(0.2, 0.2)),
            ], p=0.7),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.562454871481894, 0.8208898956471341, 0.395364053852456],
            std=[0.43727472598867456, 0.31812502566122625, 0.3796120355707891]
        )
    ]),
    'test' : transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.562454871481894, 0.8208898956471341, 0.395364053852456],
            std=[0.43727472598867456, 0.31812502566122625, 0.3796120355707891]
        )
    ])}
    print("Loading trainig set")
    dataset = AffectNet(afectdata=os.path.join(args.pathBase,'train_set'),transform=data_transforms['train'],typeExperiment='EXP',exchangeLabel=None)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize, shuffle=True)

    datasetVal = AffectNet(afectdata=os.path.join(args.pathBase,'val_set'),transform=data_transforms['test'],typeExperiment='EXP',exchangeLabel=None)
    val_loader = torch.utils.data.DataLoader(datasetVal, batch_size=args.batchSize, shuffle=False)

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(),lr = args.learningRate, momentum = 0.9)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learningRate)

    scheduler = optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.1)
    criterion = nn.CrossEntropyLoss().to(device)
    print("Started traning")
    print('Training Phase =================================================================== BTL  BVL BAC')
    bestForFold = bestForFoldTLoss = 500000
    bestRankForFold = -1
    for ep in range(args.epochs):
        ibl = ibr = ibtl = ' '
        model.train()
        lossAcc = []
        totalImages = 0
        iteration = 0
        for currBatch, currTargetBatch, _ in train_loader:
            printProgressBar(iteration,math.ceil(len(dataset.filesPath)/args.batchSize),length=50,prefix='Procesing face - training')
            totalImages += currBatch.shape[0]
            currTargetBatch, currBatch = currTargetBatch.to(device), currBatch.to(device)

            _, classification = model(currBatch)
            loss = criterion(classification, currTargetBatch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lossAcc.append(loss.item())
            iteration += 1

        lossAvg = sum(lossAcc) / len(lossAcc)
        scheduler.step()
        model.eval()
        total = 0
        correct = 0
        loss_val = []        
        with torch.no_grad():
            for data in val_loader:
                images, labels, _ = data
                _, classification = model(images.to(device))
                _, predicted = torch.max(classification.data, 1)
                loss = criterion(classification, labels.to(device))

                loss_val.append(loss)
                total += labels.size(0)
                correct += (predicted == labels.to(device)).sum().item()

        cResult = correct / total
        tLoss = sum(loss_val) / len(loss_val)
        state_dict = model.state_dict()
        opt_dict = optimizer.state_dict()

        if bestForFoldTLoss > tLoss:
            ibtl = 'X'
            fName = '%s_best_val_loss.pth.tar' % ('DAN')
            fName = os.path.join(args.output, fName)
            saveStatePytorch(fName, state_dict, opt_dict, ep + 1)
            bestForFoldTLoss = tLoss

        if bestForFold > lossAvg:
            ibl = 'X'
            fName = '%s_best_loss.pth.tar' % ('DAN')
            fName = os.path.join(args.output, fName)
            saveStatePytorch(fName, state_dict, opt_dict, ep + 1)
            bestForFold = lossAvg

        if bestRankForFold < cResult:
            ibr = 'X'
            fName = '%s_best_ccc.pth.tar' % ('DAN')
            fName = os.path.join(args.output, fName)
            saveStatePytorch(fName, state_dict, opt_dict, ep + 1)
            bestRankForFold = cResult

        print('[EPOCH %03d] T. Loss %.5f V. Loss %.5f V. Score %.5f [%c] [%c] [%c]' % (ep, lossAvg, tLoss, cResult, ibl,ibtl,ibr))

if __name__ == '__main__':
    train()