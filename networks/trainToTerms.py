import argparse, torch, os, sys, numpy as np
from torchvision import transforms
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import ccc, saveStatePytorch
from networks.DANVA import DANVA
from DatasetClasses.AffWild2 import AFF2Data
from DatasetClasses.AffectNet import AffectNet
from torch import nn, optim

def train():
    parser = argparse.ArgumentParser(description='Finetune DAN to VA')
    parser.add_argument('--resnetPretrained', help='Path for resnet pretrained weights', required=True)
    parser.add_argument('--originalWeights', help='Original Weights', required=True)
    parser.add_argument('--pathBase', help='Path for valence and arousal dataset', required=True)
    parser.add_argument('--batch', type=int, help='Size of the batch', required=True)
    parser.add_argument('--epochs', type=int,help='Epochs to be run', required=True)
    parser.add_argument('--output', default=None, help='Folder to save weights', required=True)
    parser.add_argument('--learningRate', help='Learning Rate', required=False, default=0.01, type=float)
    parser.add_argument('--tensorboardname', help='Learning Rate', required=False, default='DANVA')
    parser.add_argument('--optimizer', help='Optimizer', required=False, default="sgd")
    parser.add_argument('--freeze', help='Freeze weights', required=False, type=int, default=0)
    parser.add_argument('--dataset', help='Freeze weights', required=False, default="AFFWILD2")
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model")
    model = DANVA(num_class=8, num_head=4,pretrained=args.resnetPretrained)
    checkpoint = torch.load(args.originalWeights)
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    if args.freeze:
        print("Freezing weights")
        for param in model.parameters():
            param.requires_grad = bool(args.freeze)
    model.convertToTerms(151)
    model.to(device)
    print("Model loaded")
    print(model)
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
                transforms.RandomAffine(20, scale=(0.8, 1), translate=(0.2, 0.2)),
            ], p=0.7),

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(),
    ])
    print("Loading trainig set")
    if args.dataset == "AFFWILD2":
        afw2Train = AFF2Data(args.pathBase,'Train_Set',transform=data_transforms,type='TERMS')    
        afw2Val = AFF2Data(args.pathBase,'Validation_Set',transform=data_transforms,type='TERMS')
    elif args.dataset == "AFFECTNET":
        afw2Val = AffectNet(os.path.join(args.pathBase,'val_set'),'TERMS',transform=data_transforms)
        afw2Train = AffectNet(os.path.join(args.pathBase,'train_set'),'TERMS',transform=data_transforms)            
    gal_loader = torch.utils.data.DataLoader(afw2Train, batch_size=args.batch, shuffle=True)
    val_loader = torch.utils.data.DataLoader(afw2Val, batch_size=args.batch, shuffle=False)
    criterion = nn.CrossEntropyLoss().to(device)

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(),lr = args.learningRate, momentum = 0.9)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learningRate)

    scheduler = optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.1)

    print("Started traning")
    bestForFold = bestForFoldTLoss = 500000
    bestRankForFold = -1
    for ep in range(args.epochs):
        ibl = ibr = ibtl = ' '
        model.train()
        lossAcc = []
        totalImages = 0        
        prediction = gt = None
        correct = 0
        for currBatch, currTargetBatch, _ in gal_loader:
            totalImages += currBatch.shape[0]
            currTargetBatch, currBatch = currTargetBatch.type(torch.LongTensor).to(device), currBatch.to(device)

            output, _, _ = model(currBatch)
            loss = criterion(output, currTargetBatch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lossAcc.append(loss.item())

            _, predicted = torch.max(output.data, 1)
            correct += (predicted == currTargetBatch).sum().item()

        lossAvg = sum(lossAcc) / len(lossAcc)
        trainingResult = correct / totalImages
        scheduler.step()
        model.eval()
        total = 0
        loss_val = []
        with torch.no_grad():
            for data in val_loader:
                images, labels, _ = data
                totalImages += labels.shape[0]
                outputs, _, _ = model(images.to(device))

                loss = criterion(outputs, labels.type(torch.LongTensor).to(device))
                loss_val.append(loss)
                total += labels.size(0)

                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels.to(device)).sum().item()


        tLoss = sum(loss_val) / len(loss_val)
        vResult = correct / totalImages
        state_dict = model.state_dict()
        opt_dict = optimizer.state_dict()

        if bestForFoldTLoss > tLoss:
            ibtl = 'X'
            fName = '%s_best_val_loss.pth.tar' % ('DANTERM')
            fName = os.path.join(args.output, fName)
            saveStatePytorch(fName, state_dict, opt_dict, ep + 1)
            bestForFoldTLoss = tLoss

        if bestForFold > lossAvg:
            ibl = 'X'
            fName = '%s_best_loss.pth.tar' % ('DANTERM')
            fName = os.path.join(args.output, fName)
            saveStatePytorch(fName, state_dict, opt_dict, ep + 1)
            bestForFold = lossAvg

        print('[EPOCH %03d] T. Loss %.5f T. Acc %.5f V. Loss %.5f  V. Acc %.5f [%c] [%c]' % (ep, lossAvg, trainingResult, tLoss, vResult, ibl,ibtl))

if __name__ == '__main__':
    train()