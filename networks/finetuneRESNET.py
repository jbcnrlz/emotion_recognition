import argparse, torch, os, sys, numpy as np
from torchvision import transforms, models
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import ccc, saveStatePytorch
from networks.DANVA import DANVA
from DatasetClasses.AffWild2 import AFF2Data
from DatasetClasses.AffectNet import AffectNet
from torch import nn, optim

def train():
    parser = argparse.ArgumentParser(description='Finetune DAN to VA')
    parser.add_argument('--pathBase', help='Path for valence and arousal dataset', required=True)
    parser.add_argument('--batch', type=int, help='Size of the batch', required=True)
    parser.add_argument('--epochs', type=int,help='Epochs to be run', required=True)
    parser.add_argument('--output', default=None, help='Folder to save weights', required=True)
    parser.add_argument('--learningRate', help='Learning Rate', required=False, default=0.01, type=float)
    parser.add_argument('--tensorboardname', help='Learning Rate', required=False, default='RESNET50')
    parser.add_argument('--optimizer', help='Optimizer', required=False, default="sgd")
    parser.add_argument('--dataset', help='Freeze weights', required=False, default="AFFWILD2")
    parser.add_argument('--quantityOfTerms', help='Quantity of terms to use', required=False, default=151,type=int)
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model")
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, args.quantityOfTerms)
    model.to(device)
    print("Model loaded")
    print(model)
    data_transforms_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
                transforms.RandomAffine(20, scale=(0.8, 1), translate=(0.2, 0.2)),
            ], p=0.7),

        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(),
    ])

    data_transforms_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("Loading trainig set")
    if args.dataset == "AFFWILD2":
        afw2Train = AFF2Data(args.pathBase,'Train_Set',transform=data_transforms_train,type='TERMS',termsQuantity=args.quantityOfTerms)    
        afw2Val = AFF2Data(args.pathBase,'Validation_Set',transform=data_transforms_val,type='TERMS',termsQuantity=args.quantityOfTerms)
    elif args.dataset == "AFFECTNET":
        afw2Val = AffectNet(os.path.join(args.pathBase,'val_set'),'TERMS',transform=data_transforms_val,termsQuantity=args.quantityOfTerms)
        afw2Train = AffectNet(os.path.join(args.pathBase,'train_set'),'TERMS',transform=data_transforms_train,termsQuantity=args.quantityOfTerms)            
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

            output = model(currBatch)
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
        correct = 0
        loss_val = []
        with torch.no_grad():
            for data in val_loader:
                images, labels, _ = data                
                outputs = model(images.to(device))

                loss = criterion(outputs, labels.type(torch.LongTensor).to(device))
                loss_val.append(loss)
                total += labels.size(0)

                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels.to(device)).sum().item()


        tLoss = sum(loss_val) / len(loss_val)
        vResult = correct / total
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