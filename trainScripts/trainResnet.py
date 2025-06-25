import argparse, torch, os, sys, numpy as np, math, random
from torchvision import transforms, datasets
from torchvision.models import resnet50
from torch import nn, optim
import torch.distributions as dist
from torch.utils.data import DataLoader
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import saveStatePytorch, printProgressBar

def main():
    parser = argparse.ArgumentParser(description='Finetune resnet')
    parser.add_argument('--batchSize', type=int, help='Size of the batch', required=True)
    parser.add_argument('--epochs', type=int,help='Epochs to be run', required=True)
    parser.add_argument('--output', default=None, help='Folder to save weights', required=True)
    parser.add_argument('--learningRate', help='Learning Rate', required=False, default=0.01, type=float)
    args = parser.parse_args()


    trans = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])    
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasetTrain = datasets.CelebA(root='data', split='train', download=True, transform=trans, target_type='identity')
    train_loader = DataLoader(
        dataset=datasetTrain,
        batch_size=args.batchSize,
        shuffle=True
    )

    transVal = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])    
    ])
    datasetTest = datasets.CelebA(root='data', split='valid', download=True, transform=transVal, target_type='identity')   
    val_loader = DataLoader(
        dataset=datasetTest,
        batch_size=args.batchSize,
        shuffle=False,    
    )

    print("Loading model -- Using " + str(device))

    model = resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10177)  # Assuming 10 classes for CelebA identity
    model.to(device)    

    print("Model loaded")
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=args.learningRate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.1)
    criterion = nn.CrossEntropyLoss().to(device)

    os.system('cls' if os.name == 'nt' else 'clear')
    print("Started traning")
    print('Training Phase =================================================================== BTL  BVL BAC')

    for ep in range(args.epochs):
        ibl = ibr = ibtl = ' '
        model.train()
        lossAvg = []
        totalImages = 0
        iteration = 0
        for currBatch, currTargetBatch in train_loader:
            printProgressBar(iteration,math.ceil(len(datasetTrain)/args.batchSize),length=50,prefix='Procesing face - training')
            totalImages += currBatch.shape[0]
            currTargetBatch, currBatch = currTargetBatch.to(device), currBatch.to(device)

            classification = model(currBatch)
            loss = criterion(classification, currTargetBatch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lossAvg.append(loss.item())
            iteration += 1

        lossAvg = sum(lossAvg) / len(lossAvg)
        scheduler.step()

        model.eval()
        loss_val = []
        iteration = 0
        correct = 0
        with torch.no_grad():
            for currBatch, currTargetBatch in val_loader:
                printProgressBar(iteration,math.ceil(len(datasetTest)/args.batchSize),length=50,prefix='Procesing face - testing')

                totalImages += currBatch.shape[0]
                currTargetBatch, currBatch = currTargetBatch.to(device), currBatch.to(device)

                classification = model(currBatch)
                loss = criterion(classification, currTargetBatch)                

                loss_val.append(loss.item())

                _, classes_preditas = torch.max(classification, dim=1)
                correct += (classes_preditas == currTargetBatch).sum().item()

                iteration += 1

        loss_val = sum(loss_val) / len(loss_val)
        cResult = correct / totalImages
        state_dict = model.state_dict()
        opt_dict = optimizer.state_dict()

        if bestForFold < cResult:
            ibr = 'X'
            fName = '%s_best_rank.pth.tar' % ('GioGio')
            fName = os.path.join(args.output, fName)
            #saveStatePytorch(fName, state_dict, opt_dict, ep + 1)
            bestForFold = cResult

        if bestForFoldTLoss > loss_val:
            ibtl = 'X'
            fName = '%s_best_val_loss.pth.tar' % ('RESNETATT')
            fName = os.path.join(args.output, fName)
            saveStatePytorch(fName, state_dict, opt_dict, ep + 1)
            bestForFoldTLoss = loss_val

        if bestForFold > lossAvg:
            ibl = 'X'
            fName = '%s_best_loss.pth.tar' % ('RESNETATT')
            fName = os.path.join(args.output, fName)
            saveStatePytorch(fName, state_dict, opt_dict, ep + 1)
            bestForFold = lossAvg        

        print(f'[EPOCH {ep:03d}] Accuracy of the network f{cResult} Training Loss {lossAvg} Validation Loss {loss_val} [{ibr}] [{ibl}] [{ibtl}]')        


if __name__ == "__main__":
    main()