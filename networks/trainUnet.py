import argparse, torch, os, shutil, torch.optim as optim, torch.nn as nn, sys, numpy as np, multiprocessing
from multiprocessing.pool import Pool
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from networks.unet.unet_model import UNet
from DatasetClasses.AffWild2 import AFF2Data
from DatasetClasses.AffectNet import AffectNet
from torchvision.transforms import ToPILImage
from helper.function import saveStatePytorch


def saveImageNetwork(data,path,imageFilename,extesion='png'):
    if not os.path.exists(path):
        os.makedirs(path)

    pilTrans = ToPILImage()
    for id, d in enumerate(data):
        pilImage = pilTrans(d.cpu())
        pilImage.save(os.path.join(path,imageFilename[id].split(os.path.sep)[-1][:-3]+extesion))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Deep Models')
    parser.add_argument('--pathBase', help='Path for faces', required=True)
    parser.add_argument('--batch', type=int, default=500, help='Size of the batch', required=False)
    parser.add_argument('--epochs', type=int, default=10, help='Epochs to be run', required=False)
    parser.add_argument('--fineTuneWeights', default=None, help='Do fine tuning with weights', required=False)
    parser.add_argument('--folderSnapshots', default='trainPytorch', help='Folder for snapshots', required=False)
    parser.add_argument('--dataset', help='Dataset to train with', required=False, default="AFFECTNET")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('####### Loading Files for training #######')
    data_transforms_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
                transforms.RandomAffine(20, scale=(0.8, 1), translate=(0.2, 0.2)),
            ], p=0.7),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5],std=[0.5]),
        transforms.RandomErasing(),
    ])

    data_transforms_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5],std=[0.5]),
    ])

    print("Loading trainig set")
    if args.dataset == "AFFWILD2":
        afw2Train = AFF2Data(args.pathBase,'Train_Set',transform=data_transforms_train,type='VA')    
        afw2Val = AFF2Data(args.pathBase,'Validation_Set',transform=data_transforms_val,type='VA')
    elif args.dataset == "AFFECTNET":
        afw2Val = AffectNet(os.path.join(args.pathBase,'val_set'),'VA',transform=data_transforms_val)
        afw2Train = AffectNet(os.path.join(args.pathBase,'train_set'),'VA',transform=data_transforms_train)
    train_loader = torch.utils.data.DataLoader(afw2Train, batch_size=args.batch, shuffle=True)
    test_loader = torch.utils.data.DataLoader(afw2Val, batch_size=args.batch, shuffle=False)
    criterion = nn.CrossEntropyLoss().to(device)


    if os.path.exists(args.folderSnapshots):
        shutil.rmtree(args.folderSnapshots)
    os.makedirs(args.folderSnapshots)

    phases = ['training','validation']

    for t in phases:
        if os.path.exists('reconstructed_from_%s' % (t)):
            shutil.rmtree('reconstructed_from_%s' % (t))

        os.makedirs('reconstructed_from_%s' % (t))
    
    cc = SummaryWriter()

    print('####### Creating model #######')
    imLearning = UNet(n_channels=1,n_classes=1)
    print(imLearning)
    imLearning.to(device)

    optimizer = optim.SGD(imLearning.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    criterion = nn.L1Loss()

    bestLossTrain = 50000
    bestLossVal = 50000
    bestSimilarity = -1
    evaluator = nn.CosineSimilarity(dim=0)
    print('####### Starting train #######')
    pilTrans = transforms.ToPILImage()
    try:
        for ep in range(args.epochs):
            ibl = ibr = ibv = ' '
            imLearning.train()
            lossAcc = []
            #print(list(imLearning.parameters())[0])
            filesPath = []
            for currBatch, currTargetBatch, pathForFiles in train_loader:
                currTargetBatch, currBatch, = currTargetBatch.to(device), currBatch.to(device)
                imPred, _ = imLearning(currBatch)
                loss = criterion(imPred,currBatch)
                lossAcc.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                saveImageNetwork(imPred,'reconstructed_from_%s' % (phases[0]),pathForFiles)

            cc.add_scalar('UNet_rebuild/training_loss', sum(lossAcc) / len(lossAcc), ep)

            imLearning.eval()
            total = 0
            cosineSimilarity = 0
            labelsData = [[],[]]
            scores = []
            loss_val = []
            with torch.no_grad():
                for currBatch, currTargetBatch, pathForFiles in test_loader:
                    currTargetBatch, currBatch = currTargetBatch.to(device), currBatch.to(device)
                    imPred, _ = imLearning(currBatch)
                    loss = criterion(imPred,currBatch)
                    loss_val.append(loss.item())
                    saveImageNetwork(imPred,'reconstructed_from_%s' % (phases[1]),pathForFiles)

            lossAvg = sum(lossAcc) / len(lossAcc)
            tLoss = sum(loss_val) / len(loss_val)

            cc.add_scalar('UNet_rebuild/validation_loss', tLoss, ep)

            state_dict = imLearning.state_dict()
            opt_dict = optimizer.state_dict()

            if bestLossTrain > lossAvg:
                ibl = 'X'
                fName = '%s_best_training_loss.pth.tar' % ('unet_rebuild')
                fName = os.path.join(args.folderSnapshots, fName)
                saveStatePytorch(fName, state_dict, opt_dict, ep + 1)
                bestLossTrain = lossAvg

            if bestLossVal > tLoss:
                ibv = 'X'
                fName = '%s_best_validation_loss.pth.tar' % ('unet_rebuild')
                fName = os.path.join(args.folderSnapshots, fName)
                saveStatePytorch(fName, state_dict, opt_dict, ep + 1)
                bestLossVal = tLoss

            fName = '%s_current_loss.pth.tar' % ('unet_rebuild')
            fName = os.path.join(args.folderSnapshots, fName)
            saveStatePytorch(fName, state_dict, opt_dict, ep + 1)
            bestLossVal = tLoss

            print('[EPOCH %03d] Accuracy of the network on the %d Training Loss %.5f Validation Loss %.5f [%c] [%c] [%c]' % (ep, total, lossAvg, tLoss, ibl, ibv, ibr))

    except KeyboardInterrupt:
        fName = '%s_interrupted.pth.tar' % ('unet_rebuild')
        fName = os.path.join(args.folderSnapshots, fName)
        saveStatePytorch(fName, imLearning.state_dict(), optimizer.state_dict(), ep + 1)
        print('####### Saving weights for interrupting fase #######')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
