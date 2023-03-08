import torch,os,sys,argparse, matplotlib.pyplot as plt, numpy as np, random
from torchvision import transforms
from sklearn.decomposition import PCA
from torch.utils.tensorboard import SummaryWriter
from torch import optim, nn
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from loss.CenterLoss import CenterLoss
from loss.ArcFace import CombinedMarginLoss
from DatasetClasses.AffectNet import AffectNet
from networks.ResnetEmotionHead import ResnetEmotionHead
from helper.function import saveStatePytorch, printProgressBar

def vaCenterValues():
    return [[0,0],[0.81,0.51],[-0.63,-0.27],[0.4,0.67],[-0.64,0.6],[-0.6,0.35],[-0.43,0.67],[-0.23,0.31]]

def vaVarianceValues():
    return [[0.26,0.31],[0.21,0.26],[0.23,0.34],[0.3,0.27],[0.2,0.32],[0.2,0.41],[0.29,0.27],[0.39,0.33]]

def outputFeaturesImage(centers,features,labels):
    pcaProjection = PCA(n_components=2)
    fig = plt.Figure(figsize=(4, 4), dpi=320, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    cProjection = pcaProjection.fit_transform(centers)
    ax.scatter(cProjection[:,0],cProjection[:,1])    
    for l in np.unique(labels):
        if (features[labels == l] is None):
            continue
        if min(features[labels == l].shape) < 2:
            continue
        fProjection = pcaProjection.fit_transform(features[labels == l])
        ax.scatter(fProjection[:,0],fProjection[:,1])

    fig.set_tight_layout(True)
    return fig

def main():
    parser = argparse.ArgumentParser(description='Extract latent features with AutoencoderKL')
    parser.add_argument('--pathBase', help='Path for valence and arousal dataset', required=True)
    parser.add_argument('--epochs', help='Path for valence and arousal dataset', required=False, default=20, type=int)
    parser.add_argument('--output', help='Path for valence and arousal dataset', required=False, default='resnetEmotion')
    parser.add_argument('--batchSize', help='Path for valence and arousal dataset', required=True, type=int)
    parser.add_argument('--learningRate', help='Learning Rate', required=False, default=0.01, type=float)
    parser.add_argument('--additiveLoss', help='Path for valence and arousal dataset', required=False,default=None)
    parser.add_argument('--samplePlotSize', help='Path for valence and arousal dataset', required=False,type=int,default=500)
    parser.add_argument('--loadRandomSplits', help='Path for valence and arousal dataset', required=False,type=int,default=0)
    args = parser.parse_args()    

    alpha = 0.1

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    writer = SummaryWriter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_transforms = {
    'train': transforms.Compose([
        #transforms.Resize((256,256)),
        #transforms.RandomCrop(224),
        #transforms.RandomHorizontalFlip(),        
        transforms.ToTensor(),
    ]),
    'test' : transforms.Compose([
        #transforms.Resize((256,256)),
        transforms.ToTensor(),
    ])} 

    if args.loadRandomSplits > 0:
        loaders = []
        for idxSplit in range(args.loadRandomSplits + 1):
            dataset = AffectNet(afectdata=os.path.join(args.pathBase,'train_set'),transform=data_transforms['train'],typeExperiment='EXP',datasetNumber=idxSplit)
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize, shuffle=True)
            loaders.append(train_loader)
    else:
        dataset = AffectNet(afectdata=os.path.join(args.pathBase,'train_set'),transform=data_transforms['train'],typeExperiment='EXP')
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize, shuffle=True)

    datasetVal = AffectNet(afectdata=os.path.join(args.pathBase,'val_set'),transform=data_transforms['test'],typeExperiment='EXP')
    val_loader = torch.utils.data.DataLoader(datasetVal, batch_size=args.batchSize, shuffle=False)

    model = ResnetEmotionHead(8,resnetModel='resnet18',vaGuidance=True).to(device)
    print(model)
    criterion = nn.GaussianNLLLoss().to(device)    
    cLoss = CenterLoss(8,512).to(device)
    params = list(model.parameters()) + list(cLoss.parameters())
    centersGuideVA = torch.tensor(np.array(vaCenterValues())).type(torch.float32).to(device)
    varGuidedVA = torch.tensor(np.array(vaVarianceValues())).type(torch.float32).to(device)
    optimizer = optim.Adam(params,lr=args.learningRate)
    bestForFoldTLoss = bestForFold = 5000
    for ep in range(args.epochs):
        ibl = ibtl = ' '
        lossAcc = []
        cLossAcc = []
        gnllAcc = []
        iteration = 0
        if args.loadRandomSplits > 0:
            train_loader = loaders[random.randint(0,len(loaders) - 1)]

        for img,label,_ in train_loader:
            printProgressBar(iteration,len(dataset.filesPath),length=50,prefix='Procesing face - training')
            img = img.to(device)
            label = label.to(device)
            features, va = model(img)
            vaLabels = centersGuideVA[label]
            vastdDev = varGuidedVA[label]
            #label[label > 1] = 1

            cLossValue = cLoss(features,label)
            gnllLoss = criterion(va,vaLabels,vastdDev ** 2)
            loss = gnllLoss + (alpha * cLossValue)
 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lossAcc.append(loss.item())
            cLossAcc.append(cLossValue.item())
            gnllAcc.append(gnllLoss.item())
            iteration += img.shape[0]

        lossAvg = sum(lossAcc) / len(lossAcc)
        cLossAvg = sum(cLossAcc) / len(cLossAcc)
        gLossAvg = sum(gnllAcc) / len(gnllAcc)
        writer.add_scalar('RESNETEmo/GNLLLoss/train', gLossAvg, ep)
        writer.add_scalar('RESNETEmo/CLoss/train', cLossAvg, ep)
        writer.add_scalar('RESNETEmo/Loss/train', lossAvg, ep)
        #scheduler.step()
        model.eval()
        iteration = 0
        loss_val = []
        cLossAcc_val = []
        gnllAcc_val = []
        correct = 0
        with torch.no_grad():
            for img,label,_ in val_loader:
                printProgressBar(iteration,len(datasetVal.filesPath),length=50,prefix='Procesing face - testing')
                img = img.to(device)
                label = label.to(device)
                vaLabels = centersGuideVA[label]
                vastdDev = varGuidedVA[label]

                #label[label > 1] = 1
                features, va = model(img)
                cLossValue = cLoss(features,label)
                gnllLoss = criterion(va,vaLabels,vastdDev ** 2)
                loss = gnllLoss + (alpha * cLossValue)

                #_, predicted = torch.max(classes.data, 1)

                #correct += (predicted == label).sum().item()
                loss_val.append(loss)
                cLossAcc_val.append(cLossValue.item())
                gnllAcc_val.append(gnllLoss.item())
                iteration += img.shape[0]

            lossAvgVal = sum(loss_val) / len(loss_val)
            cLossAcc_val = sum(cLossAcc_val) / len(cLossAcc_val)
            gnllAcc_val = sum(gnllAcc_val) / len(gnllAcc_val)
            #correct = correct / iteration
            writer.add_scalar('RESNETEmo/GNLLLoss/val', gnllAcc_val, ep)
            writer.add_scalar('RESNETEmo/CLoss/val', cLossAcc_val, ep)
            writer.add_scalar('RESNETEmo/Loss/val', lossAvgVal, ep)

        state_dict = model.state_dict()
        opt_dict = optimizer.state_dict()
        fName = '%s_current.pth.tar' % ('resnet_emotion')
        fName = os.path.join(args.output, fName)

        if bestForFoldTLoss > lossAvgVal and lossAvgVal > 0:
            ibtl = 'X'
            fName = '%s_best_val_loss_neutral.pth.tar' % ('resnet_emotion')
            fName = os.path.join(args.output, fName)
            saveStatePytorch(fName, state_dict, opt_dict, ep + 1)
            bestForFoldTLoss = lossAvgVal

        if bestForFold > lossAvg and lossAvg > 0:
            ibl = 'X'
            fName = '%s_best_loss_neutral.pth.tar' % ('resnet_emotion')
            fName = os.path.join(args.output, fName)
            saveStatePytorch(fName, state_dict, opt_dict, ep + 1)
            bestForFold = lossAvg

        print('[EPOCH %03d] Training Loss %.5f Validation Loss %.5f - [%c] [%c]               ' % (ep, lossAvg, lossAvgVal,ibl,ibtl))
    
if __name__ == '__main__':
    main()