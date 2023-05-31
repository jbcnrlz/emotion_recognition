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
from networks.ResnetEmotionHead import ResnetEmotionHeadClassifier, ResnetEmotionHeadClassifierAttention
from helper.function import saveStatePytorch, printProgressBar, loadNeighFiles
from torch.nn.functional import linear, normalize
from sklearn.naive_bayes import GaussianNB

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
    parser.add_argument('--networkToUse', help='Path for valence and arousal dataset', required=False,default='resnet18')
    parser.add_argument('--additiveLoss', help='Path for valence and arousal dataset', required=False,default=None)
    parser.add_argument('--samplePlotSize', help='Path for valence and arousal dataset', required=False,type=int,default=500)
    parser.add_argument('--loadRandomSplits', help='Path for valence and arousal dataset', required=False,type=int,default=0)
    parser.add_argument('--numberOfClasses', help='Path for valence and arousal dataset', required=False,default=8,type=int)
    parser.add_argument('--neighsFiles', help='Path for valence and arousal dataset', required=False,default=None)
    parser.add_argument('--useAttention', help='Path for valence and arousal dataset', required=False,default=False, type=bool)
    args = parser.parse_args()    

    alpha = 0.1

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    writer = SummaryWriter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_transforms = {
    'train': transforms.Compose([
        #transforms.Resize((256,256)),
        #transforms.RandomCrop(120),
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
    if (not args.useAttention):
        model = ResnetEmotionHeadClassifier(args.numberOfClasses,args.networkToUse,vaGuidance=False).to(device)
    else:
        model = ResnetEmotionHeadClassifierAttention(args.numberOfClasses,args.networkToUse).to(device)
    print(model)

    nFile = None    
    if args.neighsFiles is not None:
        nFile = loadNeighFiles(args.neighsFiles)        

    criterion = nn.CrossEntropyLoss().to(device)    
    cLoss = None
    if args.additiveLoss == 'centerLoss':
        cLoss = CenterLoss(args.numberOfClasses,512).to(device)
        params = list(model.parameters()) + list(cLoss.parameters())
        optimizer = optim.Adam(params,lr=args.learningRate)
    elif args.additiveLoss == 'arcface':
        cLoss = CombinedMarginLoss(
            64,
            1.0,
            0.5,
            0.0,
            0
        ).to(device)
        optimizer = optim.AdamW(params=[{"params": model.parameters()}, {"params": cLoss.parameters()}], lr=args.learningRate, weight_decay=5e-4)
    else:
        optimizer = optim.Adam(model.parameters(),lr=args.learningRate)
    bestForFoldTLoss = bestForFold = 5000
    wVALOSS = 1    
    for ep in range(args.epochs):
        ibl = ibtl = ' '
        lossAcc = []
        otherLoss = [[],[]]
        iteration = 0
        featuresProject = None
        labelsProject = None
        alreadySampled = 0
        if args.loadRandomSplits > 0:
            train_loader = loaders[random.randint(0,len(loaders) - 1)]
        vaLossHistory = []
        for img,label,pathfile in train_loader:
            printProgressBar(iteration,len(dataset.filesPath),length=50,prefix='Procesing face - training')
            logitsClass = np.zeros((img.shape[0],args.numberOfClasses))
            if nFile is not None:
                for idx, p in enumerate(pathfile):
                    gnb = GaussianNB()
                    cFileName = p.split(os.path.sep)[-1]
                    cNs = np.array(nFile[cFileName]['neighbours'])
                    cNs[:,-1][cNs[:,-1] > 1] = 1
                    if np.all(cNs[:,0] == cNs[0,0]) and np.all(cNs[:,1] == cNs[0,1]):
                        labelsSame, countLabels = np.unique(cNs[:,-1],return_counts=True)
                        logitsClass[idx][labelsSame.astype(np.uint8)] = countLabels / np.sum(countLabels)
                    else:
                        for idxN in range(2):
                            cNs[cNs[:,idxN] == 0,idxN] = 1e-10
                        gnb.fit(cNs[:,:-1],cNs[:,-1])
                        logitsClass[idx][gnb.classes_.astype(np.uint16)] = gnb.predict_proba(np.array([nFile[cFileName]['va']]))

                logitsClass = torch.tensor(logitsClass).to(device)

            img = img.to(device)
            label = label.to(device)
            features, classes = model(img)
            if nFile is not None:
                classes = classes * logitsClass
            label[label > 1] = 1

            if cLoss is not None:
                if args.additiveLoss == 'centerLoss':
                    cLossV = alpha * cLoss(features,label)
                    ceLossV = criterion(classes, label)
                    loss = ceLossV + cLossV
                else:
                    with torch.cuda.amp.autocast(False):
                        features = normalize(features)
                    features = features.clamp(-1, 1)
                    thetas = cLoss(features,label)
                    loss = criterion(thetas,label)
            else:
                loss = criterion(classes, label)
 
            optimizer.zero_grad()
            loss.backward()
            if args.additiveLoss == 'centerLoss':
                for param in cLoss.parameters():
                    param.grad.data *= (0.0005/(alpha * args.learningRate))
            optimizer.step()

            lossAcc.append(loss.item())
            otherLoss[0].append(ceLossV.item())
            otherLoss[1].append(cLossV.item())
            iteration += img.shape[0]
            if args.additiveLoss == 'centerLoss':
                if random.randint(0,1) and alreadySampled < args.samplePlotSize:                
                    if featuresProject is None:
                        featuresProject, labelsProject = features.cpu().detach().numpy(),label.cpu().detach().numpy()
                    else:
                        featuresProject = np.concatenate((featuresProject,features.cpu().detach().numpy()))
                        labelsProject = np.concatenate((labelsProject,label.cpu().detach().numpy()))

                    alreadySampled += featuresProject.shape[0]
        if args.additiveLoss == 'centerLoss':
            projectionCloss = outputFeaturesImage(cLoss.centers.cpu().detach().numpy(),featuresProject,labelsProject)
            writer.add_figure('RESNETEmo/Features/train',projectionCloss,ep)

        lossAvg = sum(lossAcc) / len(lossAcc)
        writer.add_scalar('RESNETEmo/Loss/train', lossAvg, ep)
        writer.add_scalar('RESNETEmo/CenterLoss/train', sum(otherLoss[1]) / len(otherLoss[1]), ep)
        writer.add_scalar('RESNETEmo/CELoss/train', sum(otherLoss[0]) / len(otherLoss[0]), ep)
        #scheduler.step()
        model.eval()
        iteration = 0
        loss_val = []
        correct = 0
        otherLoss = [[],[]]
        with torch.no_grad():
            for img,label,pathfile in val_loader:
                printProgressBar(iteration,len(datasetVal.filesPath),length=50,prefix='Procesing face - testing')
                img = img.to(device)
                label = label.to(device)

                label[label > 1] = 1
                features, classes = model(img)
                if cLoss is not None:
                    if args.additiveLoss == 'centerLoss':
                        cLossV = alpha * cLoss(features,label)
                        ceLossV = criterion(classes, label)
                        loss = ceLossV + cLossV
                    else:
                        with torch.cuda.amp.autocast(False):
                            features = normalize(features)
                        features = features.clamp(-1, 1)
                        thetas = cLoss(features,label)
                        loss = criterion(thetas,label)             
                else:                    
                    loss = criterion(classes, label)
                _, predicted = torch.max(classes.data, 1)
                correct += (predicted == label).sum().item()
                otherLoss[0].append(ceLossV.item())
                otherLoss[1].append(cLossV.item())
                loss_val.append(loss)
                iteration += img.shape[0]

            lossAvgVal = sum(loss_val) / len(loss_val)
            correct = correct / iteration
            writer.add_scalar('RESNETEmo/Loss/val', lossAvgVal, ep)
            writer.add_scalar('RESNETEmo/Accuracy', correct, ep)
            writer.add_scalar('RESNETEmo/CenterLoss/val', sum(otherLoss[1]) / len(otherLoss[1]), ep)
            writer.add_scalar('RESNETEmo/CELoss/val', sum(otherLoss[0]) / len(otherLoss[0]), ep)

        #wVALOSS = (sum(vaLossHistory) / len(vaLossHistory)) / max(vaLossHistory)
        #writer.add_scalar('RESNETEmo/WeigthMSELoss', wVALOSS, ep)
        state_dict = model.state_dict()
        opt_dict = optimizer.state_dict()
        fName = '%s_current.pth.tar' % ('resnet_emotion')
        fName = os.path.join(args.output, fName)
        #saveStatePytorch(fName, state_dict, opt_dict, ep + 1)

        if bestForFoldTLoss > lossAvgVal:
            ibtl = 'X'
            fName = '%s_best_val_loss_neutral.pth.tar' % ('resnet_emotion')
            fName = os.path.join(args.output, fName)
            saveStatePytorch(fName, state_dict, opt_dict, ep + 1)
            bestForFoldTLoss = lossAvgVal

        if bestForFold > lossAvg:
            ibl = 'X'
            fName = '%s_best_loss_neutral.pth.tar' % ('resnet_emotion')
            fName = os.path.join(args.output, fName)
            saveStatePytorch(fName, state_dict, opt_dict, ep + 1)
            bestForFold = lossAvg

        print('[EPOCH %03d] Training Loss %.5f Validation Loss %.5f Accuracy %.2f - [%c] [%c]               ' % (ep, lossAvg, lossAvgVal,correct,ibl,ibtl))
    
if __name__ == '__main__':
    main()