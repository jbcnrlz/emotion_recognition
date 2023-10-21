import torch,os,sys,argparse, matplotlib.pyplot as plt, numpy as np, random, math
from torchvision import transforms
from sklearn.decomposition import PCA
from torch.utils.tensorboard import SummaryWriter
from torch import optim, nn
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from loss.CenterLoss import CenterLoss
from DatasetClasses.AffectNet import AffectNet
from networks.VAEForEmotion import VAEOurEmotion, VAEOurEmotionAttention
from helper.function import saveStatePytorch, printProgressBar, loadNeighFiles
from scipy.stats import norm
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
    parser.add_argument('--samplePlotSize', help='Path for valence and arousal dataset', required=False,type=int,default=500)
    parser.add_argument('--numberOfClasses', help='Path for valence and arousal dataset', required=False,default=8,type=int)
    parser.add_argument('--neighsFiles', help='Path for valence and arousal dataset', required=False,default=None)
    parser.add_argument('--additiveLoss', help='Path for valence and arousal dataset', required=False,default=None)
    parser.add_argument('--modelToTrain', help='Path for valence and arousal dataset', required=False,default="vae")
    args = parser.parse_args()        

    alpha = 0.1

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    writer = SummaryWriter()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    '''
    classesDist = np.array(
        [
            [0,0.0001 ],[0.81,0.21],[-0.63,0.23],[0.4,0.3   ], #neutral, happy, sad, surprise
            [-0.64,0.2],[-0.6,0.2 ],[-0.51,0.2 ],[-0.23,0.39]  #fear, disgust, anger, contempt
        ], dtype = np.float32)
    '''

    classesDist = np.array(
        [
            [0,0.0001],   [0.605,0.21],  [-0.522,0.15],[0.605,0.21], #neutral, happy, sad, surprise
            [-0.522,0.15],[-0.522,0.15 ],[-0.522,0.15],[-0.522,0.15]  #fear, disgust, anger, contempt
        ], dtype = np.float32)

    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomCrop(120),
        transforms.Resize((256,256)),
        #transforms.RandomHorizontalFlip(),        
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

    dataset = AffectNet(afectdata=os.path.join(args.pathBase,'train_set'),transform=data_transforms['train'],typeExperiment='EXP',exchangeLabel=None)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize, shuffle=True)

    datasetVal = AffectNet(afectdata=os.path.join(args.pathBase,'val_set'),transform=data_transforms['test'],typeExperiment='EXP',exchangeLabel=None)
    val_loader = torch.utils.data.DataLoader(datasetVal, batch_size=args.batchSize, shuffle=False)
    if args.modelToTrain == 'vae':
        model = VAEOurEmotion(3).to(device)
    elif args.modelToTrain == 'vaeAttn':
        model = VAEOurEmotionAttention(3).to(device)
    print(model)
    nFile = None    
    if args.neighsFiles is not None:
        nFile = loadNeighFiles(args.neighsFiles)        

    latentLOSS = nn.KLDivLoss(reduction='batchmean').to(device)
    reconsLOSS = nn.MSELoss().to(device)
    ceLOSS = nn.CrossEntropyLoss().to(device)
    cLoss = None
    if args.additiveLoss == 'centerloss':
        cLoss = CenterLoss(args.numberOfClasses,1024).to(device)
        params = list(model.parameters()) + list(cLoss.parameters())
        optimizer = optim.Adam(params,lr=args.learningRate)
    else:
        optimizer = optim.Adam(model.parameters(),lr=args.learningRate)
    bestForFoldTLoss = bestForFold = 5000
    bestResult = 0
    print('Training Phase =================================================================== BTL  BVL BAC')
    for ep in range(args.epochs):
        ibl = ibtl = ibacc = ' '
        lossAcc = []
        otherLoss = [[],[],[],[]]
        iteration = 0
        for imgTr,labelTr,pathfile in train_loader:
            logitsClass = np.zeros((imgTr.shape[0],1)).flatten()
            printProgressBar(iteration,math.ceil(len(dataset.filesPath)/args.batchSize),length=50,prefix='Procesing face - training')
            if nFile is not None:
                for idx, p in enumerate(pathfile):
                    gnb = GaussianNB()
                    cFileName = p.split(os.path.sep)[-1]
                    cNs = np.array(nFile[cFileName]['neighbours'])
                    #cNs[:,-1][cNs[:,-1] > 1] = 1
                    if np.all(cNs[:,0] == cNs[0,0]) and np.all(cNs[:,1] == cNs[0,1]):
                        logitsClass[idx] = int(cNs[:,-1][0])
                    else:
                        for idxN in range(2):
                            cNs[cNs[:,idxN] == 0,idxN] = 1e-10
                        gnb.fit(cNs[:,:-1],cNs[:,-1])
                        logitsClass[idx] = int(gnb.predict(np.array([nFile[cFileName]['va']]))[0])

                labelTr = torch.tensor(logitsClass,dtype=torch.long)

            imgTr = imgTr.to(device)
            label = labelTr.to(device)
            distTr, classModule, imgReconsTr, z = model(imgTr)
            expected = distTr.clone().detach().cpu()
            fResult = []
            for valCalc in range(expected.shape[0]):
                fResult.append(norm.pdf(expected[valCalc],loc=classesDist[labelTr[valCalc]][0],scale=classesDist[labelTr[valCalc]][1]))
            expected = torch.tensor(np.array(fResult,dtype=np.float32)).to(device)
            if args.additiveLoss is not None:
                cLossV = alpha * cLoss(z,label)
                ceLossV = ceLOSS(classModule,label)
                ltLoss = latentLOSS(distTr,expected)
                reconLoss = reconsLOSS(imgTr,imgReconsTr)
                loss_tr = ltLoss * 0.05 + reconLoss * 0.15 + (cLossV + ceLossV) * 0.8
            else:            
                ceLossV = ceLOSS(classModule,label)
                ltLoss = latentLOSS(distTr,expected)
                reconLoss = reconsLOSS(imgTr,imgReconsTr)
                loss_tr = ltLoss * 0.05 + reconLoss * 0.15 + ceLossV * 0.8                

            optimizer.zero_grad()
            loss_tr.backward()
            if args.additiveLoss == 'centerLoss':
                for param in cLoss.parameters():
                    param.grad.data *= (0.0005/(alpha * args.learningRate))

            optimizer.step()

            lossAcc.append(loss_tr.item())
            if args.additiveLoss is not None:
                otherLoss[0].append(cLossV.item())
                otherLoss[1].append(ceLossV.item())
                otherLoss[2].append(ltLoss.item())
                otherLoss[3].append(reconLoss.item())
            else:
                otherLoss[1].append(ceLossV.item())
                otherLoss[2].append(ltLoss.item())
                otherLoss[3].append(reconLoss.item())

            iteration += 1

        lossAvg = sum(lossAcc) / len(lossAcc)
        writer.add_scalar('VAEmo/Loss/train', lossAvg, ep)

        lossesName = ['centerloss','crossentropyloss','klloss','mseloss']
        for idx, otl in enumerate(otherLoss):
            if len(otl) > 0:
                vTB = sum(otl) / len(otl)
                writer.add_scalar('VAEmo/%s/train' % (lossesName[idx]), vTB, ep)
        
        model.eval()
        iteration = 0
        loss_val = []
        otherLoss = [[],[],[],[]]
        correct = 0
        total = 0
        with torch.no_grad():
            for img,label,pathfile in val_loader:
                printProgressBar(iteration,math.ceil(len(datasetVal.filesPath)/args.batchSize),prefix='Procesing face - testing',length=50)
                img = img.to(device)
                label = label.to(device)
                dist, classModule, imgRecons, z = model(img)

                expected = dist.clone().detach().cpu()
                fResult = []
                for valCalc in range(expected.shape[0]):
                    fResult.append(norm.pdf(expected[valCalc],loc=classesDist[label[valCalc]][0],scale=classesDist[label[valCalc]][1]))
                expected = torch.tensor(np.array(fResult,dtype=np.float32)).to(device)                

                if args.additiveLoss is not None:
                    cLossV = alpha * cLoss(z,label)
                    ceLossV = ceLOSS(classModule,label)
                    ltLoss = latentLOSS(dist,expected)
                    reconLoss = reconsLOSS(img,imgRecons)
                    loss = ltLoss * 0.05 + reconLoss * 0.15 + (cLossV + ceLossV) * 0.8
                else:            
                    ceLossV = ceLOSS(classModule,label)
                    ltLoss = latentLOSS(dist,expected)
                    reconLoss = reconsLOSS(img,imgRecons)
                    loss = ltLoss * 0.05 + reconLoss * 0.15 + ceLossV * 0.8                

                #loss = latentLOSS(dist,expected) * 0.05 + reconsLOSS(img,imgRecons) * 0.15 + ceLOSS(classModule,label) * 0.8
                _, predicted = torch.max(classModule.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
                loss_val.append(loss.item())
                if args.additiveLoss is not None:
                    otherLoss[0].append(cLossV.item())
                    otherLoss[1].append(ceLossV.item())
                    otherLoss[2].append(ltLoss.item())
                    otherLoss[3].append(reconLoss.item())
                else:
                    otherLoss[1].append(ceLossV.item())
                    otherLoss[2].append(ltLoss.item())
                    otherLoss[3].append(reconLoss.item())

                iteration += 1
            lossAvgVal = sum(loss_val) / len(loss_val)
            cResult = correct / total
            writer.add_scalar('VAEmo/Loss/val', lossAvgVal, ep)
            lossesName = ['centerloss','crossentropyloss','klloss','mseloss']
            for idx, otl in enumerate(otherLoss):
                if len(otl) > 0:
                    vTB = sum(otl) / len(otl)
                    writer.add_scalar('VAEmo/%s/val' % (lossesName[idx]), vTB, ep)

        state_dict = model.state_dict()
        opt_dict = optimizer.state_dict()
        fName = '%s_current.pth.tar' % ('vae_emotion')
        fName = os.path.join(args.output, fName)
        #saveStatePytorch(fName, state_dict, opt_dict, ep + 1)

        if bestForFoldTLoss > lossAvgVal:
            ibtl = 'X'
            fName = '%s_best_val_loss_neutral.pth.tar' % ('vae_emotion')
            fName = os.path.join(args.output, fName)
            saveStatePytorch(fName, state_dict, opt_dict, ep + 1)
            if args.additiveLoss is not None:
                saveStatePytorch(os.path.join(args.output,'center_loss_wtgs.pth.tar'),cLoss.state_dict(),opt_dict,ep+1)
            bestForFoldTLoss = lossAvgVal

        if bestForFold > lossAvg:
            ibl = 'X'
            fName = '%s_best_loss_neutral.pth.tar' % ('vae_emotion')
            fName = os.path.join(args.output, fName)
            saveStatePytorch(fName, state_dict, opt_dict, ep + 1)
            bestForFold = lossAvg

        if bestResult < cResult:
            fName = '%s_best_acc_neutral.pth.tar' % ('vae_emotion')
            fName = os.path.join(args.output, fName)
            saveStatePytorch(fName, state_dict, opt_dict, ep + 1)
            bestResult = cResult
            ibacc = 'X'
        
        print('[EPOCH %03d] Training Loss %.5f Validation Loss %.5f Accuracy on validation %.5f - [%c]  [%c] [%c]' % (ep, lossAvg, lossAvgVal,cResult,ibl,ibtl,ibacc))
        
    
if __name__ == '__main__':
    main()