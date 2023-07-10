import torch,os,sys,argparse, matplotlib.pyplot as plt, numpy as np, random
from torchvision import transforms
from sklearn.decomposition import PCA
from torch.utils.tensorboard import SummaryWriter
from torch import optim, nn
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from DatasetClasses.AffectNet import AffectNet
from networks.VAEForEmotion import VAEOurEmotion
from helper.function import saveStatePytorch, printProgressBar
from scipy.stats import norm

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
    args = parser.parse_args()        

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    writer = SummaryWriter()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    classesDist = np.array([[0,0.0001],[0.81,0.21],[-0.63,0.23],[0.4,0.3],[-0.64,0.2],[-0.6,0.2],[-0.51,0.2],[-0.23,0.39]], dtype = np.float32)

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

    dataset = AffectNet(afectdata=os.path.join(args.pathBase,'train_set'),transform=data_transforms['train'],typeExperiment='EXP')
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize, shuffle=True)

    datasetVal = AffectNet(afectdata=os.path.join(args.pathBase,'val_set'),transform=data_transforms['test'],typeExperiment='EXP')
    val_loader = torch.utils.data.DataLoader(datasetVal, batch_size=args.batchSize, shuffle=False)
    model = VAEOurEmotion(3).to(device)
    print(model)
    '''
    nFile = None    
    if args.neighsFiles is not None:
        nFile = loadNeighFiles(args.neighsFiles)        
    '''

    latentLOSS = nn.KLDivLoss(reduction='batchmean').to(device)
    reconsLOSS = nn.MSELoss().to(device)
    ceLOSS = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(),lr=args.learningRate)
    bestForFoldTLoss = bestForFold = 5000
    bestResult = 0
    for ep in range(args.epochs):
        ibl = ibtl = ' '
        lossAcc = []
        otherLoss = [[],[]]
        iteration = 0
        for imgTr,labelTr,_ in train_loader:
            printProgressBar(iteration,len(dataset.filesPath),length=args.batchSize,prefix='Procesing face - training')

            imgTr = imgTr.to(device)
            label = labelTr.to(device)
            distTr, classModule, imgReconsTr, _ = model(imgTr)
            expected = distTr.clone().detach().cpu()
            fResult = []
            for valCalc in range(expected.shape[0]):
                fResult.append(norm.pdf(expected[valCalc],loc=classesDist[labelTr[valCalc]][0],scale=classesDist[labelTr[valCalc]][1]))
            expected = torch.tensor(np.array(fResult,dtype=np.float32)).to(device)
            loss_tr = latentLOSS(distTr,expected) * 0.05 + reconsLOSS(imgTr,imgReconsTr) * 0.15 + ceLOSS(classModule,label) * 0.8

            optimizer.zero_grad()
            loss_tr.backward()
            optimizer.step()

            lossAcc.append(loss_tr.item())

            iteration += 1

        lossAvg = sum(lossAcc) / len(lossAcc)
        writer.add_scalar('VAEmo/Loss/train', lossAvg, ep)
        #writer.add_scalar('VAEmo/CenterLoss/train', sum(otherLoss[1]) / len(otherLoss[1]), ep)
        #writer.add_scalar('VAEmo/CELoss/train', sum(otherLoss[0]) / len(otherLoss[0]), ep)
        
        #scheduler.step()
        
        model.eval()
        iteration = 0
        loss_val = []
        correct = 0
        total = 0
        with torch.no_grad():
            for img,label,pathfile in val_loader:
                printProgressBar(iteration,len(datasetVal.filesPath),length=args.batchSize,prefix='Procesing face - testing')
                img = img.to(device)
                label = label.to(device)
                dist, classModule, imgRecons, _ = model(img)

                expected = dist.clone().detach().cpu()
                fResult = []
                for valCalc in range(expected.shape[0]):
                    fResult.append(norm.pdf(expected[valCalc],loc=classesDist[label[valCalc]][0],scale=classesDist[label[valCalc]][1]))
                expected = torch.tensor(np.array(fResult,dtype=np.float32)).to(device)                

                loss = latentLOSS(dist,expected) * 0.05 + reconsLOSS(img,imgRecons) * 0.15 + ceLOSS(classModule,label) * 0.8
                _, predicted = torch.max(classModule.data, 1)
                total += label.size(0)
                correct += (predicted == label.to(device)).sum().item()
                loss_val.append(loss.item())
                iteration += 1
            lossAvgVal = sum(loss_val) / len(loss_val)
            cResult = correct / total
            writer.add_scalar('VAEmo/Loss/val', lossAvgVal, ep)


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

        print('[EPOCH %03d] Training Loss %.5f Validation Loss %.5f Accuracy on validation %.5f - [%c] [%c]               ' % (ep, lossAvg, lossAvgVal,cResult,ibl,ibtl))
        
    
if __name__ == '__main__':
    main()