import torch,os,sys,argparse, matplotlib.pyplot as plt, numpy as np, random
from torchvision import transforms
from sklearn.decomposition import PCA
from torch.utils.tensorboard import SummaryWriter
from torch import optim, nn
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from DatasetClasses.AffectNet import AffectNet
from DatasetClasses.WFD import WFD
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
    parser.add_argument('--output', help='Path for valence and arousal dataset', required=False, default='resnetEmotion')
    parser.add_argument('--batchSize', help='Path for valence and arousal dataset', required=True, type=int)
    parser.add_argument('--weightsForVAE', help='Path for valence and arousal dataset', required=False, default=None)
    parser.add_argument('--dataset', help='Path for valence and arousal dataset', required=False, default='affectnet')
    args = parser.parse_args()        

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_transforms = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.562454871481894, 0.8208898956471341, 0.395364053852456],
            std=[0.43727472598867456, 0.31812502566122625, 0.3796120355707891]
        )
    ])

    datasetVal = None
    if args.dataset == 'affectnet':
        datasetVal = AffectNet(afectdata=os.path.join(args.pathBase,'val_set'),transform=data_transforms,typeExperiment='EXP')        
    elif args.dataset == 'wfd':
        datasetVal = WFD(afectdata=args.pathBase,transform=data_transforms)

    val_loader = torch.utils.data.DataLoader(datasetVal, batch_size=args.batchSize, shuffle=False)
    model = VAEOurEmotion(3)
    if args.weightsForVAE is not None:
        checkpoint = torch.load(args.weightsForVAE)
        model.load_state_dict(checkpoint['state_dict'],strict=True)
    model.to(device)
    print(model)

    model.eval()
    outputFile = []
    with torch.no_grad():
        for idxBtc, data in enumerate(val_loader):
            print("Extraction Batch %d" % (idxBtc))
            images, _, pathsForFiles = data
            _, _, imgRecons, latFeat = model(images.to(device))
            for f in range(latFeat.shape[0]):
                outputFile.append((latFeat[f].cpu().reshape((-1)),pathsForFiles[f]))

    generateCSVFile(args.output,outputFile)

def generateCSVFile(filePath,features):
    with open(filePath,'w') as fp:
        fp.write(','.join(list(map(str,list(range(features[0][0].shape[0])))))+',%s\n' % ('filePath'))
        for f in features:
            fp.write(','.join(list(map(str,f[0].tolist()))) + ',%s\n' % (f[1]))

if __name__ == '__main__':
    main()