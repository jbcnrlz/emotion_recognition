import torch.utils.data, os, sys, torch, argparse, logging
from torchvision import transforms
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import printProgressBar
from networks.ResnetEmotionHead import ResnetEmotionHead, ResnetEmotionHeadClassifier
from DatasetClasses.AffectNet import AffectNet
from DatasetClasses.CKPlus import CKPlus
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description='Extract latent features with RESNET')
    parser.add_argument('--pathBase', help='Path for valence and arousal dataset', required=True)
    parser.add_argument('--weightsForResnet', help='Path for valence and arousal dataset', required=True)
    parser.add_argument('--resnetModel', help='Path for valence and arousal dataset', default="resnet18")
    parser.add_argument('--outputCSVLatent', help='Path for valence and arousal dataset', required=True) 
    parser.add_argument('--networkToUse', help='Path for valence and arousal dataset', required=False,default='ResnetEmotionHead')    
    parser.add_argument('--dataset', help='Path for valence and arousal dataset', required=False,default='affectnet')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])
    if args.networkToUse == 'affectnet':
        datasetVal = AffectNet(afectdata=os.path.join(args.pathBase,'val_set'),transform=data_transforms,typeExperiment='EXP')
    else:
        datasetVal = CKPlus(ckData=args.pathBase,transform=data_transforms)
    val_loader = torch.utils.data.DataLoader(datasetVal, batch_size=50, shuffle=False)
    checkpoint = torch.load(args.weightsForResnet)
    if args.networkToUse == 'ResnetEmotionHead':
        model = ResnetEmotionHead(checkpoint['state_dict']['vaModule.2.weight'].shape[0],args.resnetModel,vaGuidance=True)
    else:
        model = ResnetEmotionHeadClassifier(checkpoint['state_dict']['softmax.2.weight'].shape[0],args.resnetModel,vaGuidance=False)    
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    model.to(device)
    model.eval()
    iteration = 0
    outputFile = []
    with torch.no_grad():
        for img,label,pathfile in val_loader:
            printProgressBar(iteration,len(datasetVal.filesPath) // 50,length=50,prefix='Procesing face - validating')
            img = img.to(device)
            features, _ = model(img)
            for f in range(features.shape[0]):
                outputFile.append((features[f].cpu(),pathfile[f]))

            iteration += 1
        
    generateCSVFile(args.outputCSVLatent,outputFile)

def generateCSVFile(filePath,features):
    with open(filePath,'w') as fp:
        fp.write(','.join(list(map(str,list(range(features[0][0].shape[0])))))+',%s\n' % ('filePath'))
        for f in features:
            fp.write(','.join(list(map(str,f[0].tolist()))) + ',%s\n' % (f[1]))

if __name__ == '__main__':
    main()