import torch.utils.data, os, sys, torch, argparse, logging
from torchvision import transforms
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import printProgressBar
from networks.ResnetEmotionHead import ResnetEmotionHead
from DatasetClasses.AffectNet import AffectNet
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description='Extract latent features with ALAE')
    parser.add_argument('--pathBase', help='Path for valence and arousal dataset', required=True)
    parser.add_argument('--weightsForResnet', help='Path for valence and arousal dataset', required=True)
    parser.add_argument('--resnetModel', help='Path for valence and arousal dataset', default="resnet18")
    parser.add_argument('--outputCSVLatent', help='Path for valence and arousal dataset', required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_transforms = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
    ])
    datasetVal = AffectNet(afectdata=os.path.join(args.pathBase,'val_set'),transform=data_transforms,typeExperiment='EXP')    
    val_loader = torch.utils.data.DataLoader(datasetVal, batch_size=1, shuffle=False)

    model = ResnetEmotionHead(2,'resnet18')
    checkpoint = torch.load(args.weightsForResnet)
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    model.to(device)
    model.eval()
    iteration = 0
    outputFile = []
    for img,label,pathfile in val_loader:
        printProgressBar(iteration,len(datasetVal.filesPath),length=50,prefix='Procesing face - validating')
        img = img.to(device)
        features, _ = model(img)
        outputFile.append((features.cpu(),pathfile[0]))
        
    generateCSVFile(args.outputCSVLatent,outputFile)

def generateCSVFile(filePath,features):
    with open(filePath,'w') as fp:
        fp.write(','.join(list(map(str,list(range(features[0][0].shape[1])))))+',%s\n' % ('filePath'))
        for f in features:
            fp.write(','.join(list(map(str,f[0].tolist()[0]))) + ',%s\n' % (f[1]))

if __name__ == '__main__':
    main()