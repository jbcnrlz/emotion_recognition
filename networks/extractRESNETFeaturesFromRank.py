import argparse, torch, os, sys, numpy as np
from torchvision import transforms
from torchvision import models
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from DatasetClasses.AffectNet import AffectNet
from DatasetClasses.AffWild2 import AFF2Data
from torch import nn

def saveToCSV(features,files,pathCSV):
    with open(pathCSV,'w') as pcsv:
        for i in range(features.shape[1]):
            pcsv.write('%d,' % (i))
        pcsv.write('file\n')
        for idx, p in enumerate(features):
            for vf in p:
                pcsv.write("%f," % (vf))
            pcsv.write('%s\n' % (files[idx]))

def getRanks(vaLabel,vaDists):
    dists = torch.cdist(vaLabel,vaDists,p=2)
    return dists

def test():
    parser = argparse.ArgumentParser(description='Finetune resnet')
    parser.add_argument('--pathBase', help='Path for valence and arousal dataset', required=True)
    parser.add_argument('--batchSize', type=int, help='Size of the batch', required=True)
    parser.add_argument('--output', default=None, help='Folder to save weights', required=True)
    parser.add_argument('--numberOfClasses', help='Freeze weights', required=False, type=int, default=0)
    parser.add_argument('--wtsLoad', default=None, help='Folder to save weights', required=True)
    parser.add_argument('--dataset', help='Dataset for feature extractoin', required=False, default="OMG")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(512, args.numberOfClasses,bias=False)
    checkpoint = torch.load(args.wtsLoad)
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    model.to(device)
    data_transforms = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    dataset = None
    if args.dataset == 'AFFECTNET':
        if (args.numberOfClasses == 7):
            dataset = AffectNet(afectdata=os.path.join(args.pathBase,'val_set'),transform=data_transforms,typeExperiment='BOTH',exchangeLabel=None,loadLastLabel=False)
        else:
            dataset = AffectNet(afectdata=os.path.join(args.pathBase,'val_set'),transform=data_transforms,typeExperiment='BOTH',exchangeLabel=None)
    elif args.dataset == 'AFFWILD2':
        dataset = AFF2Data(args.pathBase,'Validation_Set',transform=data_transforms,type='RANK')    

    val_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize, shuffle=False)
    predictions = None
    model.eval()
    pathFile = []
    with torch.no_grad():
        for data in val_loader:
            if args.dataset == 'AFFWILD2':
                images, _, pathForFiles = data
            else:
                images, _, pathForFiles, _ = data
            classification = model(images.to(device))
            prediction = classification.cpu().detach().numpy()
            predictions = prediction if predictions is None else np.concatenate((prediction,predictions))
            pathFile = pathFile + list(pathForFiles)

    saveToCSV(predictions,pathFile,args.output)


if __name__ == '__main__':
    test()