import argparse, torch, os, sys, numpy as np, pandas as pd
from torchvision import transforms
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import ccc
from DAN.networks.dan import DAN
from DatasetClasses.OMG import OMGData
from DatasetClasses.AFEWVA import AFEWVA
from DatasetClasses.TeachingDataset import TeachingDataset
from DatasetClasses.AffWild2 import AFF2Data
from DatasetClasses.AffectNet import AffectNet

def saveToCSV(preds,files,pathCSV):
    with open(pathCSV,'w') as pcsv:
        pcsv.write('valence,arousal,file\n')
        for idx, p in enumerate(preds):
            pcsv.write('%f,%f,%s\n' % (p[0],p[1],files[idx]))

def test():
    parser = argparse.ArgumentParser(description='Extract VA with DAN')
    parser.add_argument('--resnetPretrained', help='Path for resnet pretrained weights', required=True)
    parser.add_argument('--danPretrained', help='Path for DAN pretrained', required=True)
    parser.add_argument('--pathBase', help='Path for valence and arousal dataset', required=True)
    parser.add_argument('--batch', type=int, help='Size of the batch', required=True)
    parser.add_argument('--output', default=None, help='File to save csv', required=True)
    parser.add_argument('--annotationFile', help='Path for annotation file', required=False)
    parser.add_argument('--dataset', help='Dataset for feature extractoin', required=False, default="OMG")
    parser.add_argument('--datasetPart', help='Only for AffWild - train or validation', required=False, default="Validation_Set")
    parser.add_argument('--termsCSV', help='Path for the terms file', required=True)
    args = parser.parse_args()

    tFiles = ['neutral', 'joy', 'sadness', 'surprise', 'fear', 'disgust', 'anger', 'contempt']
    classesLabel = np.array(tFiles)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model")
    model = DAN(num_class=7, num_head=4,pretrained=args.resnetPretrained)
    checkpoint = torch.load(args.danPretrained)
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    model.to(device)
    print("Model loaded")
    print(model)
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
    ])
    print("Loading test set")
    if args.dataset == 'OMG':
        dataset = OMGData(omgData=args.pathBase,annotationFile=args.annotationFile,transform=data_transforms)
    elif args.dataset == 'AFEWVA':
        dataset = AFEWVA(afewdata=args.pathBase,transform=data_transforms)
    elif args.dataset == 'AFFECTNET':
        dataset = AffectNet(afectdata=args.pathBase,transform=data_transforms,typeExperiment='EXP')
    elif args.dataset == 'AFFWILD2':
        dataset = AFF2Data(args.pathBase,args.datasetPart,transform=data_transforms,type='EXPR')    
    else:
        dataset = TeachingDataset(teachingData=args.pathBase,transform=data_transforms)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=False)

    model.eval()
    fullPaths = []
    predictions = []
    with torch.no_grad():
        for idxBtc, data in enumerate(val_loader):
            print("Extraction Batch %d" % (idxBtc))
            images, _, pathsForFiles = data
            outputs, _, _ = model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            fullPaths += pathsForFiles
            predictions += list(classesLabel[predicted.cpu()])

    with open(args.output,'w') as pcsv:
        pcsv.write('emotion,file\n')
        for idx, p in enumerate(predictions):
            pcsv.write('%s,%s\n' % (p,fullPaths[idx]))


if __name__ == '__main__':
    test()