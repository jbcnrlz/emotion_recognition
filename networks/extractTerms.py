import argparse, torch, os, sys, numpy as np, pandas as pd
from importlib.resources import path
from torchvision import transforms
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import ccc
from networks.DANVA import DANVA
from DatasetClasses.OMG import OMGData
from DatasetClasses.AFEWVA import AFEWVA
from DatasetClasses.TeachingDataset import TeachingDataset
from DatasetClasses.AffWild2 import AFF2Data
from DatasetClasses.AffectNet import AffectNet

def saveToCSV(preds,files,pathCSV):
    with open(pathCSV,'w') as pcsv:
        pcsv.write('1,2\n')
        for idx, p in enumerate(preds):
            pcsv.write('%s,%s\n' % (p,files[idx]))

def test():
    parser = argparse.ArgumentParser(description='Extract VA with DAN')
    parser.add_argument('--resnetPretrained', help='Path for resnet pretrained weights', required=True)
    parser.add_argument('--weights', help='Weights', required=True)
    parser.add_argument('--pathBase', help='Path for valence and arousal dataset', required=True)
    parser.add_argument('--batch', type=int, help='Size of the batch', required=True)
    parser.add_argument('--output', default=None, help='File to save csv', required=True)
    parser.add_argument('--annotationFile', help='Path for annotation file', required=False)
    parser.add_argument('--dataset', help='Dataset for feature extractoin', required=False, default="OMG")
    parser.add_argument('--typeOutput', help='Average ou VS per frame', required=False, default="average")
    parser.add_argument('--datasetPart', help='Only for AffWild - train or validation', required=False, default="Validation_Set")
    parser.add_argument('--quantityOfTerms', help='Quantity of terms to use', required=False, default=151,type=int)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model")
    model = DANVA(num_class=8, num_head=4,pretrained=args.resnetPretrained)
    checkpoint = torch.load(args.weights)
    model.convertToTerms(args.quantityOfTerms)
    model.load_state_dict(checkpoint['state_dict'],strict=True)
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
        dataset = AffectNet(afectdata=args.pathBase,transform=data_transforms,termsQuantity=args.quantityOfTerms,typeExperiment='TERMS')
    elif args.dataset == 'AFFWILD2':
        dataset = AFF2Data(args.pathBase,args.datasetPart,transform=data_transforms,type="TERMS")    
    else:
        dataset = TeachingDataset(teachingData=args.pathBase,transform=data_transforms)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=False)

    model.eval()
    predictions = []
    pathFile = []
    with torch.no_grad():
        for idxBtc, data in enumerate(val_loader):
            print("Extraction Batch %d" % (idxBtc))
            images, _, pathsForFiles = data
            outputs, _, _ = model(images.to(device))

            _, predicted = torch.max(outputs.data, 1)  
            predictions = predictions + list(predicted)
            pathFile = pathFile + list(pathsForFiles)

    vaPerUtt = {}
    if args.dataset == 'OMG':
        for idxF, file in enumerate(pathFile):
            dirUtt = int(file.split(os.path.sep)[-2].split('_')[1])
            if dirUtt not in vaPerUtt.keys():
                vaPerUtt[dirUtt] = [predictions[idxF]]
            else:
                vaPerUtt[dirUtt].append(predictions[idxF])
    elif args.dataset == "AFFWILD2":
        terms = np.array(pd.read_csv('joinedWithDistance_%d.csv' % (args.quantityOfTerms)))[:,0]
        for idxF, file in enumerate(pathFile):            
            dirUtt = file.split(os.path.sep)[-2]
            if dirUtt not in vaPerUtt.keys():
                vaPerUtt[dirUtt] = []

            fileNameFrame = int(file.split(os.path.sep)[-1][:-4])
            while len(vaPerUtt[dirUtt]) < fileNameFrame:
                vaPerUtt[dirUtt].append('')

            vaPerUtt[dirUtt][fileNameFrame - 1] = terms[predictions[idxF].cpu().item()]

    else:
        for idxF, file in enumerate(pathFile):            
            dirUtt = file
            if dirUtt not in vaPerUtt.keys():
                vaPerUtt[dirUtt] = [predictions[idxF]]
            else:
                vaPerUtt[dirUtt].append(predictions[idxF])

    print('Generating CSV')
    if args.typeOutput == 'average':
        means = [np.mean(np.array(vaPerUtt[k]),axis=0) for k in vaPerUtt]
        utt = list(map(str,vaPerUtt.keys()))
        saveToCSV(means,utt,args.output)
    else:
        values = []
        utt = []
        for k in vaPerUtt:
            for idxTerm, i in enumerate(vaPerUtt[k]):
                values.append(i)
                utt.append(os.path.join(k,'%05d.jpg' % (idxTerm+1)))
        saveToCSV(values,utt,args.output)
        print('oi')

if __name__ == '__main__':
    test()