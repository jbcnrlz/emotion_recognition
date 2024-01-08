import argparse, torch, os, sys, numpy as np
from importlib.resources import path
from torchvision import transforms
from torch import nn
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from networks.ResnetEmotionHead import ResnetEmotionHeadClassifierAttention
from DatasetClasses.OMG import OMGData
from DatasetClasses.AFEWVA import AFEWVA
from DatasetClasses.TeachingDataset import TeachingDataset
from DatasetClasses.AffWild2 import AFF2Data
from DatasetClasses.AffectNet import AffectNet

def saveToCSV(preds,files,labels,pathCSV):
    with open(pathCSV,'w') as pcsv:
        pcsv.write('%s,file,label\n' % (','.join([str(f) for f in range(len(preds[0]))])))
        for idx, p in enumerate(preds):
            for fp in p:
                pcsv.write('%f,' % (fp))
            pcsv.write("%s,%s\n" % (files[idx],labels[idx]))

def test():
    parser = argparse.ArgumentParser(description='Extract VA with RESNETEMOTION')
    parser.add_argument('--weights', help='Weights', required=True)
    parser.add_argument('--pathBase', help='Path for valence and arousal dataset', required=True)
    parser.add_argument('--batch', type=int, help='Size of the batch', required=True)
    parser.add_argument('--output', default=None, help='File to save csv', required=True)
    parser.add_argument('--annotationFile', help='Path for annotation file', required=False)
    parser.add_argument('--dataset', help='Dataset for feature extractoin', required=False, default="OMG")
    parser.add_argument('--typeOutput', help='Average ou VS per frame', required=False, default="average")
    parser.add_argument('--datasetPart', help='Only for AffWild - train or validation', required=False, default="Validation_Set")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model")
    model = ResnetEmotionHeadClassifierAttention(classes=8, resnetModel='resnet18')
    checkpoint = torch.load(args.weights)
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    model.to(device)
    print("Model loaded")
    print(model)
    data_transforms = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.562454871481894, 0.8208898956471341, 0.395364053852456],
            std=[0.43727472598867456, 0.31812502566122625, 0.3796120355707891]
        )])
    print("Loading test set")
    if args.dataset == 'OMG':
        dataset = OMGData(omgData=args.pathBase,annotationFile=args.annotationFile,transform=data_transforms)
    elif args.dataset == 'AFEWVA':
        dataset = AFEWVA(afewdata=args.pathBase,transform=data_transforms)
    elif args.dataset == 'AFFECTNET':
        dataset = AffectNet(afectdata=args.pathBase,transform=data_transforms,typeExperiment='EXP')
    elif args.dataset == 'AFFWILD2':
        dataset = AFF2Data(args.pathBase,args.datasetPart,transform=data_transforms)    
    else:
        dataset = TeachingDataset(teachingData=args.pathBase,transform=data_transforms)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=False)

    model.eval()
    predictions = None
    pathFile = []
    labelsFile = []
    soft =  nn.Softmax(dim=1)
    with torch.no_grad():
        for idxBtc, data in enumerate(val_loader):
            print("Extraction Batch %d" % (idxBtc))
            images, labels, pathsForFiles = data
            _, outputs, _ = model(images.to(device))
            outputs = soft(outputs)

            prediction = outputs.cpu().detach().numpy()
            predictions = prediction if predictions is None else np.concatenate((prediction,predictions))

            pathFile = pathFile + list(pathsForFiles)
            labelsFile = labelsFile + list(np.array(labels))

    vaPerUtt = {}
    if args.dataset == 'OMG':
        for idxF, file in enumerate(pathFile):
            dirUtt = int(file.split(os.path.sep)[-2].split('_')[1])
            if dirUtt not in vaPerUtt.keys():
                vaPerUtt[dirUtt] = [predictions[idxF]]
            else:
                vaPerUtt[dirUtt].append(predictions[idxF])
    else:
        for idxF, file in enumerate(pathFile):            
            dirUtt = file
            if dirUtt not in vaPerUtt.keys():
                vaPerUtt[dirUtt] = {'logits' : [predictions[idxF]], 'label' : labelsFile[idxF]}
            else:
                vaPerUtt[dirUtt]['logits'].append(predictions[idxF])

    print('Generating CSV')
    if args.typeOutput == 'average':
        means = [np.mean(np.array(vaPerUtt[k]),axis=0) for k in vaPerUtt]
        utt = list(map(str,vaPerUtt.keys()))
        saveToCSV(means,utt,args.output)
    else:
        values = []
        utt = []
        lbl = []
        for k in vaPerUtt:
            values.append(vaPerUtt[k]['logits'][0])
            utt.append(k)
            lbl.append(vaPerUtt[k]['label'])
        saveToCSV(values,utt,lbl,args.output)
        print('oi')

if __name__ == '__main__':
    test()