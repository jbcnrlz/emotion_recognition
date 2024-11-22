import argparse, torch, os, sys, numpy as np
from torchvision import transforms
from torch import nn
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from DAN.networks.dan import DAN
from DatasetClasses.AffectNet import AffectNet
from DatasetClasses.AffWild2 import AFF2Data

def test():
    parser = argparse.ArgumentParser(description='Extract VA with DAN')
    parser.add_argument('--weights', help='Weights', required=True)
    parser.add_argument('--pathBase', help='Path for valence and arousal dataset', required=True)
    parser.add_argument('--batch', type=int, help='Size of the batch', required=True)
    parser.add_argument('--output', default=None, help='File to save csv', required=True)
    parser.add_argument('--classes', type=int, help='File to save csv', required=True)
    parser.add_argument('--dataset', help='File to save csv', required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model")
    model = DAN(num_head=4, num_class=args.classes, pretrained=None)
    checkpoint = torch.load(args.weights)
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
    if args.dataset == 'affwild':
        dataset = AFF2Data(args.pathBase,'Validation_Set',transform=data_transforms,type='EXPR')
    else:
        dataset = AffectNet(afectdata=args.pathBase,transform=data_transforms,typeExperiment="EXP")
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=False)

    model.eval()
    outputFile = []
    predictions = None
    pathFile = []
    labelsFile = []
    result = 0
    total = 0
    with torch.no_grad():
        for idxBtc, data in enumerate(val_loader):
            print("Extraction Batch %d" % (idxBtc))
            images, labels, pathsForFiles = data
            logits, _, attHeads = model(images.to(device))

            if args.dataset != 'affwild':
                _, pred = torch.max(logits,1)
                result += (pred == labels.to(device)).sum().item()
                total += images.shape[0]


            for f in range(attHeads.shape[0]):
                outputFile.append((attHeads[f].sum(dim=0).cpu(),pathsForFiles[f]))

            prediction = logits.clone().cpu().detach().numpy()
            predictions = prediction if predictions is None else np.concatenate((prediction,predictions))
            pathFile = list(pathsForFiles) + pathFile 
            labelsFile = list(np.array(labels)) + labelsFile


    if args.dataset != 'affwild':
        print(result / total)

    vaPerUtt = {}
    for idxF, file in enumerate(pathFile):            
        dirUtt = file
        if dirUtt not in vaPerUtt.keys():
            vaPerUtt[dirUtt] = {'logits' : [predictions[idxF]], 'label' : labelsFile[idxF]}
        else:
            vaPerUtt[dirUtt]['logits'].append(predictions[idxF])

    values = []
    utt = []
    lbl = []
    for k in vaPerUtt:
        values.append(vaPerUtt[k]['logits'][0])
        utt.append(k)
        lbl.append(vaPerUtt[k]['label'])
    saveToCSV(values,utt,lbl,"logitsDAN.csv")


    generateCSVFile(args.output,outputFile)


def generateCSVFile(filePath,features):
    with open(filePath,'w') as fp:
        fp.write(','.join(list(map(str,list(range(features[0][0].shape[0])))))+',%s\n' % ('filePath'))
        for f in features:
            fp.write(','.join(list(map(str,f[0].tolist()))) + ',%s\n' % (f[1]))


def saveToCSV(preds,files,labels,pathCSV):
    with open(pathCSV,'w') as pcsv:
        pcsv.write('%s,file,label\n' % (','.join([str(f) for f in range(len(preds[0]))])))
        for idx, p in enumerate(preds):
            for fp in p:
                pcsv.write('%f,' % (fp))
            pcsv.write("%s,%s\n" % (files[idx],labels[idx]))


if __name__ == '__main__':
    test()