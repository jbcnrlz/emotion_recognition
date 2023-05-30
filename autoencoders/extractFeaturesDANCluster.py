import argparse, torch, os, sys
from torchvision import transforms
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from DAN.networks.dan import DAN
from DatasetClasses.AffectNet import AffectNet

def saveToCSV(preds,files,pathCSV):
    with open(pathCSV,'w') as pcsv:
        pcsv.write('valence,arousal,file\n')
        for idx, p in enumerate(preds):
            pcsv.write('%f,%f,%s\n' % (p[0],p[1],files[idx]))

def test():
    parser = argparse.ArgumentParser(description='Extract VA with DAN')
    parser.add_argument('--weights', help='Weights', required=True)
    parser.add_argument('--pathBase', help='Path for valence and arousal dataset', required=True)
    parser.add_argument('--batch', type=int, help='Size of the batch', required=True)
    parser.add_argument('--output', default=None, help='File to save csv', required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model")
    model = DAN(num_head=4, num_class=8, pretrained=None)
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
    dataset = AffectNet(afectdata=args.pathBase,transform=data_transforms)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=False)

    model.eval()
    outputFile = []
    with torch.no_grad():
        for idxBtc, data in enumerate(val_loader):
            print("Extraction Batch %d" % (idxBtc))
            images, _, pathsForFiles = data
            _, _, attHeads = model(images.to(device))
            for f in range(attHeads.shape[0]):
                outputFile.append((attHeads[f].sum(dim=0).cpu(),pathsForFiles[f]))

    generateCSVFile(args.output,outputFile)


def generateCSVFile(filePath,features):
    with open(filePath,'w') as fp:
        fp.write(','.join(list(map(str,list(range(features[0][0].shape[0])))))+',%s\n' % ('filePath'))
        for f in features:
            fp.write(','.join(list(map(str,f[0].tolist()))) + ',%s\n' % (f[1]))


if __name__ == '__main__':
    test()