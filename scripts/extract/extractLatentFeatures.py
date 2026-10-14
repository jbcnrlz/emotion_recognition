import argparse, torch, os, sys, numpy as np
from torchvision import transforms
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from networks.unet.unet_model import UNet
from DatasetClasses.AffWild2 import AFF2Data
from DatasetClasses.AffectNet import AffectNet
from torchvision.transforms import ToPILImage

def saveToCSV(features,files,pathCSV):
    with open(pathCSV,'w') as pcsv:
        for i in range(features.shape[1]):
            pcsv.write('%d,' % (i))
        pcsv.write('file\n')
        for idx, p in enumerate(features):
            for vf in p:
                pcsv.write("%f," % (vf))
            pcsv.write('%s\n' % (files[idx]))

def saveImageNetwork(data,path,imageFilename,extesion='png'):
    if not os.path.exists(path):
        os.makedirs(path)

    pilTrans = ToPILImage()
    for id, d in enumerate(data):
        pilImage = pilTrans(d.cpu())
        pilImage.save(os.path.join(path,imageFilename[id].split(os.path.sep)[-1][:-3]+extesion))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Deep Models')
    parser.add_argument('--pathBase', help='Path for faces', required=True)
    parser.add_argument('--batch', type=int, default=500, help='Size of the batch', required=False)
    parser.add_argument('--weights', default=None, help='Do fine tuning with weights', required=False)
    parser.add_argument('--dataset', help='Dataset to train with', required=False, default="AFFECTNET")
    parser.add_argument('--outputFeatureFile', help='Feature file to ouput', required=False, default='autoencoder.csv')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('####### Loading Files for Extraction #######')
    data_transforms_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5],std=[0.5])
    ])

    print("Loading test set")
    if args.dataset == "AFFWILD2":
        afw2Val = AFF2Data(args.pathBase,'Validation_Set',transform=data_transforms_val,type='VA')
    elif args.dataset == "AFFECTNET":
        afw2Val = AffectNet(os.path.join(args.pathBase,'val_set'),'VA',transform=data_transforms_val)
    test_loader = torch.utils.data.DataLoader(afw2Val, batch_size=args.batch, shuffle=False)

    print('####### Creating model #######')
    imLearning = UNet(n_channels=1,n_classes=1)
    checkpoint = torch.load(args.weights)
    imLearning.load_state_dict(checkpoint['state_dict'],strict=True)
    print(imLearning)
    imLearning.to(device)

    print('####### Feature extraction #######')
    imLearning.eval()
    predictions = None
    pathFile = []
    with torch.no_grad():
        for currBatch, currTargetBatch, pathForFiles in test_loader:
            currTargetBatch, currBatch = currTargetBatch.to(device), currBatch.to(device)
            img, features = imLearning(currBatch)
            saveImageNetwork(img,'reconstructed_from_extraction',pathForFiles)
            prediction = features.cpu().detach().numpy()
            predictions = prediction if predictions is None else np.concatenate((prediction,predictions))
            pathFile = pathFile + list(pathForFiles)

    saveToCSV(predictions,pathFile,args.outputFeatureFile)