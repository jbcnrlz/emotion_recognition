import argparse, os, sys, torch, numpy as np
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from DatasetClasses.AffectNet import AffectNet, OPTAffectNet
from networks.extractFeaturesBayesian import saveToCSV
from torchvision import transforms
def main():
    parser = argparse.ArgumentParser(description='Finetune resnet')
    parser.add_argument('--pathBase', help='Path for valence and arousal dataset', required=True)
    parser.add_argument('--output', help='Path for valence and arousal dataset', required=True)
    args = parser.parse_args()

    data_transforms = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
    ])

    dataset = OPTAffectNet(
        num_workers=4, 
        preload_images=True, 
        afectdata=os.path.join(args.pathBase,'train_set'),
        transform=data_transforms,
        typeExperiment='PROBS_VAD'
    )
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False)
    pathFile = []
    predictions = None
    for idxBtc, data in enumerate(train_loader):
        _, labels, pathsForFiles = data

        prediction = labels[0].numpy()
        predictions = prediction if predictions is None else np.concatenate((prediction,predictions))

        pathFile = list(pathsForFiles) + pathFile

    saveToCSV(predictions,pathFile,args.output)
if __name__ == '__main__':
    main()