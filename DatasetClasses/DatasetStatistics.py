import argparse, os, sys, torch, numpy as np, matplotlib.pyplot as plt
from torchvision import transforms
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from DatasetClasses.AffectNet import AffectNet
from DatasetClasses.JoinedDataset import JoinedDataset

class DatasetStatistics():

    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def plotClassDistribution(self):
        datasetDistribution = {}
        for _, currTargetBatch, _ in self.dataset:
            npCBatch = currTargetBatch.numpy()
            for k in np.unique(npCBatch):
                if k not in datasetDistribution.keys():
                    datasetDistribution[k] = 0
                
                datasetDistribution[k] += np.count_nonzero(npCBatch == k)

        fig, ax = plt.subplots()
        #emotionsLabels = [ 'neutral', 'happy', 'sad', 'surprise','fear', 'disgust', 'anger', 'contempt' ]
        ax.bar(list(datasetDistribution.keys()),list(datasetDistribution.values()))

        plt.show()


def generateStatistics():
    parser = argparse.ArgumentParser(description='Generate statistics')
    parser.add_argument('--pathBase', help='Path for valence and arousal dataset', required=True)
    parser.add_argument('--dataset', help='Path for valence and arousal dataset', required=False, default='affectnet')
    args = parser.parse_args()

    data_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    print("Loading trainig set")
    if args.dataset == 'affectnet':
        dataset = AffectNet(afectdata=os.path.join(args.pathBase,'train_set'),transform=data_transforms,typeExperiment='EXP',exchangeLabel=None)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=False)
    elif args.dataset == 'joineddataset':
        dataset = JoinedDataset(args.pathBase,transform=data_transforms)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=False)
    ds = DatasetStatistics(train_loader)
    ds.plotClassDistribution()



if __name__ == '__main__':
    generateStatistics()