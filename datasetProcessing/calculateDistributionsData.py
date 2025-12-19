import argparse, torch, os, sys, numpy as np, math
from torchvision import transforms
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from DatasetClasses.AffectNet import AffectNet
from helper.function import printProgressBar

def main():
    parser = argparse.ArgumentParser(description='Finetune resnet')
    parser.add_argument('--pathBase', help='Path for valence and arousal dataset', required=True)
    args = parser.parse_args()
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ColorJitter(brightness=0.2,contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ]),
    'test' : transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])}
    print("Loading trainig set")
    dataset = AffectNet(afectdata=os.path.join(args.pathBase,'train_set'),transform=data_transforms['train'],typeExperiment='UNIVERSAL_VAD_EXP')
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=True)

    print("Started Calculating distributions data")
    it = 0
    emotionsCount = [None for _ in range(8)]
    for _, currTargetBatch, _ in train_loader:
        printProgressBar(it,math.ceil(len(dataset.filesPath)/20),length=50,prefix='Procesing face - training')
        for idx, emo in enumerate(currTargetBatch[2]):
            try:
                emotionsCount[emo] = currTargetBatch[1][idx].numpy() if emotionsCount[emo] is None else np.concatenate((emotionsCount[emo].reshape((-1,3)),currTargetBatch[1][idx].numpy().reshape((1,-1))),axis=0)
            except:
                continue
        it += 1
    for i in range(len(emotionsCount)):
        print(f"Emotion {i} - samples: {0 if emotionsCount[i] is None else emotionsCount[i].shape[0]} - Mean VAD: {0 if emotionsCount[i] is None else np.mean(emotionsCount[i],axis=0)} - Std VAD: {0 if emotionsCount[i] is None else np.std(emotionsCount[i],axis=0)}")


if __name__ == '__main__':
    main()