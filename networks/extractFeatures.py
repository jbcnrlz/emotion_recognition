import argparse, torch, os, sys, numpy as np
from torchvision import transforms
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import ccc
from networks.DANVA import DANVA
from DatasetClasses.AffWild2 import AFF2Data

def test():
    parser = argparse.ArgumentParser(description='Extract VA with DAN')
    parser.add_argument('--resnetPretrained', help='Path for resnet pretrained weights', required=True)
    parser.add_argument('--weights', help='Weights', required=True)
    parser.add_argument('--pathBase', help='Path for valence and arousal dataset', required=True)
    parser.add_argument('--batch', type=int, help='Size of the batch', required=True)
    parser.add_argument('--output', default=None, help='File to save csv', required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model")
    model = DANVA(num_class=8, num_head=4,pretrained=args.resnetPretrained)
    checkpoint = torch.load(args.weights)
    model.convertToVA()
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    model.to(device)
    print("Model loaded")
    print(model)
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
                transforms.RandomAffine(20, scale=(0.8, 1), translate=(0.2, 0.2)),
            ], p=0.7),

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(),
    ])
    print("Loading test set")
    afw2Val = AFF2Data(args.pathBase,'Validation_Set',transform=data_transforms)    
    val_loader = torch.utils.data.DataLoader(afw2Val, batch_size=args.batch, shuffle=False)

    model.eval()
    predictions = None
    gtsComp = None
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            outputs, _, _ = model(images.to(device))

            prediction = outputs.cpu().detach().numpy()
            predictions = prediction if predictions is None else np.concatenate((prediction,predictions))

            gt = labels.detach().numpy()
            gtsComp = gtsComp if gtsComp is None else np.concatenate((gt,gtsComp))

    print("CCC Valence %f - CCC Arousal %c \n" % (ccc(prediction[:,0],gtsComp[:,0]),ccc(predictions[:,1],gtsComp[:,1])))

if __name__ == '__main__':
    test()