import torch,os,sys,argparse, numpy as np, pandas as pd
from diffusers.models import AutoencoderKL
from torchvision import transforms
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from DatasetClasses.AffectNet import AffectNet
from torchvision.transforms import ToPILImage
from networks.ResnetEmotionHead import ResnetEmotionHead
from torch.utils.tensorboard import SummaryWriter
from torch import optim, nn
from helper.function import saveStatePytorch, getFirstLevel, printProgressBar

def main():
    parser = argparse.ArgumentParser(description='Extract latent features with AutoencoderKL')
    parser.add_argument('--pathBase', help='Path for valence and arousal dataset', required=True)
    parser.add_argument('--epochs', help='Path for valence and arousal dataset', required=False, default=20)
    parser.add_argument('--output', help='Path for valence and arousal dataset', required=False, default='resnetEmotion')
    parser.add_argument('--batchSize', help='Path for valence and arousal dataset', required=True)
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    writer = SummaryWriter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_transforms = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
    ])
    dataset = AffectNet(afectdata=os.path.join(args.pathBase,'train_set'),transform=data_transforms,typeExperiment='EXP')
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

    datasetVal = AffectNet(afectdata=os.path.join(args.pathBase,'val_set'),transform=data_transforms,typeExperiment='EXP')
    val_loader = torch.utils.data.DataLoader(datasetVal, batch_size=2, shuffle=False)

    model = ResnetEmotionHead(2,'resnet18')
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    criterion = nn.CrossEntropyLoss().to(device)
    bestForFoldTLoss = bestForFold = 5000
    for ep in range(args.epochs):
        ibl = ibtl = ' '
        lossAcc = []
        iteration = 0
        for img,label,pathfile in train_loader:
            printProgressBar(iteration,len(dataset.filesPath),length=50,prefix='Procesing face - training')
            img = img.to(device)
            label = label.to(device)
            features, classes = model(img)

            label[label > 1] = 1

            loss = criterion(classes, label)
 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lossAcc.append(loss.item())
            iteration += img.shape[0]

        lossAvg = sum(lossAcc) / len(lossAcc)
        writer.add_scalar('RESNETEmo/Loss/train', lossAvg, ep)
        #scheduler.step()
        model.eval()
        iteration = 0
        loss_val = []
        with torch.no_grad():
            for img,label,pathfile in val_loader:
                printProgressBar(iteration,len(datasetVal.filesPath),length=50,prefix='Procesing face - testing')
                img = img.to(device)
                label = label.to(device)
                label[label > 1] = 1
                features, classes = model(img)
                loss = criterion(classes, label)

                loss_val.append(loss)
                iteration += img.shape[0]

            lossAvgVal = sum(loss_val) / len(loss_val)
            writer.add_scalar('RESNETEmo/Loss/val', lossAvgVal, ep)

        state_dict = model.state_dict()
        opt_dict = optimizer.state_dict()
        fName = '%s_current.pth.tar' % ('resnet_emotion')
        fName = os.path.join(args.output, fName)
        #saveStatePytorch(fName, state_dict, opt_dict, ep + 1)

        if bestForFoldTLoss > lossAvgVal:
            ibtl = 'X'
            fName = '%s_best_val_loss_neutral.pth.tar' % ('resnet_emotion')
            fName = os.path.join(args.output, fName)
            saveStatePytorch(fName, state_dict, opt_dict, ep + 1)
            bestForFoldTLoss = lossAvgVal

        if bestForFold > lossAvg:
            ibl = 'X'
            fName = '%s_best_loss_neutral.pth.tar' % ('resnet_emotion')
            fName = os.path.join(args.output, fName)
            saveStatePytorch(fName, state_dict, opt_dict, ep + 1)
            bestForFold = lossAvg

        print('[EPOCH %03d] Training Loss %.5f Validation Loss %.5f - [%c] [%c]               ' % (ep, lossAvg, lossAvgVal,ibl,ibtl))
    
if __name__ == '__main__':
    main()