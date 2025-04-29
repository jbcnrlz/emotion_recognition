import argparse, torch, os, sys, numpy as np, math
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from loss.LTRLosses import approxNDCGLoss, rankNet, neuralNDCG
from DatasetClasses.AffectNet import AffectNet
from helper.function import saveStatePytorch, printProgressBar, loadNeighFiles
from torch import nn, optim
from networks.DDAM import DDAMNet
from networks.sam import SAM
import torch.nn.functional as F


eps = sys.float_info.epsilon

class AttentionLoss(nn.Module):
    def __init__(self, ):
        super(AttentionLoss, self).__init__()
    
    def forward(self, x):
        num_head = len(x)
        loss = 0
        cnt = 0
        if num_head > 1:
            for i in range(num_head-1):
                for j in range(i+1, num_head):
                    mse = F.mse_loss(x[i], x[j])
                    cnt = cnt+1
                    loss = loss+mse
            loss = cnt/(loss + eps)
        else:
            loss = 0
        return loss

def train():
    parser = argparse.ArgumentParser(description='Finetune resnet')
    parser.add_argument('--pathBase', help='Path for valence and arousal dataset', required=True)
    parser.add_argument('--batchSize', type=int, help='Size of the batch', required=True)
    parser.add_argument('--epochs', type=int,help='Epochs to be run', required=True)
    parser.add_argument('--output', default=None, help='Folder to save weights', required=True)
    parser.add_argument('--learningRate', help='Learning Rate', required=False, default=0.01, type=float)
    parser.add_argument('--tensorboardname', help='Learning Rate', required=False, default='DAMMNet_Rank')
    parser.add_argument('--optimizer', help='Optimizer', required=False, default="adam")
    parser.add_argument('--freeze', help='Freeze weights', required=False, type=int, default=0)
    parser.add_argument('--model_path', help='Freeze weights', required=False, default=0)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model -- Using " + str(device))
    model = DDAMNet(num_class=8,num_head=2,pretrained=True)

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.RandomAffine(20, scale=(0.8, 1), translate=(0.2, 0.2)),
            ], p=0.7),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=1, scale=(0.05, 0.05))
        ]),
        'test' : transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
    }

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    writer = SummaryWriter()    
    #checkpoint = torch.load(args.model_path, map_location=device)
    #model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    if args.freeze:
        print("Freezing weights")
        for param in model.parameters():
            param.requires_grad = bool(args.freeze)

    model.to(device)    
    print("Model loaded")
    #print(model)
    print("Loading trainig set")
    dataset = AffectNet(afectdata=os.path.join(args.pathBase,'train_set'),transform=data_transforms['train'],typeExperiment='EXP',exchangeLabel=None)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize, shuffle=True)

    datasetVal = AffectNet(afectdata=os.path.join(args.pathBase,'val_set'),transform=data_transforms['test'],typeExperiment='EXP',exchangeLabel=None)
    val_loader = torch.utils.data.DataLoader(datasetVal, batch_size=args.batchSize, shuffle=False)

    optimizer = SAM(model.parameters(),torch.optim.Adam,lr=args.learningRate,rho=0.05,adaptive=False)

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    criterion = nn.CrossEntropyLoss().to(device)
    criterion_at = AttentionLoss()

    os.system('cls' if os.name == 'nt' else 'clear')
    print("Started traning")
    print('Training Phase =================================================================== BTL  BVL BAC')
    bestForFold = bestForFoldTLoss = 500000
    bestRankForFold = -1
    for ep in range(args.epochs):
        ibl = ibr = ibtl = ' '
        model.train()
        lossAcc = []
        ceLossHist = []
        totalImages = 0
        iteration = 0
        for currBatch, currTargetBatch, _ in train_loader:
            printProgressBar(iteration,math.ceil(len(dataset.filesPath)/args.batchSize),length=50,prefix='Procesing face - training')
            totalImages += currBatch.shape[0]
            currTargetBatch, currBatch = currTargetBatch.to(device), currBatch.to(device)
            classification,_,heads = model(currBatch)

            clValue = criterion(classification, currTargetBatch)
            atValue = criterion_at(heads)
            loss = clValue + 0.1*atValue

            optimizer.zero_grad()
            loss.backward()
            optimizer.first_step(zero_grad=True)

            lossAcc.append(loss.item())
            ceLossHist.append(clValue.item())
            iteration += 1

        lossAvg = sum(lossAcc) / len(lossAcc)
        ceLossHist = sum(ceLossHist) / len(ceLossHist)
        writer.add_scalar('RESNETAtt/Loss/train', lossAvg, ep)
        writer.add_scalar('RESNETAtt/CELoss/train', ceLossHist,ep)
        scheduler.step()
        model.eval()
        total = 0
        correct = 0
        loss_val = []
        ceLossHist = []
        rankNetLossHist = []
        iteration = 0
        with torch.no_grad():
            for data in val_loader:
                printProgressBar(iteration,math.ceil(len(datasetVal.filesPath)/args.batchSize),length=50,prefix='Procesing face - testing')
                images, labels, _  = data
                classification,_,heads = model(images.to(device))
                _, predicted = torch.max(classification.data, 1)
                labels = labels.to(device)

                clValue = criterion(classification, labels)
                atValue = criterion_at(heads)
                loss = clValue + 0.1*atValue

                loss_val.append(loss)
                ceLossHist.append(clValue.item())
                total += labels.size(0)
                correct += (predicted == labels.to(device)).sum().item()
                iteration += 1

        cResult = correct / total
        tLoss = sum(loss_val) / len(loss_val)
        ceLossHist = sum(ceLossHist) / len(ceLossHist)
        writer.add_scalar('RESNETAtt/Loss/val', tLoss, ep)
        writer.add_scalar('RESNETAtt/CELoss/val', ceLossHist,ep)
        writer.add_scalar('RESNETAtt/Acc', cResult, ep)
        state_dict = model.state_dict()
        opt_dict = optimizer.state_dict()

        if bestForFoldTLoss > tLoss:
            ibtl = 'X'
            fName = '%s_best_val_loss.pth.tar' % ('RESNETATT')
            fName = os.path.join(args.output, fName)
            saveStatePytorch(fName, state_dict, opt_dict, ep + 1)
            bestForFoldTLoss = tLoss

        if bestForFold > lossAvg:
            ibl = 'X'
            fName = '%s_best_loss.pth.tar' % ('RESNETATT')
            fName = os.path.join(args.output, fName)
            saveStatePytorch(fName, state_dict, opt_dict, ep + 1)
            bestForFold = lossAvg

        if bestRankForFold < cResult:
            ibr = 'X'
            fName = '%s_best_ccc.pth.tar' % ('RESNETATT')
            fName = os.path.join(args.output, fName)
            saveStatePytorch(fName, state_dict, opt_dict, ep + 1)
            bestRankForFold = cResult

        print('[EPOCH %03d] T. Loss %.5f V. Loss %.5f V. Score %.5f [%c] [%c] [%c]' % (ep, lossAvg, tLoss, cResult, ibl,ibtl,ibr))

if __name__ == '__main__':
    train()