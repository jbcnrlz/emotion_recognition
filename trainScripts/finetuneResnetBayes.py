import argparse, torch, os, sys, numpy as np, math, random
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from DatasetClasses.AffectNet import AffectNet
from helper.function import saveStatePytorch, printProgressBar
from networks.EmotionResnetVA import ResnetWithBayesianHead
from torch import nn, optim
import torch.distributions as dist

def elbo_loss(output, target, model, kl_weight=1.0, sigma=1.0):
    """
    Calcula o ELBO loss para modelos Bayesianos
    
    Args:
        output: Saída do modelo (previsões)
        target: Valores reais
        model: Modelo bayesiano que implementa kl_divergence()
        kl_weight: Peso para o termo KL (útil para annealing)
        sigma: Desvio padrão para a verossimilhança Gaussiana
    
    Returns:
        Tuple: (loss negativo do ELBO, log likelihood, KL divergence)
    """
    # Calcular a verossimilhança logarítmica (log likelihood)
    likelihood = dist.Normal(output, sigma)
    log_likelihood = likelihood.log_prob(target).sum()
    
    # Obter a divergência KL total do modelo
    kl_divergence = model.kl_divergence()
    
    # Calcular o ELBO (Evidence Lower Bound)
    elbo = log_likelihood - kl_weight * kl_divergence
    
    # Retornar o negativo do ELBO (para minimização) e componentes
    return -elbo

def train():
    parser = argparse.ArgumentParser(description='Finetune resnet')
    parser.add_argument('--pathBase', help='Path for valence and arousal dataset', required=True)
    parser.add_argument('--batchSize', type=int, help='Size of the batch', required=True)
    parser.add_argument('--epochs', type=int,help='Epochs to be run', required=True)
    parser.add_argument('--output', default=None, help='Folder to save weights', required=True)
    parser.add_argument('--learningRate', help='Learning Rate', required=False, default=0.01, type=float)
    parser.add_argument('--tensorboardname', help='Learning Rate', required=False, default='DANVA')
    parser.add_argument('--optimizer', help='Optimizer', required=False, default="sgd")
    parser.add_argument('--freeze', help='Freeze weights', required=False, type=int, default=0)
    parser.add_argument('--numberOfClasses', help='Freeze weights', required=False, type=int, default=0)
    parser.add_argument('--additiveLoss', help='Adding additive Loss', required=False,default=None)
    parser.add_argument('--neighsFiles', help='File with neighbours', required=False,default=None)
    parser.add_argument('--trainDataset', help='File with neighbours', required=False,default="affectnet")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    writer = SummaryWriter()    
    print("Loading model -- Using " + str(device))
    model = ResnetWithBayesianHead(classes=args.numberOfClasses)

    model.to(device)    
    print("Model loaded")
    print(model)
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ]),
    'test' : transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])}
    print("Loading trainig set")
    dataset = AffectNet(afectdata=os.path.join(args.pathBase,'train_set'),transform=data_transforms['train'],typeExperiment='PROBS_VA')
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize, shuffle=True)
    datasetVal = AffectNet(afectdata=os.path.join(args.pathBase,'val_set'),transform=data_transforms['test'],typeExperiment='PROBS_VA')
    val_loader = torch.utils.data.DataLoader(datasetVal, batch_size=args.batchSize, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=args.learningRate)

    scheduler = optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.1)
    criterion = nn.BCEWithLogitsLoss().to(device)

    os.system('cls' if os.name == 'nt' else 'clear')
    print("Started traning")
    print('Training Phase =================================================================== BTL  BVL BAC')
    bestForFold = bestForFoldTLoss = 500000
    bestRankForFold = -1
    alpha = 0.1
    for ep in range(args.epochs):
        ibl = ibr = ibtl = ' '
        model.train()
        lossAcc = []
        totalImages = 0
        iteration = 0
        for currBatch, currTargetBatch, _ in train_loader:
            printProgressBar(iteration,math.ceil(len(dataset.filesPath)/args.batchSize),length=50,prefix='Procesing face - training')
            totalImages += currBatch.shape[0]
            currTargetBatch, currBatch, vaBatch = currTargetBatch[0].to(device), currBatch.to(device), currTargetBatch[1].to(device)

            classification, vaValueEstim = model(currBatch)
            loss = criterion(classification, currTargetBatch) + elbo_loss(vaValueEstim,vaBatch,model.bayesianHead)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lossAcc.append(loss.item())
            iteration += 1

        lossAvg = sum(lossAcc) / len(lossAcc)
        writer.add_scalar('RESNETAtt/Loss/train', lossAvg, ep)
        scheduler.step()
        model.eval()
        total = 0
        correct = 0
        loss_val = []
        iteration = 0
        with torch.no_grad():
            for currBatch, currTargetBatch, _ in val_loader:
                printProgressBar(iteration,math.ceil(len(datasetVal.filesPath)/args.batchSize),length=50,prefix='Procesing face - testing')

                totalImages += currBatch.shape[0]
                currTargetBatch, currBatch, vaBatch = currTargetBatch[0].to(device), currBatch.to(device), currTargetBatch[1].to(device)

                classification, vaValueEstim = model(currBatch)
                loss = criterion(classification, currTargetBatch) + elbo_loss(vaValueEstim,vaBatch,model.bayesianHead)

                loss_val.append(loss.item())
                iteration += 1

        tLoss = sum(loss_val) / len(loss_val)
        writer.add_scalar('RESNETAtt/Loss/val', tLoss, ep)
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

        print('[EPOCH %03d] T. Loss %.5f V. Loss %.5f [%c] [%c]' % (ep, lossAvg, tLoss, ibl,ibtl))

if __name__ == '__main__':
    train()