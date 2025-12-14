import argparse, torch, os, sys, numpy as np, math, random, re
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from DatasetClasses.AffectNet import AffectNet
from helper.function import saveStatePytorch, printProgressBar, overlay_attention_maps
from networks.EmotionResnetVA import ResnetWithBayesianGMMHead, ResNet50WithAttentionGMM, ResNet50WithAttentionLikelihood, ResNet50WithAttentionLikelihoodNoVA
from torch import nn, optim
import torch.distributions as dist, random
from torch.nn import functional as F
from loss.FocalLoss import FocalLoss
from loss.FocalConsistencyLoss import FocalConsistencyLoss

def regularized_gmm_loss(y_pred, y_true, components, n_components, alpha=0.1):
    """
    Combina MSE com regularização para:
    - Evitar colapso de componentes
    - Garantir boas propriedades numéricas
    """
    # MSE básico
    mse_loss = F.mse_loss(y_pred, y_true)
    
    # Processar parâmetros
    weights = F.softmax(components.view(components.size(0), n_components, -1)[:, :, 0], dim=1)
    cov_params = components.view(components.size(0), n_components, -1)[:, :, 3:]
    
    # Regularização para evitar pesos muito pequenos
    weight_reg = -torch.mean(torch.log(weights + 1e-6))
    
    # Regularização para evitar covariâncias singulares
    det_reg = 0.0
    for i in range(n_components):
        sx = torch.exp(cov_params[:, i, 0])
        sy = torch.exp(cov_params[:, i, 1])
        rho = torch.tanh(cov_params[:, i, 2])
        det = sx**2 * sy**2 * (1 - rho**2)
        det_reg += torch.mean(1 / (det + 1e-6))
    
    return mse_loss + alpha * (weight_reg + det_reg)

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
    parser.add_argument('--trainDataset', help='File with neighbours', required=False,default="affectnet")
    parser.add_argument('--resumeWeights', help='File with neighbours', required=False,default=None)
    parser.add_argument('--resnetSize', help='File with neighbours', required=False,default=18,type=int)
    parser.add_argument('--secondaryLossFunction', help='File with neighbours', required=False,default="ELBO")
    parser.add_argument('--model', help='File with neighbours', required=False,default="gmm")
    parser.add_argument('--pretrainedResnet', help='File with neighbours', required=False,default=None)
    parser.add_argument('--mainLossFunc', help='File with neighbours', required=False,default="BCE")
    parser.add_argument('--lambdaLASSO', help='File with neighbours', required=False,default=0,type=float)
    parser.add_argument('--useDominance', help='File with neighbours', required=False,default=False,type=bool)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.pretrainedResnet is not None:
        checkpoint = torch.load(args.pretrainedResnet,weights_only=False)

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    writer = SummaryWriter()    
    print("Loading model -- Using " + str(device))
    outVA = 2
    if args.useDominance:
        outVA = 3
    if args.model == "gmm":
        model = ResnetWithBayesianGMMHead(classes=args.numberOfClasses,resnetModel=args.resnetSize,pretrained=args.pretrainedResnet)
    elif args.model == "attgmm":
        model = ResNet50WithAttentionGMM(num_classes=args.numberOfClasses,pretrained=args.pretrainedResnet,bottleneck='none',bayesianHeadType='VA' if outVA == 2 else 'VAD')
    elif args.model == 'attbayes':        
        model = ResNet50WithAttentionLikelihood(num_classes=args.numberOfClasses,pretrained=args.pretrainedResnet,bottleneck='none',bayesianHeadType='VA' if outVA == 2 else 'VAD')
    elif args.model == 'simpleNetwork':
        model = ResNet50WithAttentionLikelihoodNoVA(num_classes=args.numberOfClasses,pretrained=args.pretrainedResnet,bottleneck='none')
    model.to(device)    
    print("Model loaded")
    print(model)
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
    #dataset = AffectNet(afectdata=os.path.join(args.pathBase,'train_set'),transform=data_transforms['train'],typeExperiment='PROBS_VA' if outVA == 2 else 'PROBS_VAD')
    dataset = AffectNet(afectdata=os.path.join(args.pathBase,'train_set'),transform=data_transforms['train'],typeExperiment='UNIVERSAL_VAD')
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize, shuffle=True)
    #datasetVal = AffectNet(afectdata=os.path.join(args.pathBase,'val_set'),transform=data_transforms['test'],typeExperiment='PROBS_VA' if outVA == 2 else 'PROBS_VAD')
    datasetVal = AffectNet(afectdata=os.path.join(args.pathBase,'val_set'),transform=data_transforms['test'],typeExperiment='UNIVERSAL_VAD')
    val_loader = torch.utils.data.DataLoader(datasetVal, batch_size=args.batchSize, shuffle=False)

    optimizer = optim.AdamW(model.parameters(), lr=args.learningRate, weight_decay=1e-2)

    scheduler = optim.lr_scheduler.StepLR(optimizer, args.epochs // 5, gamma=0.1)        
    criterion = None
    if args.mainLossFunc == "BCE":
        print("Using BCE loss")
        criterion = nn.BCEWithLogitsLoss().to(device)
    elif args.mainLossFunc == "MSE":
        print("Using MSE loss")
        criterion = nn.MSELoss().to(device)
    elif args.mainLossFunc == "FOCAL":
        print("Using Focal loss")        
        criterion = FocalLoss(alpha=0.25, gamma=2.0).to(device)
    elif args.mainLossFunc == "FOCALCONSISTENCY":
        print("Using Focal Consistency loss")        
        criterion = FocalConsistencyLoss(alpha=0.25, gamma=2.0, conflict_weight=0.5).to(device)
    secLoss = None
    lossFuncName = re.sub(r'[^a-zA-Z0-9\s]', '', str(criterion))
    if args.secondaryLossFunction != "ELBO":
        print("Using secondary loss function: " + args.secondaryLossFunction)
        secLoss= nn.L1Loss().to(device)
    else:
        print("Using ELBO loss")
    start_epoch = 0
    if args.resumeWeights is not None:
        print("Loading weights")
        checkpoint = torch.load(args.resumeWeights)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print("Weights loaded")

    os.system('cls' if os.name == 'nt' else 'clear')
    print("Started traning")
    print('Training Phase =================================================================== BTL  BVL BAC')
    bestForFold = bestForFoldTLoss = 500000
    bestRankForFold = -1
    alpha = 0.1
    for ep in range(start_epoch,args.epochs):
        ibl = ibr = ibtl = ' '
        model.train()
        lossAcc = []
        elboLoss = []
        ceLoss = []
        totalImages = 0
        iteration = 0
        for currBatch, currTargetBatch, _ in train_loader:
            printProgressBar(iteration,math.ceil(len(dataset.filesPath)/args.batchSize),length=50,prefix='Procesing face - training')
            totalImages += currBatch.shape[0]
            currTargetBatch, currBatch, vaBatch = currTargetBatch[0].to(device), currBatch.to(device), currTargetBatch[1].to(device)
            if isinstance(model,ResNet50WithAttentionLikelihoodNoVA):
                classification = model(currBatch)
                loss = criterion(classification, currTargetBatch)
            else:                                
                classification, parameters, vaValueEstim = model(currBatch)
                ceVal = criterion(classification, currTargetBatch)
                if args.secondaryLossFunction == "ELBO":
                    elboVal = elbo_loss(vaValueEstim,vaBatch,model.bayesianHead)
                    loss = 0.999 * ceVal + 0.001 * elboVal
                else:
                    elboVal = secLoss(vaValueEstim, vaBatch)
                    loss = 0.5 * ceVal + 0.5 * elboVal

                if args.lambdaLASSO > 0:
                    l1_norm = sum(p.abs().sum() for p in model.parameters())
                    loss += args.lambdaLASSO * l1_norm

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lossAcc.append(loss.item())
            if not isinstance(model,ResNet50WithAttentionLikelihoodNoVA):
                elboLoss.append(elboVal.item())            
                ceLoss.append(ceVal.item())
            iteration += 1

        lossAvg = sum(lossAcc) / len(lossAcc)
        writer.add_scalar('RESNETAtt/Loss/train', lossAvg, ep)
        if not isinstance(model,ResNet50WithAttentionLikelihoodNoVA):
            elboLoss = sum(elboLoss) / len(elboLoss)
            ceLoss = sum(ceLoss) / len(ceLoss)
            writer.add_scalar(f'RESNETAtt/{lossFuncName}/train',ceLoss, ep)
            writer.add_scalar('RESNETAtt/ELBOLoss/train',elboLoss,ep)
        scheduler.step()
        model.eval()
        elboLoss = []
        ceLoss = []
        loss_val = []
        iteration = 0
        imageAttention = None
        with torch.no_grad():
            for currBatch, currTargetBatch, _ in val_loader:
                printProgressBar(iteration,math.ceil(len(datasetVal.filesPath)/args.batchSize),length=50,prefix='Procesing face - testing')

                totalImages += currBatch.shape[0]
                currTargetBatch, currBatch, vaBatch = currTargetBatch[0].to(device), currBatch.to(device), currTargetBatch[1].to(device)

                if (random.randint(0, 100) < 5) or (imageAttention is None):
                    imageAttention = currBatch[random.randint(0,currBatch.shape[0]-1)].cpu().detach().numpy()

                if isinstance(model,ResNet50WithAttentionLikelihoodNoVA):
                    classification = model(currBatch)
                    loss = criterion(classification, currTargetBatch)
                else:

                    classification, parameters, vaValueEstim = model(currBatch)
                    ceVal = criterion(classification, currTargetBatch)
                    if args.secondaryLossFunction == "ELBO":
                        elboVal = elbo_loss(vaValueEstim,vaBatch,model.bayesianHead)
                        loss = 0.999 * ceVal + 0.001 * elboVal
                    else:
                        elboVal = secLoss(vaValueEstim, vaBatch)
                        loss = 0.5 * ceVal + 0.5 * elboVal
                    elboLoss.append(elboVal.item())                
                    ceLoss.append(ceVal.item())

                loss_val.append(loss.item())
                iteration += 1

        tLoss = sum(loss_val) / len(loss_val)
        writer.add_scalar('RESNETAtt/Loss/val', tLoss, ep)
        if not isinstance(model,ResNet50WithAttentionLikelihoodNoVA):
            elboLoss = sum(elboLoss) / len(elboLoss)
            ceLoss = sum(ceLoss) / len(ceLoss)
            writer.add_scalar(f'RESNETAtt/{lossFuncName}/val',ceLoss, ep)
            writer.add_scalar('RESNETAtt/ELBOLoss/val',elboLoss,ep)
        if imageAttention is not None:
            attentionMaps = model.attention_maps
            _, attMapsOv = overlay_attention_maps(imageAttention, attentionMaps)
            for idx, at in enumerate(attMapsOv):
                writer.add_image(f'RESNETAtt/AttentionMaps_{idx}', at, ep, dataformats='HWC')
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