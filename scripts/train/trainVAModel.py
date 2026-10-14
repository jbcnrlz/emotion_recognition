import argparse, torch, os, sys, numpy as np, math, random, re
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from DatasetClasses.AffectNet import AffectNet
from helper.function import saveStatePytorch, printProgressBar, overlay_attention_maps
from networks.EmotionResnetVA import Glint360kResNetWithCrossAttentionVA

from torch import nn, optim
from torch.nn import functional as F
from loss.VAloss import CCCLoss

def js_divergence(p_logits, q_target):
    p_probs = F.softmax(p_logits, dim=1)
    q_probs = torch.clamp(q_target, min=0.0)
    q_probs = q_probs / (q_probs.sum(dim=1, keepdim=True) + 1e-8)
    m_probs = 0.5 * (p_probs + q_probs)
    m_log_probs = torch.log(m_probs + 1e-8)
    kl_p_m = F.kl_div(m_log_probs, p_probs, reduction='batchmean')
    kl_q_m = F.kl_div(m_log_probs, q_probs, reduction='batchmean')
    jsd = 0.5 * (kl_p_m + kl_q_m)
    return jsd

def train():
    parser = argparse.ArgumentParser(description='Finetune resnet for Valence and Arousal')
    parser.add_argument('--pathBase', help='Path for valence and arousal dataset', required=True)
    parser.add_argument('--batchSize', type=int, help='Size of the batch', required=True)
    parser.add_argument('--epochs', type=int,help='Epochs to be run', required=True)
    parser.add_argument('--output', default=None, help='Folder to save weights', required=True)
    parser.add_argument('--learningRate', help='Learning Rate', required=False, default=0.01, type=float)
    parser.add_argument('--optimizer', help='Optimizer', required=False, default="sgd")
    parser.add_argument('--freeze', help='Freeze weights', required=False, type=int, default=0)
    parser.add_argument('--trainDataset', help='Train dataset', required=False,default="affectnet")
    parser.add_argument('--resumeWeights', help='Resume weights', required=False,default=None)
    parser.add_argument('--model', help='Model Architecture', required=False,default="glint360ksimpleNetworkCrossAtt")
    parser.add_argument('--pretrainedResnet', help='Pretrained Resnet', required=False,default=None)
    parser.add_argument('--mainLossFunc', help='Main Loss Function', required=False,default="MSE")
    parser.add_argument('--lambdaLASSO', help='Lambda LASSO', required=False,default=0,type=float)
    parser.add_argument('--useDominance', help='Use Dominance', required=False,default=False,type=bool)
    parser.add_argument('--typeExperiment', help='Experiment type', required=False, default="VA")
    parser.add_argument('--emotionMapping', help='Order of the emotions in the dataset', required=False, default="7,0,6,2,5,4,3,1")
    parser.add_argument('--imageSize', type=int, default=256, help='Size of the image')
    parser.add_argument('--patience', type=int, default=4, help='Early stopping patience')
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emotionsOrder = np.array(list(map(int, args.emotionMapping.split(','))))

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    writer = SummaryWriter()    
    print("Loading model -- Using " + str(device))
    outVA = 2
    if args.useDominance or "VAD" in args.typeExperiment:
        outVA = 3
    
    # Instance explicitly for VA cross attention model
    if args.model == 'glint360ksimpleNetworkCrossAtt':
        model = Glint360kResNetWithCrossAttentionVA(pretrained_path=args.pretrainedResnet, num_sectors=7)
    else:
         raise ValueError(f"Model {args.model} is not supported in this script version.")
        
    model.to(device)    
    print("Model loaded")
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((args.imageSize, args.imageSize)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), value=0),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    'test' : transforms.Compose([
        transforms.Resize((args.imageSize, args.imageSize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])}
    
    print("Loading trainig set")
    dataset = AffectNet(afectdata=os.path.join(args.pathBase,'train_set'),transform=data_transforms['train'],typeExperiment=args.typeExperiment)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize, shuffle=True)
    datasetVal = AffectNet(afectdata=os.path.join(args.pathBase,'val_set'),transform=data_transforms['test'],typeExperiment=args.typeExperiment)
    val_loader = torch.utils.data.DataLoader(datasetVal, batch_size=args.batchSize, shuffle=False)
    
    criterion = None
    if args.mainLossFunc == "MSE":
        criterion = nn.MSELoss().to(device)
    elif args.mainLossFunc == "CCC":
        print("Using Concordance Correlation Coefficient (CCC) Loss")
        criterion = CCCLoss().to(device)
    else:
        raise ValueError(f"Loss function {args.mainLossFunc} is not supported for continuous target variable.")

    optimizer = optim.AdamW(model.parameters(), lr=args.learningRate, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.epochs // 5, gamma=0.1)        
    
    start_epoch = 0
    if args.resumeWeights is not None:
        checkpoint = torch.load(args.resumeWeights)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']

    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"Started traning with the protocol {args.typeExperiment}")
    print('Training Phase =================================================================== BTL  BVL BAC')
    bestForFold = bestForFoldTLoss = 500000
    bestRank = -np.inf
    epochs_no_improve = 0
    
    is_prob_target = ("PROBS" in args.typeExperiment or "UNIVERSAL" in args.typeExperiment)
    
    for ep in range(start_epoch,args.epochs):
        if args.model == 'glint360ksimpleNetworkCrossAtt' and ep == 5:
            model.set_backbone_freeze(freeze=False)
            backbone_lr = args.learningRate * 0.1 
            optimizer = optim.AdamW([
                {'params': model.backbone.parameters(), 'lr': backbone_lr},
                {'params': model.attention_fusion.parameters(), 'lr': args.learningRate},
                {'params': model.va_head.parameters(), 'lr': args.learningRate}
            ], weight_decay=1e-2)
            scheduler = optim.lr_scheduler.StepLR(optimizer, (args.epochs - ep) // 3, gamma=0.1)

        rankT = rankA = None
        ibl = ibr = ibtl = ' '
        model.train()
        lossAcc = []
        jsdTrainAcc = [] 
        rmseTrainAcc = [] 
        totalImages = 0
        iteration = 0
        accPercentage = None
        
        for currBatch, currTargetBatch, _ in train_loader:
            printProgressBar(iteration,math.ceil(len(dataset.filesPath)/args.batchSize),length=50,prefix='Procesing face - training')
            totalImages += currBatch.shape[0]
            currBatch = currBatch.to(device)
            
            # ATUALIZADO: Lógica Robusta de Unpacking para suportar VA e VAD puro
            if isinstance(currTargetBatch, list):
                main_target = currTargetBatch[0].to(device)
                vaBatch = currTargetBatch[1].to(device)
                labels = currTargetBatch[2].to(device) if len(currTargetBatch) > 2 else None
            else:
                if "EXP" in args.typeExperiment:
                    main_target = currTargetBatch[:, :-1].to(device) 
                    vaBatch = main_target
                    labels = currTargetBatch[:, -1].to(torch.long).to(device) 
                else:
                    main_target = currTargetBatch.to(device)
                    vaBatch = main_target
                    labels = None

            classification = model(currBatch)
            loss = criterion(classification, main_target)

            if args.lambdaLASSO > 0:
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss += args.lambdaLASSO * l1_norm

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lossAcc.append(loss.item())
            
            if is_prob_target:
                jsd_val = js_divergence(classification.detach(), main_target.detach())
                jsdTrainAcc.append(jsd_val.item())
            else:
                rmse_val = torch.sqrt(F.mse_loss(classification.detach(), main_target.detach())).item()
                rmseTrainAcc.append(rmse_val)

            if labels is not None and is_prob_target:
                _, preds = torch.max(classification, 1)
                corrects = torch.sum(preds.cpu() == emotionsOrder[labels.cpu()]).item()
                accPercentage = corrects if accPercentage is None else accPercentage + corrects
            iteration += 1

        lossAvg = sum(lossAcc) / len(lossAcc)
        writer.add_scalar('RESNETAtt/Loss/train', lossAvg, ep)
        
        if is_prob_target:
            jsdTrainAvg = sum(jsdTrainAcc) / len(jsdTrainAcc)
            writer.add_scalar('RESNETAtt/JSD/train', jsdTrainAvg, ep)
        else:
            rmseTrainAvg = sum(rmseTrainAcc) / len(rmseTrainAcc)
            writer.add_scalar('RESNETAtt/RMSE/train', rmseTrainAvg, ep)

        if labels is not None and accPercentage is not None:
            accPercentage = accPercentage / totalImages
            writer.add_scalar('RESNETAtt/Accuracy/train', accPercentage, ep)
            rankT = accPercentage

        scheduler.step()
        model.eval()
        loss_val = []
        jsdValAcc = [] 
        rmseValAcc = []
        iteration = 0
        imageAttention = None
        accPercentage = None
        totalImages = 0
        
        with torch.no_grad():
            for currBatch, currTargetBatch, _ in val_loader:
                printProgressBar(iteration,math.ceil(len(datasetVal.filesPath)/args.batchSize),length=50,prefix='Procesing face - testing')
                totalImages += currBatch.shape[0]
                currBatch = currBatch.to(device)
                
                # ATUALIZADO: Lógica Robusta de Unpacking (Validação)
                if isinstance(currTargetBatch, list):
                    main_target = currTargetBatch[0].to(device)
                    vaBatch = currTargetBatch[1].to(device)
                    labels = currTargetBatch[2].to(device) if len(currTargetBatch) > 2 else None
                else:
                    if "EXP" in args.typeExperiment:
                        main_target = currTargetBatch[:, :-1].to(device)
                        vaBatch = main_target
                        labels = currTargetBatch[:, -1].to(torch.long).to(device)
                    else:
                        main_target = currTargetBatch.to(device)
                        vaBatch = main_target
                        labels = None

                if (random.randint(0, 100) < 5) or (imageAttention is None):
                    imageAttention = currBatch[random.randint(0,currBatch.shape[0]-1)].cpu().detach().numpy()

                classification = model(currBatch)
                loss = criterion(classification, main_target)
                loss_val.append(loss.item())
                
                if is_prob_target:
                    jsd_val = js_divergence(classification.detach(), main_target.detach())
                    jsdValAcc.append(jsd_val.item())
                else:
                    rmse_val = torch.sqrt(F.mse_loss(classification.detach(), main_target.detach())).item()
                    rmseValAcc.append(rmse_val)

                if labels is not None and is_prob_target:
                    _, preds = torch.max(classification, 1)
                    corrects = torch.sum(preds.cpu() == emotionsOrder[labels.cpu()]).item()
                    accPercentage = corrects if accPercentage is None else accPercentage + corrects

                iteration += 1

        tLoss = sum(loss_val) / len(loss_val)
        writer.add_scalar('RESNETAtt/Loss/val', tLoss, ep)
        
        if is_prob_target:
            jsdValAvg = sum(jsdValAcc) / len(jsdValAcc)
            writer.add_scalar('RESNETAtt/JSD/val', jsdValAvg, ep)
        else:
            rmseValAvg = sum(rmseValAcc) / len(rmseValAcc)
            writer.add_scalar('RESNETAtt/RMSE/val', rmseValAvg, ep)

        if labels is not None and accPercentage is not None:
            accPercentage = accPercentage / totalImages
            writer.add_scalar('RESNETAtt/Accuracy/val', accPercentage, ep)
            rankA = accPercentage

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
            epochs_no_improve = 0  
        else:
            epochs_no_improve += 1 

        if bestForFold > lossAvg:
            ibl = 'X'
            fName = '%s_best_loss.pth.tar' % ('RESNETATT')
            fName = os.path.join(args.output, fName)
            saveStatePytorch(fName, state_dict, opt_dict, ep + 1)
            bestForFold = lossAvg

        if (rankT is not None) and (bestRank < rankT):
            bestRank = rankT
            ibr = 'X'
            fName = '%s_best_accuracy.pth.tar' % ('RESNETATT')
            fName = os.path.join(args.output, fName)
            saveStatePytorch(fName, state_dict, opt_dict, ep + 1)        

        # Formatting Output variables depending on regression vs classification
        if is_prob_target:
            metric_t_str = f"T. JSD {jsdTrainAvg:.4f}"
            metric_v_str = f"V. JSD {jsdValAvg:.4f}"
        else:
            metric_t_str = f"T. RMSE {rmseTrainAvg:.4f}"
            metric_v_str = f"V. RMSE {rmseValAvg:.4f}"

        if rankT is not None:
            print(f"[EPOCH {ep:03d}] T. Loss {lossAvg:.5f} T. Rank {rankT:.4f} {metric_t_str} V. Loss {tLoss:.5f} V. Rank: {rankA:.5f} {metric_v_str} [{ibl}] [{ibtl}] [{ibr}] | Patience: {epochs_no_improve}/{args.patience}")
        else:
            print(f"[EPOCH {ep:03d}] T. Loss {lossAvg:.5f} {metric_t_str} V. Loss {tLoss:.5f} {metric_v_str} [{ibl}] [{ibtl}] | Patience: {epochs_no_improve}/{args.patience}")

        if epochs_no_improve >= args.patience:
            print(f"\n[Early Stopping] Treinamento interrompido na época {ep:03d}. Sem melhora na Validation Loss (tLoss) por {args.patience} épocas consecutivas.")
            break

if __name__ == '__main__':
    train()