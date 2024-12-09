import os, sys, math
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import printProgressBar
import argparse
import numpy as np
import torch
from torchvision import transforms, datasets
import torch.utils.data as data
from networks.DDAM import DDAMNet
from DatasetClasses.AffWild2 import AFF2Data
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
from DatasetClasses.AffectNet import AffectNet

def saveToCSV(preds,files,labels,pathCSV):
    with open(pathCSV,'w') as pcsv:
        pcsv.write('%s,file,label\n' % (','.join([str(f) for f in range(len(preds[0]))])))
        for idx, p in enumerate(preds):
            for fp in p:
                pcsv.write('%f,' % (fp))
            pcsv.write("%s,%s\n" % (files[idx],labels[idx]))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aff_path', type=str, default='./data/fer_112_112_v2.0/affectnet/', help='AfectNet dataset path.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--workers', default=8, type=int, help='Number of data loading workers.')
    parser.add_argument('--num_head', type=int, default=2, help='Number of attention head.')
    parser.add_argument('--num_class', type=int, default=8, help='Number of class.')
    parser.add_argument('--model_path', default = './checkpoints_ver2.0/affecnet8_epoch25_acc0.6469.pth')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--csv_path', required=True)
    return parser.parse_args()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j]*100, fmt)+'%',
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")


    plt.ylabel('Actual', fontsize=18)
    plt.xlabel('Predicted', fontsize=18)
    plt.tight_layout()


class7_names = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry'] 
class8_names = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry', 'Contempt'] 

def run_test():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = DDAMNet(num_class=args.num_class, num_head=args.num_head, pretrained=False)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    key = 'model_state_dict' if 'model_state_dict' in checkpoint.keys() else 'state_dict'
    model.load_state_dict(checkpoint[key])
    model.to(device)
    model.eval()        

    data_transforms_val = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])     
    val_dataset = None
    if args.dataset == 'affectnet':
        dataset = AffectNet(afectdata=os.path.join(args.aff_path,'val_set'),transform=data_transforms_val,typeExperiment='EXP',exchangeLabel=None)
        #val_dataset = AffectNet(afectdata=os.path.join(args.aff_path,'val_set'),transform=data_transforms_val,typeExperiment='EXP',exchangeLabel=None)    
    else:
        dataset = AFF2Data(args.aff_path,'Validation_Set',transform=data_transforms_val,type="RANK")

    val_dataset = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print('Validation set size:', val_dataset.__len__())
    
    iter_cnt = 0
    bingo_cnt = 0
    sample_cnt = 0
    predictions = None
    files = None
    labels = None
    for idx, (imgs, targets, f) in enumerate(val_dataset):
        printProgressBar(idx,math.ceil(len(dataset.filesPath)/args.batch_size),length=50,prefix='Extracting right features')
        imgs = imgs.to(device)
        targets = targets.to(device)
        out,feat,heads = model(imgs)
        predictions = out.cpu().detach().numpy() if predictions is None else np.concatenate((out.cpu().detach().numpy(),predictions))
        files = np.array(f) if files is None else np.concatenate((np.array(f),files))
        labels = targets.cpu().detach().numpy() if labels is None else np.concatenate((targets.cpu().detach().numpy(),labels))
        sample_cnt += out.size(0)     


    saveToCSV(predictions,files,labels,args.csv_path)
       
if __name__ == "__main__":
                   
    run_test()