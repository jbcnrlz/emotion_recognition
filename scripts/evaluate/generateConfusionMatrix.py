import argparse, torch, os, sys, numpy as np, matplotlib.pyplot as plt, re, itertools
from textwrap import wrap
from torchvision import transforms
from torch import nn
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from networks.ResnetEmotionHead import ResnetEmotionHeadClassifierAttention
from DatasetClasses.OMG import OMGData
from DatasetClasses.AFEWVA import AFEWVA
from DatasetClasses.TeachingDataset import TeachingDataset
from DatasetClasses.AffWild2 import AFF2Data
from DatasetClasses.AffectNet import AffectNet
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(correct_labels, predict_labels, labels, normalize=False):
    '''
    Parameters:
        correct_labels                  : These are your true classification categories.
        predict_labels                  : These are you predicted classification categories
        labels                          : This is a lit of labels which will be used to display the axix labels

    Returns:
        summary: TensorFlow summary

    Other itema to note:
        - Depending on the number of category and the data , you may have to modify the figzie, font sizes etc.
        - Currently, some of the ticks dont line up due to rotations.
    '''
    cm = confusion_matrix(correct_labels, predict_labels)
    if normalize:
        cm = cm.astype('float') * 10 / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, copy=True)
        cm = cm.astype('int')

    np.set_printoptions(precision=2)
    ###fig, ax = matplotlib.figure.Figure()

    fig = plt.Figure(figsize=(4, 4), dpi=320, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    imcf = ax.imshow(cm, cmap='Oranges')

    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
    classes = ['\n'.join(wrap(l, 40)) for l in classes]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted', fontsize=7)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=4, rotation=-90, ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=4, va='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i, j] != 0 else '.', horizontalalignment="center", fontsize=3,
                verticalalignment='center', color="black")
    fig.set_tight_layout(True)
    return fig

def test():
    parser = argparse.ArgumentParser(description='Extract VA with RESNETEMOTION')
    parser.add_argument('--weights', help='Weights', required=True)
    parser.add_argument('--confPath', help='confusion matrix path', required=True)
    parser.add_argument('--pathBase', help='Path for valence and arousal dataset', required=True)
    parser.add_argument('--batch', type=int, help='Size of the batch', required=True)
    parser.add_argument('--dataset', help='Dataset for feature extractoin', required=False, default="OMG")
    parser.add_argument('--datasetPart', help='Only for AffWild - train or validation', required=False, default="Validation_Set")
    parser.add_argument('--classesQuantity', help='Only for AffWild - train or validation', required=False, type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model")
    model = ResnetEmotionHeadClassifierAttention(classes=args.classesQuantity, resnetModel='resnet18')
    checkpoint = torch.load(args.weights)
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    model.to(device)
    print("Model loaded")
    print(model)
    data_transforms = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.562454871481894, 0.8208898956471341, 0.395364053852456],
            std=[0.43727472598867456, 0.31812502566122625, 0.3796120355707891]
        )])
    print("Loading test set")
    if args.dataset == 'OMG':
        dataset = OMGData(omgData=args.pathBase,annotationFile=args.annotationFile,transform=data_transforms)
    elif args.dataset == 'AFEWVA':
        dataset = AFEWVA(afewdata=args.pathBase,transform=data_transforms)
    elif args.dataset == 'AFFECTNET':
        dataset = AffectNet(afectdata=args.pathBase,transform=data_transforms,typeExperiment='EXP',loadLastLabel=False)
    elif args.dataset == 'AFFWILD2':
        dataset = AFF2Data(args.pathBase,args.datasetPart,transform=data_transforms)    
    else:
        dataset = TeachingDataset(teachingData=args.pathBase,transform=data_transforms)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=False)
    emotionsLabels = [ 'neutral', 'happy', 'sad', 'surprise','fear', 'disgust', 'anger']
    model.eval()
    predictions = []
    labelsFile = []
    with torch.no_grad():
        for idxBtc, data in enumerate(val_loader):
            print("Extraction Batch %d" % (idxBtc))
            images, labels, pathsForFiles = data
            _, outputs, _ = model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            predictions += list(predicted.detach().cpu().numpy())
            labelsFile += list(labels.detach().cpu().numpy())

    acc = sum(list(map(int,np.array(predictions)==np.array(labelsFile))))
    print(acc/len(predictions))
    confusionMatrix = plot_confusion_matrix(labelsFile,predictions,emotionsLabels)
    confusionMatrix.savefig(args.confPath)
    print('chupeta')

if __name__ == '__main__':
    test()