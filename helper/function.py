import os, torch, numpy as np, torch.nn.functional as F, matplotlib.pyplot as plt, cv2

def saveLandmarks(ldns,filepath):
    with open(filepath,'w') as fp:
        for l in ldns:
            fp.write(','.join(list(map(str,l)))+'\n')

def outputImageWithLandmarks(image,landmarks,pathImage):
    for landmark in landmarks:
        for x, y in landmark:
            # display landmarks on "image_cropped"
            # with white colour in BGR and thickness 1
            cv2.circle(image, (x, y), 1, (0, 0, 255), 1)

    cv2.imwrite(pathImage,image)

def getDirectoriesInPath(path):
    return [f for f in os.listdir(path) if not os.path.isfile(os.path.join(path, f))]


def getFilesInPath(path, onlyFiles=True, fullPath=True,imagesOnly=False,imageExtesions=('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
    if fullPath:
        joinFunc = lambda p, fn: os.path.join(p, fn)
    else:
        joinFunc = lambda p, fn: fn

    if onlyFiles:
        return [joinFunc(path, f) for f in os.listdir(path) if (os.path.isfile(os.path.join(path, f)) and (not imagesOnly or f.lower().endswith(imageExtesions)))]
    else:
        return [joinFunc(path, f) for f in os.listdir(path) if (not imagesOnly or f.lower().endswith(imageExtesions))]

def getFilePaths(pathBaseForFaces,imageExtesions=('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
    filesFound = []
    dirs = getDirectoriesInPath(pathBaseForFaces)
    for d in dirs:
        filesFound += getFilePaths(os.path.join(pathBaseForFaces, d),imageExtesions)

    return filesFound + getFilesInPath(pathBaseForFaces, imagesOnly=True,imageExtesions=imageExtesions)

def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    _, output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels, classes):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig

# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def arrangeFaces(faceVector):
    facesArranged = {}
    for f in faceVector:
        fileName = os.path.sep.join(f.split(os.path.sep)[-2:]).split('.')[0].split('_')
        frameSeq = '_'.join(fileName[:3])
        if frameSeq not in facesArranged.keys():
            facesArranged[frameSeq] = []

        facesArranged[frameSeq].append(f)

    return facesArranged

def readFeturesFiles(ffpath):
    returnData=[]
    with open(ffpath,'r') as fct:
        for f in fct:
            returnData.append(list(map(float,f.split(','))))
    return returnData

def saveStatePytorch(fName, stateDict, optmizerStateDict, epoch, arch='VGG'):
    torch.save({
        'epoch': epoch,
        'arch': arch,
        'state_dict': stateDict,
        'optimizer': optmizerStateDict,
    }, fName)

def ccc(x,y):
    ''' Concordance Correlation Coefficient'''
    sxy = np.sum((x - x.mean())*(y - y.mean()))/x.shape[0]
    rhoc = 2*sxy / (np.var(x) + np.var(y) + (x.mean() - y.mean())**2)
    return rhoc