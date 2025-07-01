import os, torch, numpy as np, torch.nn.functional as F, matplotlib.pyplot as plt, cv2
from shapely.geometry.polygon import Point
from shapely import affinity
from scipy.ndimage import zoom

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

def saveCSV(pathFile,header,data):
    with open(pathFile,'w') as pf:
        pf.write('class,valence mean,valence std,arousal mean,arousal std\n')
        for idxF, d in enumerate(data):
            if header[idxF] != '':
                pf.write(header[idxF]+','+','.join(list(map(str,d)))+'\n')

def create_ellipse(center, lengths, angle=0):
    """
    create a shapely ellipse. adapted from
    https://gis.stackexchange.com/a/243462
    """
    circ = Point(center).buffer(1)
    ell = affinity.scale(circ, lengths[0], lengths[1])
    ellr = affinity.rotate(ell, angle)
    return ellr

def getFirstLevel(emotions):
    returnData = []
    for e in emotions:
        if e[-1] == 1:
            returnData.append(e)

    return np.array(returnData)

def getSecondLevelFromVA(firsLevelVA,emotions):
    flvl = getFirstLevel(emotions)
    resultFinal = np.array([ np.linalg.norm(firsLevelVA-s[[1,3]]) for s in flvl]).argmin()
    fLevel = flvl[resultFinal][0]
    secondLevel = separatedSecondLevel(fLevel,emotions)
    resultFinal = np.array([ np.linalg.norm(firsLevelVA-s[[1,3]]) for s in secondLevel]).argmin()
    return fLevel, secondLevel[resultFinal][0]

def separatedSecondLevel(firstLevel,emotions):
    found = 0
    output = []
    for e in emotions:
        if firstLevel == e[0]:
            found = 1
        else:
            if found == 1 and e[1] != 0:
                output.append(e)
            elif found == 1:
                break

    return np.array(output)

def getFeatureFromText(filePath):
    returnClass= ''
    with open(filePath,'r') as fp:
        for f in fp:
            returnClass = f.strip()
            return returnClass.lower()

def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
def loadNeighFiles(pathFile):
    returnData = {}
    with open(pathFile,'r') as pf:
        currFile = ''
        for p in pf:
            fileComma = p.split(',')
            if len(fileComma) > 3:
                filePath = fileComma[0].split(os.path.sep)[-1]
                currFile = filePath
                returnData[filePath] = {'label' : int(fileComma[3]), 'va' : [float(fileComma[1]),float(fileComma[2])],'neighbours' : []}
            else:
                returnData[currFile]['neighbours'].append([float(fileComma[0]),float(fileComma[1]),int(fileComma[2])])

    return returnData


def visualizeAttentionMaps(image_tensor, attention_maps, title="Mapas de Atenção", save_dir="attention_maps_output", image_name="input_image"):
    # Cria o diretório de saída se ele não existir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    num_maps = len(attention_maps)
    
    image_display = image_tensor.permute(1, 2, 0).cpu().numpy()
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_display = image_display * std + mean
    image_display = np.clip(image_display, 0, 1)

    # Figura para exibir todos os mapas juntos
    fig, axes = plt.subplots(1, num_maps + 1, figsize=(4 * (num_maps + 1), 4))
    
    # Salva a imagem original
    plt.imsave(os.path.join(save_dir, f"{image_name}_original.png"), image_display)
    axes[0].imshow(image_display)
    axes[0].set_title("Imagem Original")
    axes[0].axis('off')

    for i, attention_map_tensor in enumerate(attention_maps):
        attention_map_np = attention_map_tensor.squeeze().cpu().numpy()
        
        if attention_map_np.ndim == 3:
            attention_map_np = np.mean(attention_map_np, axis=0)
        
        resized_attention_map = zoom(attention_map_np, 
                                     (image_display.shape[0] / attention_map_np.shape[0],
                                      image_display.shape[1] / attention_map_np.shape[1]),
                                     order=3) 

        # Cria uma nova figura para cada mapa de atenção para salvar individualmente
        fig_single, ax_single = plt.subplots(1, 1, figsize=(6, 6))
        ax_single.imshow(image_display)
        # Salva o mapa de calor puro
        plt.imsave(os.path.join(save_dir, f"{image_name}_attention_map_layer_{i+1}_heatmap_only.png"), resized_attention_map, cmap='jet')
        
        # Salva o mapa de atenção sobreposto à imagem original
        ax_single.imshow(resized_attention_map, cmap='jet', alpha=0.5) 
        ax_single.set_title(f"Atenção Layer {i+1}")
        ax_single.axis('off')
        plt.savefig(os.path.join(save_dir, f"{image_name}_attention_map_layer_{i+1}_overlay.png"), bbox_inches='tight', pad_inches=0)
        plt.close(fig_single) # Fecha a figura individual para não sobrecarregar a memória

        # Adiciona ao plot geral
        axes[i+1].imshow(image_display) 
        axes[i+1].imshow(resized_attention_map, cmap='jet', alpha=0.5) 
        axes[i+1].set_title(f"Atenção Layer {i+1}")
        axes[i+1].axis('off')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_dir, f"{image_name}_all_attention_maps.png"), bbox_inches='tight') # Salva a figura com todos os mapas

def overlay_attention_maps(image_tensor, attention_maps):
    """
    Retorna imagens originais sobrepostas com mapas de atenção como arrays NumPy.
    
    Args:
        image_tensor: torch.Tensor shape [C, H, W] (normalizado)
        attention_maps: Lista de torch.Tensor com mapas de atenção
        
    Returns:
        tuple: (image_np, overlays)
            - image_np: imagem original como array [H, W, 3]
            - overlays: lista de arrays [H, W, 3] com sobreposições
    """
    # Converte e desnormaliza a imagem
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = image_np * std + mean
    image_np = np.clip(image_np, 0, 1)
    
    overlays = []
    
    for attention_map_tensor in attention_maps:
        # Processa o mapa de atenção
        attn_map = attention_map_tensor.squeeze().cpu().numpy()
        
        if attn_map.ndim == 3:
            attn_map = np.mean(attn_map, axis=0)
        
        # Redimensiona para o tamanho da imagem
        h_ratio = image_np.shape[0] / attn_map.shape[0]
        w_ratio = image_np.shape[1] / attn_map.shape[1]
        resized_attn = zoom(attn_map, (h_ratio, w_ratio), order=3)
        
        # Cria a sobreposição
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image_np)
        ax.imshow(resized_attn, cmap='jet', alpha=0.5)
        ax.axis('off')
        
        # Converte a figura para array NumPy
        fig.canvas.draw()
        overlay_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        overlay_np = overlay_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        overlays.append(overlay_np)
    
    return image_np, overlays