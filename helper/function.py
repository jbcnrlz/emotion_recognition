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
    image_np = np.transpose(image_tensor, (1, 2, 0))
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
        
        # Usando renderer.buffer_rgba() em vez de tostring_rgb()
        buf = fig.canvas.buffer_rgba()
        overlay_np = np.asarray(buf)
        overlay_np = overlay_np[..., :3]  # Remove o canal alpha se existir
        plt.close(fig)
        
        overlays.append(overlay_np)
    
    return image_np, overlays

def visualize_conflict_matrix(matrix, class_names=None, title=None, 
                             save_path=None, figsize=(12, 10), dpi=100):
    """
    Visualiza uma matriz de conflito
    
    Args:
        matrix: Matriz numpy ou tensor
        class_names: Lista com nomes das classes
        title: Título do gráfico
        save_path: Caminho para salvar a figura
        figsize: Tamanho da figura
        dpi: Resolução da figura
    """
    # Converter tensor para numpy se necessário
    if torch.is_tensor(matrix):
        matrix_np = matrix.cpu().detach().numpy()
    else:
        matrix_np = np.array(matrix)
    
    # Se class_names não for fornecido, criar nomes padrão
    if class_names is None:
        n_classes = matrix_np.shape[0]
        class_names = [f'Class {i}' for i in range(n_classes)]
    
    # Criar figura
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Criar heatmap
    im = ax.imshow(matrix_np, cmap='RdYlBu_r', vmin=0, vmax=1)
    
    # Configurar ticks
    n_classes = len(class_names)
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    
    # Adicionar grade
    ax.set_xticks(np.arange(n_classes + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_classes + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', size=0)
    
    # Adicionar valores nas células
    for i in range(n_classes):
        for j in range(n_classes):
            value = matrix_np[i, j]
            # Escolher cor do texto baseado no fundo
            text_color = 'white' if value > 0.5 else 'black'
            text = ax.text(j, i, f'{value:.2f}', 
                          ha='center', va='center', 
                          color=text_color, fontsize=8)
    
    # Adicionar barra de cores
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Conflict Weight', fontsize=12)
    
    # Título
    if title is None:
        title = f'Conflict Matrix ({n_classes} Classes)'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Salvar figura se path for fornecido
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig, ax

def plot_top_conflicts(matrix, class_names=None, top_k=10, 
                       figsize=(10, 6), save_path=None):
    """
    Plota os maiores conflitos de forma de barras
    
    Args:
        matrix: Matriz de conflito
        class_names: Nomes das classes
        top_k: Número de maiores conflitos para mostrar
        figsize: Tamanho da figura
        save_path: Caminho para salvar
    """
    # Converter tensor para numpy se necessário
    if torch.is_tensor(matrix):
        matrix_np = matrix.cpu().detach().numpy()
    else:
        matrix_np = np.array(matrix)
    
    # Se class_names não for fornecido
    if class_names is None:
        n_classes = matrix_np.shape[0]
        class_names = [f'Class {i}' for i in range(n_classes)]
    
    # Extrair pares de conflito
    conflicts = []
    n = len(class_names)
    
    for i in range(n):
        for j in range(i + 1, n):  # Apenas triângulo superior
            conflicts.append((i, j, matrix_np[i, j]))
    
    # Ordenar por peso
    conflicts.sort(key=lambda x: x[2], reverse=True)
    
    # Pegar top_k
    top_conflicts = conflicts[:top_k]
    
    # Criar figura
    fig, ax = plt.subplots(figsize=figsize)
    
    # Preparar dados para o gráfico de barras
    bars = []
    values = []
    
    for i, j, weight in top_conflicts:
        bar_label = f'{class_names[i]} - {class_names[j]}'
        bars.append(bar_label)
        values.append(weight)
    
    # Criar gráfico de barras
    y_pos = np.arange(len(bars))
    bars_plot = ax.barh(y_pos, values, color='steelblue', alpha=0.8)
    
    # Adicionar valores nas barras
    for bar, value in zip(bars_plot, values):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{value:.3f}', ha='left', va='center', fontsize=9)
    
    # Configurar eixo Y
    ax.set_yticks(y_pos)
    ax.set_yticklabels(bars, fontsize=10)
    ax.invert_yaxis()  # Maior valor no topo
    
    # Configurar eixo X
    ax.set_xlabel('Conflict Weight', fontsize=12)
    ax.set_xlim(0, min(1.0, max(values) * 1.2))
    
    # Título
    ax.set_title(f'Top {top_k} Conflict Pairs', fontsize=14, fontweight='bold')
    
    # Adicionar grade
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Salvar se necessário
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    return fig, ax

def plot_conflict_evolution(conflict_history, class_names=None, 
                           figsize=(14, 10), save_path=None):
    """
    Plota a evolução dos conflitos ao longo do treinamento
    
    Args:
        conflict_history: Lista de matrizes de conflito por época
        class_names: Nomes das classes
        figsize: Tamanho da figura
        save_path: Caminho para salvar
    """
    n_epochs = len(conflict_history)
    n_classes = conflict_history[0].shape[0]
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(n_classes)]
    
    # Criar figura com subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # 1. Evolução da média dos pesos de conflito
    mean_weights = [matrix.mean() for matrix in conflict_history]
    axes[0].plot(range(n_epochs), mean_weights, 'b-o', linewidth=2, markersize=4)
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Mean Conflict Weight', fontsize=11)
    axes[0].set_title('Evolução da Média dos Conflitos', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Evolução do número de conflitos fortes (> 0.5)
    strong_counts = [(matrix > 0.5).sum() for matrix in conflict_history]
    axes[1].plot(range(n_epochs), strong_counts, 'r-o', linewidth=2, markersize=4)
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Número de Conflitos Fortes', fontsize=11)
    axes[1].set_title('Evolução dos Conflitos Fortes', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Matriz inicial vs final
    cmap = plt.cm.RdYlBu_r
    im1 = axes[2].imshow(conflict_history[0], cmap=cmap, vmin=0, vmax=1)
    axes[2].set_title('Matriz Inicial (Época 0)', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im1, ax=axes[2], fraction=0.046, pad=0.04)
    
    im2 = axes[3].imshow(conflict_history[-1], cmap=cmap, vmin=0, vmax=1)
    axes[3].set_title(f'Matriz Final (Época {n_epochs-1})', fontsize=12, fontweight='bold')
    axes[3].axis('off')
    plt.colorbar(im2, ax=axes[3], fraction=0.046, pad=0.04)
    
    plt.suptitle('Evolução da Matriz de Conflito durante o Treinamento', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Salvar se necessário
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    
    return fig, axes