import numpy as np, argparse, os, sys, shutil
import matplotlib.pyplot as plt
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getFilesInPath, printProgressBar, getDirectoriesInPath
import matplotlib.pyplot as plt

def fillDictFromDataset(pathBaseForFaces):
    imagePath = os.path.join(os.path.sep.join(pathBaseForFaces.split(os.path.sep)[:-3]),'cropped_aligned')
    returnDict = {}
    dirFaces = getFilesInPath(pathBaseForFaces)
    for cf, d in enumerate(dirFaces):
        printProgressBar(iteration=cf, total=len(dirFaces), prefix='Loading annotations', suffix='Complete', length=50)
        folderName = d.split(os.path.sep)[-1][:-4]
        returnDict[folderName] = [int(fileName.split(os.path.sep)[-1][:-4]) for fileName in getFilesInPath(os.path.join(imagePath,folderName),imagesOnly=True)]
        returnDict[folderName].sort()
    print("")
    return returnDict

def loadAffWildDataset(pathAnnotations,filesIdx):    
    files = [os.path.join(pathAnnotations,f+'.txt') for f in list(filesIdx.keys())]
    vaValues = []
    idxs = []
    fileName = []
    for cf, f in enumerate(files):
        imageVA = []
        printProgressBar(iteration=cf, total=len(files), prefix='Loading Files', suffix='Complete', length=50)
        fileName.append(f.split(os.path.sep)[-1])
        with open(f, 'r') as fct:
            for iLine, line in enumerate(fct):
                line = line.split(',')
                try:
                    line = list(map(float, line))
                    if line not in imageVA:
                        idxs.append([filesIdx[fileName[-1][:-4]][iLine],fileName[-1]])
                        imageVA.append(line)
                        vaValues.append(line)
                except Exception as e:
                    print(e)

                if (len(vaValues) != len(idxs)):
                    print("Isso não devia acontecer")
    return np.array(vaValues), idxs, fileName

def calcular_densidade_pontos(limite_x, limite_y, pontos_total, mostrar_grafico=False):
    
    # Extrai os limites
    x_min, x_max = limite_x
    y_min, y_max = limite_y
    
    # Verifica se os limites são válidos
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("Os limites mínimos devem ser menores que os máximos")
    
    # Calcula a área total
    area_total = (x_max - x_min) * (y_max - y_min)
    
    # Gera pontos aleatórios dentro dos limites
    maskX = (pontos_total[:,0] > x_min) & (pontos_total[:,0] < x_max)
    maskY = (pontos_total[:,1] > y_min) & (pontos_total[:,1] < y_max)
    finalMask = maskX & maskY
    pontos_x = pontos_total[finalMask,0]
    pontos_y = pontos_total[finalMask,1]
    
    # Se quiser mostrar o gráfico
    if mostrar_grafico:
        plt.figure(figsize=(8, 6))
        plt.scatter(pontos_x, pontos_y, s=1, alpha=0.5)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xlabel('Eixo X')
        plt.ylabel('Eixo Y')
        plt.title(f'Distribuição de pontos')
        plt.grid(True)
        plt.show()
    
    # Calcula a densidade (pontos por unidade de área)
    densidade = pontos_total[finalMask].shape[0] / area_total
    
    return densidade

def determinar_quadrante(x, y):
    if x > 0 and y > 0:
        return 0
    elif x < 0 and y > 0:
        return 1
    elif x < 0 and y < 0:
        return 2
    elif x > 0 and y < 0:
        return 3
    else:
        return 0

def main():
    parser = argparse.ArgumentParser(description='Fuse distributions')
    parser.add_argument('--pathBase', help='Path for valence and arousal dataset', required=True)
    parser.add_argument('--pathWild', help='Path for valence and arousal dataset', required=True)
    parser.add_argument('--pathFusedDataset', help='Path for valence and arousal dataset', required=True)
    args = parser.parse_args()

    annFiles = getFilesInPath(os.path.join(args.pathBase,'images'))
    indexesImages = fillDictFromDataset(args.pathWild)
    imageFiles = args.pathWild.split(os.path.sep)[:-3]
    imageFiles.append('cropped_aligned')
    imageFiles = os.path.sep.join(imageFiles)
    annotationsValues = np.zeros((len(annFiles),2))
    print("Loading VA Values for AffWild")
    for idx, a in enumerate(annFiles):
        printProgressBar(iteration=idx, total=len(annFiles), prefix='Loading VA Files', suffix='Complete', length=50)
        fileName = a.split(os.path.sep)[-1].split('.')[0]
        annFolder = os.path.sep.join([args.pathBase,'annotations'])
        annotationsValues[idx][0] = np.load(os.path.join(annFolder,f'{fileName}_val.npy'))
        annotationsValues[idx][1] = np.load(os.path.join(annFolder,f'{fileName}_aro.npy'))

    densidades = []
    quadsLims = [[(0,1),(0,1)],[(-1,0),(0,1)],[(-1,0),(-1,0)],[(0,1),(-1,0)]]
    for i in quadsLims:
        densidades.append(calcular_densidade_pontos(i[0], i[1], annotationsValues, mostrar_grafico=False))        

    maxDensidade = max(densidades)
    # Load AffWild dataset
    print("Loading AffWild dataset")
    vaValues, idxs, _ = loadAffWildDataset(args.pathWild,indexesImages)
    for idx, v in enumerate(vaValues):
        cQ = determinar_quadrante(v[0],v[1])
        if (densidades[cQ] < maxDensidade):
            folderFile = idxs[idx][-1][:-4]
            shutil.copyfile(os.path.join(imageFiles,folderFile,"{:05d}.jpg".format(idxs[idx][0])), os.path.join(args.pathFusedDataset,'images',f"{idxs[idx][0]}_affwild.jpg"))
            np.save(os.path.join(args.pathFusedDataset,'annotations',f"{idxs[idx][0]}_affwild_val.npy"),v[0])
            np.save(os.path.join(args.pathFusedDataset,'annotations',f"{idxs[idx][0]}_affwild_aro.npy"),v[1])
            annotationsValues = np.vstack((annotationsValues, v))
            densidades[cQ] = calcular_densidade_pontos(quadsLims[cQ][0], quadsLims[cQ][1], annotationsValues, mostrar_grafico=False)
    print('vaValues',vaValues)

if __name__ == '__main__':
    main()