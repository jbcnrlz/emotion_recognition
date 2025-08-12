import numpy as np, argparse, os, sys, shutil
import matplotlib.pyplot as plt
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getFilesInPath, printProgressBar, getDirectoriesInPath
import matplotlib.pyplot as plt

def calcular_densidade_pontos(limite_x, limite_y, pontos_total, mostrar_grafico=False):
    
    # Extrai os limites
    x_min, x_max = limite_x
    y_min, y_max = limite_y
    
    # Verifica se os limites são válidos
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("Os limites mínimos devem ser menores que os máximos")
    
    # Calcula a área total
    area_total = (x_max - x_min) * (y_max - y_min)
    # Calcula a densidade (pontos por unidade de área)    
    
    # Gera pontos aleatórios dentro dos limites
    maskX = (pontos_total[:,0] > x_min) & (pontos_total[:,0] < x_max)
    maskY = (pontos_total[:,1] > y_min) & (pontos_total[:,1] < y_max)
    finalMask = maskX & maskY
    pontos_x = pontos_total[finalMask,0]
    pontos_y = pontos_total[finalMask,1]

    densidade = pontos_total[finalMask].shape[0] / area_total
    
    # Se quiser mostrar o gráfico
    if mostrar_grafico:
        # Calcula densidade para cada quadrante
        mid_x = (x_min + x_max) / 2
        mid_y = (y_min + y_max) / 2
        
        # Quadrantes: Q1 (top-right), Q2 (top-left), Q3 (bottom-left), Q4 (bottom-right)
        quadrant_masks = [
            (pontos_total[:,0] > mid_x) & (pontos_total[:,1] > mid_y),  # Q1
            (pontos_total[:,0] <= mid_x) & (pontos_total[:,1] > mid_y),  # Q2
            (pontos_total[:,0] <= mid_x) & (pontos_total[:,1] <= mid_y), # Q3
            (pontos_total[:,0] > mid_x) & (pontos_total[:,1] <= mid_y)   # Q4
        ]
        
        quadrant_areas = [
            (x_max - mid_x) * (y_max - mid_y),  # Q1
            (mid_x - x_min) * (y_max - mid_y),   # Q2
            (mid_x - x_min) * (mid_y - y_min),   # Q3
            (x_max - mid_x) * (mid_y - y_min)    # Q4
        ]
        
        quadrant_densities = []
        for q_mask, q_area in zip(quadrant_masks, quadrant_areas):
            q_points = pontos_total[q_mask].shape[0]
            quadrant_densities.append(q_points / q_area if q_area > 0 else 0)

        
        plt.figure(figsize=(10, 8))
        plt.scatter(pontos_x, pontos_y, s=10, alpha=0.5)
        
        # Adiciona linhas de divisão dos quadrantes
        plt.axvline(x=mid_x, color='red', linestyle='--', linewidth=1)
        plt.axhline(y=mid_y, color='red', linestyle='--', linewidth=1)
        
        # Adiciona rótulos e densidades para cada quadrante
        quadrant_positions = [
            ((mid_x + x_max)/2, (mid_y + y_max)/2),  # Q1
            ((x_min + mid_x)/2, (mid_y + y_max)/2),   # Q2
            ((x_min + mid_x)/2, (y_min + mid_y)/2),   # Q3
            ((mid_x + x_max)/2, (y_min + mid_y)/2)    # Q4
        ]
        
        quadrant_names = ['Q1 (Valence +, Arousal +)', 
                         'Q2 (Valence -, Arousal +)',
                         'Q3 (Valence -, Arousal -)', 
                         'Q4 (Valence +, Arousal -)']
        
        for pos, name, density in zip(quadrant_positions, quadrant_names, quadrant_densities):
            plt.text(pos[0], pos[1], f'{name}\nDensity: {density:.4f}', 
                    ha='center', va='center', 
                    fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xlabel('Valence')
        plt.ylabel('Arousal')
        plt.title('VA Space Distribution with Densities by Quadrant')
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout()
        plt.show()
    
    
    return densidade

def main():
    parser = argparse.ArgumentParser(description='Fuse distributions')
    parser.add_argument('--pathBase', help='Path for valence and arousal dataset', required=True)
    args = parser.parse_args()

    annFiles = getFilesInPath(os.path.join(args.pathBase,'images'))
    annotationsValues = np.zeros((len(annFiles),2))
    print("Loading VA Values for Dataset")
    for idx, a in enumerate(annFiles):
        printProgressBar(iteration=idx, total=len(annFiles), prefix='Loading VA Files', suffix='Complete', length=50)
        fileName = a.split(os.path.sep)[-1].split('.')[0]
        annFolder = os.path.sep.join([args.pathBase,'annotations'])
        annotationsValues[idx][0] = np.load(os.path.join(annFolder,f'{fileName}_val.npy'))
        annotationsValues[idx][1] = np.load(os.path.join(annFolder,f'{fileName}_aro.npy'))

    calcular_densidade_pontos([-1,1], [-1,1], annotationsValues, mostrar_grafico=True)


if __name__ == '__main__':
    main()