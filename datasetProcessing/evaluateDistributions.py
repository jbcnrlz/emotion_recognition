import numpy as np, argparse, torch, os, sys
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from DatasetClasses.AffectNet import AffectNet
from matplotlib.patches import Ellipse
from matplotlib.patches import Ellipse
from PIL import Image  # Para carregar imagens

def saveToCSV(preds,labels,files,pathCSV):
    emotions = ["neutral","happy","sad","surprised","fear","disgust","angry","contempt","serene","contemplative","secure","untroubled","quiet"]
    with open(pathCSV,'w') as pcsv:
        pcsv.write('%s,exp,file\n' % (','.join([emotions[f] for f in range(len(preds[0]))])))
        for idx, p in enumerate(preds):
            for fp in p:
                pcsv.write(f'{fp},')
            pcsv.write(f"{emotions[labels[idx]]},{files[idx]}\n")

def saveRankFile(probs,paths):
    for idx, p in enumerate(probs):
        splitedPath = paths[idx].split(os.path.sep)
        fileName = splitedPath[-1].split(".")[0]
        annotationsFolder = os.path.join(os.path.sep.join(splitedPath[0:-2]),'annotations')
        with open(os.path.join(annotationsFolder,f"{fileName}_prob_rank.txt"),'w') as f:
            joinedProbs = ','.join([str(x) for x in p])
            f.write(joinedProbs+'\n')


def generateTextForLLM(values,txtFile,pathImage):
    with open(txtFile,'a') as f:
        f.write(f"File path: {pathImage} ")
        f.write("Given the set of terms below describing the emotional state of the attached picture, each term being associated with a probability, generate a caption describing the emotional state. ")
        for v in values:
            f.write(f"{v}: {values[v]} ")
        f.write("\n")

def save_probability_histograms(probs, labels, classesDist, vas, face_images, output_folder='hist_probs'):
    """
    Salva histogramas das probabilidades para cada imagem, com anotação da emoção correta e a imagem da face.

    Args:
        probs (numpy.ndarray): Array de probabilidades com shape (n_imagens, n_emocoes).
        labels (numpy.ndarray): Array de rótulos corretos com shape (n_imagens,).
        classesDist (numpy.ndarray): Array com as médias e variâncias das distribuições.
        vas (numpy.ndarray): Array de valores de Valence e Arousal para cada imagem.
        face_images (list): Lista de caminhos ou arrays das imagens das faces.
        output_folder (str): Pasta onde os histogramas serão salvos.
    """
    # Verifica se a pasta existe, caso contrário, cria a pasta
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Nomes das emoções (ajuste conforme necessário)
    emotions = ["neutral", "happy", "sad", "surprised", "fear", "disgust", "angry", "contempt","serene","contemplative","secure","untroubled","quiet"]
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'cyan', 'magenta', 'brown', 'pink', 'gray', 'olive', 'lime']

    # Itera sobre as probabilidades de cada imagem
    for i, (prob, label, va, face_image) in enumerate(zip(probs, labels, vas, face_images)):
        # Cria uma nova figura e subplots a cada iteração
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # Subgráfico 1: Histograma das probabilidades
        ax1.bar(emotions, prob, color='blue')
        ax1.set_xlabel('Emotions')
        ax1.set_ylabel('Probability')
        ax1.set_title(f'Histogram of Probabilities - Image {i}')
        ax1.tick_params(axis='x', rotation=45)  # Rotaciona os rótulos do eixo x

        # Adiciona o texto da emoção correta
        true_emotion = emotions[label]
        ax1.text(
            x=0.5, y=0.95,  # Posição do texto (normalizada entre 0 e 1)
            s=f'Anotated Emotion: {true_emotion}',
            transform=ax1.transAxes,  # Usa coordenadas do eixo do ax1
            fontsize=12,
            ha='center',  # Alinhamento horizontal
            va='top',  # Alinhamento vertical
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')  # Caixa de fundo
        )

        # Subgráfico 2: Centros das distribuições e elipses
        for idx, (emotion, dist) in enumerate(zip(emotions, classesDist)):
            # Extrair centro (mean) e covariância (cov)
            mean = [dist[0], dist[2]]  # Valence e Arousal
            cov = [[dist[1]**2, 0], [0, dist[3]**2]]  # Matriz de covariância

            # Plotar o centro da distribuição
            ax2.scatter(mean[0], mean[1], color=colors[idx], label=emotion, s=100)

            # Calcular ângulo e tamanho da elipse
            lambda_, v = np.linalg.eig(cov)  # Autovalores e autovetores
            angle = np.degrees(np.arctan2(v[1, 0], v[0, 0]))  # Ângulo de rotação
            width, height = 2 * np.sqrt(lambda_)  # Tamanho da elipse

            # Desenhar a elipse
            ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                              edgecolor=colors[idx], facecolor='none', linestyle='--', alpha=0.7)
            ax2.add_patch(ellipse)

        # Plotar o ponto de Valence e Arousal da imagem atual
        ax2.scatter(va[0], va[1], color='black', label='Emotion Point', s=100)

        # Configurações do subgráfico 2
        ax2.set_xlabel('Valence')
        ax2.set_ylabel('Arousal')
        ax2.set_title('Distribution Centers with Ellipses')
        ax2.grid(True)
        #x2.legend(loc='lower right')
        ax2.set_aspect('equal', 'box')

        # Subgráfico 3: Imagem da face
        if isinstance(face_image, str):  # Se for um caminho de arquivo
            img = Image.open(face_image)
        else:  # Se for um array NumPy (imagem já carregada)
            img = face_image

        ax3.imshow(img)
        ax3.axis('off')  # Remove os eixos
        ax3.set_title('Face Image')

        # Ajustar layout e salvar a figura
        plt.tight_layout()
        output_path = os.path.join(output_folder, f'histogram_{i}.png')
        fig.savefig(output_path)
        plt.close(fig)  # Fecha a figura para liberar memória

def main():
    parser = argparse.ArgumentParser(description='Generate Emotion Ranks')
    parser.add_argument('--pathBase', help='Path for valence and arousal dataset', required=True)
    parser.add_argument('--batchSize', type=int, help='Size of the batch', required=True)
    args = parser.parse_args()
    data_transforms = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    datasetVal = AffectNet(afectdata=os.path.join(args.pathBase,'train_set'),transform=data_transforms,typeExperiment='VA',exchangeLabel=None)
    classesDist = np.array([
        [0,0,0,0],
        [0.81,0.21,0.51,0.26], #happy
        [-0.63,0.23,-0.27,0.34], #sad
        [0.4,0.3,0.67,0.27], #surprised
        [-0.64,0.2,0.6,0.32],#fear
        [-0.6,0.2,0.35,0.41],#disgust
        [-0.51,0.2,0.59,0.33],#angry
        [-0.23,0.39,0.31,0.33],#contempt
        [0.65,0.29,-0.33,0.36],#leisurely-protected-relaxed
        [0.15,0.41,-0.18,0.30],#aloof-consoled-humble-modest-nonchalant-reserved-reverent-sheltered-solemn
        [0.74,0.22,-0.13,0.32],#secure
        [0.79,0.25,-0.01,0.46],#untroubled
        [0.19,0.57,-0.4,0.21]#quiet
    ])

    emotions = {"neutral" : [],"happy" : [] ,"sad" : [],"surprised" : [],"fear" : [],"disgust":[],"angry":[],"contempt": [], "serene" : [], "contemplative" : [], "secure" : [], "untroubled" : [], "quiet" : []}
    idx = -1
    covm = []
    means = []
    for k in emotions:
        idx += 1
        emotions[k] = [[classesDist[idx][0],classesDist[idx][2]], [[classesDist[idx][1]**2,0],[0,classesDist[idx][3]**2]]]
        covm.append([[classesDist[idx][1]**2,0],[0,classesDist[idx][3]**2]])
        means.append([classesDist[idx][0],classesDist[idx][2]])

    X = []
    labels = []
    for i, (emotion, (mean, cov)) in enumerate(emotions.items()):
        samples = np.random.multivariate_normal(mean, cov, 1000)
        X.append(samples)
        labels.extend([i] * len(samples))

    X = np.vstack(X)
    n_components = len(emotions)  # Número de estados emocionais
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(X)
    gmm.covariances_ = np.array(covm)
    gmm.means_ = np.array(means)

    val_loader = torch.utils.data.DataLoader(datasetVal, batch_size=args.batchSize, shuffle=False)
    outputData = None
    probs = None
    lbls = None
    vas = None    
    pts = None
    for data in val_loader:
        #_, labels, paths, vaBatch = data
        _, vaBatch, paths = data
        probs = np.concatenate((probs,gmm.predict_proba(vaBatch.numpy()))) if probs is not None else gmm.predict_proba(vaBatch.numpy())
        #lbls = np.concatenate((lbls,labels.numpy())) if lbls is not None else labels.numpy()
        outputData = np.concatenate((outputData,vaBatch.numpy())) if outputData is not None else vaBatch.numpy()        
        vas = np.concatenate((vas,vaBatch.numpy())) if vas is not None else vaBatch.numpy()
        pts = np.concatenate((pts,paths)) if pts is not None else paths

    #saveToCSV(probs,lbls,pts,'annotatedAffectNet.csv')

    #save_probability_histograms(probs, lbls, classesDist, vas, pts, output_folder='hist_probs')
    saveRankFile(probs,pts)
    '''
    for idx, p in enumerate(probs):
        valuesGenerateText = {}
        for idx2, v in enumerate(p):
            valuesGenerateText[list(emotions.keys())[idx2]] = v
        generateTextForLLM(valuesGenerateText,'llm.txt',pts[idx])
    '''
    '''
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'cyan', 'magenta', 'brown', 'pink']
    emotions = list(emotions.keys())
    # 📊 Visualizando os clusters e o novo ponto
    added_labels = {}  # Dicionário para rastrear rótulos já adicionados
    fig, ax = plt.subplots()
    for idx, l in enumerate(outputData):
        emotion_idx = (-probs[idx]).argsort()[0]
        emotion_label = emotions[emotion_idx]
        plot_two_color_point(ax, outputData[idx, 0], outputData[idx, 1], colors[emotion_idx], colors[lbls[idx]], size=2)
    
        if emotion_label not in added_labels:
            plt.scatter(outputData[idx, 0], outputData[idx, 1], c=colors[emotion_idx], alpha=0.5, label=emotion_label)
            added_labels[emotion_label] = True
        else:
            plt.scatter(outputData[idx, 0], outputData[idx, 1], c=colors[emotion_idx], alpha=0.5)
    
    plt.scatter(classesDist[:,0],classesDist[:,2],color='purple',marker='x',s=200,label="Distribution center")
    #plt.scatter(outputData[:, 0], outputData[:, 1], color='red', marker='x', s=200, label="New point")
    ax.set_aspect('equal', 'box')
    plt.xlabel("Valence")
    plt.ylabel("Arousal")
    plt.title("Clusters de Emoções via GMM")
    plt.legend()
    plt.show()
    '''
# Função para criar um ponto com duas cores
def plot_two_color_point(ax, x, y, color1, color2, size=100):
    # Cria um semicírculo para a primeira cor
    theta1 = np.linspace(0, np.pi, 100)
    x1 = x + np.cos(theta1) * (size / 200)
    y1 = y + np.sin(theta1) * (size / 200)
    ax.fill(x1, y1, color=color1, edgecolor='none')

    # Cria um semicírculo para a segunda cor
    theta2 = np.linspace(np.pi, 2 * np.pi, 100)
    x2 = x + np.cos(theta2) * (size / 200)
    y2 = y + np.sin(theta2) * (size / 200)
    ax.fill(x2, y2, color=color2, edgecolor='none')

if __name__ == '__main__':
    main()