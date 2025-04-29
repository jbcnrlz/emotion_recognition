import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
import os

def gerar_grafico_com_imagem(url_imagem, descricao):
    # Baixar a imagem da URL
    img = Image.open(url_imagem)
    imageName = url_imagem.split(os.path.sep)[-1]
    
    # Criar a figura e os eixos
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Exibir a imagem
    ax.imshow(img)
    ax.axis('off')  # Remover eixos
    
    # Adicionar descrição abaixo da imagem
    plt.figtext(0.5, 0.05, descricao, 
                ha='center', va='center', 
                fontsize=12, wrap=True,
                bbox=dict(facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join('imagesLLM',imageName),dpi=300, bbox_inches='tight')
    plt.close()

# Exemplo de uso
with open("emotions_gt.txt",'r') as fil:
    for f in fil:
        dadosImage = f.split(' - ')
        dados = os.path.join("C:\\Users\\joaoc\\AffectNet\\val_set\\images",dadosImage[1].strip())
        gerar_grafico_com_imagem(dados, dadosImage[0])