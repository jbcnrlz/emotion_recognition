import argparse, csv
import base64
import os

def ler_arquivo_tsv(caminho_arquivo):
    """
    Função para ler um arquivo TSV e retornar seu conteúdo como uma lista de linhas.
    
    Args:
        caminho_arquivo (str): Caminho para o arquivo TSV.
    
    Returns:
        list: Lista contendo as linhas do arquivo TSV, onde cada linha é uma lista de valores.
    """
    pathDirMSCeleb = os.path.dirname(caminho_arquivo)
    try:
        with open(caminho_arquivo, 'r', encoding='utf-8') as arquivo:
            reader = csv.reader(arquivo, delimiter='\t')
            i = 0
            for row in reader:
                MID, imgSearchRank, faceID, data = row[0], row[1], row[4], base64.b64decode(row[-1])

                saveDir = os.path.join(pathDirMSCeleb,MID)
                savePath = os.path.join(saveDir, "{}-{}.jpg".format(imgSearchRank, faceID))

                # assert(magic.from_buffer(data) == 'JPEG image data, JFIF standard 1.01')

                os.makedirs(saveDir, exist_ok=True)
                with open(savePath, 'wb') as f:
                    f.write(data)

                i += 1

                if i % 1000 == 0:
                    print("Extracted {} images.".format(i))
                        
    except FileNotFoundError:
        print(f"Erro: Arquivo '{caminho_arquivo}' não encontrado.")        
    except Exception as e:
        print(f"Erro ao ler o arquivo: {e}")        
    

def train():
    parser = argparse.ArgumentParser(description='Extract features from resnet emotion')
    parser.add_argument('--tsvFile', help='Weights', required=True)
    args = parser.parse_args()

    ler_arquivo_tsv(args.tsvFile)
    

if __name__ == '__main__':
    train()