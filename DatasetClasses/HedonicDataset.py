import os
import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from tqdm import tqdm  # Importando a biblioteca de barra de progresso

class VideoFramesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str ou Path): Diretório raiz do dataset que contém a pasta 'videos/' 
                                    e os arquivos JSON dos experimentos.
            transform (callable, optional): Transformações para aplicar nas imagens.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []  # Armazenará tuplas (caminho_da_imagem, (chave, valor))

        self._discover_and_load()

    def _discover_and_load(self):
        # Busca automaticamente todos os arquivos .json a partir da pasta raiz
        json_paths = list(self.root_dir.rglob("*.json"))
        all_annotations = []
        
        # 1. Primeiro, lemos todos os arquivos JSON e juntamos todos os itens em uma única lista
        for json_path in json_paths:
            with open(json_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_annotations.extend(data)
                except json.JSONDecodeError:
                    print(f"Aviso: Não foi possível ler o JSON {json_path}")
                    continue

        # 2. Iteramos sobre as anotações usando o tqdm para gerar a barra de progresso
        for item in tqdm(all_annotations, desc="Carregando frames e anotações", unit="item"):
            # Extrair a tupla (chave, valor) do label
            annotations = item.get("annotations", {})
            characteristics = annotations.get("characteristics", {})
            questions = annotations.get("questions", {})

            label_tuple = None
            
            # Pega o primeiro par chave-valor encontrado
            if characteristics:
                key = list(characteristics.keys())[0]
                val = characteristics[key]
                label_tuple = (key, val)
            elif questions:
                key = list(questions.keys())[0]
                val = questions[key]
                label_tuple = (key, val)
            
            # Se não houver anotação, pula esse item
            if label_tuple is None:
                continue

            # Fazer o parse do caminho para encontrar a pasta dos frames
            video_path = item.get("video_path", "")
            parts = Path(video_path).parts
            
            if len(parts) >= 4:
                # parts[0] = 'videos'
                exp_folder = parts[1]
                participant_folder = parts[2]
                # Remove a extensão (.webm, .mp4, etc) para pegar o nome da pasta do aspecto
                aspect_folder = Path(parts[3]).stem 

                # Monta o caminho real onde os frames estão armazenados
                frames_dir = os.path.join(self.root_dir, "videos", exp_folder, "cropped_aligned", participant_folder, aspect_folder)

                # Mapear todas as imagens dentro dessa pasta
                if os.path.exists(frames_dir) and os.path.isdir(frames_dir):
                    for img_name in os.listdir(frames_dir):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(frames_dir, img_name)
                            self.samples.append((str(img_path), label_tuple))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_tuple = self.samples[idx]
        
        # Abre a imagem e garante que está em RGB
        image = Image.open(img_path).convert("RGB")

        # Aplica as transformações (ex: Resize, ToTensor)
        if self.transform:
            image = self.transform(image)

        # CORREÇÃO: Converter a chave e o valor para string.
        # Isso garante consistência no batch e impede o PyTorch de tentar criar Tensors com textos.
        safe_label_tuple = (str(label_tuple[0]), str(label_tuple[1]))

        # Retorna a imagem e a tupla (chave, valor)
        return image, safe_label_tuple, img_path