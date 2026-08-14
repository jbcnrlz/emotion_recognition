import os, re, torch, pandas as pd, numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from scipy.signal import savgol_filter

class MIntPainFrameDataset(Dataset):
    def __init__(self, root_dir, modality='RGB', transform=None):
        """
        Inicializa o Dataset PyTorch focado em carregar frames individuais.
        
        Args:
            root_dir (str): Caminho para a raiz do dataset (ex: './MIntPain')
            modality (str): Qual modalidade carregar ('RGB', 'D' ou 'T').
            transform (callable, optional): Transformações do torchvision para aplicar nas imagens.
        """
        self.root_dir = Path(root_dir)
        self.modality = modality
        self.transform = transform
        self.samples = []
        
        # Extensões válidas
        self.valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        
        if not self.root_dir.exists():
            raise FileNotFoundError(f"O diretório '{self.root_dir}' não foi encontrado.")
            
        self._index_dataset()

    def _index_dataset(self):
        """Varre a estrutura e cria um índice plano (1 item = 1 frame)."""
        pattern = re.compile(r'(Sub\d+)_(Trial\d+)_(Sweep\d+)_Label(\d+)')

        for subject_dir in self.root_dir.iterdir():
            if not subject_dir.is_dir(): continue
            
            for trial_dir in subject_dir.iterdir():
                if not trial_dir.is_dir(): continue
                
                for sweep_dir in trial_dir.iterdir():
                    if not sweep_dir.is_dir(): continue
                    
                    match = pattern.search(sweep_dir.name)
                    if match:
                        sub_id, trial_id, sweep_id, label_str = match.groups()
                        
                        mod_path = sweep_dir / self.modality
                        if mod_path.exists() and mod_path.is_dir():
                            # Coleta todos os frames válidos desta pasta
                            frames = sorted([
                                f for f in mod_path.iterdir() 
                                if f.is_file() and f.suffix.lower() in self.valid_extensions
                            ])
                            
                            # Para CADA FRAME, cria uma entrada individual no dataset
                            for frame_path in frames:
                                self.samples.append({
                                    'subject_id': sub_id,
                                    'trial_id': trial_id,
                                    'sweep_id': sweep_id,
                                    'frame_name': frame_path.name, # Identificador do frame
                                    'pain_label': int(label_str),
                                    'frame_path': str(frame_path)
                                })

    def __len__(self):
        # Agora o len() retorna o total de IMAGENS, não de vídeos.
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retorna o tensor da imagem individual e seus respectivos metadados.
        """
        sample = self.samples[idx]
        
        # Carrega a imagem
        img = Image.open(sample['frame_path']).convert('RGB')
        
        # Aplica os transforms (Resize, ToTensor, Normalize, etc)
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
            
        # Rótulo em formato PyTorch
        label = torch.tensor(sample['pain_label'], dtype=torch.long)
            
        return {
            'image': img,        # Shape: [C, H, W]
            'label': label,      # Shape: []
            'metadata': sample   # Útil para salvar as features com o nome correto depois
        }
        
    def get_dataframe(self):
        """Exporta o índice para Pandas DataFrame."""
        return pd.DataFrame(self.samples)

class MIntPAINProtocolDataset(Dataset):
    def __init__(self, features_csv, protocol_txt, window_size=30, sg_window=11, sg_poly=3):
        """
        Args:
            features_csv (str): Caminho para o CSV unificado com as features de todos os vídeos.
            protocol_txt (str): Caminho para o TXT de protocolo que dita a divisão (train ou val).
            window_size (int): Tamanho da janela temporal causal (T).
            sg_window (int): Janela do filtro Savitzky-Golay (deve ser ímpar).
            sg_poly (int): Ordem do polinômio do filtro.
        """
        self.window_size = window_size
        self.emotions = [
            'happy', 'contempt', 'surprised', 'angry', 
            'disgusted', 'fearful', 'sad', 'neutral'
        ]
        
        # 1. Leitura do Protocolo de Divisão (Lidando com espaços no path)
        protocol_data = []
        with open(protocol_txt, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # rsplit(' ', 2) divide a linha começando da direita, estritamente 2 vezes.
                # Isso isola o video_id e o label, mantendo o file_path completo no índice 0
                parts = line.rsplit(' ', 2)
                
                if len(parts) == 3:
                    protocol_data.append(parts)
        
        # Converte a lista processada em um DataFrame do Pandas
        prot_df = pd.DataFrame(protocol_data, columns=['file_path', 'label', 'video_id'])
        
        # Garante a tipagem correta das colunas numéricas
        prot_df['label'] = prot_df['label'].astype(int)
        prot_df['video_id'] = prot_df['video_id'].astype(int)
        
        # 2. Leitura do Repositório Central de Features
        feat_df = pd.read_csv(features_csv)
        
        # 3. Isolamento da Chave de Merge (Nome do Arquivo)
        # Ignora as diferenças de pastas base entre Windows, Linux e partições /mnt/
        prot_df['filename'] = prot_df['file_path'].apply(lambda x: os.path.basename(x.replace('\\', '/')))
        feat_df['filename'] = feat_df['file'].apply(lambda x: os.path.basename(x.replace('\\', '/')))
        
        # 4. Cruzamento de Dados (Filtragem pelo Protocolo)
        # O 'inner' garante que só sobram no DataFrame as imagens listadas no TXT carregado
        df = pd.merge(feat_df, prot_df, on='filename', how='inner')
        
        self.windows_x = []
        self.windows_y = []
        
        # 5. Processamento Temporal por Vídeo
        grouped = df.groupby('video_id')
        
        for video_id, group_df in grouped:
            # Ordenação alfabética do nome do arquivo garante a linha do tempo (RGB-HH-MM-SS-MS.jpg)
            group_df = group_df.sort_values('filename') 
            
            # Suavização para remoção do ruído (flickering) das features
            for col in self.emotions:
                w_len = min(sg_window, len(group_df))
                w_len = w_len if w_len % 2 != 0 else w_len - 1
                if w_len > sg_poly:
                    group_df[col] = savgol_filter(group_df[col].values, window_length=w_len, polyorder=sg_poly)
            
            video_features = group_df[self.emotions].values
            video_labels = group_df['label'].values
            
            num_frames = len(video_features)
            
            # 6. Fatiamento Causal (Many-to-One)
            if num_frames >= window_size:
                for i in range(num_frames - window_size + 1):
                    window_x = video_features[i : i + window_size]
                    
                    # O target é sempre a anotação correspondente ao frame mais recente da janela
                    window_y = video_labels[i + window_size - 1]
                    
                    self.windows_x.append(window_x)
                    self.windows_y.append(window_y)
        
        self.windows_x = torch.tensor(np.array(self.windows_x), dtype=torch.float32)
        self.windows_y = torch.tensor(np.array(self.windows_y), dtype=torch.float32)

    def __len__(self):
        return len(self.windows_x)

    def __getitem__(self, idx):
        return self.windows_x[idx], self.windows_y[idx]