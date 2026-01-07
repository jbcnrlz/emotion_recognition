import torch.utils.data as data, os, torch, numpy as np, sys, pandas as pd, random, glob
from tqdm import tqdm
from PIL import Image as im
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getDirectoriesInPath, getFilesInPath, printProgressBar
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import threading
#from generateDenseOpticalFlow import runDenseOpFlow

class AffectNet(data.Dataset):    
    """Versão otimizada mantendo compatibilidade total com a original."""
    
    def __init__(self, afectdata, typeExperiment="VA", transform=None, 
                 termsQuantity=151, exchangeLabel=None, loadLastLabel=True,
                 use_cache=True):
        """
        Args compatíveis com a versão original.
        use_cache: Novo parâmetro opcional para habilitar cache
        """
        self.exchangeLabel = exchangeLabel
        self.terms = None if typeExperiment != 'TERMS' else self._load_terms_file(termsQuantity)
        self.transform = transform
        self.label = []
        self.filesPath = []
        self.seconLabel = [] if typeExperiment == 'BOTH' else None
        self.typeExperiment = typeExperiment
        self.loadLastLabel = loadLastLabel
        self.use_cache = use_cache
        
        # Cache para labels
        self._label_cache = {}
        
        # Carregar imagens
        images_dir = os.path.join(afectdata, 'images')
        annotations_dir = os.path.join(afectdata, 'annotations')
        
        # Listar imagens de forma mais eficiente
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        faces = []
        for ext in image_extensions:
            faces.extend(glob.glob(os.path.join(images_dir, f'*{ext}')))
            faces.extend(glob.glob(os.path.join(images_dir, f'*{ext.upper()}')))
        
        print(f"Loading {len(faces)} face images")
        
        # Carregar labels com barra de progresso
        for f in tqdm(faces, desc='Loading dataset'):
            image_number = os.path.basename(f).rsplit('.', 1)[0]
            
            # Determinar tipo de carregamento baseado no experimento
            label_data = self._load_label_data(annotations_dir, image_number, f)
            
            if label_data is None:
                continue
                
            self.label.append(label_data)
            self.filesPath.append(f)
        
        print(f"Loaded {len(self.filesPath)} samples")
    
    def _load_label_data(self, annotations_dir, image_number, image_path):
        """Carrega dados do label de forma otimizada."""
        type_exp = self.typeExperiment
        
        # VA
        if "VA" in type_exp and 'VAD' not in type_exp and 'PROBS_' not in type_exp and 'UNIVERSAL_' not in type_exp:
            val, aro = self._load_va(annotations_dir, image_number)
            if val is None or aro is None:
                return None
                
            label = [val, aro]
            if '_EXP' in type_exp:
                exp = self._load_exp(annotations_dir, image_number)
                label.append(exp)
            return label
        
        # VAD_ADJUSTED
        elif "VAD_ADJUSTED" in type_exp and 'PROBS_' not in type_exp and 'UNIVERSAL_' not in type_exp:
            val, aro, dom = self._load_vad_adjusted(annotations_dir, image_number)
            if val is None or aro is None or dom is None:
                return None
                
            label = [val, aro, dom]
            if '_EXP' in type_exp:
                exp = self._load_exp(annotations_dir, image_number)
                label.append(exp)
            return label
        
        # VAD padrão
        elif "VAD" in type_exp and 'PROBS_' not in type_exp and 'UNIVERSAL_' not in type_exp and 'ORIGINAL_' not in type_exp:
            val, aro, dom = self._load_vad(annotations_dir, image_number)
            if val is None or aro is None or dom is None:
                return None
                
            label = [val, aro, dom]
            if '_EXP' in type_exp:
                exp = self._load_exp(annotations_dir, image_number)
                label.append(exp)
            return label
        
        # EXP
        elif type_exp == "EXP":
            exp_label = self._load_exp(annotations_dir, image_number)
            if exp_label == 7 and not self.loadLastLabel:
                return None
            return int(exp_label)
        
        # BOTH
        elif type_exp == "BOTH":
            exp_label = self._load_exp(annotations_dir, image_number)
            if exp_label == 7 and not self.loadLastLabel:
                return None
                
            val, aro = self._load_va(annotations_dir, image_number)
            if val is None or aro is None:
                return None
            
            # Armazenar secundário separadamente
            if self.seconLabel is not None:
                self.seconLabel.append([val, aro])
            
            return int(exp_label)
        
        # RANK
        elif type_exp == 'RANK':
            rank_path = os.path.join(annotations_dir, f'{image_number}_rank.txt')
            return rank_path if os.path.exists(rank_path) else None
        
        # RANDOM
        elif type_exp == 'RANDOM':
            return np.random.randint(0, 8)
        
        # PROBS
        elif type_exp == 'PROBS':
            prob_path = os.path.join(annotations_dir, f'{image_number}_prob_rank.txt')
            return prob_path if os.path.exists(prob_path) else None
        
        # PROBS_VAD
        elif 'PROBS_VAD' in type_exp:
            return self._load_probs_vad(annotations_dir, image_number)
        
        # PROBS_VA
        elif 'PROBS_VA' in type_exp:
            return self._load_probs_va(annotations_dir, image_number)
        
        # UNIVERSAL
        elif type_exp == 'UNIVERSAL':
            uni_path = os.path.join(annotations_dir, f'{image_number}_prob_rank_universal.txt')
            return uni_path if os.path.exists(uni_path) else None
        
        # UNIVERSAL_VAD_ADJUSTED
        elif 'UNIVERSAL_VAD_ADJUSTED' in type_exp:
            return self._load_universal_vad_adjusted(annotations_dir, image_number)
        
        # UNIVERSAL_VAD
        elif 'UNIVERSAL_VAD' in type_exp:
            return self._load_universal_vad(annotations_dir, image_number)
        
        # ORIGINAL_VAD - CORRIGIDO para incluir _EXP
        elif 'ORIGINAL_VAD' in type_exp:
            return self._load_original_vad(annotations_dir, image_number)
        
        # TERMS (default)
        else:
            term_path = os.path.join(annotations_dir, f'{image_number}_terms.txt')
            term = self._load_term_data(term_path)
            if term and self.terms is not None:
                indices = np.where(self.terms == term)[0]
                return indices[0] if len(indices) > 0 else None
            return None
    
    def _load_npy_cached(self, filepath):
        """Carrega arquivo .npy com cache."""
        if not os.path.exists(filepath):
            return None
        
        if self.use_cache and filepath in self._label_cache:
            return self._label_cache[filepath]
        
        try:
            data = np.load(filepath)
            if self.use_cache:
                self._label_cache[filepath] = data
            return data
        except:
            return None
    
    def _load_va(self, annotations_dir, image_number):
        """Carrega valores VA."""
        # Tentar diferentes padrões de nome de arquivo
        val_paths = [
            os.path.join(annotations_dir, f'{image_number}_val.npy'),
            os.path.join(annotations_dir, f'{int(image_number)}_val.npy') if image_number.isdigit() else None
        ]
        
        aro_paths = [
            os.path.join(annotations_dir, f'{image_number}_aro.npy'),
            os.path.join(annotations_dir, f'{int(image_number)}_aro.npy') if image_number.isdigit() else None
        ]
        
        val = None
        for path in val_paths:
            if path and os.path.exists(path):
                val = self._load_npy_cached(path)
                if val is not None:
                    break
        
        aro = None
        for path in aro_paths:
            if path and os.path.exists(path):
                aro = self._load_npy_cached(path)
                if aro is not None:
                    break
        
        return val, aro
    
    def _load_vad(self, annotations_dir, image_number):
        """Carrega valores VAD."""
        val, aro = self._load_va(annotations_dir, image_number)
        
        dom_paths = [
            os.path.join(annotations_dir, f'{image_number}_dom.npy'),
            os.path.join(annotations_dir, f'{int(image_number)}_dom.npy') if image_number.isdigit() else None
        ]
        
        dom = None
        for path in dom_paths:
            if path and os.path.exists(path):
                dom = self._load_npy_cached(path)
                if dom is not None:
                    break
        
        return val, aro, dom
    
    def _load_vad_adjusted(self, annotations_dir, image_number):
        """Carrega valores VAD ajustados."""
        val_paths = [
            os.path.join(annotations_dir, f'{image_number}_adjusted_val.npy'),
            os.path.join(annotations_dir, f'{int(image_number)}_adjusted_val.npy') if image_number.isdigit() else None
        ]
        
        aro_paths = [
            os.path.join(annotations_dir, f'{image_number}_adjusted_aro.npy'),
            os.path.join(annotations_dir, f'{int(image_number)}_adjusted_aro.npy') if image_number.isdigit() else None
        ]
        
        val = None
        for path in val_paths:
            if path and os.path.exists(path):
                val = self._load_npy_cached(path)
                if val is not None:
                    break
        
        aro = None
        for path in aro_paths:
            if path and os.path.exists(path):
                aro = self._load_npy_cached(path)
                if aro is not None:
                    break
        
        dom_paths = [
            os.path.join(annotations_dir, f'{image_number}_dom.npy'),
            os.path.join(annotations_dir, f'{int(image_number)}_dom.npy') if image_number.isdigit() else None
        ]
        
        dom = None
        for path in dom_paths:
            if path and os.path.exists(path):
                dom = self._load_npy_cached(path)
                if dom is not None:
                    break
        
        return val, aro, dom
    
    def _load_exp(self, annotations_dir, image_number):
        """Carrega expressão."""
        exp_paths = [
            os.path.join(annotations_dir, f'{image_number}_exp.npy'),
            os.path.join(annotations_dir, f'{int(image_number)}_exp.npy') if image_number.isdigit() else None
        ]
        
        for path in exp_paths:
            if path and os.path.exists(path):
                exp = self._load_npy_cached(path)
                if exp is not None:
                    return int(exp)
        
        return 255
    
    def _load_probs_vad(self, annotations_dir, image_number):
        """Carrega PROBS_VAD."""
        prob_paths = [
            os.path.join(annotations_dir, f'{image_number}_prob_rank.txt'),
            os.path.join(annotations_dir, f'{int(image_number)}_prob_rank.txt') if image_number.isdigit() else None
        ]
        
        prob_path = None
        for path in prob_paths:
            if path and os.path.exists(path):
                prob_path = path
                break
        
        if not prob_path:
            return None
        
        val, aro, dom = self._load_vad(annotations_dir, image_number)
        if val is None or aro is None or dom is None:
            return None
        
        label = [prob_path, float(val), float(aro), float(dom)]
        
        if '_EXP' in self.typeExperiment:
            exp = self._load_exp(annotations_dir, image_number)
            label.append(exp)
        
        return label
    
    def _load_probs_va(self, annotations_dir, image_number):
        """Carrega PROBS_VA."""
        prob_paths = [
            os.path.join(annotations_dir, f'{image_number}_prob_rank.txt'),
            os.path.join(annotations_dir, f'{int(image_number)}_prob_rank.txt') if image_number.isdigit() else None
        ]
        
        prob_path = None
        for path in prob_paths:
            if path and os.path.exists(path):
                prob_path = path
                break
        
        if not prob_path:
            return None
        
        val, aro = self._load_va(annotations_dir, image_number)
        if val is None or aro is None:
            return None
        
        label = [prob_path, float(val), float(aro)]
        
        if '_EXP' in self.typeExperiment:
            exp = self._load_exp(annotations_dir, image_number)
            label.append(exp)
        
        return label
    
    def _load_universal_vad_adjusted(self, annotations_dir, image_number):
        """Carrega UNIVERSAL_VAD_ADJUSTED."""
        uni_paths = [
            os.path.join(annotations_dir, f'{image_number}_prob_rank_universal.txt'),
            os.path.join(annotations_dir, f'{int(image_number)}_prob_rank_universal.txt') if image_number.isdigit() else None
        ]
        
        uni_path = None
        for path in uni_paths:
            if path and os.path.exists(path):
                uni_path = path
                break
        
        if not uni_path:
            return None
        
        val, aro, dom = self._load_vad_adjusted(annotations_dir, image_number)
        if val is None or aro is None or dom is None:
            return None
        
        label = [uni_path, float(val), float(aro), float(dom)]
        
        if '_EXP' in self.typeExperiment:
            exp = self._load_exp(annotations_dir, image_number)
            label.append(exp)
        
        return label
    
    def _load_universal_vad(self, annotations_dir, image_number):
        """Carrega UNIVERSAL_VAD."""
        uni_paths = [
            os.path.join(annotations_dir, f'{image_number}_prob_rank_universal.txt'),
            os.path.join(annotations_dir, f'{int(image_number)}_prob_rank_universal.txt') if image_number.isdigit() else None
        ]
        
        uni_path = None
        for path in uni_paths:
            if path and os.path.exists(path):
                uni_path = path
                break
        
        if not uni_path:
            return None
        
        # Determinar prefixo baseado no tipo de experimento
        prefix = '_adjusted' if '_ADJUSTED' in self.typeExperiment else ''
        
        # Carregar valores
        val_path = os.path.join(annotations_dir, f'{image_number}{prefix}_val.npy')
        aro_path = os.path.join(annotations_dir, f'{image_number}{prefix}_aro.npy')
        dom_path = os.path.join(annotations_dir, f'{image_number}_dom.npy')
        
        val = self._load_npy_cached(val_path)
        aro = self._load_npy_cached(aro_path)
        dom = self._load_npy_cached(dom_path)
        
        if val is None or aro is None or dom is None:
            return None
        
        label = [uni_path, float(val), float(aro), float(dom)]
        
        if '_EXP' in self.typeExperiment:
            exp = self._load_exp(annotations_dir, image_number)
            label.append(exp)
        
        return label
    
    def _load_original_vad(self, annotations_dir, image_number):
        """Carrega ORIGINAL_VAD - CORRIGIDO para funcionar como a versão original."""
        orig_paths = [
            os.path.join(annotations_dir, f'{image_number}_prob_rank_original.txt'),
            os.path.join(annotations_dir, f'{int(image_number)}_prob_rank_original.txt') if image_number.isdigit() else None
        ]
        
        orig_path = None
        for path in orig_paths:
            if path and os.path.exists(path):
                orig_path = path
                break
        
        if not orig_path:
            return None
        
        # Determinar prefixo como na versão original
        # Na versão original: prefix = '' if typeExperiment not in '_ADJUSTED' else '_adjusted'
        # Isso está errado, deveria ser: '' if '_ADJUSTED' not in typeExperiment else '_adjusted'
        prefix = '_adjusted' if '_ADJUSTED' in self.typeExperiment else ''
        
        # Tentar carregar valores - seguir o mesmo padrão da versão original
        try:
            # Primeiro tentar com int(image_number)
            if image_number.isdigit():
                val_path = os.path.join(annotations_dir, f'{int(image_number)}{prefix}_val.npy')
                aro_path = os.path.join(annotations_dir, f'{int(image_number)}{prefix}_aro.npy')
                dom_path = os.path.join(annotations_dir, f'{int(image_number)}_dom.npy')
            else:
                val_path = os.path.join(annotations_dir, f'{image_number}{prefix}_val.npy')
                aro_path = os.path.join(annotations_dir, f'{image_number}{prefix}_aro.npy')
                dom_path = os.path.join(annotations_dir, f'{image_number}_dom.npy')
            
            val = self._load_npy_cached(val_path)
            aro = self._load_npy_cached(aro_path)
            dom = self._load_npy_cached(dom_path)
            
            # Se algum for None, tentar sem o prefixo
            if val is None or aro is None or dom is None:
                val_path = os.path.join(annotations_dir, f'{image_number}_val.npy')
                aro_path = os.path.join(annotations_dir, f'{image_number}_aro.npy')
                
                val = self._load_npy_cached(val_path)
                aro = self._load_npy_cached(aro_path)
                
                if val is None or aro is None or dom is None:
                    return None
            
            label = [orig_path, float(val), float(aro), float(dom)]
            
            # Adicionar expressão se _EXP estiver no typeExperiment
            if '_EXP' in self.typeExperiment:
                exp = self._load_exp(annotations_dir, image_number)
                label.append(exp)
            
            return label
        except:
            return None
    
    def _load_terms_file(self, termsQuantity):
        """Carrega arquivo de termos."""
        try:
            # Verificar se o arquivo existe
            filename = f'joinedWithDistance_{termsQuantity}.csv'
            if not os.path.exists(filename):
                # Tentar carregar de forma diferente
                return np.array([])
            
            df = pd.read_csv(filename)
            return np.array(df)[:, 0]
        except Exception as e:
            print(f"Error loading terms file: {e}")
            return np.array([])
    
    def _load_term_data(self, pathData):
        """Carrega dados de termo."""
        if not os.path.exists(pathData):
            return ""
        try:
            with open(pathData, 'r') as f:
                return f.readline().strip()
        except:
            return ""
    
    def _load_rank_file(self, rankPath):
        """Carrega arquivo de ranking."""
        try:
            with open(rankPath, 'r') as f:
                return list(map(int, f.readline().strip().split(',')))
        except:
            return []
    
    def _load_prob_file(self, probPath):
        """Carrega arquivo de probabilidades."""
        try:
            with open(probPath, 'r') as f:
                return list(map(float, f.readline().strip().split(',')))
        except:
            return []
    
    def __len__(self):
        return len(self.filesPath)
    
    def __getitem__(self, idx):
        """Mantém exatamente a mesma interface da versão original."""
        path = self.filesPath[idx]
        image = im.open(path)
        
        valenceLabel = None
        label_data = self.label[idx]
        
        # Processamento idêntico ao original
        if self.typeExperiment == 'TERMS':
            label = torch.from_numpy(np.array(label_data)).to(torch.float32)
        elif self.typeExperiment == 'EXP' or self.typeExperiment == 'BOTH':
            if self.exchangeLabel is not None:
                valenceLabel = torch.from_numpy(np.array(self.exchangeLabel[label_data])).to(torch.long)
            if self.typeExperiment == 'BOTH':
                valenceLabel = torch.from_numpy(
                    np.array([self.seconLabel[idx][0].astype(np.float32), 
                             self.seconLabel[idx][1].astype(np.float32)])).to(torch.float32)
            label = torch.from_numpy(np.array(label_data).astype(np.uint8)).to(torch.long)
        elif self.typeExperiment == "RANK":
            label = torch.from_numpy(np.array(self._load_rank_file(label_data)).astype(np.uint8)).to(torch.long)
        elif self.typeExperiment == "RANDOM":
            label = torch.from_numpy(np.array(np.random.randint(0, 8))).to(torch.long)
        elif self.typeExperiment == "PROBS":
            label = torch.from_numpy(np.array(self._load_prob_file(label_data)).astype(np.float32)).to(torch.float32)
        elif self.typeExperiment == "UNIVERSAL":
            label = torch.from_numpy(np.array(self._load_prob_file(label_data)).astype(np.float32)).to(torch.float32)
        elif 'PROBS_VAD' in self.typeExperiment:
            label = [
                torch.from_numpy(np.array(self._load_prob_file(label_data[0])).astype(np.float32)).to(torch.float32),
                torch.from_numpy(np.array([label_data[1], label_data[2], label_data[3]]))
            ]
            if '_EXP' in self.typeExperiment and len(label_data) > 4:
                label.append(torch.from_numpy(np.array(label_data[4]).astype(np.uint8)).to(torch.long))
        elif 'PROBS_VA' in self.typeExperiment:
            label = [
                torch.from_numpy(np.array(self._load_prob_file(label_data[0])).astype(np.float32)).to(torch.float32),
                torch.from_numpy(np.array([label_data[1], label_data[2]]))
            ]
            if '_EXP' in self.typeExperiment and len(label_data) > 3:
                label.append(torch.from_numpy(np.array(label_data[3]).astype(np.uint8)).to(torch.long))
        elif 'UNIVERSAL_VAD' in self.typeExperiment:
            label = [
                torch.from_numpy(np.array(self._load_prob_file(label_data[0])).astype(np.float32)).to(torch.float32),
                torch.from_numpy(np.array([label_data[1], label_data[2], label_data[3]]))
            ]
            if '_EXP' in self.typeExperiment and len(label_data) > 4:
                label.append(torch.from_numpy(np.array(label_data[4]).astype(np.uint8)).to(torch.long))
        # ADICIONADO: Caso para ORIGINAL_VAD - funciona igual ao UNIVERSAL_VAD
        elif 'ORIGINAL_VAD' in self.typeExperiment:
            label = [
                torch.from_numpy(np.array(self._load_prob_file(label_data[0])).astype(np.float32)).to(torch.float32),
                torch.from_numpy(np.array([label_data[1], label_data[2], label_data[3]]))
            ]
            if '_EXP' in self.typeExperiment and len(label_data) > 4:
                label.append(torch.from_numpy(np.array(label_data[4]).astype(np.uint8)).to(torch.long))
        elif "VAD" in self.typeExperiment:
            label = torch.from_numpy(
                np.array([label_data[0].astype(np.float32), 
                         label_data[1].astype(np.float32), 
                         label_data[2].astype(np.float32)])).to(torch.float32)
            if '_EXP' in self.typeExperiment:
                label_part2 = torch.from_numpy(np.array(label_data[-1]).astype(np.uint8)).to(torch.long)
                label = torch.cat([label, label_part2.unsqueeze(0)])
        elif "VA" in self.typeExperiment:
            label = torch.from_numpy(
                np.array([label_data[0].astype(np.float32), 
                         label_data[1].astype(np.float32)])).to(torch.float32)
            if '_EXP' in self.typeExperiment:
                label_part2 = torch.from_numpy(np.array(label_data[-1]).astype(np.uint8)).to(torch.long)
                label = torch.cat([label, label_part2.unsqueeze(0)])
        else:
            label = torch.from_numpy(
                np.array([label_data[0].astype(np.float32), 
                         label_data[1].astype(np.float32)])).to(torch.float32)
        
        if self.transform is not None:
            image = self.transform(image)

        if valenceLabel is not None:
            return image, label, self.filesPath[idx], valenceLabel
        else:
            return image, label, self.filesPath[idx]
    
    def sample(self, classes, exclude):
        """Método sample idêntico ao original."""
        returnValue = [0] * (max(classes) + 1)
        for c in classes:
            available = [idx for idx, cl in enumerate(self.label) if cl == c]
            if not available:
                continue
                
            sortedIdx = random.randint(0, len(available) - 1)
            while self.filesPath[available[sortedIdx]] in exclude:
                sortedIdx = random.randint(0, len(available) - 1)

            i, _, _ = self.__getitem__(available[sortedIdx])
            returnValue[c] = i

        return torch.stack(returnValue)
    '''
    def __init__(self, afectdata, typeExperiment="VA", transform=None, termsQuantity=151,exchangeLabel=None,loadLastLabel=True):
        self.exchangeLabel = exchangeLabel
        self.terms = None if typeExperiment != 'TERMS' else self.loadTermsFile(termsQuantity)
        self.transform = transform
        self.label = []
        self.filesPath = []
        self.seconLabel = []
        self.typeExperiment = typeExperiment
        quantitylabels = None
        faces = getFilesInPath(os.path.join(afectdata,'images'),imagesOnly=True)
        print(f"Loading {len(faces)} face images")
        for idx, f in enumerate(faces):
            printProgressBar(idx,len(faces),length=50,prefix='Loading Faces...')
            imageNumber = f.split(os.path.sep)[-1][:-4]            
            if "VA" in typeExperiment and 'VAD' not in typeExperiment and 'PROBS_' not in typeExperiment and 'UNIVERSAL_' not in typeExperiment:
                try:
                    valValue = np.load(os.path.join(afectdata,'annotations','%d_val.npy' % (int(imageNumber))))
                    aroValue = np.load(os.path.join(afectdata,'annotations','%d_aro.npy' % (int(imageNumber))))
                except:
                    valValue = np.load(os.path.join(afectdata,'annotations',f'{imageNumber}_val.npy'))
                    aroValue = np.load(os.path.join(afectdata,'annotations',f'{imageNumber}_aro.npy'))
                self.label.append([valValue,aroValue])
                if '_EXP' in typeExperiment:
                    try:
                        lbl = np.load(os.path.join(afectdata,'annotations' ,f'{imageNumber}_exp.npy'))
                        self.label[-1].append(int(lbl))
                    except:
                        self.label[-1].append(255)
            elif "VAD_ADJUSTED" in typeExperiment and 'PROBS_' not in typeExperiment and 'UNIVERSAL_' not in typeExperiment:
                try:
                    valValue = np.load(os.path.join(afectdata,'annotations','%d_adjusted_val.npy' % (int(imageNumber))))
                    aroValue = np.load(os.path.join(afectdata,'annotations','%d_adjusted_aro.npy' % (int(imageNumber))))
                    domValue = np.load(os.path.join(afectdata,'annotations','%d_dom.npy' % (int(imageNumber))))
                except:
                    valValue = np.load(os.path.join(afectdata,'annotations',f'{imageNumber}_adjusted_val.npy'))
                    aroValue = np.load(os.path.join(afectdata,'annotations',f'{imageNumber}_adjusted_aro.npy'))
                    try:
                        domValue = np.load(os.path.join(afectdata,'annotations',f'{imageNumber}_dom.npy'))
                    except:
                        continue
                self.label.append([valValue,aroValue,domValue])
                if '_EXP' in typeExperiment:
                    try:
                        lbl = np.load(os.path.join(afectdata,'annotations' ,f'{imageNumber}_exp.npy'))
                        self.label[-1].append(int(lbl))
                    except:
                        self.label[-1].append(255)
            elif "VAD" in typeExperiment and 'PROBS_' not in typeExperiment and 'UNIVERSAL_' not in typeExperiment:
                try:
                    valValue = np.load(os.path.join(afectdata,'annotations','%d_val.npy' % (int(imageNumber))))
                    aroValue = np.load(os.path.join(afectdata,'annotations','%d_aro.npy' % (int(imageNumber))))
                    domValue = np.load(os.path.join(afectdata,'annotations','%d_dom.npy' % (int(imageNumber))))
                except:
                    valValue = np.load(os.path.join(afectdata,'annotations',f'{imageNumber}_val.npy'))
                    aroValue = np.load(os.path.join(afectdata,'annotations',f'{imageNumber}_aro.npy'))
                    try:
                        domValue = np.load(os.path.join(afectdata,'annotations',f'{imageNumber}_dom.npy'))
                    except:
                        continue
                self.label.append([valValue,aroValue,domValue])
                if '_EXP' in typeExperiment:
                    try:
                        lbl = np.load(os.path.join(afectdata,'annotations' ,f'{imageNumber}_exp.npy'))
                        self.label[-1].append(int(lbl))
                    except:
                        self.label[-1].append(255)
            elif typeExperiment == "EXP":
                currLabel = np.load(os.path.join(afectdata,'annotations' ,'%d_exp.npy' % (int(imageNumber))))
                if int(currLabel) == 7 and not loadLastLabel:
                    continue
                self.label.append(int(currLabel))
            elif typeExperiment == "BOTH":
                currLabel = np.load(os.path.join(afectdata,'annotations' ,'%d_exp.npy' % (int(imageNumber))))
                if int(currLabel) == 7 and not loadLastLabel:
                    continue
                try:
                    valValue = np.load(os.path.join(afectdata,'annotations','%d_val.npy' % (int(imageNumber))))
                    aroValue = np.load(os.path.join(afectdata,'annotations','%d_aro.npy' % (int(imageNumber))))
                except:
                    valValue = np.load(os.path.join(afectdata,'annotations',f'{imageNumber}_val.npy'))
                    aroValue = np.load(os.path.join(afectdata,'annotations',f'{imageNumber}_aro.npy'))
                self.label.append(currLabel)
                self.seconLabel.append([valValue,aroValue])
            elif typeExperiment == 'RANK':
                currLabel = os.path.join(afectdata,'annotations' ,'%d_rank.txt' % (int(imageNumber)))
                self.label.append(currLabel)
            elif typeExperiment == 'RANDOM':
                self.label.append(np.random.randint(0,8))
            elif typeExperiment == 'PROBS':
                currLabel = os.path.join(afectdata,'annotations' ,'%d_prob_rank.txt' % (int(imageNumber)))
                self.label.append(currLabel)
            elif 'PROBS_VAD' in typeExperiment:
                try:
                    currLabel = os.path.join(afectdata,'annotations' ,'%d_prob_rank.txt' % (int(imageNumber)))
                    valValue = np.load(os.path.join(afectdata,'annotations','%d_val.npy' % (int(imageNumber)))).astype(np.float64)
                    aroValue = np.load(os.path.join(afectdata,'annotations','%d_aro.npy' % (int(imageNumber)))).astype(np.float64)
                    domValue = np.load(os.path.join(afectdata,'annotations','%d_dom.npy' % (int(imageNumber)))).astype(np.float64)
                except:
                    currLabel = os.path.join(afectdata,'annotations' ,f'{imageNumber}_prob_rank.txt')
                    valValue = np.load(os.path.join(afectdata,'annotations',f'{imageNumber}_val.npy' )).astype(np.float64)
                    domValue = np.load(os.path.join(afectdata,'annotations',f'{imageNumber}_aro.npy')).astype(np.float64)
                    try:
                        domValue = np.load(os.path.join(afectdata,'annotations',f'{imageNumber}_dom.npy')).astype(np.float64)
                    except:
                        continue
                self.label.append([currLabel,valValue,aroValue,domValue])
                if '_EXP' in typeExperiment:
                    try:
                        lbl = np.load(os.path.join(afectdata,'annotations' ,f'{imageNumber}_exp.npy'))
                        self.label[-1].append(int(lbl))
                    except:
                        self.label[-1].append(255)
            elif 'PROBS_VA' in typeExperiment:
                try:
                    currLabel = os.path.join(afectdata,'annotations' ,'%d_prob_rank.txt' % (int(imageNumber)))
                    valValue = np.load(os.path.join(afectdata,'annotations','%d_val.npy' % (int(imageNumber)))).astype(np.float64)
                    aroValue = np.load(os.path.join(afectdata,'annotations','%d_aro.npy' % (int(imageNumber)))).astype(np.float64)
                except:
                    currLabel = os.path.join(afectdata,'annotations' ,f'{imageNumber}_prob_rank.txt')
                    valValue = np.load(os.path.join(afectdata,'annotations',f'{imageNumber}_val.npy' )).astype(np.float64)
                    aroValue = np.load(os.path.join(afectdata,'annotations',f'{imageNumber}_aro.npy')).astype(np.float64)
                self.label.append([currLabel,valValue,aroValue])                    
                if '_EXP' in typeExperiment:
                    try:
                        lbl = np.load(os.path.join(afectdata,'annotations' ,f'{imageNumber}_exp.npy'))
                        self.label[-1].append(int(lbl))
                    except:
                        self.label[-1].append(255)
            elif typeExperiment == 'UNIVERSAL':
                currLabel = os.path.join(afectdata,'annotations' ,'%d_prob_rank_universal.txt' % (int(imageNumber)))
                self.label.append(currLabel)
            elif 'UNIVERSAL_VAD_ADJUSTED' in typeExperiment:
                try:
                    currLabel = os.path.join(afectdata,'annotations' ,'%d_prob_rank_universal.txt' % (int(imageNumber)))
                    valValue = np.load(os.path.join(afectdata,'annotations','%d_adjusted_val.npy' % (int(imageNumber)))).astype(np.float64)
                    aroValue = np.load(os.path.join(afectdata,'annotations','%d_adjusted_aro.npy' % (int(imageNumber)))).astype(np.float64)
                    domValue = np.load(os.path.join(afectdata,'annotations','%d_dom.npy' % (int(imageNumber)))).astype(np.float64)
                except:
                    currLabel = os.path.join(afectdata,'annotations' ,f'{imageNumber}_prob_rank_universal.txt')
                    valValue = np.load(os.path.join(afectdata,'annotations',f'{imageNumber}_adjusted_val.npy' )).astype(np.float64)
                    aroValue = np.load(os.path.join(afectdata,'annotations',f'{imageNumber}_adjusted_aro.npy')).astype(np.float64)
                    try:
                        domValue = np.load(os.path.join(afectdata,'annotations',f'{imageNumber}_dom.npy')).astype(np.float64)
                    except:
                        continue
                self.label.append([currLabel,valValue,aroValue,domValue])
                if '_EXP' in typeExperiment:
                    try:
                        lbl = np.load(os.path.join(afectdata,'annotations' ,f'{imageNumber}_exp.npy'))
                        self.label[-1].append(int(lbl))
                    except:
                        self.label[-1].append(255)
                                        
            elif 'UNIVERSAL_VAD' in typeExperiment:
                prefix = '' if typeExperiment not in '_ADJUSTED' else '_adjusted'
                try:
                    currLabel = os.path.join(afectdata,'annotations' ,'%d_prob_rank_universal.txt' % (int(imageNumber)))
                    valValue = np.load(os.path.join(afectdata,'annotations',f'{int(imageNumber)}{prefix}_val.npy')).astype(np.float64)
                    aroValue = np.load(os.path.join(afectdata,'annotations',f'{int(imageNumber)}{prefix}_aro.npy')).astype(np.float64)
                    domValue = np.load(os.path.join(afectdata,'annotations','%d_dom.npy' % (int(imageNumber)))).astype(np.float64)
                except:
                    currLabel = os.path.join(afectdata,'annotations' ,f'{imageNumber}_prob_rank_universal.txt')
                    valValue = np.load(os.path.join(afectdata,'annotations',f'{imageNumber}_val.npy' )).astype(np.float64)
                    aroValue = np.load(os.path.join(afectdata,'annotations',f'{imageNumber}_aro.npy')).astype(np.float64)
                    try:
                        domValue = np.load(os.path.join(afectdata,'annotations',f'{imageNumber}_dom.npy')).astype(np.float64)
                    except:
                        continue
                self.label.append([currLabel,valValue,aroValue,domValue])
                if '_EXP' in typeExperiment:
                    try:
                        lbl = np.load(os.path.join(afectdata,'annotations' ,f'{imageNumber}_exp.npy'))
                        self.label[-1].append(int(lbl))
                    except:
                        self.label[-1].append(255)
            elif 'ORIGINAL_VAD' in typeExperiment:
                prefix = '' if typeExperiment not in '_ADJUSTED' else '_adjusted'
                try:
                    currLabel = os.path.join(afectdata,'annotations' ,'%d_prob_rank_original.txt' % (int(imageNumber)))
                    valValue = np.load(os.path.join(afectdata,'annotations',f'{int(imageNumber)}{prefix}_val.npy')).astype(np.float64)
                    aroValue = np.load(os.path.join(afectdata,'annotations',f'{int(imageNumber)}{prefix}_aro.npy')).astype(np.float64)
                    domValue = np.load(os.path.join(afectdata,'annotations','%d_dom.npy' % (int(imageNumber)))).astype(np.float64)
                except:
                    currLabel = os.path.join(afectdata,'annotations' ,f'{imageNumber}_prob_rank_original.txt')
                    valValue = np.load(os.path.join(afectdata,'annotations',f'{imageNumber}{prefix}_val.npy' )).astype(np.float64)
                    aroValue = np.load(os.path.join(afectdata,'annotations',f'{imageNumber}{prefix}_aro.npy')).astype(np.float64)
                    domValue = np.load(os.path.join(afectdata,'annotations',f'{imageNumber}_dom.npy')).astype(np.float64)
                    try:
                        domValue = np.load(os.path.join(afectdata,'annotations',f'{imageNumber}_dom.npy')).astype(np.float64)
                    except:
                        continue
                self.label.append([currLabel,valValue,aroValue,domValue])
                if '_EXP' in typeExperiment:
                    try:
                        lbl = np.load(os.path.join(afectdata,'annotations' ,f'{imageNumber}_exp.npy'))
                        self.label[-1].append(int(lbl))
                    except:
                        self.label[-1].append(255)
                    
            else:
                currLabel = self.loadTermData(os.path.join(afectdata,'annotations_%d' % (termsQuantity),'%d_terms.txt' % (int(imageNumber))))
                self.label.append(np.where(self.terms == currLabel)[0][0])
            if quantitylabels is not None and quantitylabels[int(currLabel)] > 500 and not random.randint(0,1):
                continue
            self.filesPath.append(f)

            if quantitylabels is not None and sum(quantitylabels) >= 1000:
                break
            
    def loadTermsFile(self,termsQuantity):
        return np.array(pd.read_csv('joinedWithDistance_%d.csv' % (termsQuantity)))[:,0]

    def __len__(self):
        return len(self.filesPath)

    def loadTermData(self,pathData):
        returnData = ""
        with open(pathData,'r') as pd:
            for p in pd:
                returnData = p

        return returnData

    def loadRankFile(self,rankPath):
        dataOutput = None
        with open(str(rankPath),'r') as rf:
            dataOutput = list(map(int,rf.readline().strip().split(',')))
        return dataOutput

    def loadProbFile(self,rankPath):
        dataOutput = None
        with open(str(rankPath),'r') as rf:
            dataOutput = list(map(float,rf.readline().strip().split(',')))
        return dataOutput

    def __getitem__(self, idx):
        path = self.filesPath[idx]
        image = im.open(path)
        valenceLabel = None
        if self.typeExperiment == 'TERMS':
            label = torch.from_numpy(np.array(self.label[idx])).to(torch.float32)
        elif self.typeExperiment == 'EXP' or self.typeExperiment == 'BOTH':
            if self.exchangeLabel is not None:
                valenceLabel = torch.from_numpy(np.array(self.exchangeLabel[self.label[idx]])).to(torch.long)
            if self.typeExperiment == 'BOTH':
                valenceLabel = torch.from_numpy(np.array( [self.seconLabel[idx][0].astype(np.float32),self.seconLabel[idx][1].astype(np.float32)] )).to(torch.float32)
            label = torch.from_numpy(np.array(self.label[idx]).astype(np.uint8)).to(torch.long)
        elif self.typeExperiment == "RANK":
            label = torch.from_numpy(np.array(self.loadRankFile(self.label[idx])).astype(np.uint8)).to(torch.long)
        elif self.typeExperiment == "RANDOM":
            label = torch.from_numpy(np.array(np.random.randint(0,8))).to(torch.long)
        elif self.typeExperiment == "PROBS":
            label = torch.from_numpy(np.array(self.loadProbFile(self.label[idx])).astype(np.float32)).to(torch.float32)
        elif self.typeExperiment == "UNIVERSAL":
            label = torch.from_numpy(np.array(self.loadProbFile(self.label[idx])).astype(np.float32)).to(torch.float32)
        elif 'PROBS_VAD' in self.typeExperiment:
            label = [
                torch.from_numpy(np.array(self.loadProbFile(self.label[idx][0])).astype(np.float32)).to(torch.float32),
                torch.from_numpy(np.array([self.label[idx][1],self.label[idx][2],self.label[idx][3]]))
            ]
            if '_EXP' in self.typeExperiment:
                label.append(torch.from_numpy(np.array(self.label[idx][4]).astype(np.uint8)).to(torch.long))
        elif 'PROBS_VA' in self.typeExperiment:
            label = [
                torch.from_numpy(np.array(self.loadProbFile(self.label[idx][0])).astype(np.float32)).to(torch.float32),
                torch.from_numpy(np.array([self.label[idx][1],self.label[idx][2]]))
            ]
            if '_EXP' in self.typeExperiment:
                label.append(torch.from_numpy(np.array(self.label[idx][3]).astype(np.uint8)).to(torch.long))
        elif 'UNIVERSAL_VAD' in self.typeExperiment:
            label = [
                torch.from_numpy(np.array(self.loadProbFile(self.label[idx][0])).astype(np.float32)).to(torch.float32),
                torch.from_numpy(np.array([self.label[idx][1],self.label[idx][2],self.label[idx][3]]))
            ]
            if '_EXP' in self.typeExperiment:
                label.append(torch.from_numpy(np.array(self.label[idx][4]).astype(np.uint8)).to(torch.long))
        elif "VAD" in self.typeExperiment:
            label = torch.from_numpy(
                np.array( 
                    [self.label[idx][0].astype(np.float32),self.label[idx][1].astype(np.float32),self.label[idx][2].astype(np.float32)] )).to(torch.float32)
            if '_EXP' in self.typeExperiment:
                label_part2 = torch.from_numpy(np.array(self.label[idx][-1]).astype(np.uint8)).to(torch.long)
                label = torch.cat([label, label_part2.unsqueeze(0)])                
        elif "VA" in self.typeExperiment:
            label = torch.from_numpy(
                np.array( 
                    [self.label[idx][0].astype(np.float32),self.label[idx][1].astype(np.float32)] )).to(torch.float32)
            if '_EXP' in self.typeExperiment:
                label_part2 = torch.from_numpy(np.array(self.label[idx][-1]).astype(np.uint8)).to(torch.long)
                label = torch.cat([label, label_part2.unsqueeze(0)])
        elif 'UNIVERSAL_VAD' in self.typeExperiment:
            label = [
                torch.from_numpy(np.array(self.loadProbFile(self.label[idx][0])).astype(np.float32)).to(torch.float32),
                torch.from_numpy(np.array([self.label[idx][1],self.label[idx][2],self.label[idx][3]]))
            ]
            if '_EXP' in self.typeExperiment:
                label.append(torch.from_numpy(np.array(self.label[idx][4]).astype(np.uint8)).to(torch.long))
        else:
            label = torch.from_numpy(np.array( [self.label[idx][0].astype(np.float32),self.label[idx][1].astype(np.float32)] )).to(torch.float32)
        if self.transform is not None:
            image = self.transform(image)

        if valenceLabel is not None:
            return image, label, self.filesPath[idx], valenceLabel
        else:
            return image, label, self.filesPath[idx]
        
    def sample(self,classes,exclude):
        returnValue = [0] * (max(classes) + 1)
        for c in classes:
            availble = [ idxc for idxc, cl in enumerate(self.label) if cl == c]
            sortedIdx = random.randint(0,len(availble)-1)
            while self.filesPath[availble[sortedIdx]] in exclude:
                sortedIdx = random.randint(0,len(availble)-1)

            i,_,_ = self.__getitem__(availble[sortedIdx])
            returnValue[c] = i

        return torch.stack(returnValue)
    '''
class OPTAffectNet(data.Dataset):    
    def __init__(self, afectdata, typeExperiment="VA", transform=None, termsQuantity=151, 
                 exchangeLabel=None, loadLastLabel=True, preload_images=False, num_workers=4,
                 max_open_files=1000):
        self.exchangeLabel = exchangeLabel
        self.terms = None if typeExperiment != 'TERMS' else self.loadTermsFile(termsQuantity)
        self.transform = transform
        self.label = []
        self.filesPath = []
        self.seconLabel = []
        self.typeExperiment = typeExperiment
        self.preload_images = preload_images
        self.images_cache = {}
        self.annotation_cache = {}
        self.cache_lock = threading.Lock()
        self.loadLastLabel = loadLastLabel
        self.max_open_files = max_open_files
        
        faces = getFilesInPath(os.path.join(afectdata,'images'), imagesOnly=True)
        print(f"Loading {len(faces)} face images")
        
        # Extrair números das imagens (mantendo o nome completo do arquivo sem extensão)
        image_numbers = [os.path.splitext(os.path.basename(f))[0] for f in faces]
        
        # Carregar anotações em paralelo com barra de progresso
        self.load_annotations_parallel(afectdata, image_numbers, faces, num_workers)
        
        # Pré-carregar imagens se solicitado com barra de progresso
        if preload_images:
            self.preload_images_to_memory(faces, num_workers)
    
    def get_possible_annotation_paths(self, afectdata, image_number, annotation_type):
        """Gera possíveis caminhos para arquivos de anotação"""
        base_name = str(image_number)
        
        if annotation_type == 'prob_rank':
            return [
                os.path.join(afectdata, 'annotations', f'{base_name}_prob_rank.txt'),
            ]
        elif annotation_type == 'val':
            return [
                os.path.join(afectdata, 'annotations', f'{base_name}_val.npy'),
            ]
        elif annotation_type == 'aro':
            return [
                os.path.join(afectdata, 'annotations', f'{base_name}_aro.npy'),
            ]
        elif annotation_type == 'dom':
            return [
                os.path.join(afectdata, 'annotations', f'{base_name}_dom.npy'),
            ]
        elif annotation_type == 'exp':
            return [
                os.path.join(afectdata, 'annotations', f'{base_name}_exp.npy'),
            ]
        elif annotation_type == 'rank':
            return [
                os.path.join(afectdata, 'annotations', f'{base_name}_rank.txt'),
            ]
        return []

    def safe_file_operation(self, file_path, operation, *args):
        """Operação segura de arquivo com controle de limite"""
        try:
            if operation == 'read_text':
                with open(file_path, 'r') as f:
                    return f.readline().strip().split(',')
            elif operation == 'read_numpy':
                return np.load(file_path)
            elif operation == 'read_terms':
                with open(file_path, 'r') as f:
                    return f.readline().strip()
        except Exception as e:
            print(f"Error in file operation {operation} on {file_path}: {e}")
            return None

    def load_annotations_parallel(self, afectdata, image_numbers, faces, num_workers):
        """Carrega anotações em paralelo com controle de limite de arquivos"""
        print("Loading annotations...")
        total_files = len(image_numbers)
        processed_files = 0
        successful_files = 0
        
        # Processar em lotes para evitar muitos arquivos abertos
        batch_size = min(200, total_files // 10 or 50)
        
        for batch_start in range(0, total_files, batch_size):
            batch_end = min(batch_start + batch_size, total_files)
            batch_images = image_numbers[batch_start:batch_end]
            batch_faces = faces[batch_start:batch_end]
            
            # Usar menos workers para reduzir arquivos abertos simultaneamente
            with ThreadPoolExecutor(max_workers=min(num_workers, 2)) as executor:
                futures = {}
                
                for i, (image_number, face_path) in enumerate(zip(batch_images, batch_faces)):
                    if self.typeExperiment == "PROBS_VAD":
                        future = executor.submit(self.load_probs_vad_annotation, afectdata, image_number, face_path)
                    elif self.typeExperiment == "PROBS_VA":
                        future = executor.submit(self.load_probs_va_annotation, afectdata, image_number, face_path)
                    elif self.typeExperiment == "VA":
                        future = executor.submit(self.load_va_annotation, afectdata, image_number, face_path)
                    elif self.typeExperiment == "VAD":
                        future = executor.submit(self.load_vad_annotation, afectdata, image_number, face_path)
                    elif self.typeExperiment == "EXP":
                        future = executor.submit(self.load_exp_annotation, afectdata, image_number, face_path)
                    elif self.typeExperiment == "BOTH":
                        future = executor.submit(self.load_both_annotation, afectdata, image_number, face_path)
                    elif self.typeExperiment == "RANK":
                        future = executor.submit(self.load_rank_annotation, afectdata, image_number, face_path)
                    elif self.typeExperiment == "PROBS":
                        future = executor.submit(self.load_probs_annotation, afectdata, image_number, face_path)
                    elif self.typeExperiment == "TERMS":
                        future = executor.submit(self.load_terms_annotation, afectdata, image_number, face_path)
                    else:
                        future = executor.submit(self.load_default_annotation, afectdata, image_number, face_path)
                    
                    futures[future] = (face_path, image_number)
                
                # Processar resultados do batch
                for future in as_completed(futures):
                    face_path, image_number = futures[future]
                    processed_files += 1
                    printProgressBar(processed_files, total_files, prefix='Loading annotations:', suffix=f'{processed_files}/{total_files}', length=50)
                    
                    try:
                        result = future.result()
                        if result is not None:
                            if isinstance(result, tuple) and len(result) > 1:
                                self.label.append(result[0])
                                self.seconLabel.append(result[1])
                            else:
                                self.label.append(result)
                            self.filesPath.append(face_path)
                            successful_files += 1
                    except Exception as e:
                        print(f"\nError loading annotation for {face_path}: {e}")
            
            # Limpar cache periódicamente para liberar recursos
            if len(self.annotation_cache) > self.max_open_files:
                print("Clearing annotation cache to free resources...")
                self.annotation_cache.clear()
        
        print(f"\nSuccessfully loaded {successful_files} annotations out of {total_files}")

    def load_probs_vad_annotation(self, afectdata, image_number, face_path):
        """Carrega anotações PROBS_VAD de forma otimizada"""
        try:
            base_name = str(image_number)
            
            # Gerar possíveis caminhos
            prob_paths = self.get_possible_annotation_paths(afectdata, base_name, 'prob_rank')
            val_paths = self.get_possible_annotation_paths(afectdata, base_name, 'val')
            aro_paths = self.get_possible_annotation_paths(afectdata, base_name, 'aro')
            dom_paths = self.get_possible_annotation_paths(afectdata, base_name, 'dom')
            
            # Encontrar arquivos que existem
            prob_path = self.find_existing_file(prob_paths)
            val_path = self.find_existing_file(val_paths)
            aro_path = self.find_existing_file(aro_paths)
            dom_path = self.find_existing_file(dom_paths)
            
            if not all([prob_path, val_path, aro_path, dom_path]):
                return None
            
            # Carregar dados com cache
            with self.cache_lock:
                # Carregar probabilidades
                if prob_path not in self.annotation_cache:
                    prob_data = self.safe_file_operation(prob_path, 'read_text')
                    if prob_data:
                        self.annotation_cache[prob_path] = list(map(float, prob_data))
                
                # Carregar valores VAD
                if val_path not in self.annotation_cache:
                    val_data = self.safe_file_operation(val_path, 'read_numpy')
                    if val_data is not None:
                        self.annotation_cache[val_path] = val_data.astype(np.float64)
                
                if aro_path not in self.annotation_cache:
                    aro_data = self.safe_file_operation(aro_path, 'read_numpy')
                    if aro_data is not None:
                        self.annotation_cache[aro_path] = aro_data.astype(np.float64)
                
                if dom_path not in self.annotation_cache:
                    dom_data = self.safe_file_operation(dom_path, 'read_numpy')
                    if dom_data is not None:
                        self.annotation_cache[dom_path] = dom_data.astype(np.float64)
            
            # Verificar se todos os dados foram carregados
            if not all([prob_path in self.annotation_cache, val_path in self.annotation_cache, 
                       aro_path in self.annotation_cache, dom_path in self.annotation_cache]):
                return None
            
            return [prob_path, 
                    self.annotation_cache[val_path], 
                    self.annotation_cache[aro_path], 
                    self.annotation_cache[dom_path]]
                    
        except Exception as e:
            print(f"Error loading PROBS_VAD for {image_number}: {e}")
            return None

    def find_existing_file(self, file_paths):
        """Encontra o primeiro arquivo que existe na lista de caminhos"""
        for path in file_paths:
            if os.path.exists(path):
                return path
        return None

    def load_probs_va_annotation(self, afectdata, image_number, face_path):
        """Carrega anotações PROBS_VA de forma otimizada"""
        try:
            base_name = str(image_number)
            
            prob_paths = self.get_possible_annotation_paths(afectdata, base_name, 'prob_rank')
            val_paths = self.get_possible_annotation_paths(afectdata, base_name, 'val')
            aro_paths = self.get_possible_annotation_paths(afectdata, base_name, 'aro')
            
            prob_path = self.find_existing_file(prob_paths)
            val_path = self.find_existing_file(val_paths)
            aro_path = self.find_existing_file(aro_paths)
            
            if not all([prob_path, val_path, aro_path]):
                return None
            
            with self.cache_lock:
                if prob_path not in self.annotation_cache:
                    prob_data = self.safe_file_operation(prob_path, 'read_text')
                    if prob_data:
                        self.annotation_cache[prob_path] = list(map(float, prob_data))
                
                if val_path not in self.annotation_cache:
                    val_data = self.safe_file_operation(val_path, 'read_numpy')
                    if val_data is not None:
                        self.annotation_cache[val_path] = val_data.astype(np.float64)
                
                if aro_path not in self.annotation_cache:
                    aro_data = self.safe_file_operation(aro_path, 'read_numpy')
                    if aro_data is not None:
                        self.annotation_cache[aro_path] = aro_data.astype(np.float64)
            
            if not all([prob_path in self.annotation_cache, val_path in self.annotation_cache, 
                       aro_path in self.annotation_cache]):
                return None
            
            return [prob_path, 
                    self.annotation_cache[val_path], 
                    self.annotation_cache[aro_path]]
                    
        except Exception as e:
            print(f"Error loading PROBS_VA for {image_number}: {e}")
            return None

    def preload_images_to_memory(self, faces, num_workers):
        """Pré-carrega imagens na memória com controle de limite"""
        print("Preloading images to memory...")
        total_images = len(faces)
        processed_images = 0
        successful_images = 0
        
        # Processar em lotes menores
        batch_size = min(100, total_images // 10 or 50)
        
        for batch_start in range(0, total_images, batch_size):
            batch_end = min(batch_start + batch_size, total_images)
            batch_faces = faces[batch_start:batch_end]
            
            def load_image(face_path):
                try:
                    # Fechar arquivo imediatamente após carregar
                    with im.open(face_path) as img:
                        image = img.copy()
                    if self.transform:
                        image = self.transform(image)
                    return face_path, image, True
                except Exception as e:
                    return face_path, None, False
            
            with ThreadPoolExecutor(max_workers=min(num_workers, 2)) as executor:
                future_to_path = {executor.submit(load_image, face_path): face_path for face_path in batch_faces}
                
                for future in as_completed(future_to_path):
                    face_path = future_to_path[future]
                    processed_images += 1
                    printProgressBar(processed_images, total_images, prefix='Preloading images:', suffix=f'{processed_images}/{total_images}', length=50)
                    
                    try:
                        face_path, image, success = future.result()
                        if success and image is not None:
                            self.images_cache[face_path] = image
                            successful_images += 1
                    except Exception as e:
                        print(f"\nError processing image {face_path}: {e}")
            
            # Limpar cache de imagens periodicamente se necessário
            if len(self.images_cache) > self.max_open_files:
                print("Clearing image cache to free resources...")
                # Manter apenas as imagens mais recentes
                keys_to_keep = list(self.images_cache.keys())[-self.max_open_files//2:]
                self.images_cache = {k: self.images_cache[k] for k in keys_to_keep}
        
        print(f"\nSuccessfully preloaded {successful_images} images out of {total_images}")

    @lru_cache(maxsize=300)  # Cache menor para evitar muitos arquivos abertos
    def load_prob_file_cached(self, rank_path):
        """Versão em cache do loadProbFile com controle de recursos"""
        try:
            with open(rank_path, 'r') as rf:
                data_output = list(map(float, rf.readline().strip().split(',')))
            return data_output
        except Exception as e:
            print(f"Error loading cached prob file {rank_path}: {e}")
            return None

    def loadProbFile(self, rankPath):
        """Carrega arquivo de probabilidades com controle de recursos"""
        try:
            with open(str(rankPath), 'r') as rf:
                data_output = list(map(float, rf.readline().strip().split(',')))
            return data_output
        except Exception as e:
            print(f"Error loading prob file {rankPath}: {e}")
            return None

    def __getitem__(self, idx):
        path = self.filesPath[idx]
        
        # Usar cache se disponível
        if self.preload_images and path in self.images_cache:
            image = self.images_cache[path]
        else:
            try:
                # Fechar arquivo imediatamente após carregar
                with im.open(path) as img:
                    image = img.copy()
                if self.transform:
                    image = self.transform(image)
            except Exception as e:
                print(f"Error loading image {path}: {e}")
                # Retornar uma imagem dummy em caso de erro
                image = im.new('RGB', (224, 224), color='gray')
        
        valenceLabel = None
        
        # Processar labels baseado no tipo de experimento
        try:
            if self.typeExperiment == 'TERMS':
                label = torch.from_numpy(np.array(self.label[idx])).to(torch.float32)
            elif self.typeExperiment == 'EXP' or self.typeExperiment == 'BOTH':
                if self.exchangeLabel is not None:
                    valenceLabel = torch.from_numpy(np.array(self.exchangeLabel[self.label[idx]])).to(torch.long)
                if self.typeExperiment == 'BOTH':
                    valenceLabel = torch.from_numpy(np.array([self.seconLabel[idx][0].astype(np.float32), 
                                                            self.seconLabel[idx][1].astype(np.float32)])).to(torch.float32)
                label = torch.from_numpy(np.array(self.label[idx]).astype(np.uint8)).to(torch.long)
            elif self.typeExperiment == "RANK":
                label = torch.from_numpy(np.array(self.loadRankFile(self.label[idx])).astype(np.uint8)).to(torch.long)
            elif self.typeExperiment == "RANDOM":
                label = torch.from_numpy(np.array(np.random.randint(0,8))).to(torch.long)
            elif self.typeExperiment == "PROBS":
                label = torch.from_numpy(np.array(self.loadProbFile(self.label[idx])).astype(np.float32)).to(torch.float32)
            elif self.typeExperiment == 'PROBS_VA':
                prob_path, val, aro = self.label[idx]
                prob_data = self.load_prob_file_cached(prob_path)
                if prob_data is None:
                    prob_data = self.loadProbFile(prob_path)
                label = [
                    torch.from_numpy(np.array(prob_data).astype(np.float32)).to(torch.float32),
                    torch.from_numpy(np.array([val, aro]))
                ]
            elif self.typeExperiment == 'PROBS_VAD':
                prob_path, val, aro, dom = self.label[idx]
                prob_data = self.load_prob_file_cached(prob_path)
                if prob_data is None:
                    prob_data = self.loadProbFile(prob_path)
                label = [
                    torch.from_numpy(np.array(prob_data).astype(np.float32)).to(torch.float32),
                    torch.from_numpy(np.array([val, aro, dom]))
                ]
            elif self.typeExperiment == "VAD":
                label = torch.from_numpy(np.array([self.label[idx][0].astype(np.float32), 
                                                self.label[idx][1].astype(np.float32), 
                                                self.label[idx][2].astype(np.float32)])).to(torch.float32)
            else:  # VA
                label = torch.from_numpy(np.array([self.label[idx][0].astype(np.float32), 
                                                self.label[idx][1].astype(np.float32)])).to(torch.float32)
        except Exception as e:
            print(f"Error processing label for {path}: {e}")
            # Retornar label dummy em caso de erro
            if self.typeExperiment == "VA":
                label = torch.zeros(2)
            elif self.typeExperiment == "VAD":
                label = torch.zeros(3)
            elif self.typeExperiment == "PROBS_VA" or self.typeExperiment == "PROBS_VAD":
                label = [torch.zeros(10), torch.zeros(2 if self.typeExperiment == "PROBS_VA" else 3)]
            else:
                label = torch.zeros(1)
        
        if valenceLabel is not None:
            return image, label, self.filesPath[idx], valenceLabel
        else:
            return image, label, self.filesPath[idx]


    def __len__(self):
        return len(self.filesPath)

    def sample(self, classes, exclude):
        returnValue = [0] * (max(classes) + 1)
        for c in classes:
            available = [ idxc for idxc, cl in enumerate(self.label) if cl == c]
            if not available:
                continue
            sortedIdx = random.randint(0, len(available)-1)
            while self.filesPath[available[sortedIdx]] in exclude:
                sortedIdx = random.randint(0, len(available)-1)
            i, _, _ = self.__getitem__(available[sortedIdx])
            returnValue[c] = i
        return torch.stack(returnValue)
    
    def loadTermsFile(self, termsQuantity):
        return np.array(pd.read_csv('joinedWithDistance_%d.csv' % (termsQuantity)))[:,0]

    def loadTermData(self, pathData):
        returnData = ""
        with open(pathData,'r') as pd:
            for p in pd:
                returnData = p
        return returnData

    def loadRankFile(self, rankPath):
        dataOutput = None
        with open(str(rankPath),'r') as rf:
            dataOutput = list(map(int, rf.readline().strip().split(',')))
        return dataOutput

    def loadProbFile(self, rankPath):
        dataOutput = None
        with open(str(rankPath),'r') as rf:
            dataOutput = list(map(float, rf.readline().strip().split(',')))
        return dataOutput