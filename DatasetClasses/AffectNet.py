import torch.utils.data as data, os, torch, numpy as np, sys, pandas as pd, random
from PIL import Image as im
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import getDirectoriesInPath, getFilesInPath, printProgressBar
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import threading
#from generateDenseOpticalFlow import runDenseOpFlow

class AffectNet(data.Dataset):    
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