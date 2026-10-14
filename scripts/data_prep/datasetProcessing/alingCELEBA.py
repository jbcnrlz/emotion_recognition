import argparse, sys, os, cv2, numpy as np
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import printProgressBar, getFilesInPath
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel
from helper.function import printProgressBar, getFilesInPath

class FaceAlignmentDataset(Dataset):
    def __init__(self, files, rank, world_size):
        self.files = files
        self.rank = rank
        self.world_size = world_size
        # Distribui os arquivos igualmente entre os processos
        self.local_files = self.files[self.rank::self.world_size]
        
        # Carrega o modelo de detecção uma vez por processo
        model_file = 'faceDetection/res10_300x300_ssd_iter_140000_fp16.caffemodel'
        config_file = 'faceDetection/deploy.prototxt'
        self.net = cv2.dnn.readNetFromCaffe(config_file, model_file)
        if torch.cuda.is_available():
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def __len__(self):
        return len(self.local_files)

    def __getitem__(self, idx):
        file = self.local_files[idx]
        try:
            frame = cv2.imread(file)
            if frame is None:
                return None, None
                
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, 
                                       (300, 300), [104, 117, 123])
            self.net.setInput(blob)
            detections = self.net.forward()

            best_roi = None
            for j in range(detections.shape[2]):
                confidence = detections[0, 0, j, 2]
                if confidence > 0.7:
                    box = detections[0, 0, j, 3:7] * np.array([w, h, w, h])
                    (x, y, x2, y2) = box.astype("int")
                    best_roi = frame[y:y2, x:x2]
                    break

            return file, best_roi
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            return None, None

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size, args):
    setup(rank, world_size)
    
    files = getFilesInPath(args.input, imagesOnly=True)
    dataset = FaceAlignmentDataset(files, rank, world_size)
    
    if rank == 0 and not os.path.exists(args.output):
        os.makedirs(args.output)
    dist.barrier()  # Espera todos os processos chegarem aqui

    processed = 0
    total_files = len(dataset)
    
    for i in range(len(dataset)):
        file, roi = dataset[i]
        if file is None:
            continue
            
        output_file = os.path.join(args.output, os.path.basename(file))
        if roi is not None:
            cv2.imwrite(output_file, roi)
        
        processed += 1
        if rank == 0:  # Apenas o rank 0 imprime a barra de progresso
            global_processed = torch.tensor([processed * world_size], dtype=torch.float32)
            dist.reduce(global_processed, dst=0, op=dist.ReduceOp.SUM)
            printProgressBar(global_processed.item(), len(files), 
                           prefix=f'Rank {rank} Aligning:', 
                           suffix='Complete', length=50)

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distributed Face Alignment')
    parser.add_argument('--input', help='Input folder with images', required=True)
    parser.add_argument('--output', help='Output folder for aligned images', required=True)
    args = parser.parse_args()

    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    torch.multiprocessing.spawn(main, args=(world_size, args), nprocs=world_size)