import cv2
import shutil
import os
import face_alignment
import argparse
import sys
import numpy as np
from collections import defaultdict
import threading
from queue import Queue
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from helper.function import getFilesInPath

class VideoWriterManager:
    def __init__(self, base_path, fps=30, frame_size=(160, 160)):
        self.base_path = base_path
        self.fps = fps
        self.frame_size = frame_size
        self.writers = {}
        self.frame_queues = {}
        self.writer_threads = {}
        self.running = True
        
        if not os.path.exists(base_path):
            os.makedirs(base_path)
    
    def add_video_writer(self, roi_id):
        """Adiciona um novo video writer para um ROI"""
        if roi_id in self.writers:
            return
            
        output_path = os.path.join(self.base_path, f'roi_{roi_id:03d}.avi')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        
        writer = cv2.VideoWriter(output_path, fourcc, self.fps, self.frame_size)
        self.writers[roi_id] = writer
        
        # Criar queue e thread para escrita assíncrona
        self.frame_queues[roi_id] = Queue(maxsize=50)
        thread = threading.Thread(target=self._write_frames, args=(roi_id,))
        thread.daemon = True
        thread.start()
        self.writer_threads[roi_id] = thread
    
    def _write_frames(self, roi_id):
        """Thread para escrita assíncrona de frames"""
        while self.running or not self.frame_queues[roi_id].empty():
            try:
                frame = self.frame_queues[roi_id].get(timeout=1.0)
                if frame is None:  # Sinal para parar
                    break
                self.writers[roi_id].write(frame)
            except:
                continue
    
    def write_frame(self, roi_id, frame):
        """Adiciona frame à queue de escrita"""
        if roi_id in self.writers and roi_id in self.frame_queues:
            try:
                # Redimensionar para tamanho consistente
                resized_frame = cv2.resize(frame, self.frame_size)
                self.frame_queues[roi_id].put_nowait(resized_frame)
            except:
                pass  # Queue cheia, descartar frame
    
    def release_all(self):
        """Libera todos os recursos"""
        self.running = False
        
        # Sinalizar para as threads pararem
        for roi_id in self.frame_queues:
            self.frame_queues[roi_id].put(None)
        
        # Aguardar threads terminarem
        for thread in self.writer_threads.values():
            thread.join(timeout=2.0)
        
        # Liberar writers
        for writer in self.writers.values():
            writer.release()
        
        self.writers.clear()
        self.frame_queues.clear()
        self.writer_threads.clear()

class FaceTracker:
    def __init__(self, max_distance=30):
        self.next_roi_id = 0
        self.rois = {}  # {roi_id: position}
        self.max_distance = max_distance
    
    def find_roi(self, landmarks_points):
        """Encontra ROI correspondente para landmarks usando cálculo vetorizado"""
        return_data = []
        
        if not self.rois:
            for point in landmarks_points:
                roi_id = self.register(point)
                return_data.append(roi_id)
            return return_data
        
        # Cálculo vetorizado de distâncias
        current_rois = np.array(list(self.rois.values()))
        landmarks_array = np.array(landmarks_points)
        
        # Calcular todas as distâncias de uma vez
        if current_rois.size > 0 and landmarks_array.size > 0:
            # Broadcasting para cálculo eficiente
            dists = np.linalg.norm(current_rois[:, np.newaxis] - landmarks_array, axis=2)
            min_dists = dists.min(axis=0)
            best_matches = dists.argmin(axis=0)
            
            for i, (min_dist, best_match) in enumerate(zip(min_dists, best_matches)):
                if min_dist < self.max_distance:
                    # Atualizar posição do ROI
                    roi_id = list(self.rois.keys())[best_match]
                    self.rois[roi_id] = landmarks_array[i]
                    return_data.append(roi_id)
                else:
                    # Novo ROI
                    roi_id = self.register(landmarks_array[i])
                    return_data.append(roi_id)
        
        return return_data
    
    def register(self, point):
        """Registra novo ROI"""
        roi_id = self.next_roi_id
        self.rois[roi_id] = np.array(point)
        self.next_roi_id += 1
        return roi_id

def resize_frame_for_detection(frame, max_dim=640):
    """Redimensiona frame para detecção mantendo aspect ratio"""
    h, w = frame.shape[:2]
    
    if max(h, w) <= max_dim:
        return frame, 1.0
    
    scale = max_dim / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized_frame = cv2.resize(frame, (new_w, new_h))
    return resized_frame, scale

def scale_detections_back(detections, landmarks, scale):
    """Converte detecções e landmarks de volta para a escala original"""
    if scale == 1.0:
        return detections, landmarks
    
    scale_back = 1.0 / scale
    
    # Converter detecções (bounding boxes)
    if detections is not None and len(detections) > 0:
        if isinstance(detections, list):
            detections = [np.array(det) * scale_back for det in detections]
        else:
            detections = detections * scale_back
    
    # Converter landmarks
    if landmarks is not None and len(landmarks) > 0:
        if isinstance(landmarks, list):
            # landmarks é uma lista de arrays, converter cada um
            landmarks = [landmark * scale_back for landmark in landmarks]
        else:
            landmarks = landmarks * scale_back
    
    return detections, landmarks

def get_device():
    """Determina o dispositivo a ser usado (CPU ou GPU)"""
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'

def process_frame_batch(fa, frames_batch, detection_interval=5):
    """Processa um lote de frames de forma otimizada"""
    results = []
    
    for i, (frame_number, frame) in enumerate(frames_batch):
        # Detectar faces apenas em intervalos específicos para performance
        if i % detection_interval == 0:
            try:
                # Reduzir resolução para detecção mais rápida
                frame_resized, scale = resize_frame_for_detection(frame)
                
                # Detectar faces
                faces = fa.face_detector.detect_from_image(frame_resized)
                
                # Obter landmarks se faces foram detectadas
                landmarks = None
                if len(faces) > 0:
                    landmarks = fa.get_landmarks(frame_resized)
                
                # Converter de volta para escala original
                faces, landmarks = scale_detections_back(faces, landmarks, scale)
                
                results.append((frame_number, frame, faces, landmarks))
                
            except Exception as e:
                print(f"Erro na detecção do frame {frame_number}: {e}")
                # Em caso de erro, adicionar frame sem detecção
                results.append((frame_number, frame, None, None))
        else:
            # Para frames sem detecção, reutilizar detecção anterior
            results.append((frame_number, frame, None, None))
    
    return results

def convert_to_numpy_arrays(faces, landmarks):
    """Converte faces e landmarks para arrays numpy consistentes"""
    faces_np = None
    landmarks_np = None
    
    if faces is not None and len(faces) > 0:
        if isinstance(faces, list):
            faces_np = np.array(faces)
        else:
            faces_np = faces
    
    if landmarks is not None and len(landmarks) > 0:
        if isinstance(landmarks, list):
            # landmarks é uma lista de arrays, converter para array 3D
            landmarks_np = np.array(landmarks)
        else:
            landmarks_np = landmarks
    
    return faces_np, landmarks_np

def main():
    parser = argparse.ArgumentParser(description='Extract faces from videos')
    parser.add_argument('--pathBase', help='Path for videos', required=True)
    parser.add_argument('--output_size', type=int, default=160, help='Output frame size')
    parser.add_argument('--detection_interval', type=int, default=5, help='Face detection interval')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for processing')
    parser.add_argument('--max_dim', type=int, default=640, help='Maximum dimension for detection')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto', help='Device to use for processing')
    args = parser.parse_args()
    
    # Determinar dispositivo
    if args.device == 'auto':
        device = get_device()
    else:
        device = args.device
    
    print(f"Inicializando FaceAlignment com dispositivo: {device}")
    
    # Inicializar face alignment com configurações otimizadas
    try:
        fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._2D, 
            flip_input=False,
            device=device
        )
        print(f"✓ FaceAlignment inicializado com sucesso no dispositivo: {device}")
    except Exception as e:
        print(f"✗ Erro ao inicializar FaceAlignment: {e}")
        print("Tentando inicializar com CPU...")
        try:
            fa = face_alignment.FaceAlignment(
                face_alignment.LandmarksType._2D, 
                flip_input=False,
                device='cpu'
            )
            print("✓ FaceAlignment inicializado com CPU como fallback")
        except Exception as e2:
            print(f"✗ Erro crítico: Não foi possível inicializar FaceAlignment: {e2}")
            return

    videos = getFilesInPath(args.pathBase)

    for video_path in videos:
        if video_path.endswith('.txt'):
            continue

        file_video_name = os.path.splitext(video_path)[0]
        video_output_dir = file_video_name + "_videos"

        print(f"\nProcessando vídeo: {file_video_name}")

        # Limpar e criar diretório de saída
        if os.path.exists(video_output_dir):
            shutil.rmtree(video_output_dir)
        os.makedirs(video_output_dir)

        # Abrir vídeo
        vcap = cv2.VideoCapture(video_path)
        if not vcap.isOpened():
            print(f"✗ Erro ao abrir vídeo: {video_path}")
            continue

        # Obter propriedades do vídeo
        fps = int(vcap.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            fps = 30
        
        total_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📹 Propriedades do vídeo: {width}x{height} @ {fps}fps | Total de frames: {total_frames}")
        
        # Inicializar tracker e video writer
        tracker = FaceTracker(max_distance=25)
        video_manager = VideoWriterManager(
            video_output_dir, 
            fps=fps, 
            frame_size=(args.output_size, args.output_size)
        )

        frame_number = 0
        batch = []
        batch_size = args.batch_size

        # Variáveis para manter a última detecção válida
        last_valid_faces = None
        last_valid_landmarks = None

        try:
            while True:
                ret, frame = vcap.read()
                if not ret:
                    break

                # Adicionar frame ao batch
                batch.append((frame_number, frame))
                
                # Processar batch quando atingir o tamanho
                if len(batch) >= batch_size:
                    results = process_frame_batch(fa, batch, args.detection_interval)
                    
                    for frame_num, frame_data, faces, landmarks in results:
                        # Usar detecção atual ou manter a última válida
                        current_faces = faces if faces is not None else last_valid_faces
                        current_landmarks = landmarks if landmarks is not None else last_valid_landmarks
                        
                        if current_faces is not None and current_landmarks is not None and len(current_faces) > 0:
                            # Converter para arrays numpy consistentes
                            current_faces_np, current_landmarks_np = convert_to_numpy_arrays(current_faces, current_landmarks)
                            
                            if current_faces_np is not None and current_landmarks_np is not None:
                                # Atualizar última detecção válida
                                last_valid_faces = current_faces_np
                                last_valid_landmarks = current_landmarks_np
                                
                                # Encontrar ROIs correspondentes
                                if len(current_landmarks_np) > 0:
                                    # landmarks_np tem shape (num_faces, 68, 2)
                                    # Pegar o ponto 33 (nariz) para cada face
                                    landmark_points = current_landmarks_np[:, 33, :]  # Shape: (num_faces, 2)
                                    roi_indices = tracker.find_roi(landmark_points)
                                    
                                    # Processar cada face detectada
                                    for idx, (face, roi_id) in enumerate(zip(current_faces_np, roi_indices)):
                                        try:
                                            face = [int(coord) for coord in face]
                                            # Extrair ROI com margem
                                            margin_w = int((face[2] - face[0]) * 0.1)
                                            margin_h = int((face[3] - face[1]) * 0.1)
                                            
                                            x1 = max(0, face[0] - margin_w)
                                            y1 = max(0, face[1] - margin_h)
                                            x2 = min(frame_data.shape[1], face[2] + margin_w)
                                            y2 = min(frame_data.shape[0], face[3] + margin_h)
                                            
                                            face_roi = frame_data[y1:y2, x1:x2]
                                            
                                            if face_roi.size > 0:
                                                # Adicionar video writer se necessário
                                                if roi_id not in video_manager.writers:
                                                    video_manager.add_video_writer(roi_id)
                                                    print(f"➕ Novo ROI detectado: {roi_id}")
                                                
                                                # Escrever frame no vídeo correspondente
                                                video_manager.write_frame(roi_id, face_roi)
                                                
                                        except Exception as e:
                                            print(f"⚠️ Erro ao processar face {roi_id} no frame {frame_num}: {e}")
                    
                    batch = []
                
                frame_number += 1
                
                # Mostrar progresso
                if frame_number % 100 == 0:
                    progress = (frame_number / total_frames * 100) if total_frames > 0 else 0
                    print(f"📊 Progresso: {progress:.1f}% | Frames: {frame_number}/{total_frames} | ROIs: {len(video_manager.writers)}")

            # Processar batch final
            if batch:
                results = process_frame_batch(fa, batch, args.detection_interval)
                for frame_num, frame_data, faces, landmarks in results:
                    current_faces = faces if faces is not None else last_valid_faces
                    current_landmarks = landmarks if landmarks is not None else last_valid_landmarks
                    
                    if current_faces is not None and current_landmarks is not None and len(current_faces) > 0:
                        current_faces_np, current_landmarks_np = convert_to_numpy_arrays(current_faces, current_landmarks)
                        
                        if current_faces_np is not None and current_landmarks_np is not None and len(current_landmarks_np) > 0:
                            landmark_points = current_landmarks_np[:, 33, :]
                            roi_indices = tracker.find_roi(landmark_points)
                            
                            for idx, (face, roi_id) in enumerate(zip(current_faces_np, roi_indices)):
                                try:
                                    face = [int(coord) for coord in face]
                                    margin_w = int((face[2] - face[0]) * 0.1)
                                    margin_h = int((face[3] - face[1]) * 0.1)
                                    
                                    x1 = max(0, face[0] - margin_w)
                                    y1 = max(0, face[1] - margin_h)
                                    x2 = min(frame_data.shape[1], face[2] + margin_w)
                                    y2 = min(frame_data.shape[0], face[3] + margin_h)
                                    
                                    face_roi = frame_data[y1:y2, x1:x2]
                                    
                                    if face_roi.size > 0:
                                        if roi_id not in video_manager.writers:
                                            video_manager.add_video_writer(roi_id)
                                        video_manager.write_frame(roi_id, face_roi)
                                        
                                except Exception as e:
                                    print(f"⚠️ Erro ao processar face final {roi_id}: {e}")

        except KeyboardInterrupt:
            print("⏹️ Processamento interrompido pelo usuário")
        except Exception as e:
            print(f"❌ Erro durante o processamento: {e}")
        finally:
            # Liberar recursos
            vcap.release()
            video_manager.release_all()
            
            print(f"✅ Vídeo concluído: {len(video_manager.writers)} ROIs salvos em {video_output_dir}")

if __name__ == '__main__':
    main()