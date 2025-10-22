
import cv2
import numpy as np
import os
import urllib.request
import ssl
from collections import defaultdict

class DNNFaceDetector:
    def __init__(self):
        self.net = None
        self.load_detector()
    
    def load_detector(self):
        """Carrega o detector DNN, baixando se necessário"""
        config_file = "deploy.prototxt"
        model_file = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
        
        # Verificar se os arquivos existem
        if not os.path.exists(config_file) or not os.path.exists(model_file):
            print("Arquivos do detector DNN não encontrados. Baixando...")
            if not self.download_dnn_files():
                print("Não foi possível baixar os arquivos do DNN. Usando Haar Cascade.")
                return None
        
        try:
            self.net = cv2.dnn.readNetFromCaffe(config_file, model_file)
            print("✓ Detector DNN carregado com sucesso!")
            return self.net
        except Exception as e:
            print(f"✗ Erro ao carregar detector DNN: {e}")
            return None
    
    def download_dnn_files(self):
        """Baixa os arquivos do detector DNN com tratamento de erro melhorado"""
        # Criar contexto SSL para evitar erros de certificado
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        files = {
            'deploy.prototxt': [
                'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt',
                'https://gist.githubusercontent.com/opencv/opencv/raw/master/samples/dnn/face_detector/deploy.prototxt'
            ],
            'res10_300x300_ssd_iter_140000_fp16.caffemodel': [
                'https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel',
                'https://storage.googleapis.com/opencv-dnn/models/face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel'
            ]
        }
        
        success = True
        
        for filename, urls in files.items():
            if os.path.exists(filename):
                print(f"✓ {filename} já existe")
                continue
                
            downloaded = False
            for url in urls:
                try:
                    print(f"Tentando baixar {filename} de: {url}")
                    # Usar contexto SSL personalizado
                    with urllib.request.urlopen(url, context=ssl_context) as response, open(filename, 'wb') as out_file:
                        data = response.read()
                        out_file.write(data)
                    print(f"✓ {filename} baixado com sucesso!")
                    downloaded = True
                    break
                except Exception as e:
                    print(f"✗ Falha no download de {url}: {e}")
                    continue
            
            if not downloaded:
                print(f"✗ Não foi possível baixar {filename} de nenhuma fonte")
                success = False
        
        return success
    
    def detect_faces(self, frame, confidence_threshold=0.7):
        """Detecta faces usando DNN"""
        if self.net is None:
            return [], []
        
        h, w = frame.shape[:2]
        
        try:
            # Criar blob da imagem
            blob = cv2.dnn.blobFromImage(
                frame, 
                1.0, 
                (300, 300), 
                [104, 117, 123], 
                swapRB=False, 
                crop=False
            )
            
            self.net.setInput(blob)
            detections = self.net.forward()
            
            faces = []
            confidences = []
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                if confidence > confidence_threshold:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x1, y1, x2, y2 = box.astype("int")
                    
                    # Garantir que as coordenadas estão dentro dos limites
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)
                    
                    width = x2 - x1
                    height = y2 - y1
                    
                    if width > 20 and height > 20:  # Filtro de tamanho mínimo
                        faces.append((x1, y1, width, height))
                        confidences.append(float(confidence))
            
            return faces, confidences
        except Exception as e:
            print(f"Erro na detecção DNN: {e}")
            return [], []

class FaceTracker:
    def __init__(self, max_disappeared=30, max_distance=100):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
    
    def register(self, centroid):
        object_id = self.next_object_id
        self.objects[object_id] = centroid
        self.disappeared[object_id] = 0
        self.next_object_id += 1
        return object_id
    
    def deregister(self, object_id):
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]
    
    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
        
        if len(self.objects) == 0:
            for rect in rects:
                centroid = self._calculate_centroid(rect)
                self.register(centroid)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            input_centroids = [self._calculate_centroid(rect) for rect in rects]
            
            D = self._calculate_distances(object_centroids, input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows, used_cols = set(), set()
            
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue
                
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)
            
            # Objetos não correspondidos
            for row in set(range(D.shape[0])).difference(used_rows):
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            # Novos objetos
            for col in set(range(D.shape[1])).difference(used_cols):
                centroid = input_centroids[col]
                self.register(centroid)
        
        return self.objects
    
    def _calculate_centroid(self, rect):
        x, y, w, h = rect
        return (int(x + w/2), int(y + h/2))
    
    def _calculate_distances(self, centroids1, centroids2):
        D = np.zeros((len(centroids1), len(centroids2)))
        for i, c1 in enumerate(centroids1):
            for j, c2 in enumerate(centroids2):
                D[i, j] = np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
        return D

def detect_faces_haar(frame):
    """Detecta faces usando Haar Cascade"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    if face_cascade.empty():
        print("Erro ao carregar classificador Haar Cascade")
        return []
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    return faces

def download_dnn_files_manual():
    """Script separado apenas para baixar os arquivos DNN"""
    print("=== DOWNLOAD MANUAL DOS ARQUIVOS DNN ===")
    
    # Criar contexto SSL para evitar erros de certificado
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    files = {
        'deploy.prototxt': [
            'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt',
            'https://gist.githubusercontent.com/opencv/opencv/raw/master/samples/dnn/face_detector/deploy.prototxt'
        ],
        'res10_300x300_ssd_iter_140000_fp16.caffemodel': [
            'https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel',
            'https://storage.googleapis.com/opencv-dnn/models/face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel'
        ]
    }
    
    for filename, urls in files.items():
        if os.path.exists(filename):
            print(f"✓ {filename} já existe")
            continue
            
        print(f"\nBaixando {filename}...")
        downloaded = False
        
        for i, url in enumerate(urls):
            try:
                print(f"Tentando fonte {i+1}: {url}")
                with urllib.request.urlopen(url, context=ssl_context) as response, open(filename, 'wb') as out_file:
                    data = response.read()
                    out_file.write(data)
                print(f"✓ {filename} baixado com sucesso!")
                downloaded = True
                break
            except Exception as e:
                print(f"✗ Falha: {e}")
                continue
        
        if not downloaded:
            print(f"✗ Não foi possível baixar {filename}")
            return False
    
    print("\n✓ Todos os arquivos foram baixados com sucesso!")
    return True

def process_video(input_video_path, output_dir='output_faces'):
    """Processa o vídeo usando detector DNN ou Haar Cascade"""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Verificar se o vídeo existe
    if not os.path.exists(input_video_path):
        print(f"Erro: Arquivo de vídeo não encontrado: {input_video_path}")
        return
    
    # Inicializar detector
    dnn_detector = DNNFaceDetector()
    use_dnn = dnn_detector.net is not None
    
    if use_dnn:
        print("Usando detector DNN (mais preciso)")
    else:
        print("Usando detector Haar Cascade")
    
    # Abrir vídeo
    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        print("Erro ao abrir o vídeo")
        return
    
    # Obter propriedades do vídeo
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 25  # Valor padrão se não for detectado
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Vídeo: {width}x{height} @ {fps}fps")
    
    # Inicializar tracker
    tracker = FaceTracker(max_disappeared=30, max_distance=150)
    video_writers = {}
    video_sizes = {}
    
    frame_count = 0
    print("Processando vídeo...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detectar faces a cada 3 frames para melhor performance
            if frame_count % 3 == 0:
                if use_dnn:
                    faces, confidences = dnn_detector.detect_faces(frame)
                else:
                    faces = detect_faces_haar(frame)
                    confidences = [1.0] * len(faces)
            else:
                faces = []
                confidences = []
            
            # Atualizar tracker
            tracked_objects = tracker.update(faces)
            
            # Processar cada face detectada
            for object_id, centroid in tracked_objects.items():
                matching_face = None
                for i, (x, y, w, h) in enumerate(faces):
                    face_centroid = (int(x + w/2), int(y + h/2))
                    distance = np.sqrt((centroid[0]-face_centroid[0])**2 + (centroid[1]-face_centroid[1])**2)
                    if distance < tracker.max_distance:
                        matching_face = (x, y, w, h)
                        break
                
                if matching_face:
                    x, y, w, h = matching_face
                    
                    # Margem adaptativa
                    margin_w = int(w * 0.4)
                    margin_h = int(h * 0.4)
                    
                    x1 = max(0, x - margin_w)
                    y1 = max(0, y - margin_h)
                    x2 = min(width, x + w + margin_w)
                    y2 = min(height, y + h + margin_h)
                    
                    face_width = x2 - x1
                    face_height = y2 - y1
                    
                    # Criar video writer se necessário
                    if object_id not in video_writers:
                        output_path = os.path.join(output_dir, f'pessoa_{object_id:03d}.avi')
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        video_width = max(face_width, 100)
                        video_height = max(face_height, 100)
                        video_writers[object_id] = cv2.VideoWriter(
                            output_path, fourcc, fps, (video_width, video_height)
                        )
                        video_sizes[object_id] = (video_width, video_height)
                    
                    # Extrair e redimensionar ROI
                    face_roi = frame[y1:y2, x1:x2]
                    if face_roi.size > 0:
                        target_size = video_sizes[object_id]
                        if face_roi.shape[1] != target_size[0] or face_roi.shape[0] != target_size[1]:
                            face_roi = cv2.resize(face_roi, target_size)
                        video_writers[object_id].write(face_roi)
                    
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Frames processados: {frame_count}, Pessoas detectadas: {len(video_writers)}")
            
                
    except KeyboardInterrupt:
        print("Processamento interrompido pelo usuário")
    except Exception as e:
        print(f"Erro durante o processamento: {e}")
    
    finally:
        # Liberar recursos
        cap.release()
        for writer in video_writers.values():
            writer.release()
        #cv2.destroyAllWindows()
        
        print(f"Processamento concluído! {len(video_writers)} pessoas detectadas.")
        print(f"Vídeos salvos em: {output_dir}")

def main():
    print("=== DETECTOR DE FACES E SEPARADOR DE VÍDEOS ===")
    
    # Opção para baixar arquivos DNN manualmente
    print("\n1. Baixar arquivos do detector DNN")
    print("2. Processar vídeo")
    escolha = input("Escolha uma opção (1 ou 2): ").strip()
    
    if escolha == "1":
        download_dnn_files_manual()
        return
    
    # Processar vídeo
    input_video = input("\nDigite o caminho do vídeo de entrada: ").strip()
    
    if not input_video:
        print("Caminho do vídeo não fornecido!")
        return
    
    process_video(input_video)

if __name__ == "__main__":
    main()