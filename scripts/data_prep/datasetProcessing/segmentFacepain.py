import os
import argparse
from pathlib import Path
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Segmentação e extração de faces de um dataset.")
    parser.add_argument('--input_dir', required=True, help="Diretório raiz do dataset original.")
    parser.add_argument('--output_dir', required=True, help="Diretório onde as faces recortadas serão salvas.")
    parser.add_argument('--image_size', type=int, default=256, help="Tamanho (resolução) da face recortada salva.")
    parser.add_argument('--margin', type=int, default=30, help="Margem de pixels ao redor da face para não cortar o queixo/testa.")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)

    if not input_path.exists():
        print(f"❌ Erro: O diretório de entrada '{input_path}' não existe.")
        return

    # Configura o dispositivo (GPU se disponível, senão CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Iniciando MTCNN no dispositivo: {device.type.upper()}")

    # Inicializa o detector MTCNN
    # keep_all=False: Mantém apenas a face com maior probabilidade na imagem
    # post_process=False: Não normaliza a imagem salva, mantém os pixels originais (0-255)
    mtcnn = MTCNN(
        image_size=args.image_size, 
        margin=args.margin, 
        keep_all=False, 
        select_largest=True, 
        post_process=False, 
        device=device
    )

    # Extensões válidas para buscar imagens
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    
    # Encontra todos os arquivos de imagem dentro de input_dir recursivamente
    all_image_files = [
        f for f in input_path.rglob('*') 
        if f.is_file() and f.suffix.lower() in valid_extensions
    ]
    
    print(f"🔍 Foram encontradas {len(all_image_files)} imagens para processar.")
    
    faces_encontradas = 0
    faces_nao_encontradas = 0

    # Barra de progresso para acompanhar o processamento
    for img_file in tqdm(all_image_files, desc="Processando imagens"):
        try:
            # Mantém a árvore de diretórios idêntica à original
            # Calcula o caminho relativo do arquivo em relação à pasta de entrada
            rel_path = img_file.relative_to(input_path)
            save_path = output_path / rel_path
            
            # Cria as pastas de destino se não existirem
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Carrega a imagem
            img = Image.open(img_file).convert('RGB')
            
            # O MTCNN detecta, recorta e já salva a imagem no caminho especificado
            face = mtcnn(img, save_path=str(save_path))
            
            if face is not None:
                faces_encontradas += 1
            else:
                faces_nao_encontradas += 1
                
        except Exception as e:
            # Caso a imagem esteja corrompida, apenas relata e continua
            print(f"\n⚠️ Falha ao processar {img_file.name}: {e}")
            faces_nao_encontradas += 1

    # Resumo final
    print("\n" + "="*40)
    print("📊 RESUMO DA EXTRAÇÃO DE FACES")
    print("="*40)
    print(f"Total de imagens processadas: {len(all_image_files)}")
    print(f"Faces recortadas e salvas:    {faces_encontradas}")
    print(f"Faces NÃO detectadas/Erros:   {faces_nao_encontradas}")
    print(f"Salvo em: {output_path.absolute()}")
    print("="*40)

if __name__ == "__main__":
    main()