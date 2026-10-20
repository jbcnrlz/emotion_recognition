#!/usr/bin/env python3
import os
import glob
import hashlib
import argparse
import pandas as pd
import numpy as np
from PIL import Image

def compute_sha256(filepath):
    """Calcula hash SHA-256 binário exato do arquivo em blocos."""
    h = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        print(f"[ERRO] Falha ao ler {filepath}: {e}")
        return None

def get_pixel_signature(filepath, size=(64, 64)):
    """Gera assinatura de pixels em tons de cinza para pareamento perceptual."""
    try:
        with Image.open(filepath) as img:
            return np.array(img.convert('L').resize(size), dtype=np.float32)
    except Exception:
        return None

def parse_args():
    parser = argparse.ArgumentParser(
        description="Cruza imagens de avaliações com base original (ex: AffectNet) via SHA-256 e fallback perceptual."
    )
    parser.add_argument(
        "--faces-dir", 
        type=str, 
        required=True, 
        help="Caminho para o diretório local media/faces/"
    )
    parser.add_argument(
        "--affectnet-dir", 
        type=str, 
        required=True, 
        help="Caminho para o diretório de imagens originais (ex: AffectNet val_set/images/)"
    )
    parser.add_argument(
        "--ratings-csv", 
        type=str, 
        default="ratings_export_20261019_201809.csv", 
        help="Caminho do CSV de avaliações dos participantes (default: ratings_export_20261019_201809.csv)"
    )
    parser.add_argument(
        "--output-csv", 
        type=str, 
        default="mapping_ratings_affectnet.csv", 
        help="Arquivo CSV de saída com o mapeamento de-para (default: mapping_ratings_affectnet.csv)"
    )
    parser.add_argument(
        "--disable-fallback", 
        action="store_true", 
        help="Desativa o fallback perceptual (compara apenas hashes binários exatos)"
    )
    parser.add_argument(
        "--mse-threshold", 
        type=float, 
        default=1.0, 
        help="Limiar máximo de MSE para aceitar correspondência perceptual (default: 1.0)"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Validação de caminhos
    if not os.path.isdir(args.faces_dir):
        raise FileNotFoundError(f"Diretório não encontrado: {args.faces_dir}")
    if not os.path.isdir(args.affectnet_dir):
        raise FileNotFoundError(f"Diretório não encontrado: {args.affectnet_dir}")
    if not os.path.isfile(args.ratings_csv):
        raise FileNotFoundError(f"Arquivo CSV não encontrado: {args.ratings_csv}")

    print("1. Carregando dados de avaliações...")
    df_ratings = pd.read_csv(args.ratings_csv)
    unique_faces = df_ratings[['image_code', 'image_filename']].drop_duplicates()
    print(f"Total de imagens únicas a serem mapeadas: {len(unique_faces)}")

    # Indexação das imagens de ratings
    print(f"\n2. Indexando imagens em {args.faces_dir}...")
    target_hashes = {}       # sha256 -> (image_code, image_filename)
    target_signatures = {}   # filename -> pixel array

    for _, row in unique_faces.iterrows():
        fpath = os.path.join(args.faces_dir, row['image_filename'])
        if os.path.exists(fpath):
            sha = compute_sha256(fpath)
            if sha:
                target_hashes[sha] = (row['image_code'], row['image_filename'])
            if not args.disable_fallback:
                sig = get_pixel_signature(fpath)
                if sig is not None:
                    target_signatures[row['image_filename']] = sig
        else:
            print(f"[AVISO] Arquivo ausente em disco: {fpath}")

    # Varredura do dataset de referência
    print(f"\n3. Varrendo imagens em {args.affectnet_dir} via SHA-256...")
    mapped_results = []
    matched_filenames = set()

    all_dataset_files = glob.glob(os.path.join(args.affectnet_dir, '*.jpg')) + \
                        glob.glob(os.path.join(args.affectnet_dir, '*.png'))
    print(f"Total de imagens encontradas na pasta de referência: {len(all_dataset_files)}")

    for aff_path in all_dataset_files:
        aff_name = os.path.basename(aff_path)
        aff_sha = compute_sha256(aff_path)

        if aff_sha in target_hashes:
            img_code, img_fname = target_hashes[aff_sha]
            mapped_results.append({
                'image_code': img_code,
                'image_filename': img_fname,
                'affectnet_file': aff_name,
                'affectnet_full_path': aff_path,
                'match_method': 'exact_sha256',
                'confidence': 1.0
            })
            matched_filenames.add(img_fname)
            print(f"  [MATCH SHA-256] {img_fname} <--> {aff_name}")

        if len(matched_filenames) == len(target_hashes):
            print("\nTodas as imagens foram pareadas com sucesso por SHA-256!")
            break

    # Fallback perceptual
    unmatched = set(target_signatures.keys()) - matched_filenames
    if unmatched and not args.disable_fallback:
        print(f"\n4. Executando fallback perceptual (MSE <= {args.mse_threshold}) para {len(unmatched)} arquivo(s)...")
        for aff_path in all_dataset_files:
            aff_name = os.path.basename(aff_path)
            aff_sig = get_pixel_signature(aff_path)
            if aff_sig is None:
                continue

            for img_fname in list(unmatched):
                target_sig = target_signatures[img_fname]
                mse = np.mean((aff_sig - target_sig) ** 2)
                
                if mse <= args.mse_threshold:
                    img_code = unique_faces.loc[unique_faces['image_filename'] == img_fname, 'image_code'].iloc[0]
                    mapped_results.append({
                        'image_code': img_code,
                        'image_filename': img_fname,
                        'affectnet_file': aff_name,
                        'affectnet_full_path': aff_path,
                        'match_method': 'perceptual_mse',
                        'confidence': float(1.0 / (1.0 + mse))
                    })
                    print(f"  [MATCH MSE={mse:.4f}] {img_fname} <--> {aff_name}")
                    unmatched.remove(img_fname)

            if not unmatched:
                break

    # Salva os resultados
    df_result = pd.DataFrame(mapped_results)
    df_result.to_csv(args.output_csv, index=False)
    print(f"\nProcesso concluído! Mapeamento exportado para: {args.output_csv}")
    print(f"Total pareado: {len(df_result)} de {len(unique_faces)}")

if __name__ == '__main__':
    main()