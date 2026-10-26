#!/usr/bin/env python3
"""
match_and_map.py
----------------
Identifica diretamente no dataset quais faces fizeram parte do user study
a partir dos arquivos de imagem físicos presentes no diretório do estudo.

Princípio Central:
- As faces do user study são detectadas DIRETAMENTE do diretório de imagens (--faces-dir).
- O arquivo .csv de ratings NÃO é usado para detectar faces (é puramente opcional para enriquecer metadados).
- As imagens do dataset de referência (--dataset-dir) são varridas recursivamente.
- O pareamento cruza as imagens físicas do estudo com as imagens do dataset através de pipeline multi-tier.

Estratégias de Pareamento (Multi-Tier Matching):
1. Exact Filename: correspondência exata de nome de arquivo (ex: 1086.jpg <-> 1086.jpg).
2. Stem Match: correspondência do radical sem extensão (ex: 1086.png <-> 1086.jpg ou 1086 <-> 1086.jpg).
3. Exact SHA-256: arquivos idênticos bit a bit caso tenham sido renomeados.
4. Perceptual dHash (64-bit Difference Hash): resistente a recompressão JPEG web, redimensionamento e conversão de formato.
5. Pixel Signature / MSE: fallback de erro quadrático médio em assinaturas normalizadas.

Recursos:
- Varredura recursiva completa de pastas e subpastas em ambos os diretórios.
- Suporte a múltiplas extensões (.jpg, .jpeg, .png, .bmp, .webp, .tiff, case-insensitive).
- Geração do arquivo de mapeamento de-para (--output-csv).
- Sinalização / tagging opcional no CSV do dataset com a coluna 'in_user_study' (--tag-dataset-csv).
- Exportação de subconjunto exclusivo (--export-subset-csv).
- Cópia ou criação de symlinks das faces identificadas no dataset (--copy-matches-to / --symlink-matches-to).
"""

import os
import sys
import glob
import hashlib
import argparse
import shutil
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
from PIL import Image

# Extensões de imagem suportadas (case-insensitive)
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}


def compute_sha256(filepath):
    """Calcula hash SHA-256 binário exato do arquivo em blocos."""
    h = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def compute_dhash(img_or_path, hash_size=8):
    """
    Calcula o Difference Hash (dHash) de 64 bits.
    Rápido e resistente a recompressões JPEG, artefatos web e redimensionamentos.
    """
    try:
        if isinstance(img_or_path, (str, Path)):
            with Image.open(img_or_path) as img:
                img_gray = img.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
                pixels = np.array(img_gray, dtype=np.int32)
        else:
            img_gray = img_or_path.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
            pixels = np.array(img_gray, dtype=np.int32)

        # Compara pixels horizontais adjacentes
        diff = pixels[:, 1:] > pixels[:, :-1]
        val = 0
        for bit in diff.flatten():
            val = (val << 1) | int(bit)
        return val
    except Exception:
        return None


def hamming_distance(h1, h2):
    """Calcula a distância de Hamming entre dois hashes inteiros de 64 bits."""
    if h1 is None or h2 is None:
        return 999
    return bin(h1 ^ h2).count('1')


def get_pixel_signature(filepath, size=(32, 32)):
    """Gera assinatura normalizada de pixels em tons de cinza para fallback de MSE."""
    try:
        with Image.open(filepath) as img:
            arr = np.array(img.convert('L').resize(size), dtype=np.float32)
            std = np.std(arr)
            if std > 1e-4:
                return (arr - np.mean(arr)) / std
            return arr - np.mean(arr)
    except Exception:
        return None


def scan_images_recursively(base_dir):
    """Varre recursivamente todas as imagens em base_dir e retorna uma lista de caminhos absolutos."""
    base_path = Path(base_dir).resolve()
    images = []
    if not base_path.exists():
        return images

    for root, _, files in os.walk(base_path):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in VALID_EXTENSIONS:
                images.append(os.path.join(root, f))
    return sorted(images)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Identifica no dataset quais faces físicas do diretório do user study fizeram parte do estudo."
    )
    parser.add_argument(
        "--faces-dir", "--study-dir",
        dest="faces_dir",
        type=str,
        required=True,
        help="Diretório onde estão as imagens de faces do user study (busca recursiva em subpastas)."
    )
    parser.add_argument(
        "--dataset-dir", "--affectnet-dir",
        dest="dataset_dir",
        type=str,
        required=True,
        help="Diretório raiz do dataset de referência (ex: AffectNetFullMerge/val_set/ ou val_set/images/)."
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="mapping_ratings_affectnet.csv",
        help="Arquivo CSV de saída com o mapeamento das faces (default: mapping_ratings_affectnet.csv)."
    )
    parser.add_argument(
        "--ratings-csv",
        type=str,
        default=None,
        help="CSV opcional de avaliações apenas para enriquecer os metadados com códigos/ratings (NÃO usado para detectar faces)."
    )
    parser.add_argument(
        "--tag-dataset-csv",
        type=str,
        default=None,
        help="CSV opcional de anotações do dataset para adicionar a coluna booleana 'in_user_study'."
    )
    parser.add_argument(
        "--tagged-output-csv",
        type=str,
        default=None,
        help="Caminho para salvar o CSV do dataset anotado com as tags de user study."
    )
    parser.add_argument(
        "--export-subset-csv",
        type=str,
        default=None,
        help="Exporta um CSV contendo apenas as instâncias do dataset identificadas no user study."
    )
    parser.add_argument(
        "--copy-matches-to",
        type=str,
        default=None,
        help="Diretório opcional para copiar as faces do dataset identificadas no estudo."
    )
    parser.add_argument(
        "--symlink-matches-to",
        type=str,
        default=None,
        help="Diretório opcional para criar links simbólicos (symlinks) das faces identificadas no dataset."
    )
    parser.add_argument(
        "--hamming-threshold",
        type=int,
        default=6,
        help="Limiar máximo de distância de Hamming para aceitar dHash perceptual (padrão: 6 de 64 bits)."
    )
    parser.add_argument(
        "--disable-fallback",
        action="store_true",
        help="Desativa o pareamento perceptual (compara apenas nomes de arquivo e hashes exatos)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("================================================================================")
    print("      IDENTIFICADOR E MAPEADOR DE FACES DO USER STUDY NO DATASET               ")
    print("================================================================================")

    # 1. Validar diretório de faces do estudo
    faces_dir = os.path.abspath(args.faces_dir)
    if not os.path.isdir(faces_dir):
        raise FileNotFoundError(f"[ERRO] Diretório de faces do estudo não encontrado: {faces_dir}")

    # 2. Validar diretório do dataset
    dataset_dir = os.path.abspath(args.dataset_dir)
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"[ERRO] Diretório do dataset não encontrado: {dataset_dir}")

    print(f"[*] Diretório de faces do estudo: {faces_dir}")
    print(f"[*] Diretório do dataset de referência: {dataset_dir}")

    # 3. Detectar faces do estudo DIRETAMENTE a partir dos arquivos físicos no diretório
    print("\n[Etapa 1/4] Varrendo imagens físicas no diretório do user study...")
    study_image_paths = scan_images_recursively(faces_dir)
    print(f"[*] Total de arquivos de imagem encontrados no diretório do estudo: {len(study_image_paths)}")

    if not study_image_paths:
        raise FileNotFoundError(f"[ERRO] Nenhuma imagem com extensões {VALID_EXTENSIONS} foi encontrada em: {faces_dir}")

    target_faces = []
    for fpath in study_image_paths:
        bname = os.path.basename(fpath)
        stem = os.path.splitext(bname)[0]
        rel_path = os.path.relpath(fpath, faces_dir)

        # Detecta se está organizado em subpasta de emoção (ex: Happy/1086.jpg)
        parts = Path(rel_path).parts
        parent_name = parts[-2] if len(parts) > 1 else ""
        emotion = parent_name if parent_name.lower() in [
            'happy', 'sad', 'anger', 'angry', 'neutral', 'surprise', 'surprised',
            'fear', 'fearful', 'contempt', 'disgust', 'disgusted'
        ] else None

        target_faces.append({
            'image_code': stem,
            'image_filename': bname,
            'user_study_rel_path': rel_path,
            'user_study_full_path': fpath,
            'stem': stem,
            'emotion': emotion
        })

    # Opcional: Se fornecido ratings_csv, enriquecer com códigos ou ratings sem filtrar a lista de faces
    if args.ratings_csv and os.path.isfile(args.ratings_csv):
        print(f"[*] Enriquecendo metadados a partir de {args.ratings_csv} (modo opcional)...")
        try:
            df_r = pd.read_csv(args.ratings_csv)
            # Mapa de filename -> image_code se existir coluna específica
            code_col = next((c for c in ['image_code', 'code', 'face_id', 'id'] if c in df_r.columns), None)
            fname_col = next((c for c in ['image_filename', 'filename', 'image', 'file'] if c in df_r.columns), None)
            if code_col and fname_col:
                name_to_code = {}
                for _, r in df_r[[fname_col, code_col]].drop_duplicates().iterrows():
                    fn = os.path.basename(str(r[fname_col]).strip().replace('\\', '/'))
                    name_to_code[fn] = str(r[code_col]).strip()
                    name_to_code[os.path.splitext(fn)[0]] = str(r[code_col]).strip()

                for tf in target_faces:
                    if tf['image_filename'] in name_to_code:
                        tf['image_code'] = name_to_code[tf['image_filename']]
                    elif tf['stem'] in name_to_code:
                        tf['image_code'] = name_to_code[tf['stem']]
                print("    Metadados de códigos enriquecidos com sucesso.")
        except Exception as e:
            print(f"    [!] Aviso ao ler ratings CSV: {e}. Prosseguindo normalmente com as faces físicas.")

    print(f"[*] Total de faces do estudo a identificar no dataset: {len(target_faces)}")

    # 4. Varredura recursiva do dataset de referência
    print("\n[Etapa 2/4] Varrendo imagens no dataset de referência...")
    dataset_files = scan_images_recursively(dataset_dir)
    print(f"[*] Total de imagens encontradas no dataset: {len(dataset_files)}")

    if not dataset_files:
        raise FileNotFoundError(f"[ERRO] Nenhuma imagem com extensões {VALID_EXTENSIONS} foi encontrada em {dataset_dir}")

    # Indexação rápida do dataset por nome e radical
    print("[*] Indexando nomes e radicais do dataset...")
    dataset_by_name = defaultdict(list)
    dataset_by_name_lower = defaultdict(list)
    dataset_by_stem = defaultdict(list)
    dataset_by_stem_lower = defaultdict(list)

    for d_path in dataset_files:
        bname = os.path.basename(d_path)
        stem = os.path.splitext(bname)[0]
        dataset_by_name[bname].append(d_path)
        dataset_by_name_lower[bname.lower()].append(d_path)
        dataset_by_stem[stem].append(d_path)
        dataset_by_stem_lower[stem.lower()].append(d_path)

    # Estruturas para resultados
    matched_results = []
    matched_target_indices = set()
    matched_dataset_paths = set()

    # --------------------------------------------------------------------------
    # FASE 1: Pareamento por Nome Exato ou Stem (Instantâneo)
    # --------------------------------------------------------------------------
    print("\n[Etapa 3/4 - Fase 1] Pareamento por Nome de Arquivo e Stem...")
    for idx, tf in enumerate(target_faces):
        fname = tf['image_filename']
        fname_lower = fname.lower()
        stem = tf['stem']
        stem_lower = stem.lower()

        matched_path = None
        method = None

        if fname in dataset_by_name:
            matched_path = dataset_by_name[fname][0]
            method = 'exact_filename'
        elif fname_lower in dataset_by_name_lower:
            matched_path = dataset_by_name_lower[fname_lower][0]
            method = 'exact_filename_case_insensitive'
        elif stem in dataset_by_stem:
            matched_path = dataset_by_stem[stem][0]
            method = 'stem_match'
        elif stem_lower in dataset_by_stem_lower:
            matched_path = dataset_by_stem_lower[stem_lower][0]
            method = 'stem_match_case_insensitive'

        if matched_path:
            rel_path = os.path.relpath(matched_path, dataset_dir)
            matched_results.append({
                'image_code': tf['image_code'],
                'image_filename': tf['image_filename'],
                'user_study_rel_path': tf['user_study_rel_path'],
                'user_study_full_path': tf['user_study_full_path'],
                'dataset_file': os.path.basename(matched_path),
                'affectnet_file': os.path.basename(matched_path),  # alias para retrocompatibilidade
                'dataset_relative_path': rel_path,
                'dataset_full_path': matched_path,
                'match_method': method,
                'confidence': 1.0,
                'user_study_emotion': tf.get('emotion')
            })
            matched_target_indices.add(idx)
            matched_dataset_paths.add(matched_path)

    print(f"  -> {len(matched_target_indices)} / {len(target_faces)} faces pareadas por Nome/Stem.")

    # --------------------------------------------------------------------------
    # FASE 2: Pareamento por SHA-256 Binário (para arquivos com nomes trocados)
    # --------------------------------------------------------------------------
    unmatched_indices = [i for i in range(len(target_faces)) if i not in matched_target_indices]

    if unmatched_indices:
        print(f"\n[Etapa 3/4 - Fase 2] Calculando SHA-256 para os {len(unmatched_indices)} alvos restantes...")
        target_sha_map = {}
        for i in unmatched_indices:
            tf = target_faces[i]
            sha = compute_sha256(tf['user_study_full_path'])
            if sha:
                target_sha_map[sha] = (i, tf)

        if target_sha_map:
            print(f"  Varrendo dataset via SHA-256...")
            for d_path in dataset_files:
                if d_path in matched_dataset_paths:
                    continue

                d_sha = compute_sha256(d_path)
                if d_sha and d_sha in target_sha_map:
                    idx, tf = target_sha_map[d_sha]
                    if idx not in matched_target_indices:
                        rel_path = os.path.relpath(d_path, dataset_dir)
                        matched_results.append({
                            'image_code': tf['image_code'],
                            'image_filename': tf['image_filename'],
                            'user_study_rel_path': tf['user_study_rel_path'],
                            'user_study_full_path': tf['user_study_full_path'],
                            'dataset_file': os.path.basename(d_path),
                            'affectnet_file': os.path.basename(d_path),
                            'dataset_relative_path': rel_path,
                            'dataset_full_path': d_path,
                            'match_method': 'exact_sha256',
                            'confidence': 1.0,
                            'user_study_emotion': tf.get('emotion')
                        })
                        matched_target_indices.add(idx)
                        matched_dataset_paths.add(d_path)
                        print(f"  [MATCH SHA-256] {tf['image_filename']} <--> {os.path.basename(d_path)}")

                if len(matched_target_indices) == len(target_faces):
                    break

        print(f"  -> Total acumulado: {len(matched_target_indices)} / {len(target_faces)} faces pareadas.")

    # --------------------------------------------------------------------------
    # FASE 3: Pareamento Perceptual via dHash de 64 bits (para recompressão/redimensionamento)
    # --------------------------------------------------------------------------
    unmatched_indices = [i for i in range(len(target_faces)) if i not in matched_target_indices]

    if unmatched_indices and not args.disable_fallback:
        print(f"\n[Etapa 3/4 - Fase 3] Pareamento Perceptual (dHash <= {args.hamming_threshold}) para {len(unmatched_indices)} alvos restantes...")

        target_dhash_list = []
        for i in unmatched_indices:
            tf = target_faces[i]
            dh = compute_dhash(tf['user_study_full_path'])
            if dh is not None:
                target_dhash_list.append((i, tf, dh))

        if target_dhash_list:
            processed_count = 0
            total_remaining_ds = len(dataset_files) - len(matched_dataset_paths)

            for d_path in dataset_files:
                if d_path in matched_dataset_paths:
                    continue

                processed_count += 1
                if processed_count % 5000 == 0:
                    print(f"  Progresso: {processed_count}/{total_remaining_ds} imagens varridas ({len(matched_target_indices)} pareadas)...")

                d_hash = compute_dhash(d_path)
                if d_hash is None:
                    continue

                for item in list(target_dhash_list):
                    idx, tf, t_dh = item
                    if idx in matched_target_indices:
                        continue

                    dist = hamming_distance(t_dh, d_hash)
                    if dist <= args.hamming_threshold:
                        rel_path = os.path.relpath(d_path, dataset_dir)
                        confidence = float(1.0 - (dist / 64.0))
                        matched_results.append({
                            'image_code': tf['image_code'],
                            'image_filename': tf['image_filename'],
                            'user_study_rel_path': tf['user_study_rel_path'],
                            'user_study_full_path': tf['user_study_full_path'],
                            'dataset_file': os.path.basename(d_path),
                            'affectnet_file': os.path.basename(d_path),
                            'dataset_relative_path': rel_path,
                            'dataset_full_path': d_path,
                            'match_method': f'perceptual_dhash (dist={dist})',
                            'confidence': round(confidence, 4),
                            'user_study_emotion': tf.get('emotion')
                        })
                        matched_target_indices.add(idx)
                        matched_dataset_paths.add(d_path)
                        print(f"  [MATCH PERCEPTUAL dHash dist={dist}] {tf['image_filename']} <--> {os.path.basename(d_path)}")
                        target_dhash_list.remove(item)
                        break

                if len(matched_target_indices) == len(target_faces):
                    print("  Todas as imagens foram pareadas!")
                    break

        print(f"  -> Total acumulado: {len(matched_target_indices)} / {len(target_faces)} faces pareadas.")

    # --------------------------------------------------------------------------
    # 5. Salvar Mapeamento Principal
    # --------------------------------------------------------------------------
    print("\n[Etapa 4/4] Gravando resultados e identificadores...")
    df_mapped = pd.DataFrame(matched_results)
    output_csv = os.path.abspath(args.output_csv)
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    df_mapped.to_csv(output_csv, index=False)
    print(f"\n[✔] Mapeamento exportado com sucesso para: {output_csv}")
    print(f"    Total de faces do estudo: {len(target_faces)}")
    print(f"    Faces identificadas no dataset: {len(df_mapped)} ({len(df_mapped)/len(target_faces)*100:.1f}%)")

    # Listar eventuais não pareadas
    unmatched_final = [tf['image_filename'] for i, tf in enumerate(target_faces) if i not in matched_target_indices]
    if unmatched_final:
        print(f"\n[!] Faces não encontradas no dataset ({len(unmatched_final)}):")
        for u in unmatched_final[:10]:
            print(f"    - {u}")
        if len(unmatched_final) > 10:
            print(f"    ... e mais {len(unmatched_final) - 10} arquivo(s).")

    # --------------------------------------------------------------------------
    # 6. Tagging no CSV do Dataset (Se solicitado via --tag-dataset-csv)
    # --------------------------------------------------------------------------
    if args.tag_dataset_csv and os.path.isfile(args.tag_dataset_csv):
        print(f"\n[*] Marcando 'in_user_study' no CSV do dataset: {args.tag_dataset_csv}...")
        df_ds = pd.read_csv(args.tag_dataset_csv)

        # Conjunto de nomes de arquivo correspondentes no dataset
        matched_files_set = set(df_mapped['dataset_file'].tolist()) | set(df_mapped['dataset_relative_path'].tolist())
        matched_to_study_name = dict(zip(df_mapped['dataset_file'], df_mapped['image_filename']))

        # Detectar coluna de arquivo no CSV do dataset
        target_col = None
        for col in ['file', 'path', 'image', 'affectnet_file', 'filename', 'image_path']:
            if col in df_ds.columns:
                target_col = col
                break

        if target_col:
            df_ds['in_user_study'] = df_ds[target_col].apply(lambda x: os.path.basename(str(x)) in matched_files_set)
            df_ds['user_study_filename'] = df_ds[target_col].apply(lambda x: matched_to_study_name.get(os.path.basename(str(x)), ""))

            tagged_out = args.tagged_output_csv or args.tag_dataset_csv.replace(".csv", "_with_user_study.csv")
            df_ds.to_csv(tagged_out, index=False)
            total_flagged = df_ds['in_user_study'].sum()
            print(f"[✔] Dataset anotado exportado para: {tagged_out}")
            print(f"    Linhas marcadas como 'in_user_study = True': {total_flagged}")
        else:
            print(f"[!] Coluna de arquivo não encontrada em {args.tag_dataset_csv}. Colunas disponíveis: {df_ds.columns.tolist()}")

    # --------------------------------------------------------------------------
    # 7. Exportar apenas o subset identificado (Se solicitado via --export-subset-csv)
    # --------------------------------------------------------------------------
    if args.export_subset_csv:
        subset_out = os.path.abspath(args.export_subset_csv)
        df_mapped.to_csv(subset_out, index=False)
        print(f"[✔] Subset exclusivo exportado para: {subset_out}")

    # --------------------------------------------------------------------------
    # 8. Copiar / Symlink das faces identificadas no dataset
    # --------------------------------------------------------------------------
    copy_dir = args.copy_matches_to
    symlink_dir = args.symlink_matches_to

    if copy_dir or symlink_dir:
        dest_dir = os.path.abspath(copy_dir or symlink_dir)
        is_symlink = bool(symlink_dir)
        os.makedirs(dest_dir, exist_ok=True)
        action_name = "Criando links simbólicos (symlinks)" if is_symlink else "Copiando imagens"
        print(f"\n[*] {action_name} das faces identificadas no dataset para: {dest_dir}...")

        count_copied = 0
        for res in matched_results:
            src_file = res['dataset_full_path']
            if os.path.isfile(src_file):
                dst_file = os.path.join(dest_dir, res['dataset_file'])
                try:
                    if is_symlink:
                        if os.path.lexists(dst_file):
                            os.remove(dst_file)
                        os.symlink(src_file, dst_file)
                    else:
                        shutil.copy2(src_file, dst_file)
                    count_copied += 1
                except Exception as e:
                    print(f"[!] Erro ao processar {src_file}: {e}")

        print(f"[✔] {count_copied} arquivos processados com sucesso em: {dest_dir}")

    print("\n================================================================================")
    print("                      PROCESSAMENTO CONCLUÍDO COM SUCESSO                       ")
    print("================================================================================")


if __name__ == '__main__':
    main()
