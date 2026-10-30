#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para compilar anotações de probabilidades universais de um dataset em um arquivo CSV,
no mesmo formato do train_set_emotion_distribution_LIKELIHOOD_OLD.csv.

Filtra estritamente os arquivos com padrão: {idImagem}_prob_rank_universal.txt
onde idImagem é o identificador da imagem correspondente no dataset (ex: 9 -> 9.jpg).
"""

import os
import sys
import csv
import re
import argparse
from typing import Dict, List, Optional, Tuple, Any

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable
    HAS_TQDM = False

# Ordem padrão das colunas de probabilidades conforme universal_distros.CSV + neutral
DEFAULT_EMOTION_COLUMNS = [
    'happy', 'sad', 'surprised', 'fearful', 'disgusted', 'angry', 'contempt', 'neutral'
]

# Mapeamento oficial de labels AffectNet (0-7 e 255)
AFFECTNET_EMOTIONS = [
    "neutral", "happy", "sad", "surprised", "fear", "disgust", "angry", "contempt"
]

# Cabeçalho padrão completo exatamente como train_set_emotion_distribution_LIKELIHOOD_OLD.csv
CSV_HEADER = DEFAULT_EMOTION_COLUMNS + ['valence', 'arousal', 'dominance', 'emotion', 'path']


def natural_sort_key(s: str) -> List[Any]:
    """Chave para ordenação natural de strings (ex: '2' antes de '10')."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def parse_probs_file(filepath: str) -> Optional[List[float]]:
    """Lê arquivo de probabilidades separado por vírgula ou espaço."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            line = f.readline().strip()
            if not line:
                return None
            parts = line.replace(',', ' ').split()
            return [float(x) for x in parts]
    except Exception as e:
        return None


def load_float_from_file(base_dir: str, image_id: str, suffixes: List[str]) -> Optional[float]:
    """Tenta carregar um float a partir de arquivos .npy ou .txt."""
    for s in suffixes:
        npy_path = os.path.join(base_dir, f"{image_id}{s}.npy")
        if os.path.exists(npy_path) and HAS_NUMPY:
            try:
                val = np.load(npy_path)
                return float(val)
            except Exception:
                pass
                
        txt_path = os.path.join(base_dir, f"{image_id}{s}.txt")
        if os.path.exists(txt_path):
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    return float(f.readline().strip())
            except Exception:
                pass
    return None


def load_exp_label(base_dir: str, image_id: str) -> Optional[str]:
    """Carrega o label de expressão a partir de {image_id}_exp.npy ou .txt."""
    suffixes = ['_exp', '_emotion']
    for s in suffixes:
        npy_path = os.path.join(base_dir, f"{image_id}{s}.npy")
        if os.path.exists(npy_path) and HAS_NUMPY:
            try:
                val = int(np.load(npy_path))
                if 0 <= val < len(AFFECTNET_EMOTIONS):
                    return AFFECTNET_EMOTIONS[val]
                elif val == 255:
                    return 'affwild'
                return str(val)
            except Exception:
                pass

        txt_path = os.path.join(base_dir, f"{image_id}{s}.txt")
        if os.path.exists(txt_path):
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    content = f.readline().strip()
                    if content.isdigit():
                        val = int(content)
                        if 0 <= val < len(AFFECTNET_EMOTIONS):
                            return AFFECTNET_EMOTIONS[val]
                        elif val == 255:
                            return 'affwild'
                        return str(val)
                    elif content:
                        return content
            except Exception:
                pass
    return None


def resolve_directories(input_path: str, user_images_dir: Optional[str] = None) -> Tuple[str, str]:
    """
    Resolve e valida os diretórios de anotações e imagens.
    Suporta:
    - Passar a pasta raiz do split (ex: train_set), contendo annotations/ e images/
    - Passar a pasta annotations diretamente (ex: train_set/annotations)
    """
    abs_input = os.path.abspath(input_path)
    
    # 1. Resolver pasta de anotações
    if os.path.isdir(os.path.join(abs_input, 'annotations')):
        annotations_dir = os.path.join(abs_input, 'annotations')
        parent_dir = abs_input
    else:
        annotations_dir = abs_input
        parent_dir = os.path.dirname(abs_input)

    # 2. Resolver pasta de imagens
    if user_images_dir:
        images_dir = os.path.abspath(user_images_dir)
    else:
        # Tenta candidatos padrões
        candidate_images = [
            os.path.join(parent_dir, 'images'),
            os.path.join(annotations_dir, 'images'),
            annotations_dir
        ]
        images_dir = candidate_images[0]
        for c in candidate_images:
            if os.path.isdir(c) and c != annotations_dir:
                images_dir = c
                break

    return annotations_dir, images_dir


def find_image_path(images_dir: str, image_id: str, default_ext: str = '.jpg', 
                    path_prefix: Optional[str] = None) -> str:
    """Retorna o caminho completo da imagem."""
    # Se path_prefix customizado foi fornecido, usa-o diretamente
    if path_prefix:
        return os.path.join(path_prefix, f"{image_id}{default_ext}")

    # Se a pasta de imagens existe fisicamente, busca se o arquivo já tem extensão
    if os.path.isdir(images_dir):
        candidate_exts = [default_ext, '.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.bmp']
        for ext in candidate_exts:
            full_path = os.path.join(images_dir, f"{image_id}{ext}")
            if os.path.exists(full_path):
                return full_path

    # Fallback padrão
    return os.path.join(images_dir, f"{image_id}{default_ext}")


def load_metadata_csv(csv_path: str) -> Dict[str, Dict[str, Any]]:
    """Carrega metadados pré-existentes de um CSV indexado por id/basename."""
    metadata = {}
    if not csv_path or not os.path.exists(csv_path):
        return metadata

    print(f"[*] Carregando metadados de apoio de '{csv_path}'...")
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Tenta obter a chave do id a partir de 'path' ou coluna de id
                raw_path = row.get('path', '')
                if raw_path:
                    base_id = os.path.splitext(os.path.basename(raw_path))[0]
                else:
                    base_id = row.get('id', row.get('image_id', ''))
                
                if base_id:
                    metadata[base_id] = {
                        'valence': row.get('valence'),
                        'arousal': row.get('arousal'),
                        'dominance': row.get('dominance'),
                        'emotion': row.get('emotion')
                    }
    except Exception as e:
        print(f"[!] Aviso: Erro ao carregar metadata CSV: {e}")
    return metadata


def compile_annotations(
    annotations_dir: str,
    images_dir: str,
    output_csv: str,
    metadata_csv: Optional[str] = None,
    default_ext: str = '.jpg',
    path_prefix: Optional[str] = None,
    fallback_emotion: str = 'argmax',
    sort_output: bool = True
) -> int:
    """
    Compila todas as anotações {idImagem}_prob_rank_universal.txt em um arquivo CSV.
    """
    if not os.path.isdir(annotations_dir):
        raise FileNotFoundError(f"Diretório de anotações não encontrado: {annotations_dir}")

    print("=" * 70)
    print(" Compilação de Anotações Universais -> CSV")
    print("=" * 70)
    print(f"[*] Diretório de Anotações : {annotations_dir}")
    print(f"[*] Diretório de Imagens   : {images_dir}")
    print(f"[*] Arquivo de Saída       : {output_csv}")
    if path_prefix:
        print(f"[*] Prefixo do Caminho     : {path_prefix}")
    print("=" * 70)

    # 1. Carregar CSV de metadados se fornecido
    metadata_cache = load_metadata_csv(metadata_csv) if metadata_csv else {}

    # 2. Varrer arquivos com sufixo _prob_rank_universal.txt
    suffix = '_prob_rank_universal.txt'
    matched_entries = []
    
    print("[*] Varrendo diretório em busca de arquivos '*_prob_rank_universal.txt'...")
    with os.scandir(annotations_dir) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith(suffix):
                # Extrai o id da imagem exatamente antes de _prob_rank_universal.txt
                image_id = entry.name[:-len(suffix)]
                matched_entries.append((image_id, entry.path))

    total_files = len(matched_entries)
    if total_files == 0:
        print(f"[!] Nenhum arquivo correspondente a '*{suffix}' encontrado em: {annotations_dir}")
        return 0

    print(f"[*] Total de arquivos identificados: {total_files}")

    # Ordenar se solicitado
    if sort_output:
        matched_entries.sort(key=lambda x: natural_sort_key(x[0]))

    # Criar diretório de saída caso não exista
    out_dir = os.path.dirname(os.path.abspath(output_csv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 3. Processar cada arquivo e gravar no CSV
    written_count = 0
    with open(output_csv, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(CSV_HEADER)

        iterator = tqdm(matched_entries, desc="Compilando anotações") if HAS_TQDM else matched_entries

        for image_id, prob_file_path in iterator:
            probs = parse_probs_file(prob_file_path)
            if probs is None:
                continue

            # Se houver menos ou mais de 8 probabilidades, preenche ou ajusta
            if len(probs) < len(DEFAULT_EMOTION_COLUMNS):
                probs = probs + [0.0] * (len(DEFAULT_EMOTION_COLUMNS) - len(probs))
            elif len(probs) > len(DEFAULT_EMOTION_COLUMNS):
                probs = probs[:len(DEFAULT_EMOTION_COLUMNS)]

            # Obter metadados VAD e Emotion
            meta = metadata_cache.get(image_id, {})

            # Valence
            val = meta.get('valence')
            if val is None or val == "":
                val = load_float_from_file(annotations_dir, image_id, ['_val', '_adjusted_val'])
            val_str = f"{val}" if val is not None else ""

            # Arousal
            aro = meta.get('arousal')
            if aro is None or aro == "":
                aro = load_float_from_file(annotations_dir, image_id, ['_aro', '_adjusted_aro'])
            aro_str = f"{aro}" if aro is not None else ""

            # Dominance
            dom = meta.get('dominance')
            if dom is None or dom == "":
                dom = load_float_from_file(annotations_dir, image_id, ['_dom'])
            dom_str = f"{dom}" if dom is not None else ""

            # Emotion
            emotion = meta.get('emotion')
            if not emotion:
                emotion = load_exp_label(annotations_dir, image_id)
            if not emotion:
                if fallback_emotion == 'argmax' and probs:
                    max_idx = int(np.argmax(probs)) if HAS_NUMPY else probs.index(max(probs))
                    emotion = DEFAULT_EMOTION_COLUMNS[max_idx]
                elif fallback_emotion == 'unknown':
                    emotion = 'unknown'
                else:
                    emotion = ''

            # Caminho da imagem no dataset
            img_path = find_image_path(images_dir, image_id, default_ext=default_ext, path_prefix=path_prefix)

            # Gravar linha no formato exato
            row = probs + [val_str, aro_str, dom_str, emotion, img_path]
            writer.writerow(row)
            written_count += 1

    print("\n" + "=" * 70)
    print(f"[+] Sucesso! Arquivo compilado com sucesso: {output_csv}")
    print(f"[+] Linhas escritas: {written_count} / {total_files}")
    print("=" * 70)
    return written_count


def main():
    parser = argparse.ArgumentParser(
        description="Compila arquivos {idImagem}_prob_rank_universal.txt em um CSV no formato de train_set_emotion_distribution_LIKELIHOOD_OLD.csv"
    )
    parser.add_argument(
        'path', 
        nargs='?', 
        default=None, 
        help="Diretório de anotações ou raiz do split (ex: /caminho/train_set ou /caminho/train_set/annotations)"
    )
    parser.add_argument(
        '--annotationsDir', '-d', 
        dest='annotations_dir',
        default=None, 
        help="Diretório contendo os arquivos de anotação (alternativa ao argumento posicional)"
    )
    parser.add_argument(
        '--imagesDir', '-i', 
        default=None, 
        help="Diretório onde as imagens estão armazenadas (padrão: auto-detectado em ../images)"
    )
    parser.add_argument(
        '--outputFile', '-o', 
        default=None, 
        help="Caminho do arquivo CSV de saída (padrão: train_set_emotion_distribution_LIKELIHOOD_OLD.csv ou baseado no nome do split)"
    )
    parser.add_argument(
        '--metadataCSV', 
        default=None, 
        help="CSV opcional com metadados adicionais pré-existentes (valence, arousal, dominance, emotion)"
    )
    parser.add_argument(
        '--imageExt', 
        default='.jpg', 
        help="Extensão padrão das imagens no dataset (padrão: .jpg)"
    )
    parser.add_argument(
        '--pathPrefix', 
        default=None, 
        help="Prefixo customizado para o caminho das imagens na coluna 'path'"
    )
    parser.add_argument(
        '--fallbackEmotion', 
        choices=['argmax', 'unknown', 'none'], 
        default='argmax', 
        help="Estratégia para a coluna 'emotion' se _exp.npy não for encontrado (padrão: argmax)"
    )
    parser.add_argument(
        '--noSort', 
        action='store_true', 
        help="Desativa a ordenação natural dos arquivos por idImagem"
    )

    args = parser.parse_args()

    input_path = args.annotations_dir or args.path
    if not input_path:
        parser.print_help()
        print("\n[Erro] Você deve fornecer o caminho do diretório de anotações.")
        sys.exit(1)

    # Resolver diretórios de anotações e imagens
    annotations_dir, images_dir = resolve_directories(input_path, args.imagesDir)

    # Definir nome do arquivo de saída padrão se não especificado
    output_file = args.outputFile
    if not output_file:
        # Detectar split no nome do diretório
        norm_dir = os.path.normpath(annotations_dir)
        parent_name = os.path.basename(os.path.dirname(norm_dir)).lower()
        curr_name = os.path.basename(norm_dir).lower()
        
        split_name = "train_set"
        for name in [curr_name, parent_name]:
            if "val" in name:
                split_name = "val_set"
                break
            elif "train" in name:
                split_name = "train_set"
                break

        output_file = f"{split_name}_emotion_distribution_LIKELIHOOD_OLD.csv"

    # Executar compilação
    compile_annotations(
        annotations_dir=annotations_dir,
        images_dir=images_dir,
        output_csv=output_file,
        metadata_csv=args.metadataCSV,
        default_ext=args.imageExt,
        path_prefix=args.pathPrefix,
        fallback_emotion=args.fallbackEmotion,
        sort_output=not args.noSort
    )


if __name__ == '__main__':
    main()

