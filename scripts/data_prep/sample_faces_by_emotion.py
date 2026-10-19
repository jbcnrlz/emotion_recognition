#!/usr/bin/env python3
"""
sample_faces_by_emotion.py
--------------------------
Seleciona aleatoriamente uma quantidade X de imagens de faces para cada estado emocional
a partir de um dataset especificado.

Suporta:
- 'auto'        : Detecta automaticamente a estrutura do dataset
- 'folder'      : Estrutura ImageFolder (dataset/<emocao>/<imagem>)
- 'affectnet'   : Estrutura AffectNet (images/ + annotations/*_exp.npy ou CSV)
- 'ckplus'      : Estrutura CK+ (cohn-kanade-images/ + Emotion/)
- 'csv'         : Arquivo CSV de anotações com colunas de imagem e emoção

Exemplos de uso via CLI:
    python scripts/data_prep/sample_faces_by_emotion.py \
        --dataset-type auto \
        --dataset-path /caminho/do/dataset \
        --samples-per-emotion 50 \
        --output-dir ./amostras_selecionadas \
        --mode copy \
        --seed 42

Uso via Módulo Python:
    from src.data import sample_dataset_per_emotion
    # ou: from scripts.data_prep.sample_faces_by_emotion import sample_dataset_per_emotion

    summary = sample_dataset_per_emotion(
        dataset_path="/caminho/do/dataset",
        samples_per_emotion=50,
        output_dir="./amostras",
        mode="copy" # ou 'symlink'
    )
"""

import os
import sys
import glob
import json
import random
import shutil
import argparse
from collections import defaultdict
from pathlib import Path

# Imagens suportadas
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

# Mapeamentos padronizados de nomes de emoções
AFFECTNET_EMOTIONS = {
    0: 'Neutral',
    1: 'Happy',
    2: 'Sad',
    3: 'Surprise',
    4: 'Fear',
    5: 'Disgust',
    6: 'Anger',
    7: 'Contempt'
}

CKPLUS_EMOTIONS = {
    1: 'Anger',
    2: 'Contempt',
    3: 'Disgust',
    4: 'Fear',
    5: 'Happy',
    6: 'Sadness',
    7: 'Surprise'
}

DFEW_EMOTIONS = {
    0: 'Happy',
    1: 'Sad',
    2: 'Neutral',
    3: 'Anger',
    4: 'Surprise',
    5: 'Disgust',
    6: 'Fear'
}


class EmotionSampler:
    """Carrega imagens por estado emocional de múltiplos formatos de datasets e realiza amostragem."""

    def __init__(self, dataset_path, dataset_type='auto', csv_path=None, image_col=None, label_col=None):
        self.dataset_path = os.path.abspath(dataset_path)
        self.dataset_type = dataset_type.lower()
        self.csv_path = csv_path
        self.image_col = image_col
        self.label_col = label_col
        
        # Mapeamento: {emotion_name: [list_of_image_paths]}
        self.emotions_to_images = defaultdict(list)
        self.detected_type = None

        self._load_dataset()

    def _load_dataset(self):
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Caminho do dataset não encontrado: {self.dataset_path}")

        dtype = self.dataset_type
        if dtype == 'auto':
            dtype = self._detect_dataset_type()

        self.detected_type = dtype
        print(f"[Dataset Loader] Tipo detectado/selecionado: '{dtype}' em: {self.dataset_path}")

        if dtype == 'folder':
            self._load_from_folder_structure()
        elif dtype == 'affectnet':
            self._load_from_affectnet()
        elif dtype == 'ckplus':
            self._load_from_ckplus()
        elif dtype == 'csv':
            self._load_from_csv()
        else:
            raise ValueError(f"Tipo de dataset desconhecido: '{dtype}'. Opções: 'auto', 'folder', 'affectnet', 'ckplus', 'csv'")

        total_images = sum(len(v) for v in self.emotions_to_images.values())
        if total_images == 0:
            raise RuntimeError(f"Nenhuma imagem válida encontrada no caminho '{self.dataset_path}' para o formato '{dtype}'.")

        print(f"[Dataset Loader] Total carregado: {total_images} imagens em {len(self.emotions_to_images)} estados emocionais.")

    def _detect_dataset_type(self):
        """Detecta o formato examinando as pastas presentes no caminho."""
        subdirs = [d.lower() for d in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, d))]
        
        # Checar AffectNet
        if 'images' in subdirs and 'annotations' in subdirs:
            return 'affectnet'
            
        # Checar CK+
        if 'emotion' in subdirs and ('cohn-kanade-images' in subdirs or 'images' in subdirs):
            return 'ckplus'

        # Checar se as subpastas contêm imagens diretamente (ImageFolder)
        has_subfolder_images = False
        for s in os.listdir(self.dataset_path):
            sub_p = os.path.join(self.dataset_path, s)
            if os.path.isdir(sub_p):
                imgs = [f for f in os.listdir(sub_p) if f.lower().endswith(IMAGE_EXTENSIONS)]
                if imgs:
                    has_subfolder_images = True
                    break

        if has_subfolder_images:
            return 'folder'

        # Checar se há CSVs de anotação
        csv_files = [f for f in os.listdir(self.dataset_path) if f.lower().endswith('.csv')]
        if csv_files or self.csv_path:
            return 'csv'

        return 'folder'

    def _load_from_folder_structure(self):
        """Carrega datasets no formato pasta/emocao/imagem.ext."""
        subdirs = sorted([d for d in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, d))])
        for emotion in subdirs:
            folder_p = os.path.join(self.dataset_path, emotion)
            for root, _, files in os.walk(folder_p):
                for f in files:
                    if f.lower().endswith(IMAGE_EXTENSIONS):
                        self.emotions_to_images[emotion].append(os.path.join(root, f))

    def _load_from_affectnet(self):
        """Carrega dataset AffectNet com annotations/*_exp.npy."""
        images_dir = os.path.join(self.dataset_path, 'images')
        annotations_dir = os.path.join(self.dataset_path, 'annotations')

        if not os.path.exists(images_dir):
            images_dir = self.dataset_path

        import numpy as np

        image_files = []
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(glob.glob(os.path.join(images_dir, f'*{ext}')))
            image_files.extend(glob.glob(os.path.join(images_dir, f'*{ext.upper()}')))

        for img_path in image_files:
            img_num = os.path.basename(img_path).rsplit('.', 1)[0]
            exp_file = os.path.join(annotations_dir, f"{img_num}_exp.npy")
            if not os.path.exists(exp_file) and img_num.isdigit():
                exp_file = os.path.join(annotations_dir, f"{int(img_num)}_exp.npy")

            if os.path.exists(exp_file):
                try:
                    exp_id = int(np.load(exp_file))
                    emotion_name = AFFECTNET_EMOTIONS.get(exp_id, f"Emotion_{exp_id}")
                    self.emotions_to_images[emotion_name].append(img_path)
                except Exception:
                    continue

    def _load_from_ckplus(self):
        """Carrega dataset CK+."""
        labels_folder = os.path.join(self.dataset_path, 'Emotion')
        images_folder = os.path.join(self.dataset_path, 'cohn-kanade-images')
        if not os.path.exists(images_folder):
            images_folder = os.path.join(self.dataset_path, 'images')

        for subject in os.listdir(labels_folder):
            subj_p = os.path.join(labels_folder, subject)
            if not os.path.isdir(subj_p):
                continue
            for seq in os.listdir(subj_p):
                seq_p = os.path.join(subj_p, seq)
                if not os.path.isdir(seq_p):
                    continue
                for f in os.listdir(seq_p):
                    if f.endswith('_emotion.txt'):
                        lbl_path = os.path.join(seq_p, f)
                        with open(lbl_path, 'r') as fp:
                            content = fp.read().strip()
                            if content:
                                emo_id = int(float(content.split()[0]))
                                emo_name = CKPLUS_EMOTIONS.get(emo_id, f"Emotion_{emo_id}")
                                # O último frame da sequência expressa a emoção máxima
                                img_prefix = '_'.join(f.split('_')[:-1])
                                img_candidates = glob.glob(os.path.join(images_folder, subject, seq, f"{img_prefix}*"))
                                img_candidates = [c for c in img_candidates if c.lower().endswith(IMAGE_EXTENSIONS)]
                                if img_candidates:
                                    self.emotions_to_images[emo_name].append(sorted(img_candidates)[-1])

    def _load_from_csv(self):
        """Carrega dataset a partir de um arquivo CSV de anotações."""
        csv_file = self.csv_path
        if not csv_file:
            candidates = [f for f in os.listdir(self.dataset_path) if f.lower().endswith('.csv')]
            if candidates:
                csv_file = os.path.join(self.dataset_path, candidates[0])
            else:
                raise FileNotFoundError("Nenhum CSV encontrado. Especifique com --csv-path.")

        import csv
        with open(csv_file, 'r', encoding='utf-8', errors='ignore') as fp:
            reader = csv.DictReader(fp)
            headers = [h.strip() for h in reader.fieldnames]

            img_col = self.image_col
            if not img_col:
                for cand in ['image_path', 'subDirectory_filePath', 'pth', 'image', 'file', 'filename', 'id', 'path']:
                    for h in headers:
                        if cand.lower() in h.lower():
                            img_col = h
                            break
                    if img_col:
                        break

            lbl_col = self.label_col
            if not lbl_col:
                for cand in ['expression', 'emotion', 'label', 'class', 'target']:
                    for h in headers:
                        if cand.lower() in h.lower():
                            lbl_col = h
                            break
                    if lbl_col:
                        break

            if not img_col or not lbl_col:
                raise ValueError(f"Não foi possível identificar colunas de imagem/label no CSV {csv_file}. Headers: {headers}")

            print(f"[CSV Loader] Usando coluna de imagem: '{img_col}', coluna de emoção: '{lbl_col}'")

            for row in reader:
                img_rel = row[img_col].strip()
                emo = row[lbl_col].strip()

                # Tentar mapear número para nome se for numérico
                if emo.isdigit():
                    emo_int = int(emo)
                    emo = AFFECTNET_EMOTIONS.get(emo_int, emo)

                full_p = os.path.join(self.dataset_path, img_rel)
                if not os.path.exists(full_p):
                    # Tentar em images/
                    full_p = os.path.join(self.dataset_path, 'images', img_rel)

                if os.path.exists(full_p):
                    self.emotions_to_images[emo].append(full_p)

    def sample(self, samples_per_emotion, seed=None, allow_fewer=True, replace=False):
        """
        Seleciona aleatoriamente `samples_per_emotion` imagens para cada emoção.
        
        Args:
            samples_per_emotion (int): Quantidade X de amostras por emoção.
            seed (int, opcional): Seed para reprodutibilidade.
            allow_fewer (bool): Se True, pega o máximo disponível se houver menos que X.
            replace (bool): Se True, permite amostragem com reposição quando houver menos que X.

        Returns:
            dict: {emotion_name: [list_of_selected_image_paths]}
        """
        if seed is not None:
            random.seed(seed)

        sampled = {}
        for emotion, images in sorted(self.emotions_to_images.items()):
            n_avail = len(images)
            if n_avail == 0:
                sampled[emotion] = []
                continue

            if n_avail >= samples_per_emotion:
                sampled[emotion] = random.sample(images, samples_per_emotion)
            else:
                if replace:
                    sampled[emotion] = random.choices(images, k=samples_per_emotion)
                elif allow_fewer:
                    # Copia todas as disponíveis embaralhadas
                    shuffled = list(images)
                    random.shuffle(shuffled)
                    sampled[emotion] = shuffled
                else:
                    raise ValueError(
                        f"A classe '{emotion}' possui apenas {n_avail} imagens, menos do que as {samples_per_emotion} solicitadas. "
                        f"Use allow_fewer=True ou replace=True."
                    )

        return sampled

    def print_summary(self, sampled=None):
        """Imprime tabela formatada com estatísticas de amostragem."""
        print("\n" + "=" * 65)
        print(f"{'Estado Emocional':<20} | {'Disponíveis':<12} | {'Amostradas':<12} | {'% Amostrada':<10}")
        print("-" * 65)
        
        total_avail = 0
        total_sampled = 0

        for emotion, images in sorted(self.emotions_to_images.items()):
            n_avail = len(images)
            n_samp = len(sampled[emotion]) if sampled and emotion in sampled else 0
            total_avail += n_avail
            total_sampled += n_samp
            pct = (n_samp / n_avail * 100) if n_avail > 0 else 0.0

            print(f"{emotion:<20} | {n_avail:<12} | {n_samp:<12} | {pct:>9.1f}%")

        print("=" * 65)
        total_pct = (total_sampled / total_avail * 100) if total_avail > 0 else 0.0
        print(f"{'TOTAL':<20} | {total_avail:<12} | {total_sampled:<12} | {total_pct:>9.1f}%")
        print("=" * 65 + "\n")

    @staticmethod
    def export(sampled_dict, output_dir=None, mode='copy', csv_output=None, json_output=None):
        """Exporta as amostras selecionadas para pasta (cópia/symlink) e/ou arquivos de índice."""
        if output_dir:
            output_dir = os.path.abspath(output_dir)
            os.makedirs(output_dir, exist_ok=True)
            print(f"[Export] Criando arquivos na pasta destino: {output_dir} (modo: {mode})")

            for emotion, paths in sampled_dict.items():
                emo_folder = os.path.join(output_dir, emotion)
                os.makedirs(emo_folder, exist_ok=True)

                for p in paths:
                    dst_filename = os.path.basename(p)
                    dst_path = os.path.join(emo_folder, dst_filename)

                    if os.path.exists(dst_path):
                        # Evitar colisões de nomes idênticos adicionando hash ou contador
                        base, ext = os.path.splitext(dst_filename)
                        dst_path = os.path.join(emo_folder, f"{base}_{abs(hash(p)) % 10000}{ext}")

                    if mode == 'symlink':
                        if os.path.exists(dst_path):
                            os.remove(dst_path)
                        os.symlink(p, dst_path)
                    elif mode == 'move':
                        shutil.move(p, dst_path)
                    else:  # copy
                        shutil.copy2(p, dst_path)

            print(f"[Export] Concluído! Imagens organizadas por emoção em: {output_dir}")

        if csv_output:
            csv_output = os.path.abspath(csv_output)
            os.makedirs(os.path.dirname(csv_output), exist_ok=True)
            import csv
            with open(csv_output, 'w', newline='', encoding='utf-8') as fp:
                writer = csv.writer(fp)
                writer.writerow(['image_path', 'emotion'])
                for emotion, paths in sampled_dict.items():
                    for p in paths:
                        writer.writerow([p, emotion])
            print(f"[Export] Arquivo CSV gerado em: {csv_output}")

        if json_output:
            json_output = os.path.abspath(json_output)
            os.makedirs(os.path.dirname(json_output), exist_ok=True)
            with open(json_output, 'w', encoding='utf-8') as fp:
                json.dump(sampled_dict, fp, indent=2, ensure_ascii=False)
            print(f"[Export] Arquivo JSON gerado em: {json_output}")


def sample_dataset_per_emotion(dataset_path, samples_per_emotion, dataset_type='auto',
                              output_dir=None, mode='copy', csv_output=None, json_output=None,
                              seed=42, allow_fewer=True, replace=False):
    """
    Função utilitária em nível superior para fácil importação em scripts e notebooks.
    """
    sampler = EmotionSampler(dataset_path=dataset_path, dataset_type=dataset_type)
    sampled = sampler.sample(
        samples_per_emotion=samples_per_emotion,
        seed=seed,
        allow_fewer=allow_fewer,
        replace=replace
    )
    sampler.print_summary(sampled)

    if output_dir or csv_output or json_output:
        sampler.export(
            sampled_dict=sampled,
            output_dir=output_dir,
            mode=mode,
            csv_output=csv_output,
            json_output=json_output
        )

    return sampled


def parse_args():
    parser = argparse.ArgumentParser(
        description="Seleciona aleatoriamente X imagens de faces para cada estado emocional a partir de um dataset."
    )
    parser.add_argument(
        '--dataset-path', '-p', type=str, required=False, default=None,
        help="Caminho raiz do dataset."
    )
    parser.add_argument(
        '--samples-per-emotion', '-n', type=int, required=False, default=None,
        help="Quantidade X de imagens a serem selecionadas por emoção."
    )
    parser.add_argument(
        '--dataset-type', '-t', type=str, default='auto',
        choices=['auto', 'folder', 'affectnet', 'ckplus', 'csv'],
        help="Tipo do dataset. 'auto' detecta automaticamente (padrão: auto)."
    )
    parser.add_argument(
        '--output-dir', '-o', type=str, default=None,
        help="Diretório onde salvar as imagens amostradas organizadas em subpastas por emoção."
    )
    parser.add_argument(
        '--mode', type=str, default='copy', choices=['copy', 'symlink', 'move'],
        help="Modo de exportação das imagens: 'copy' (padrão), 'symlink' (não ocupa espaço extra), ou 'move'."
    )
    parser.add_argument(
        '--csv-output', type=str, default=None,
        help="Caminho para salvar um arquivo CSV com os caminhos das imagens selecionadas."
    )
    parser.add_argument(
        '--json-output', type=str, default=None,
        help="Caminho para salvar um arquivo JSON com o dicionário de amostras."
    )
    parser.add_argument(
        '--seed', '-s', type=int, default=42,
        help="Seed aleatória para reprodutibilidade (padrão: 42)."
    )
    parser.add_argument(
        '--csv-path', type=str, default=None,
        help="Caminho do arquivo CSV de anotações (obrigatório se --dataset-type csv)."
    )
    parser.add_argument(
        '--image-col', type=str, default=None,
        help="Nome da coluna de imagem no CSV."
    )
    parser.add_argument(
        '--label-col', type=str, default=None,
        help="Nome da coluna de emoção no CSV."
    )
    parser.add_argument(
        '--replace', action='store_true',
        help="Permite amostragem com reposição quando uma classe tiver menos que X imagens."
    )
    parser.add_argument(
        '--interactive', '-i', action='store_true',
        help="Executa em modo interativo guiado por perguntas no terminal."
    )
    return parser.parse_args()


def interactive_mode():
    """Modo interativo pelo terminal caso o usuário execute sem argumentos."""
    print("\n" + "=" * 60)
    print("      SELETOR ALEATÓRIO DE FACES POR ESTADO EMOCIONAL")
    print("=" * 60)

    dataset_path = input("\n1. Informe o caminho do dataset: ").strip()
    while not dataset_path or not os.path.exists(dataset_path):
        print(f"Erro: Caminho '{dataset_path}' não existe. Tente novamente.")
        dataset_path = input("1. Informe o caminho do dataset: ").strip()

    print("\n2. Tipos de dataset suportados:")
    print("   [1] Auto-detectar (Recomendado)")
    print("   [2] Subpastas por emoção (ImageFolder: dataset/<emocao>/<imagem>)")
    print("   [3] AffectNet (images/ + annotations/*_exp.npy)")
    print("   [4] CK+ (cohn-kanade-images/ + Emotion/)")
    print("   [5] Arquivo CSV com anotações")
    dtype_choice = input("Selecione o tipo [1-5] (padrão 1): ").strip()
    type_map = {'1': 'auto', '2': 'folder', '3': 'affectnet', '4': 'ckplus', '5': 'csv'}
    dataset_type = type_map.get(dtype_choice, 'auto')

    csv_path = None
    if dataset_type == 'csv':
        csv_path = input("Informe o caminho do arquivo CSV de anotações: ").strip()

    samples_str = input("\n3. Quantidade X de imagens por estado emocional: ").strip()
    while not samples_str.isdigit() or int(samples_str) <= 0:
        samples_str = input("Por favor, digite um número inteiro positivo: ").strip()
    samples_per_emotion = int(samples_str)

    print("\n4. O que deseja fazer com as imagens selecionadas?")
    print("   [1] Copiar para uma nova pasta (organizada por emoção)")
    print("   [2] Criar atalhos/symlinks em uma nova pasta (não consome disco)")
    print("   [3] Apenas gerar um arquivo CSV com as imagens selecionadas")
    print("   [4] Apenas visualizar estatísticas na tela")
    action_choice = input("Escolha a opção [1-4] (padrão 1): ").strip()

    output_dir = None
    mode = 'copy'
    csv_output = None

    if action_choice in ('1', ''):
        output_dir = input("Informe o diretório de destino: ").strip() or "./amostras_selecionadas"
        mode = 'copy'
    elif action_choice == '2':
        output_dir = input("Informe o diretório de destino: ").strip() or "./symlinks_amostras"
        mode = 'symlink'
    elif action_choice == '3':
        csv_output = input("Informe o caminho do CSV de saída (ex: selecionadas.csv): ").strip() or "./selecionadas.csv"

    want_csv = input("\nDeseja também gerar um CSV índice com a lista de selecionadas? (s/N): ").strip().lower()
    if want_csv in ('s', 'sim', 'y', 'yes') and not csv_output:
        csv_output = input("Caminho do CSV índice (padrão: ./amostras_indice.csv): ").strip() or "./amostras_indice.csv"

    print("\nIniciando amostragem...")
    sample_dataset_per_emotion(
        dataset_path=dataset_path,
        samples_per_emotion=samples_per_emotion,
        dataset_type=dataset_type,
        output_dir=output_dir,
        mode=mode,
        csv_output=csv_output,
        seed=42
    )


def main():
    args = parse_args()

    # Se chamado com --interactive ou sem argumentos principais, abre menu interativo
    if args.interactive:
        interactive_mode()
        return

    if not args.dataset_path or args.samples_per_emotion is None:
        print("Erro: --dataset-path e --samples-per-emotion são obrigatórios quando não em modo interativo.")
        print("Use: python scripts/data_prep/sample_faces_by_emotion.py --help ou --interactive")
        sys.exit(1)

    sampler = EmotionSampler(
        dataset_path=args.dataset_path,
        dataset_type=args.dataset_type,
        csv_path=args.csv_path,
        image_col=args.image_col,
        label_col=args.label_col
    )

    sampled = sampler.sample(
        samples_per_emotion=args.samples_per_emotion,
        seed=args.seed,
        allow_fewer=True,
        replace=args.replace
    )

    sampler.print_summary(sampled)

    if args.output_dir or args.csv_output or args.json_output:
        sampler.export(
            sampled_dict=sampled,
            output_dir=args.output_dir,
            mode=args.mode,
            csv_output=args.csv_output,
            json_output=args.json_output
        )


if __name__ == '__main__':
    main()

