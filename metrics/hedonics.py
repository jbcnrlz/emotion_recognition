#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser(
        description="Calcula o Grau de Confiança na Avaliação Sensorial filtrando frames neutros."
    )
    parser.add_argument(
        '-e', '--emocoes', 
        required=True, 
        help="Caminho para o arquivo CSV com as emoções extraídas dos frames."
    )
    parser.add_argument(
        '-c', '--consolidado', 
        required=True, 
        help="Caminho para o arquivo CSV com as respostas consolidadas dos usuários."
    )
    parser.add_argument(
        '-o', '--output', 
        default="resultado_confianca.csv", 
        help="Caminho para o arquivo CSV de saída (padrão: resultado_confianca.csv)."
    )
    parser.add_argument(
        '-t', '--threshold', 
        type=float,
        default=0.80, 
        help="Limiar máximo para a classe Neutro (0 a 1). Frames acima disso são ignorados. Padrão: 0.80"
    )
    return parser.parse_args()

def process_sensory_data(emocoes_path, consolidado_path, output_path, limiar_neutro):
    print(f"Lendo arquivo de emoções: {emocoes_path}")
    try:
        df_emocoes = pd.read_csv(emocoes_path)
    except FileNotFoundError:
        print(f"Erro: Arquivo de emoções não encontrado em {emocoes_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Lendo arquivo consolidado: {consolidado_path}")
    try:
        df_consolidado = pd.read_csv(consolidado_path, sep=';')
        
        if len(df_consolidado.columns) == 1:
             print("Aviso: Separador ';' não funcionou. Tentando com ','.")
             df_consolidado = pd.read_csv(consolidado_path, sep=',')
    except FileNotFoundError:
        print(f"Erro: Arquivo consolidado não encontrado em {consolidado_path}", file=sys.stderr)
        sys.exit(1)

    print("Extraindo Tokens e Aspectos...")
    partes_caminho = df_emocoes['file'].str.split('/')
    
    if len(partes_caminho.iloc[0]) < 3:
         print("Erro crítico: A coluna 'file' não possui a estrutura de diretórios esperada (/<Token>/<Aspecto>/frame).", file=sys.stderr)
         sys.exit(1)

    df_emocoes['Token'] = partes_caminho.str[-3]
    df_emocoes['Aspecto'] = partes_caminho.str[-2].str.lower()

    print("Reestruturando dados hedônicos...")
    mapa_aspectos = {
        'aparência': 'CARAC_Aparência_Quanto você gostou ou desgostou desse aspecto do produto?',
        'aroma': 'CARAC_Aroma_Quanto você gostou ou desgostou desse aspecto do produto.',
        'cor': 'CARAC_Cor_Quanto você gostou ou desgostou desse aspecto do produto?',
        'sabor': 'CARAC_Sabor_Quanto você gostou ou desgostou desse aspecto do produto?',
        'textura': 'CARAC_Textura_Quanto você gostou ou desgostou desse aspecto do produto?'
    }

    colunas_presentes = set(df_consolidado.columns)
    colunas_necessarias = list(mapa_aspectos.values())
    colunas_faltantes = [col for col in colunas_necessarias if col not in colunas_presentes]
    
    if colunas_faltantes:
        print(f"Erro crítico: As seguintes colunas não foram encontradas no CSV consolidado: {colunas_faltantes}", file=sys.stderr)
        sys.exit(1)

    df_notas = df_consolidado[['Token'] + colunas_necessarias].copy()

    df_notas_long = df_notas.melt(
        id_vars=['Token'], 
        value_vars=colunas_necessarias,
        var_name='Coluna_Original', 
        value_name='Nota_Hedonica'
    )

    inv_mapa = {v: k for k, v in mapa_aspectos.items()}
    df_notas_long['Aspecto'] = df_notas_long['Coluna_Original'].map(inv_mapa)

    print("Cruzando dados (Merge)...")
    df_merged = pd.merge(df_emocoes, df_notas_long, on=['Token', 'Aspecto'], how='inner')
    
    df_merged['Nota_Hedonica'] = pd.to_numeric(df_merged['Nota_Hedonica'], errors='coerce')
    df_merged = df_merged.dropna(subset=['Nota_Hedonica']).copy()

    if df_merged.empty:
         print("Aviso: O cruzamento resultou em um dataframe vazio. Verifique os Tokens.", file=sys.stderr)
         sys.exit(1)

    print("Calculando Métricas Fisiológicas...")
    # A. Normalização da Nota Hedônica
    df_merged['H_norm'] = 2 * ((df_merged['Nota_Hedonica'] - 1) / (9 - 1)) - 1

    # B. Valência Emocional
    e_pos = df_merged['happy']
    e_neg = df_merged[['contempt', 'angry', 'disgusted', 'fearful', 'sad']].sum(axis=1)
    df_merged['V'] = e_pos - e_neg

    # C. Entropia e Certeza
    emocoes = ['happy', 'contempt', 'surprised', 'angry', 'disgusted', 'fearful', 'sad', 'neutral']
    P = df_merged[emocoes] + 1e-9 
    
    df_merged['Entropia'] = -np.sum(P * np.log2(P), axis=1)
    df_merged['W'] = 1 - (df_merged['Entropia'] / 3)

    # D. Concordância e Confiança
    df_merged['M'] = 1 - (abs(df_merged['H_norm'] - df_merged['V']) / 2)
    df_merged['Confianca_Frame'] = df_merged['M'] * df_merged['W']

    print(f"Filtrando frames neutros (Limiar de Repouso: {limiar_neutro})...")
    df_expressivo = df_merged[df_merged['neutral'] < limiar_neutro].copy()
    print(f" -> {len(df_merged)} frames totais reduzidos para {len(df_expressivo)} frames expressivos.")

    print("Agregando resultados...")
    # Passo 1: Metadados globais do vídeo
    df_base = df_merged.groupby(['Token', 'Aspecto']).agg(
        Nota_Declarada=('Nota_Hedonica', 'first'),
        Total_Frames_Video=('file', 'count')
    ).reset_index()

    # Passo 2: Métricas calculadas apenas sobre os frames expressivos
    df_metricas = df_expressivo.groupby(['Token', 'Aspecto']).agg(
        Certeza_Media_Rede=('W', 'mean'),
        Confianca_Final=('Confianca_Frame', 'mean'),
        Frames_Expressivos=('file', 'count')
    ).reset_index()

    # Passo 3: Join para manter registros que foram 100% neutros
    df_final = pd.merge(df_base, df_metricas, on=['Token', 'Aspecto'], how='left')

    # Tratamento de valores resultantes de vídeos sem frames expressivos
    df_final['Frames_Expressivos'] = df_final['Frames_Expressivos'].fillna(0).astype(int)
    df_final['Sinal_Insuficiente'] = df_final['Frames_Expressivos'] == 0

    # Conversão para porcentagem
    df_final['Confianca_Final_%'] = (df_final['Confianca_Final'] * 100).round(2)
    df_final['Certeza_Media_Rede_%'] = (df_final['Certeza_Media_Rede'] * 100).round(2)

    # Ordenação das colunas
    colunas_finais = [
        'Token', 'Aspecto', 'Nota_Declarada', 'Total_Frames_Video', 
        'Frames_Expressivos', 'Sinal_Insuficiente', 'Certeza_Media_Rede_%', 'Confianca_Final_%'
    ]
    df_final = df_final[colunas_finais]

    df_final.to_csv(output_path, index=False)
    print(f"\nSucesso! Resultados salvos em: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    args = parse_args()
    process_sensory_data(args.emocoes, args.consolidado, args.output, args.threshold)