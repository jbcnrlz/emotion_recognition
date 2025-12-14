import argparse, os, sys, numpy as np, math
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# Configurar o matplotlib ANTES de importar plt
import matplotlib
matplotlib.use('TkAgg')  # Ou 'Qt5Agg' dependendo do seu sistema

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import matplotlib.patches as mpatches
from helper.function import getFilesInPath, printProgressBar

def plot_valence_arousal_with_emotions(data, emotion_mapping=None):
    """
    Plota todos os valores de Valence e Arousal coloridos por emoção.
    
    Args:
        data (dict): Dicionário no formato {fileName: {'emotions': [], 'vaValues': []}}
        emotion_mapping (dict, optional): Mapeamento de emoções para cores. Se None, 
                                          será criado automaticamente.
    """
    
    # Coletar todos os dados
    all_valence = []
    all_arousal = []
    all_emotions = []
    
    print("Coletando dados de valence-arousal...")
    total_files = len(data)
    file_count = 0
    
    for file_name, file_data in data.items():
        file_count += 1
        printProgressBar(file_count, total_files, prefix='Coletando dados:', suffix=f'Arquivo {file_count}/{total_files}', length=50)
        
        if 'vaValues' in file_data and file_data['vaValues']:
            # Assumindo que vaValues é uma lista de pares [valence, arousal]
            for va_pair in file_data['vaValues']:
                if len(va_pair) >= 2:
                    all_valence.append(va_pair[0])
                    all_arousal.append(va_pair[1])
                    
                    # Associar com a emoção correspondente se disponível
                    if 'emotions' in file_data and file_data['emotions']:
                        # Assumindo mesma ordem entre emotions e vaValues
                        idx = file_data['vaValues'].index(va_pair)
                        if idx < len(file_data['emotions']):
                            all_emotions.append(file_data['emotions'][idx])
                        else:
                            all_emotions.append('unknown')
                    else:
                        all_emotions.append('unknown')
    
    if not all_valence or not all_arousal:
        print("Nenhum dado de valence-arousal encontrado!")
        return None, None, None
    
    print(f"\n✅ Total de pontos coletados: {len(all_valence)}")
    
    # Converter para arrays numpy
    valence_array = np.array(all_valence)
    arousal_array = np.array(all_arousal)
    
    # Criar mapeamento de cores para emoções
    if emotion_mapping is None:
        # Obter emoções únicas
        unique_emotions = list(set(all_emotions))
        print(f"🎨 Emoções únicas encontradas: {len(unique_emotions)}")
        
        # Criar mapa de cores
        cmap = get_cmap('tab20', len(unique_emotions))
        emotion_mapping = {emotion: cmap(i) for i, emotion in enumerate(unique_emotions)}
    
    # Criar figura
    print("\nCriando figura...")
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plotar pontos em lotes para evitar congelamento
    print("Plotando pontos...")
    total_points = len(all_valence)
    
    # Organizar pontos por emoção para plotagem mais eficiente
    print("Organizando pontos por emoção...")
    points_by_emotion = {}
    for i, (valence, arousal, emotion) in enumerate(zip(all_valence, all_arousal, all_emotions)):
        if emotion not in points_by_emotion:
            points_by_emotion[emotion] = {'valence': [], 'arousal': []}
        points_by_emotion[emotion]['valence'].append(valence)
        points_by_emotion[emotion]['arousal'].append(arousal)
    
    # Plotar por emoção com barra de progresso
    print("\nPlotando por emoção:")
    emotion_count = 0
    total_emotions = len(points_by_emotion)
    
    for emotion, points in points_by_emotion.items():
        emotion_count += 1
        printProgressBar(emotion_count, total_emotions, 
                        prefix='Plotando:', 
                        suffix=f'{emotion} ({len(points["valence"])} pontos)', 
                        length=50)
        
        color = emotion_mapping.get(emotion, 'gray')
        ax.scatter(points['valence'], points['arousal'], 
                  color=color, s=30, alpha=0.6, edgecolors='black', 
                  linewidth=0.3, label=emotion)
        
        # Atualizar a tela periodicamente para evitar congelamento
        if emotion_count % 5 == 0:
            plt.pause(0.01)  # Pequena pausa para atualizar a interface
    
    # Adicionar linhas de referência para quadrantes
    print("\nAdicionando elementos gráficos...")
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=0.8)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.3, linewidth=0.8)
    
    # Configurar limites do gráfico (assumindo range -1 a 1)
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    
    # Adicionar grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Adicionar rótulos dos quadrantes
    ax.text(0.8, 0.8, 'High Arousal\nPositive Valence', 
            ha='center', va='center', fontsize=10, alpha=0.7)
    ax.text(0.8, -0.8, 'Low Arousal\nPositive Valence', 
            ha='center', va='center', fontsize=10, alpha=0.7)
    ax.text(-0.8, 0.8, 'High Arousal\nNegative Valence', 
            ha='center', va='center', fontsize=10, alpha=0.7)
    ax.text(-0.8, -0.8, 'Low Arousal\nNegative Valence', 
            ha='center', va='center', fontsize=10, alpha=0.7)
    
    # Configurar labels e título
    ax.set_xlabel('Valence (Negative ↔ Positive)', fontsize=12)
    ax.set_ylabel('Arousal (Low ↔ High)', fontsize=12)
    ax.set_title(f'Valence-Arousal Space Colored by Emotion\nTotal: {total_points} pontos', fontsize=14, fontweight='bold')
    
    # Criar legenda para as emoções
    print("Criando legenda...")
    legend_patches = []
    for emotion, color in emotion_mapping.items():
        # Contar quantos pontos têm esta emoção
        count = all_emotions.count(emotion)
        patch = mpatches.Patch(color=color, label=f'{emotion} ({count})')
        legend_patches.append(patch)
    
    # Adicionar legenda
    ax.legend(handles=legend_patches, title='Emotions', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adicionar estatísticas
    stats_text = f'Total Samples: {len(all_valence)}\n'
    stats_text += f'Valence Range: [{valence_array.min():.3f}, {valence_array.max():.3f}]\n'
    stats_text += f'Arousal Range: [{arousal_array.min():.3f}, {arousal_array.max():.3f}]\n'
    stats_text += f'Média Valence: {valence_array.mean():.3f}\n'
    stats_text += f'Média Arousal: {arousal_array.mean():.3f}'
    
    ax.text(1.05, 0.3, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Ajustar layout
    print("Ajustando layout...")
    plt.tight_layout()
    
    print("\n✅ Gráfico criado com sucesso!")
    
    return fig, ax, emotion_mapping

def main():
    parser = argparse.ArgumentParser(description='Process Aff-Wild dataset for emotion recognition')
    parser.add_argument('--pathBase', help='Path for valence and arousal dataset', required=True)
    parser.add_argument('--sample_size', type=int, default=0, 
                       help='Número máximo de pontos para plotar (0 para todos)')
    args = parser.parse_args()

    filesTXT = getFilesInPath(os.path.join(args.pathBase,'annotations','EXPR_Set','Train_Set'))
    print(f"📁 Processando {len(filesTXT)} arquivos em {args.pathBase}")

    emotions = None
    data = {}

    print("Lendo arquivos de expressão...")
    for idx, f in enumerate(filesTXT):
        printProgressBar(idx+1, len(filesTXT), prefix='Lendo expressões:', suffix=f'Arquivo {idx+1}/{len(filesTXT)}', length=50)
        fileName = f.split(os.sep)[-1]
        data[fileName] = {'emotions': [], 'vaValues': []}
        with open(f, 'r') as file:
            for idxLine, fCont in enumerate(file):
                if emotions is None:
                    emotions = {}
                    for emoName in fCont.strip().split(','):
                        emotions[emoName] = 0
                elif idxLine > 0:
                    emoNames = list(emotions.keys())
                    emotions[emoNames[int(fCont.strip())]] += 1
                    data[fileName]['emotions'].append(emoNames[int(fCont.strip())])

    print("\n🎭 Distribuição das emoções:")
    total = sum(emotions.values())
    for emoName, count in emotions.items():
        percentage = (count / total) * 100 if total > 0 else 0
        print(f"   - {emoName}: {count} ({percentage:.2f}%)")

    print("\nLendo arquivos de valence-arousal...")
    filesTXT = getFilesInPath(os.path.join(args.pathBase,'annotations','VA_Set','Train_Set'))
    
    total_files = len(filesTXT)
    for idx, f in enumerate(filesTXT):
        printProgressBar(idx+1, total_files, prefix='Lendo VA:', suffix=f'Arquivo {idx+1}/{total_files}', length=50)
        fileName = f.split(os.sep)[-1]
        with open(f, 'r') as file:
            for idxLine, fCont in enumerate(file):
                if idxLine > 0:
                    try:
                        data[fileName]['vaValues'].append(list(map(float, fCont.strip().split(','))))
                    except:
                        continue

    # Verificar se há dados suficientes
    total_va_points = sum(len(v['vaValues']) for v in data.values() if 'vaValues' in v)
    print(f"\n📊 Total de pontos VA encontrados: {total_va_points}")
    
    # Se for muito grande, amostrar aleatoriamente
    if args.sample_size > 0 and total_va_points > args.sample_size:
        print(f"\n⚠️  Amostrando {args.sample_size} pontos de {total_va_points}...")
        # Criar amostra aleatória
        sampled_data = {}
        all_points = []
        
        for file_name, file_data in data.items():
            if 'vaValues' in file_data and file_data['vaValues']:
                for i, va_pair in enumerate(file_data['vaValues']):
                    emotion = file_data['emotions'][i] if 'emotions' in file_data and i < len(file_data['emotions']) else 'unknown'
                    all_points.append({
                        'file': file_name,
                        'valence': va_pair[0],
                        'arousal': va_pair[1],
                        'emotion': emotion
                    })
        
        # Amostrar aleatoriamente
        if len(all_points) > args.sample_size:
            indices = np.random.choice(len(all_points), args.sample_size, replace=False)
            sampled_points = [all_points[i] for i in indices]
            
            # Reorganizar os dados amostrados
            sampled_data = {'sampled': {'emotions': [], 'vaValues': []}}
            for point in sampled_points:
                sampled_data['sampled']['emotions'].append(point['emotion'])
                sampled_data['sampled']['vaValues'].append([point['valence'], point['arousal']])
            
            data = sampled_data
            print(f"✅ Amostra de {args.sample_size} pontos criada")
    
    # Plotar com matplotlib
    print("\n" + "="*50)
    print("INICIANDO PLOTAGEM DO GRÁFICO")
    print("="*50)
    
    fig1, ax1, emotion_map1 = plot_valence_arousal_with_emotions(data)
    
    if fig1 is not None:
        print("\n✅ Gráfico pronto! Exibindo...")
        plt.show()
        print("✅ Gráfico fechado pelo usuário")
    else:
        print("❌ Não foi possível criar o gráfico. Verifique se há dados de valence-arousal.")

if __name__ == "__main__":
    main()