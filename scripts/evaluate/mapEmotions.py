import pandas as pd
import numpy as np
import sys
import os
import argparse

# Mapeamento de 13 classes para 7 classes (neutral é mantido separado)
mapping_13_to_7 = {
    'happy': ['happy', 
              'elated + concentrating + responsible + egotistical + domineering + alone with responsibility + serious + bold + aggressive + mighty + strong + activated + alert + interested + joyful + friendly + affectionate + inspired + excited + triumphant + vigorous + kind + capable + proud + influential + dignified + powerful + masterful + controlling + self-satisfied + free + carefree + admired + enjoyment + useful + lucky',
              'hopeful + devoted + cooperative + respectful + thankful + appreciative + grateful + impressed + relaxed + leisurely + untroubled + secure',
              'loved + in love + sexually excited + fascinated'],
    'contempt': ['contempt'],
    'surprised': ['surprised', 'astonished'],
    'angry': ['angry'],
    'disgusted': ['disgusted'],
    'fearful': ['fearful'],
    'sad': ['sad',
            'proud + aloof + modest + nonchalant + reserved + solemn + quiet + humble + protected + consoled + reverent + sheltered + meek + subdued + weary',
            'fatigued + listless + ennui + blase + uninterested + quietly indignant + timid + inhibited + impotent + feeble + deactived + discontented + discouraged + bored + lonely + depressed + despairing']
}

# Nomes das 13 classes originais (na ordem do seu CSV)
classes_13 = [
    'happy',
    'contempt',
    'elated',  # Nome abreviado para a classe longa
    'hopeful',  # Nome abreviado para a classe longa  
    'surprised',
    'proud',   # Nome abreviado para a classe longa
    'loved',   # Nome abreviado para a classe longa
    'angry',
    'astonished',
    'disgusted',
    'fearful',
    'sad',
    'fatigued'  # Nome abreviado para a classe longa
]

# Nomes das 7 classes agregadas + neutral
classes_7 = list(mapping_13_to_7.keys())

def softmax(x):
    """Aplica softmax para converter logits em probabilidades"""
    exp_x = np.exp(x - np.max(x))  # para estabilidade numérica
    return exp_x / np.sum(exp_x)

def aggregate_emotions(row):
    """
    Agrega as 13 probabilidades emocionais em 7 classes + mantém neutral separado
    """
    # Extrai as probabilidades das 13 classes
    probs_13 = row[classes_13].values
    
    # Inicializa o vetor de 7 logits (sem neutral)
    logits_7 = np.zeros(7)
    
    # Mapeamento dos nomes abreviados para as classes completas
    full_name_mapping = {
        'elated': 'elated + concentrating + responsible + egotistical + domineering + alone with responsibility + serious + bold + aggressive + mighty + strong + activated + alert + interested + joyful + friendly + affectionate + inspired + excited + triumphant + vigorous + kind + capable + proud + influential + dignified + powerful + masterful + controlling + self-satisfied + free + carefree + admired + enjoyment + useful + lucky',
        'hopeful': 'hopeful + devoted + cooperative + respectful + thankful + appreciative + grateful + impressed + relaxed + leisurely + untroubled + secure',
        'proud': 'proud + aloof + modest + nonchalant + reserved + solemn + quiet + humble + protected + consoled + reverent + sheltered + meek + subdued + weary',
        'loved': 'loved + in love + sexually excited + fascinated',
        'fatigued': 'fatigued + listless + ennui + blase + uninterested + quietly indignant + timid + inhibited + impotent + feeble + deactived + discontented + discouraged + bored + lonely + depressed + despairing'
    }
    
    # Agrega os logits conforme o mapeamento
    for i, target_class in enumerate(classes_7):
        for source_class in mapping_13_to_7[target_class]:
            # Verifica se é um nome completo ou abreviado
            if source_class in classes_13:
                idx = classes_13.index(source_class)
                logits_7[i] += probs_13[idx]
            else:
                # Procura pelo nome abreviado correspondente
                for abbrev, full_name in full_name_mapping.items():
                    if full_name == source_class:
                        idx = classes_13.index(abbrev)
                        logits_7[i] += probs_13[idx]
                        break
    
    # Aplica softmax para obter probabilidades finais das 7 emoções
    probs_7 = softmax(logits_7)
    
    # Cria um array com as 7 emoções + neutral
    final_probs = np.zeros(8)
    final_probs[:7] = probs_7
    final_probs[7] = row['neutral']  # Mantém a probabilidade neutral original
    
    # Nomes das 8 classes finais
    final_classes = classes_7 + ['neutral']
    
    return pd.Series(final_probs, index=final_classes)

def main():
    # Verifica se o nome do arquivo foi passado como parâmetro
    parser = argparse.ArgumentParser(description='Map distributions from 13 to 7 emotions')
    parser.add_argument('--pathBase', help='Path for valence and arousal dataset', required=True)
    parser.add_argument('--output', default=None, help='File to save csv', required=True)
    args = parser.parse_args()
    
    # Obtém o nome do arquivo de entrada
    input_file = args.pathBase
    
    # Verifica se o arquivo existe
    if not os.path.exists(input_file):
        print(f"Erro: Arquivo '{input_file}' não encontrado!")
        sys.exit(1)
    
    # Obtém o nome do arquivo de saída (opcional)
    output_file = args.output
    
    try:
        # Carrega o arquivo CSV
        print(f"Carregando arquivo: {input_file}")
        df = pd.read_csv(input_file)
        
        # Verifica se todas as colunas necessárias existem
        required_columns = classes_13 + ['neutral']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Erro: Colunas faltando no arquivo: {missing_columns}")
            print(f"Colunas disponíveis: {list(df.columns)}")
            sys.exit(1)
        
        # Aplica a agregação para cada linha
        print("Processando agregação de emoções...")
        df_aggregated = df.apply(aggregate_emotions, axis=1)
        
        # Combina com as colunas originais (exceto as 13 classes emocionais)
        columns_to_keep = [col for col in ['valence', 'arousal', 'dominance', 'file'] 
                          if col in df.columns]
        df_final = pd.concat([df_aggregated, df[columns_to_keep]], axis=1)
        
        # Reordena as colunas para melhor visualização
        final_emotion_columns = classes_7 + ['neutral']
        column_order = final_emotion_columns + [col for col in ['valence', 'arousal', 'dominance', 'file'] 
                                  if col in df.columns]
        df_final = df_final[column_order]
        
        # Salva o resultado
        df_final.to_csv(output_file, index=False)
        
        print(f"Agregação concluída! Resultado salvo em '{output_file}'")
        print(f"Shape original: {df.shape}")
        print(f"Shape após agregação: {df_final.shape}")
        print(f"\nPrimeiras 3 linhas do resultado:")
        print(df_final.head(3))
        
        # Verifica se a soma das probabilidades é 1
        print(f"\nVerificação - Soma das probabilidades emocionais (primeira linha): {df_final[final_emotion_columns].iloc[0].sum():.6f}")
        
    except Exception as e:
        print(f"Erro durante o processamento: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()