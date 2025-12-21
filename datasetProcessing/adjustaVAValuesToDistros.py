import numpy as np, argparse, os, sys, torch, csv
from torchvision import transforms
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from DatasetClasses.AffectNet import AffectNet
from helper.function import printProgressBar

def adjust_point_to_ellipse(v0, a0, mu_v, mu_a, sigma_v, sigma_a, r=1.0):
    """
    Ajusta um ponto (v0, a0) para o limite de uma elipse definida por médias (mu_v, mu_a),
    desvios padrão (sigma_v, sigma_a) e raio r no espaço padronizado.
    
    Parâmetros:
    - v0, a0: coordenadas originais do ponto.
    - mu_v, mu_a: médias.
    - sigma_v, sigma_a: desvios padrão.
    - r: raio da elipse no espaço padronizado (padrão=1 para 1 desvio padrão).
    
    Retorna:
    - v_adj, a_adj: coordenadas ajustadas.
    - distance_ratio: razão entre a norma original e r (útil para verificação).
    """
    # Transformação para coordenadas padronizadas
    z_v = (v0 - mu_v) / sigma_v
    z_a = (a0 - mu_a) / sigma_a
    
    # Norma no espaço padronizado
    norm_z = np.sqrt(z_v**2 + z_a**2)
    
    # Se a norma for maior que r, projeta no círculo de raio r
    if norm_z > r:
        z_v_adj = (r / norm_z) * z_v
        z_a_adj = (r / norm_z) * z_a
    else:
        # Se já estiver dentro, pode-se manter ou projetar para o limite?
        # Aqui, mantemos o ponto original (mas pode-se forçar para o limite se desejar)
        z_v_adj = z_v
        z_a_adj = z_a
    
    # Transformação de volta para o espaço original
    v_adj = mu_v + sigma_v * z_v_adj
    a_adj = mu_a + sigma_a * z_a_adj
    
    return v_adj, a_adj, norm_z / r

def load_emotional_distributions(csv_file):
    """
    Carrega as distribuições emocionais do arquivo CSV.
    Retorna um dicionário com as médias e desvios padrão para cada classe.
    """
    distributions = {}
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_name = row['class']
            distributions[class_name] = {
                'valence_mean': float(row['valence mean']),
                'valence_std': float(row['valence std']),
                'arousal_mean': float(row['arousal mean']),
                'arousal_std': float(row['arousal std']),
                'dominance_mean': float(row['dominance mean']),
                'dominance_std': float(row['dominance std'])
            }
    
    # Adicionar distribuição para neutro (como mencionado)
    distributions['neutral'] = {
        'valence_mean': 0.0,
        'valence_std': 0.1,
        'arousal_mean': 0.0,
        'arousal_std': 0.1,
        'dominance_mean': 0.0,
        'dominance_std': 0.1
    }
    
    return distributions

def map_affectnet_class_to_name(class_idx):
    """
    Mapeia o índice da classe do AffectNet para o nome da emoção.
    Baseado na documentação do AffectNet.
    """
    class_mapping = {
        0: 'neutral',
        1: 'happy',
        2: 'sad',
        3: 'surprised',
        4: 'fearful',
        5: 'disgusted',
        6: 'angry',
        7: 'contempt'
    }
    return class_mapping.get(class_idx, None)

# Exemplo de uso:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adjust VA values to fit within distributions')
    parser.add_argument('--pathBase', help='Path for valence and arousal dataset', required=True)
    parser.add_argument('--distroFile', help='Path for valence and arousal dataset', required=True)
    parser.add_argument('--outputFile', help='Path to save adjusted VA values', default='adjusted_VA_values.csv')
    parser.add_argument('--radius', type=float, help='Radius for ellipse (standard deviations)', default=1.0)
    args = parser.parse_args()
    
    data_transforms = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    
    print("Loading training set")
    dataset = AffectNet(afectdata=os.path.join(args.pathBase,'train_set'),
                       transform=data_transforms,
                       typeExperiment='PROBS_VA_EXP')
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=False)
    
    # Carregar distribuições emocionais
    print(f"Loading emotional distributions from {args.distroFile}")
    distributions = load_emotional_distributions(args.distroFile)
    
    # Lista para armazenar resultados
    all_adjusted_values = []
    
    print("Adjusting VA values to fit within emotional distributions...")
    total_batches = len(train_loader)
    
    for batch_idx, (images, values, pathFiles) in enumerate(train_loader):
        va_values = values[1]
        class_indices = values[2]
        printProgressBar(batch_idx + 1, total_batches, prefix='Progress:', suffix='Complete', length=50)
        
        batch_size = va_values.shape[0]
        
        for i in range(batch_size):
            # Obter valores originais
            v_original = va_values[i, 0].item()
            a_original = va_values[i, 1].item()
            class_idx = class_indices[i].item()
            
            # Mapear classe para nome
            emotion_name = map_affectnet_class_to_name(class_idx)
            if emotion_name is None:
                print(f"Warning: Unknown class index {class_idx} at {pathFiles[i]}")
                continue
            
            # Obter distribuição para esta emoção
            if emotion_name in distributions:
                dist = distributions[emotion_name]
                
                # Ajustar ponto para ficar na elipse
                v_adjusted, a_adjusted, ratio = adjust_point_to_ellipse(
                    v_original, a_original,
                    dist['valence_mean'], dist['arousal_mean'],
                    dist['valence_std'], dist['arousal_std'],
                    r=args.radius
                )
                fileName = os.path.basename(pathFiles[i])[:-4]  # Remover extensão
                np.save(os.path.join(args.pathBase,'train_set','annotations',f'{fileName}_adjusted_val.npy'), v_adjusted)
                np.save(os.path.join(args.pathBase,'train_set','annotations',f'{fileName}_adjusted_aro.npy'), a_adjusted)
                # Armazenar resultados
                all_adjusted_values.append({
                    'emotion': emotion_name,
                    'class_idx': class_idx,
                    'valence_original': v_original,
                    'arousal_original': a_original,
                    'valence_adjusted': v_adjusted,
                    'arousal_adjusted': a_adjusted,
                    'distance_ratio': ratio,
                    'was_adjusted': ratio > 1.0
                })
            else:
                print(f"Warning: Emotion '{emotion_name}' not found in distributions")
    
    # Exibir estatísticas
    print("\n=== Adjustment Statistics ===")
    total_samples = len(all_adjusted_values)
    adjusted_samples = sum(1 for item in all_adjusted_values if item['was_adjusted'])
    print(f"Total samples processed: {total_samples}")
    print(f"Samples adjusted: {adjusted_samples} ({adjusted_samples/total_samples*100:.2f}%)")
    
    # Calcular médias de ajuste por emoção
    print("\n=== Adjustment by Emotion ===")
    emotions = set(item['emotion'] for item in all_adjusted_values)
    for emotion in sorted(emotions):
        emotion_items = [item for item in all_adjusted_values if item['emotion'] == emotion]
        adjusted_count = sum(1 for item in emotion_items if item['was_adjusted'])
        avg_ratio = np.mean([item['distance_ratio'] for item in emotion_items])
        print(f"{emotion:12s}: {len(emotion_items):4d} samples, "
              f"{adjusted_count:4d} adjusted ({adjusted_count/len(emotion_items)*100:5.1f}%), "
              f"avg ratio: {avg_ratio:.3f}")
    
    # Salvar resultados em CSV
    print(f"\nSaving adjusted values to {args.outputFile}")
    with open(args.outputFile, 'w', newline='') as f:
        fieldnames = ['emotion', 'class_idx', 'valence_original', 'arousal_original',
                     'valence_adjusted', 'arousal_adjusted', 'distance_ratio', 'was_adjusted']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_adjusted_values)
    
    print("Done!")
    
    # Exemplo de visualização para algumas amostras
    if total_samples > 0:
        print("\n=== Sample Adjustments ===")
        for i in range(min(5, total_samples)):
            item = all_adjusted_values[i]
            print(f"Emotion: {item['emotion']}")
            print(f"  Original:  valence={item['valence_original']:.3f}, arousal={item['arousal_original']:.3f}")
            print(f"  Adjusted:  valence={item['valence_adjusted']:.3f}, arousal={item['arousal_adjusted']:.3f}")
            print(f"  Distance ratio: {item['distance_ratio']:.3f} {'(adjusted)' if item['was_adjusted'] else '(inside)'}")
            print()