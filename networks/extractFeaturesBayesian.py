import argparse, torch, os, sys, numpy as np, math, matplotlib.pyplot as plt, cv2
from PIL import Image
from torch.nn import functional as F
from matplotlib import cm
from torchvision import transforms
from torchvision import models
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from DatasetClasses.AffectNet import AffectNet
from torch import nn
from networks.EmotionResnetVA import ResnetWithBayesianHead, ResnetWithBayesianGMMHead, ResNet50WithAttentionGMM, ResNet50WithAttentionLikelihood, ResNet50WithAttentionLikelihoodNoVA
from helper.function import visualizeAttentionMaps, printProgressBar

def saveToCSV(preds, files, pathCSV, vad=None, emoLabels=None, mapping=None, num_classes=None, emoHeaders=None):
    # Se num_classes não for fornecido, tenta inferir do shape de preds
    if num_classes is None:
        if preds is not None and len(preds) > 0:
            num_classes = preds.shape[1] if hasattr(preds, 'shape') and len(preds.shape) > 1 else len(preds[0])
        else:
            num_classes = 0
    
    # Lista de emoções para os cabeçalhos - agora dinâmica
    emotion_headers = []
    
    # Adiciona cabeçalhos para as classes de predição
    if emoHeaders is None:
        for i in range(num_classes):
            emotion_headers.append(f'class_{i:03d}')
    else:
        emotion_headers = list(emoHeaders)
    
    # Adiciona cabeçalhos para VAD (se existir)
    if vad is not None and len(vad) > 0:
        vad_columns = vad.shape[1] if hasattr(vad, 'shape') and len(vad.shape) > 1 else 3
        for i in range(vad_columns):
            if i == 0:
                emotion_headers.append('valence')
            elif i == 1:
                emotion_headers.append('arousal')
            elif i == 2:
                emotion_headers.append('dominance')
            else:
                emotion_headers.append(f'vad_{i:03d}')
    
    # Adiciona cabeçalho para rótulo emocional (se existir)
    if emoLabels is not None and 'emotion_label' not in emotion_headers:
        emotion_headers.append('emotion_label')
    
    # Escreve o cabeçalho no arquivo CSV
    with open(pathCSV, 'w') as pcsv:
        pcsv.write('%s,file\n' % (','.join(emotion_headers)))
        
        # Escreve os dados
        for idx, p in enumerate(preds):
            # Escreve as predições de classe
            if hasattr(p, '__len__'):
                for fp in p:
                    pcsv.write(f'{fp},')
            else:
                pcsv.write(f'{p},')
            
            # Escreve os valores VAD (se existirem)
            if vad is not None and idx < len(vad):
                vad_row = vad[idx]
                if hasattr(vad_row, '__len__'):
                    for val in vad_row:
                        pcsv.write(f'{val},')
                else:
                    pcsv.write(f'{vad_row},')
            
            # Escreve o rótulo emocional (se existir)
            if emoLabels is not None and idx < len(emoLabels):
                if mapping is not None and emoLabels[idx] in mapping:
                    pcsv.write(f'{mapping[emoLabels[idx]]},')
                else:
                    pcsv.write(f'{emoLabels[idx]},')
            
            # Escreve o nome do arquivo
            pcsv.write(f"{files[idx]}\n")

def get_gradcam_for_model(model, input_tensor, target_class=None, layer_name='layer4', model_type='attention'):
    """
    Função genérica para Grad-CAM que funciona com diferentes tipos de modelo
    """
    # NÃO colocar modelo em modo de treino - manter em eval
    original_mode = model.training
    model.eval()  # <-- Mudar para eval
    
    # Ativar gradientes para os parâmetros (ainda funciona em eval)
    for param in model.parameters():
        param.requires_grad = True
    
    # Ativar gradientes na entrada
    if not input_tensor.requires_grad:
        input_tensor = input_tensor.clone().requires_grad_(True)

    # Selecionar camada alvo baseada no tipo de modelo
    if hasattr(model, 'model'):  # Modelos baseados em ResNet
        if layer_name == 'layer1':
            target_layer = model.model.layer1[-1].conv3
        elif layer_name == 'layer2':
            target_layer = model.model.layer2[-1].conv3
        elif layer_name == 'layer3':
            target_layer = model.model.layer3[-1].conv3
        elif layer_name == 'layer4':
            target_layer = model.model.layer4[-1].conv3
        elif layer_name == 'conv1':
            target_layer = model.model.conv1
        else:
            target_layer = model.model.layer4[-1].conv3
    else:
        raise ValueError("Modelo não suportado para Grad-CAM")
    
    # Hook para capturar ativações
    activations = []
    gradients = []
    
    def forward_hook(module, input, output):
        activations.append(output)
        return None
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
        return None
    
    # Registrar hooks
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    
    try:
        # Forward pass baseado no tipo de modelo
        if model_type == 'attention_likelihood':
            # ResNet50WithAttentionLikelihood
            features = model.model.conv1(input_tensor)
            features = model.model.bn1(features)
            features = model.model.relu(features)
            features = model.model.maxpool(features)
            
            features = model.model.layer1(features)
            features = model.model.layer2(features)
            features = model.model.layer3(features)
            features = model.model.layer4(features)
            
            features = model.model.avgpool(features)
            features = torch.flatten(features, 1)
            
            distribution_params = model.likelihood_head(features)
            probs = model.probabilities(distribution_params)
            va = model.bayesianHead(probs)
            
        elif model_type == 'attention_no_va':
            # ResNet50WithAttentionLikelihoodNoVA
            features = model.model.conv1(input_tensor)
            features = model.model.bn1(features)
            features = model.model.relu(features)
            features = model.model.maxpool(features)
            
            features = model.model.layer1(features)
            features = model.model.layer2(features)
            features = model.model.layer3(features)
            features = model.model.layer4(features)
            
            features = model.model.avgpool(features)
            features = torch.flatten(features, 1)
            
            probs = model.likelihood_head(features)
            va = None  # Este modelo não tem VAD
            
        else:
            # Outros modelos
            if not isinstance(model, ResNet50WithAttentionLikelihoodNoVA):
                probs, _, va = model(input_tensor)
            else:
                probs = model(input_tensor)
                va = None
        
        # Aplicar softmax
        softmax = nn.Softmax(dim=1)
        probs_softmax = softmax(probs)
        
        # Determinar classe alvo
        if target_class is None:
            target_class = probs_softmax.argmax(dim=1)
        
        # Criar one-hot encoding
        one_hot = torch.zeros_like(probs_softmax)
        one_hot.scatter_(1, target_class.view(-1, 1), 1.0)
        
        # Backward pass
        model.zero_grad()
        probs_softmax.backward(gradient=one_hot, retain_graph=True)
        
        # Extrair ativações e gradientes
        if len(activations) > 0 and len(gradients) > 0:
            activation = activations[0]
            gradient = gradients[0]
            
            # Calcular pesos dos gradientes
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            
            # Calcular Grad-CAM
            cam = torch.sum(weights * activation, dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        else:
            # Fallback: criar CAM vazio
            cam = torch.zeros(1, 1, input_tensor.shape[2]//32, input_tensor.shape[3]//32)
            cam = cam.to(input_tensor.device)
        
        return cam.detach(), probs_softmax.detach(), va.detach() if va is not None else None
        
    finally:
        # Remover hooks
        forward_handle.remove()
        backward_handle.remove()
        
        # Restaurar modo original
        if not original_mode:
            model.eval()
        else:
            model.train()

def create_effective_gradcam_overlay(image, heatmap, alpha=0.5, threshold=0.3):
    """
    Cria um overlay efetivo do Grad-CAM que mostra claramente as áreas focadas
    """
    # Garantir que a imagem está em uint8 RGB
    if image.dtype != np.uint8:
        image = np.uint8(255 * image)
    
    # Redimensionar heatmap para o tamanho da imagem
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Normalizar heatmap para 0-255
    heatmap_norm = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
    heatmap_uint8 = np.uint8(255 * heatmap_norm)
    
    # Aplicar colormap JET
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Método 1: Overlay apenas nas áreas acima do threshold
    mask = heatmap_norm > threshold
    mask_3ch = np.stack([mask, mask, mask], axis=-1)
    
    # Aplicar heatmap apenas nas áreas da máscara
    overlay = image.copy()
    overlay[mask_3ch] = heatmap_rgb[mask_3ch]
    
    # Método 2: Blending com alpha variável baseado na intensidade do heatmap
    alpha_map = heatmap_norm * alpha
    alpha_map = np.clip(alpha_map, 0, alpha)
    
    alpha_map_3ch = np.stack([alpha_map, alpha_map, alpha_map], axis=-1)
    
    overlay_variable = image.astype(float) * (1 - alpha_map_3ch) + heatmap_rgb.astype(float) * alpha_map_3ch
    overlay_variable = np.uint8(np.clip(overlay_variable, 0, 255))
    
    # Método 3: Contornos das áreas ativas sobre a imagem
    heatmap_threshold = np.uint8(255 * (heatmap_norm > threshold))
    contours, _ = cv2.findContours(heatmap_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_with_contours = image.copy()
    # Desenhar contornos em verde
    cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 2)
    
    return {
        'masked_overlay': overlay,
        'variable_overlay': overlay_variable,
        'contours': img_with_contours,
        'heatmap_raw': heatmap_resized,
        'heatmap_color': heatmap_rgb
    }

def visualize_gradcam_for_model_type(model, image_path, save_path=None, layer_name='layer4', 
                                    threshold=0.3, model_type='attention', emotionsName=["neutral", "happy", "sad", "surprise", "fear", "disgust", "angry", "contempt"]):
    """
    Visualização do Grad-CAM adaptada para diferentes tipos de modelo
    """
    
    # Carregar imagem
    img = Image.open(image_path).convert('RGB')
    
    # Transformações
    transform_model = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_vis = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    img_tensor = transform_model(img).unsqueeze(0).cuda()
    img_vis = transform_vis(img).unsqueeze(0).cuda()
    
    # Obter Grad-CAM baseado no tipo de modelo
    print(f"Computing Grad-CAM for {model_type} model...")
    
    if isinstance(model, ResNet50WithAttentionLikelihood):
        model_type_str = 'attention_likelihood'
    elif isinstance(model, ResNet50WithAttentionLikelihoodNoVA):
        model_type_str = 'attention_no_va'
    else:
        model_type_str = 'generic'
    
    cam, probs, va = get_gradcam_for_model(
        model, img_tensor, 
        layer_name=layer_name,
        model_type=model_type_str
    )
    
    # Converter para numpy
    cam_np = cam.squeeze().cpu().numpy()
    probs_np = probs.squeeze(0).cpu().numpy() if len(probs.shape) > 1 else probs.cpu().numpy()
    
    # Preparar imagem original
    img_np = img_vis.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_uint8 = np.uint8(255 * img_np)
    
    # Criar overlays efetivos
    overlays = create_effective_gradcam_overlay(img_uint8, cam_np, alpha=0.6, threshold=threshold)
    
    # Estatísticas do heatmap
    print(f"  Heatmap stats - Min: {cam_np.min():.4f}, Max: {cam_np.max():.4f}, Mean: {cam_np.mean():.4f}")
    print(f"  Threshold used: {threshold}")
    
    # Top predições
    top_k = min(5, len(probs_np))
    top_indices = np.argsort(probs_np)[-top_k:][::-1]
    top_probs = probs_np[top_indices]
    
    # Emoções base (pode ser personalizado)
    default_emotions = emotionsName
    
    # Ajustar para o número de classes do modelo
    if len(default_emotions) < len(probs_np):
        emotions = default_emotions + [f'class_{i}' for i in range(len(default_emotions), len(probs_np))]
    else:
        emotions = default_emotions[:len(probs_np)]
    
    top_emotions = [emotions[i] for i in top_indices]
    
    # Criar visualização
    fig = plt.figure(figsize=(20, 12))
    
    # Layout da figura
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. Imagem original
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_uint8)
    ax1.set_title('Original Image', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # 2. Heatmap colorido
    ax2 = fig.add_subplot(gs[0, 1])
    heatmap_display = overlays['heatmap_raw']
    im2 = ax2.imshow(heatmap_display, cmap='jet')
    ax2.set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # 3. Overlay com máscara
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(overlays['masked_overlay'])
    ax3.set_title(f'Masked Overlay (threshold={threshold})', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # 4. Overlay com alpha variável
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(overlays['variable_overlay'])
    ax4.set_title('Variable Alpha Overlay', fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    # 5. Imagem com contornos
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.imshow(overlays['contours'])
    ax5.set_title('Image with Heatmap Contours', fontsize=12, fontweight='bold')
    ax5.axis('off')
    
    # 6. Histograma do heatmap
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.hist(heatmap_display.flatten(), bins=50, color='blue', alpha=0.7)
    ax6.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold={threshold}')
    ax6.set_title('Heatmap Activation Distribution', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Activation Value')
    ax6.set_ylabel('Frequency')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Top predições
    ax7 = fig.add_subplot(gs[1, 2])
    colors_bar = plt.cm.viridis(np.linspace(0.2, 0.8, top_k))
    bars = ax7.barh(range(top_k), top_probs, color=colors_bar)
    ax7.set_yticks(range(top_k))
    ax7.set_yticklabels(top_emotions, fontsize=10)
    ax7.set_xlabel('Probability', fontsize=11)
    ax7.set_title(f'Top-{top_k} Predictions', fontsize=12, fontweight='bold')
    ax7.invert_yaxis()
    
    # Adicionar valores nas barras
    for i, (bar, prob) in enumerate(zip(bars, top_probs)):
        width = bar.get_width()
        ax7.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{prob:.3f}', va='center', fontsize=9)
    
    # 8. Ativações por região
    ax8 = fig.add_subplot(gs[1, 3])
    region_size = 8
    h, w = heatmap_display.shape
    h_regions = h // region_size
    w_regions = w // region_size
    
    region_activations = np.zeros((h_regions, w_regions))
    for i in range(h_regions):
        for j in range(w_regions):
            region = heatmap_display[i*region_size:(i+1)*region_size, 
                                    j*region_size:(j+1)*region_size]
            region_activations[i, j] = region.mean()
    
    im8 = ax8.imshow(region_activations, cmap='YlOrRd', aspect='auto')
    ax8.set_title(f'Average Activation by Region', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Region X')
    ax8.set_ylabel('Region Y')
    plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)
    
    # 9. Informações do modelo (2 colunas)
    ax9 = fig.add_subplot(gs[2, :2])
    ax9.axis('off')
    
    # Texto informativo
    pred_class = np.argmax(probs_np)
    pred_prob = probs_np[pred_class]
    pred_emotion = emotions[pred_class] if pred_class < len(emotions) else f'class_{pred_class}'
    
    info_text = f"""
    MODEL INFORMATION
    {'='*40}
    • Model Type: {model.__class__.__name__}
    • Layer Used: {layer_name}
    • Image: {os.path.basename(image_path)}
    • Num Classes: {len(probs_np)}
    
    PREDICTION
    {'='*40}
    • Predicted Class: {pred_class}
    • Predicted Emotion: {pred_emotion}
    • Confidence: {pred_prob:.4f}
    • Top-3 Predictions:
    """
    
    for i in range(min(3, len(top_emotions))):
        info_text += f"   {i+1}. {top_emotions[i]}: {top_probs[i]:.4f}\n"
    
    if va is not None:
        va_np = va.cpu().numpy() if torch.is_tensor(va) else va
        info_text += f"""
    VAD ESTIMATES
    {'='*40}
    • Valence: {va_np[0][0]:.3f}
    • Arousal: {va_np[0][1]:.3f}
    • Dominance: {va_np[0][2]:.3f}
    """
    
    ax9.text(0.05, 0.95, info_text, transform=ax9.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 10. Heatmap 3D
    ax10 = fig.add_subplot(gs[2, 2:], projection='3d')
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    Z = heatmap_display
    
    surf = ax10.plot_surface(X, Y, Z, cmap='jet', alpha=0.8, linewidth=0, antialiased=True)
    ax10.set_title('3D Heatmap Visualization', fontsize=12, fontweight='bold')
    ax10.set_xlabel('Width')
    ax10.set_ylabel('Height')
    ax10.set_zlabel('Activation')
    
    # Título principal
    plt.suptitle(f'GRAD-CAM ANALYSIS: {os.path.basename(image_path)}', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved Grad-CAM visualization to {save_path}")
    
    return overlays['masked_overlay'], probs_np, va

def train():
    parser = argparse.ArgumentParser(description='Extract features from resnet emotion')
    parser.add_argument('--weights', help='Weights', required=True)
    parser.add_argument('--pathBase', help='Path for valence and arousal dataset', required=True)
    parser.add_argument('--batch', type=int, help='Size of the batch', required=True)
    parser.add_argument('--output', default=None, help='File to save csv', required=True)
    parser.add_argument('--dataset', help='Dataset for feature extraction', required=False, default="OMG")
    parser.add_argument('--resnetInnerModel', help='Model for feature extraction', required=False, type=int, default=18)
    parser.add_argument('--emotionModel', help='Model for feature extraction', required=False, default="resnetBayesGMM")
    parser.add_argument('--classQuantity', help='Model for feature extraction', required=False, type=int, default=14)
    parser.add_argument('--visualize_gradcam', action='store_true', help='Visualize Grad-CAM')
    parser.add_argument('--gradcam_samples', type=int, default=5, help='Number of samples to visualize Grad-CAM')
    parser.add_argument('--gradcam_layer', type=str, default='layer4', 
                       help='Layer to use for Grad-CAM (layer1, layer2, layer3, layer4, conv1)')
    parser.add_argument('--emotion_labels', type=str, default=None,
                       help='Custom emotion labels (comma-separated)')
    parser.add_argument('--typeExperiment', type=str, default='ORIGINAL_VAD_EXP',
                       help='Type of experiment')
    args = parser.parse_args()

    checkpoint = torch.load(args.weights)

    model = None
    if args.emotionModel == "resnetBayesGMM":
        model = ResnetWithBayesianGMMHead(classes=args.classQuantity, resnetModel=args.resnetInnerModel)
    elif args.emotionModel == "resnetBayes":
        model = ResnetWithBayesianHead(args.classQuantity, resnetModel=args.resnetInnerModel)
    elif args.emotionModel == "resnetAttentionGMM":
        model = ResNet50WithAttentionGMM(num_classes=args.classQuantity, bottleneck='none', bayesianHeadType='VAD')
    elif args.emotionModel == "resnetAttentionLikelihood":
        model = ResNet50WithAttentionLikelihood(num_classes=args.classQuantity, bottleneck='none', bayesianHeadType='VAD')
    elif args.emotionModel == "simpleNetwork":
        model = ResNet50WithAttentionLikelihoodNoVA(num_classes=args.classQuantity, bottleneck='none')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.weights)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.to(device)
    
    print("Model loaded")
    print(f"Model type: {type(model)}")
    print(f"Number of classes: {args.classQuantity}")
    print(f"Type of experiment: {args.typeExperiment}")
    
    # Processar labels de emoções personalizadas se fornecidas
    emotion_labels = None
    if args.emotion_labels:
        emotion_labels = [label.strip() for label in args.emotion_labels.split(',')]
        print(f"Custom emotion labels: {emotion_labels}")
    
    # Configurar transformações
    data_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = AffectNet(afectdata=args.pathBase, transform=data_transforms, typeExperiment=args.typeExperiment)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=False)
    model.eval()
    
    # Inicializar variáveis para armazenar resultados
    pathFile = []
    predictions = None
    labelsGrouped = None
    labelTrue = None
    labelPreds = None
    vadPreds = None
    vadTrue = None
    
    soft = nn.Softmax(dim=1)
    
    # Processar batch por batch
    with torch.no_grad():
        for idxBtc, data in enumerate(val_loader):
            printProgressBar(idxBtc, math.ceil(len(dataset) / args.batch), length=50, prefix='Processing batches...')
            images, (labels, vadLabels, emotion), pathsForFiles = data
            
            # Forward pass normal (sem gradientes)
            if not isinstance(model, ResNet50WithAttentionLikelihoodNoVA):
                outputs, _, vad = model(images.to(device))
            else:
                outputs = model(images.to(device))
                vad = None
            
            outputs = soft(outputs)
            
            # Armazenar resultados
            _, pred = torch.max(outputs.data, 1)
            labelPreds = pred.cpu().numpy() if labelPreds is None else np.concatenate((pred.cpu().numpy(), labelPreds))
            labelTrue = emotion.cpu().numpy() if labelTrue is None else np.concatenate((emotion.cpu().numpy(), labelTrue))
            
            prediction = outputs.cpu().numpy()
            predictions = prediction if predictions is None else np.concatenate((prediction, predictions))
            
            if vad is not None:
                vp = vad.cpu().numpy()
                vadPreds = vp if vadPreds is None else np.concatenate((vp, vadPreds))
            
            currLabel = labels.cpu().numpy()
            labelsGrouped = currLabel if labelsGrouped is None else np.concatenate((currLabel, labelsGrouped))
            
            if vadLabels is not None:
                vadC = vadLabels.cpu().numpy()
                vadTrue = vadC if vadTrue is None else np.concatenate((vadC, vadTrue))
            
            pathFile.extend(list(pathsForFiles))
    
    # Salvar resultados CSV
    saveToCSV(predictions, pathFile, args.output, vad=vadPreds, emoLabels=labelPreds, 
              mapping=[1, 7, 3, 6, 5, 4, 2, 0], num_classes=args.classQuantity,emoHeaders=emotion_labels)
    

    saveToCSV(labelsGrouped, pathFile, args.output[:-4] + "_labels.csv", 
              vad=vadTrue, emoLabels=labelTrue, num_classes=args.classQuantity,emoHeaders=emotion_labels)
    
    # Gerar visualizações Grad-CAM se solicitado
    if args.visualize_gradcam:
        # Verificar quais modelos suportam Grad-CAM
        supported_models = [
            "resnetAttentionLikelihood",
            "simpleNetwork",  # ResNet50WithAttentionLikelihoodNoVA
            "resnetAttentionGMM"
        ]
        
        if args.emotionModel in supported_models:
            print("\n" + "="*60)
            print(f"Generating Grad-CAM visualizations for {args.emotionModel}...")
            print("="*60)
            
            # Criar diretório de saída
            output_dir = f'gradcam_results_{args.emotionModel}'
            os.makedirs(output_dir, exist_ok=True)
            
            # Selecionar algumas imagens aleatórias
            import random
            if len(pathFile) > args.gradcam_samples:
                selected_indices = random.sample(range(len(pathFile)), args.gradcam_samples)
            else:
                selected_indices = range(len(pathFile))
            
            for idx in selected_indices[:args.gradcam_samples]:
                image_path = pathFile[idx]
                filename = os.path.basename(image_path)
                save_path = f'{output_dir}/{filename}_gradcam_{args.gradcam_layer}.png'
                
                print(f"\nProcessing: {filename}")
                
                try:
                    # Usar a função adaptada para diferentes modelos
                    cam, probs, va = visualize_gradcam_for_model_type(
                        model=model,
                        image_path=image_path,
                        save_path=save_path,
                        layer_name=args.gradcam_layer,
                        threshold=0.3,
                        emotionsName=emotion_labels
                    )
                    
                    if cam is not None:
                        print(f"✓ Successfully generated Grad-CAM")
                        print(f"  True label: {labelTrue[idx]}")
                        print(f"  Predicted: {labelPreds[idx]}")
                        if va is not None:
                            va_np = va.cpu().numpy() if torch.is_tensor(va) else va
                            print(f"  VAD: {va_np}")
                
                except Exception as e:
                    print(f"✗ Error: {str(e)}")
                    import traceback
                    traceback.print_exc()

        else:
            print(f"\n⚠ Warning: Grad-CAM not supported for model type: {args.emotionModel}")
            print("Supported models: resnetAttentionLikelihood, simpleNetwork, resnetAttentionGMM")

if __name__ == '__main__':
    train()