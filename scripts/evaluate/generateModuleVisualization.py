import torch, os, sys
import torch.nn as nn
import torchvision.models as models
from torchviz import make_dot
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from networks.EmotionResnetVA import ResNet50WithAttentionGMM

if __name__ == "__main__":
    # Instancie a rede
    num_classes_example = 10 # Um número de classes de exemplo
    # Teste com diferentes opções de bottleneck: 'first', 'second', 'both', ou None
    # model = ResNet50WithAttentionGMM(num_classes=num_classes_example, bottleneck='both')
    model = ResNet50WithAttentionGMM(num_classes=13, bottleneck='none') # Testa com atenção na conv1
    
    # Crie um tensor de entrada de exemplo (batch_size, canais, altura, largura)
    # Ex: (1, 3, 224, 224) para imagens RGB de tamanho padrão da ResNet
    dummy_input = torch.randn(1, 3, 224, 224)

    # Passe o tensor de entrada pelo modelo
    # Para make_dot, precisamos de uma única saída do grafo.
    # Como seu forward retorna uma tupla, pegamos o primeiro elemento ou combinamos.
    # Para uma visualização completa, é melhor passar a última saída que abrange todo o grafo.
    
    # Desempacota as saídas do forward
    probs, distributions, va = model(dummy_input)

    # O make_dot precisa de um único tensor para rastrear.
    # Vamos usar `va` (a saída final da cabeça Bayesiana) para garantir que todo o caminho
    # seja visualizado.
    dot = make_dot(va, params=dict(model.named_parameters()))

    # Salve a visualização em um arquivo (ex: PNG, PDF)
    # dot.render("resnet50_attention_gmm_graph", format="pdf", view=True) # Para abrir automaticamente
    dot.render("resnet50_attention_gmm_graph", format="png") # Salva como PNG

    print("Visualização da rede 'ResNet50WithAttentionGMM' salva como 'resnet50_attention_gmm_graph.png'")
    print("Verifique o arquivo gerado para ver a estrutura da rede.")