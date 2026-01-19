import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class FocalConsistencyLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, conflict_weight=1.0, reduction='mean'):
        """
        Args:
            gamma (float): Fator de foco. Quanto maior, mais a loss ignora exemplos fáceis. 
                           (Padrão da literatura é 2.0)
            alpha (float): Fator de balanceamento para a classe positiva.
            conflict_weight (float): Peso da penalidade por contradição emocional.
            reduction (str): 'mean' ou 'sum' para a agregação final.
        """
        super(FocalConsistencyLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.conflict_weight = conflict_weight
        self.reduction = reduction
        
        # Mapa de índices (baseado no seu output):
        # 0:happy, 1:contempt, 2:surprised, 3:angry, 
        # 4:disgusted, 5:fearful, 6:sad, 7:neutral
        
        # Pares de conflito (Valência Positiva vs Negativa)
        self.conflict_pairs = [
            (0, 6), # Happy vs Sad
            (0, 3), # Happy vs Angry
            (0, 4), # Happy vs Disgusted
            (0, 5), # Happy vs Fearful
            #(2, 4), # Surprised vs Disgusted
            #(2, 5), # Surprised vs Fearful
            # Opcional: Neutral (7) vs Todas as outras fortes.
            # Se a rede diz que é muito Neutral, não deve ser muito Angry.
            (7, 3), (7, 0), (7, 6) 
        ]

    def forward(self, logits, targets):
        # --- PARTE 1: Focal Loss ---
        # BCE with Logits calcula a cross entropy básica
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Obtemos a probabilidade da classe verdadeira (p_t)
        # p_t = p se y=1, senão (1-p)
        # Matematicamente, isso é equivalente a exp(-bce_loss)
        p_t = torch.exp(-bce_loss)
        
        # Termo de modulação focal: (1 - p_t)^gamma
        focal_term = (1 - p_t) ** self.gamma
        
        # Aplicar alpha balancing (opcional, mas recomendado)
        # Se target=1 usa alpha, se target=0 usa (1-alpha)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        focal_loss = alpha_t * focal_term * bce_loss
        
        if self.reduction == 'mean':
            focal_loss = focal_loss.mean()
        elif self.reduction == 'sum':
            focal_loss = focal_loss.sum()
            
        # --- PARTE 2: Consistency Loss ---
        probs = torch.sigmoid(logits)
        consistency_loss = 0.0
        
        for i, j in self.conflict_pairs:
            # Penaliza se ambas as emoções do par forem altas
            joint_prob = probs[:, i] * probs[:, j]
            consistency_loss += joint_prob.mean()
            
        # --- COMBINAÇÃO FINAL ---
        total_loss = focal_loss + (self.conflict_weight * consistency_loss)
        
        return total_loss
    
class RegularizedLearnedConsistencyLoss(nn.Module):
    def __init__(self, num_classes=8, gamma=2.0, alpha=0.25, 
                 sparsity_weight=0.01, reduction='mean'):
        super(RegularizedLearnedConsistencyLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.num_classes = num_classes
        self.sparsity_weight = sparsity_weight
        self.reduction = reduction
        
        # Matriz de conflito simétrica
        self.conflict_logits = nn.Parameter(torch.zeros(num_classes, num_classes))
        
        # Inicializar com alguns conflitos conhecidos
        self._initialize_with_prior_knowledge()
    
    def _initialize_with_prior_knowledge(self):
        """Inicializa com conhecimento prévio sobre conflitos emocionais"""
        conflict_pairs = [(0,6), (0,3), (0,4), (0,5), (2,4), (2,5)]
        with torch.no_grad():
            for i, j in conflict_pairs:
                self.conflict_logits[i, j] = 1.0
                self.conflict_logits[j, i] = 1.0

    def forward(self, logits, targets):
        # Focal Loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = torch.exp(-bce_loss)
        focal_term = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * focal_term * bce_loss
        
        if self.reduction == 'mean':
            focal_loss = focal_loss.mean()
        elif self.reduction == 'sum':
            focal_loss = focal_loss.sum()
        
        # Consistency Loss com pesos aprendidos
        probs = torch.sigmoid(logits)
        
        # Obter pesos usando sigmoid (0 a 1)
        conflict_weights = torch.sigmoid(self.conflict_logits)
        
        # Calcular loss de consistência
        probs_i = probs.unsqueeze(2)  # [batch, classes, 1]
        probs_j = probs.unsqueeze(1)  # [batch, 1, classes]
        joint_probs = probs_i * probs_j
        
        # Usar apenas metade para evitar dupla contagem
        mask = torch.triu(torch.ones_like(conflict_weights), diagonal=1)
        consistency_terms = (joint_probs * conflict_weights.unsqueeze(0) * mask.unsqueeze(0)).sum(dim=(1,2))
        
        if self.reduction == 'mean':
            consistency_loss = consistency_terms.mean()
        elif self.reduction == 'sum':
            consistency_loss = consistency_terms.sum()
        
        # Regularização para esparsidade
        sparsity_loss = torch.mean(torch.abs(self.conflict_logits))
        
        # Loss total
        total_loss = focal_loss + consistency_loss + self.sparsity_weight * sparsity_loss
        
        return total_loss
    
    def visualize_conflict_matrix(self):
        """Visualiza a matriz de conflito aprendida"""
        
        with torch.no_grad():
            weights = torch.sigmoid(self.conflict_logits).cpu().numpy()
            
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(weights, cmap='Reds')
        
        # Configurações do gráfico
        emotion_names = ['happy', 'contempt', 'surprised', 'angry', 
                        'disgusted', 'fearful', 'sad', 'neutral']
        ax.set_xticks(range(len(emotion_names)))
        ax.set_yticks(range(len(emotion_names)))
        ax.set_xticklabels(emotion_names, rotation=45)
        ax.set_yticklabels(emotion_names)
        ax.set_title("Matriz de Conflito Aprendida")
        
        plt.colorbar(im)
        plt.tight_layout()
        return fig