import torch
import torch.nn as nn
import torch.nn.functional as F

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