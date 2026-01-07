import torch
import torch.nn as nn
import torch.nn.functional as F

class KnowledgeGuidedConsistencyLoss(nn.Module):
    def __init__(self, num_classes=8, gamma=2.0, alpha=0.25, 
                 prior_strength=0.5, learnable_scale=True, reduction='mean'):
        """
        prior_strength: Quanto o conhecimento prévio influencia (0-1)
        learnable_scale: Se a escala dos conflitos é aprendível
        """
        super(KnowledgeGuidedConsistencyLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.num_classes = num_classes
        self.prior_strength = prior_strength
        self.reduction = reduction
        
        # Matriz de conhecimento prévio (não aprendível)
        self.prior_matrix = self._create_prior_matrix()
        
        # Matriz de aprendizado residual (aprendível)
        self.residual_matrix = nn.Parameter(torch.zeros(num_classes, num_classes))
        
        # Escala global aprendível
        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.scale = 1.0
        
        # Bias por classe (algumas classes são mais conflitantes)
        self.class_bias = nn.Parameter(torch.zeros(num_classes))
        
    def _create_prior_matrix(self):
        """Cria matriz de conhecimento prévio baseada em psicologia"""
        prior = torch.zeros(self.num_classes, self.num_classes)
        
        # Mapeamento de emoções
        emotions = {
            0: 'happy', 1: 'contempt', 2: 'surprised', 3: 'angry',
            4: 'disgusted', 5: 'fearful', 6: 'sad', 7: 'neutral'
        }
        
        # Conflitos conhecidos da literatura
        strong_conflicts = [
            (0, 3), (0, 4), (0, 5), (0, 6),  # happy vs negative emotions
            (2, 4), (2, 5),  # surprised vs disgust/fear
            (3, 0), (3, 7),  # angry vs happy/neutral
            (6, 0), (6, 7),  # sad vs happy/neutral
        ]
        
        moderate_conflicts = [
            (1, 0), (1, 3), (1, 4), (1, 7),  # contempt
            (4, 0), (4, 2), (4, 7),  # disgusted
            (5, 0), (5, 2), (5, 7),  # fearful
            (7, 0), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6),  # neutral
        ]
        
        # Aplicar conflitos
        for i, j in strong_conflicts:
            if i < self.num_classes and j < self.num_classes:
                prior[i, j] = 0.8
                prior[j, i] = 0.8
                
        for i, j in moderate_conflicts:
            if i < self.num_classes and j < self.num_classes:
                prior[i, j] = 0.4
                prior[j, i] = 0.4
        
        # Auto-conflito zero
        for i in range(self.num_classes):
            prior[i, i] = 0.0
            
        return prior
    
    def get_conflict_matrix(self):
        """Combina conhecimento prévio com aprendizado"""
        # Matriz residual simetrizada
        residual = (self.residual_matrix + self.residual_matrix.t()) / 2
        
        # Aplicar ativação suave
        residual = torch.tanh(residual) * 0.5  # Limita a [-0.5, 0.5]
        
        # Combinação com prior
        combined = self.prior_strength * self.prior_matrix.to(residual.device)
        combined += (1 - self.prior_strength) * (residual + 0.5)  # Mapeia para [0, 1]
        
        # Aplicar escala e bias
        scaled = self.scale * combined
        
        # Adicionar bias por classe
        bias_matrix = self.class_bias.view(-1, 1) + self.class_bias.view(1, -1)
        scaled = scaled + 0.1 * bias_matrix
        
        # Garantir não-negatividade
        conflict_weights = F.softplus(scaled)
        
        # Normalizar
        conflict_weights = conflict_weights / (conflict_weights.max() + 1e-8)
        
        return conflict_weights
    
    def forward(self, logits, targets):
        # Obter matriz de conflito
        conflict_weights = self.get_conflict_matrix()
        
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
        
        # Consistency Loss
        probs = torch.sigmoid(logits)
        
        # Calcular produto pareado eficiente
        probs_expanded = probs.unsqueeze(2)
        probs_t = probs.unsqueeze(1)
        joint_probs = probs_expanded * probs_t
        
        # Usar apenas triângulo superior
        mask = torch.triu(torch.ones_like(conflict_weights), diagonal=1)
        consistency_loss = (joint_probs * conflict_weights.unsqueeze(0) * mask.unsqueeze(0))
        consistency_loss = consistency_loss.sum(dim=(1,2)).mean()
        
        # Regularização para suavidade
        smoothness_loss = torch.mean(torch.abs(self.residual_matrix - self.residual_matrix.t()))
        
        # Loss total
        total_loss = focal_loss + consistency_loss + 0.001 * smoothness_loss
        
        return total_loss