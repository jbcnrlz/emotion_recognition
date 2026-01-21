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
    

class FocalConsistencyLoss14Emotions(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, conflict_weight=1.0, reduction='mean'):
        """
        Args:
            gamma (float): Fator de foco para Focal Loss.
            alpha (float): Fator de balanceamento para classe positiva.
            conflict_weight (float): Peso da penalidade por contradição emocional.
            reduction (str): 'mean' ou 'sum' para agregação final.
        """
        super(FocalConsistencyLoss14Emotions, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.conflict_weight = conflict_weight
        self.reduction = reduction
        
        # Mapeamento das 14 emoções:
        # 0:happy, 1:contempt, 2:elated, 3:hopeful, 4:surprised,
        # 5:proud, 6:loved, 7:angry, 8:astonished, 9:disgusted,
        # 10:fearful, 11:sad, 12:fatigued, 13:neutral
        
        # Pares de conflito - regras de consistência emocional:
        self.conflict_pairs = [
            # Emoções positivas vs negativas básicas
            (0, 7),   # Happy vs Angry
            (0, 9),   # Happy vs Disgusted
            (0, 10),  # Happy vs Fearful
            (0, 11),  # Happy vs Sad
            (0, 12),  # Happy vs Fatigued
            
            # Elated (elevado/alegre intenso) vs negativas
            (2, 7),   # Elated vs Angry
            (2, 9),   # Elated vs Disgusted
            (2, 10),  # Elated vs Fearful
            (2, 11),  # Elated vs Sad
            (2, 12),  # Elated vs Fatigued
            
            # Hopeful (esperançoso) vs negativas
            (3, 7),   # Hopeful vs Angry
            (3, 9),   # Hopeful vs Disgusted
            (3, 10),  # Hopeful vs Fearful
            (3, 12),  # Hopeful vs Fatigued
            
            # Proud (orgulhoso) vs negativas
            (5, 7),   # Proud vs Angry
            (5, 9),   # Proud vs Disgusted
            (5, 10),  # Proud vs Fearful
            (5, 11),  # Proud vs Sad
            
            # Loved (amado) vs negativas
            (6, 7),   # Loved vs Angry
            (6, 9),   # Loved vs Disgusted
            (6, 10),  # Loved vs Fearful
            (6, 11),  # Loved vs Sad
            
            # Neutral vs emoções intensas (positivas e negativas)
            (13, 0),  # Neutral vs Happy
            (13, 2),  # Neutral vs Elated
            (13, 4),  # Neutral vs Surprised
            (13, 7),  # Neutral vs Angry
            (13, 8),  # Neutral vs Astonished
            (13, 9),  # Neutral vs Disgusted
            (13, 10), # Neutral vs Fearful
            (13, 11), # Neutral vs Sad
            
            # Conflitos entre surpresa e emoções negativas intensas
            (4, 9),   # Surprised vs Disgusted
            (4, 10),  # Surprised vs Fearful
            (8, 9),   # Astonished vs Disgusted
            (8, 10),  # Astonished vs Fearful
            
            # Conflitos entre emoções positivas e desprezo
            (0, 1),   # Happy vs Contempt
            (2, 1),   # Elated vs Contempt
            (5, 1),   # Proud vs Contempt
            (6, 1),   # Loved vs Contempt
            
            # Conflitos entre fadiga e emoções energéticas
            (12, 0),  # Fatigued vs Happy
            (12, 2),  # Fatigued vs Elated
            (12, 4),  # Fatigued vs Surprised
            (12, 8),  # Fatigued vs Astonished
        ]

    def forward(self, logits, targets):
        # --- PARTE 1: Focal Loss ---
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        p_t = torch.exp(-bce_loss)
        focal_term = (1 - p_t) ** self.gamma
        
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
            # Penaliza se ambas as emoções do par tiverem alta probabilidade
            joint_prob = probs[:, i] * probs[:, j]
            consistency_loss += joint_prob.mean()
            
        # --- COMBINAÇÃO FINAL ---
        total_loss = focal_loss + (self.conflict_weight * consistency_loss)
        
        return total_loss
    
class RegularizedLearnedConsistencyLoss14Emotions(nn.Module):
    def __init__(self, num_classes=14, gamma=2.0, alpha=0.25, 
                 sparsity_weight=0.01, prior_strength=1.0, reduction='mean'):
        """
        Args:
            prior_strength (float): Força da inicialização com conhecimento prévio
        """
        super(RegularizedLearnedConsistencyLoss14Emotions, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.num_classes = num_classes
        self.sparsity_weight = sparsity_weight
        self.prior_strength = prior_strength
        self.reduction = reduction
        
        # Matriz de conflito simétrica
        self.conflict_logits = nn.Parameter(torch.zeros(num_classes, num_classes))
        
        # Inicializar com conhecimento prévio hierárquico
        self._initialize_with_hierarchical_prior()
    
    def _initialize_with_hierarchical_prior(self):
        """Inicialização com conhecimento prévio hierárquico"""
        with torch.no_grad():
            # Mapeamento de categorias de emoções
            positive_emotions = [0, 2, 3, 5, 6]  # happy, elated, hopeful, proud, loved
            negative_emotions = [7, 9, 10, 11, 12]  # angry, disgusted, fearful, sad, fatigued
            surprise_emotions = [4, 8]  # surprised, astonished
            neutral_idx = 13
            contempt_idx = 1
            
            # Conflitos fortes: positivas vs negativas
            for pos in positive_emotions:
                for neg in negative_emotions:
                    if pos == 0 and neg == 11:  # Happy vs Sad - conflito muito forte
                        self.conflict_logits[pos, neg] = 2.0 * self.prior_strength
                        self.conflict_logits[neg, pos] = 2.0 * self.prior_strength
                    else:
                        self.conflict_logits[pos, neg] = 1.0 * self.prior_strength
                        self.conflict_logits[neg, pos] = 1.0 * self.prior_strength
            
            # Conflitos médios: neutral vs emoções intensas
            for intense in positive_emotions + negative_emotions + surprise_emotions:
                self.conflict_logits[neutral_idx, intense] = 0.8 * self.prior_strength
                self.conflict_logits[intense, neutral_idx] = 0.8 * self.prior_strength
            
            # Conflitos específicos para desprezo
            for pos in positive_emotions:
                self.conflict_logits[pos, contempt_idx] = 0.6 * self.prior_strength
                self.conflict_logits[contempt_idx, pos] = 0.6 * self.prior_strength
            
            # Conflitos entre surpresa e medo/nojo
            for surprise in surprise_emotions:
                self.conflict_logits[surprise, 9] = 0.7 * self.prior_strength  # disgusted
                self.conflict_logits[9, surprise] = 0.7 * self.prior_strength
                self.conflict_logits[surprise, 10] = 0.7 * self.prior_strength  # fearful
                self.conflict_logits[10, surprise] = 0.7 * self.prior_strength

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
        
        # Consistency Loss
        probs = torch.sigmoid(logits)
        conflict_weights = torch.sigmoid(self.conflict_logits)
        
        probs_i = probs.unsqueeze(2)
        probs_j = probs.unsqueeze(1)
        joint_probs = probs_i * probs_j
        
        mask = torch.triu(torch.ones_like(conflict_weights), diagonal=1)
        consistency_terms = (joint_probs * conflict_weights.unsqueeze(0) * mask.unsqueeze(0)).sum(dim=(1,2))
        
        if self.reduction == 'mean':
            consistency_loss = consistency_terms.mean()
        elif self.reduction == 'sum':
            consistency_loss = consistency_terms.sum()
        
        # Regularização L1 para esparsidade
        sparsity_loss = torch.mean(torch.abs(self.conflict_logits))
        
        # Loss total
        total_loss = focal_loss + consistency_loss + self.sparsity_weight * sparsity_loss
        
        return total_loss