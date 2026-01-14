import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from helper.function import visualize_conflict_matrix, plot_top_conflicts

class KnowledgeGuidedConsistencyLoss(nn.Module):
    def __init__(self, num_classes=8, gamma=2.0, alpha=0.25, 
                 prior_strength=0.5, learnable_scale=True, reduction='mean'):
        """
        Versão automática que escolhe a matriz correta baseada em num_classes
        """
        super(KnowledgeGuidedConsistencyLoss, self).__init__()
        
        self.num_classes = num_classes
        self.reduction = reduction
        
        # Escolher a implementação correta baseada no número de classes
        if num_classes == 8:
            self.loss_module = KnowledgeGuidedConsistencyLoss8Classes(
                num_classes=num_classes, gamma=gamma, alpha=alpha,
                prior_strength=prior_strength, learnable_scale=learnable_scale
            )
        elif num_classes == 14:
            self.loss_module = KnowledgeGuidedConsistencyLoss14Classes(
                num_classes=num_classes, gamma=gamma, alpha=alpha,
                prior_strength=prior_strength, learnable_scale=learnable_scale
            )
        else:
            print(f"⚠️ Aviso: Número de classes não suportado: {num_classes}")
            print("Usando matriz de conflito genérica (sem conhecimento prévio)")
            
            # Versão genérica sem conhecimento prévio
            self.loss_module = GenericConsistencyLoss(
                num_classes=num_classes, gamma=gamma, alpha=alpha,
                prior_strength=0.0, learnable_scale=learnable_scale
            )
    
    def forward(self, logits, targets):
        return self.loss_module.forward(logits, targets, reduction=self.reduction)
    
    def get_conflict_matrix(self):
        return self.loss_module.get_conflict_matrix()
    
    def analyze_conflicts(self, threshold=0.3):
        if hasattr(self.loss_module, 'analyze_conflicts'):
            return self.loss_module.analyze_conflicts(threshold)
        else:
            with torch.no_grad():
                matrix = self.get_conflict_matrix().cpu().numpy()
                #print(f"\nMatriz de conflito ({self.num_classes} classes):")
                #print(matrix)
                return matrix
    
    def visualize_conflict_matrix(self, title=None, save_path=None, figsize=(12, 10)):
        """Visualiza a matriz de conflito atual"""
        with torch.no_grad():
            matrix = self.get_conflict_matrix()
            class_names = getattr(self.loss_module, 'class_names', None)
            
            if title is None:
                title = f'Conflict Matrix ({self.num_classes} Classes)'
            
            fig, ax = visualize_conflict_matrix(
                matrix, 
                class_names=class_names,
                title=title,
                save_path=save_path,
                figsize=figsize
            )
            return fig, ax
    
    def plot_top_conflicts(self, top_k=10, save_path=None, figsize=(10, 6)):
        """Plota os maiores conflitos"""
        with torch.no_grad():
            matrix = self.get_conflict_matrix()
            class_names = getattr(self.loss_module, 'class_names', None)
            
            fig, ax = plot_top_conflicts(
                matrix,
                class_names=class_names,
                top_k=top_k,
                save_path=save_path,
                figsize=figsize
            )
            return fig, ax
    
    def get_class_names(self):
        """Retorna os nomes das classes se disponível"""
        return getattr(self.loss_module, 'class_names', None)

class KnowledgeGuidedConsistencyLoss8Classes(nn.Module):
    def __init__(self, num_classes=8, gamma=2.0, alpha=0.25, 
                 prior_strength=0.5, learnable_scale=True):
        """
        Ordem para 8 classes:
        0: happy, 1: contempt, 2: surprised, 3: angry, 
        4: disgusted, 5: fearful, 6: sad, 7: neutral
        """
        super(KnowledgeGuidedConsistencyLoss8Classes, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.num_classes = num_classes
        self.prior_strength = prior_strength
        
        # Verificar se num_classes está correto
        if num_classes != 8:
            print(f"⚠️ Aviso: Esta loss foi projetada para 8 classes, mas num_classes={num_classes}")
        
        # Nomes das classes para referência
        self.class_names = ['happy', 'contempt', 'surprised', 'angry',
                           'disgusted', 'fearful', 'sad', 'neutral']
        
        # Índices das emoções
        self.idx_happy = 0
        self.idx_contempt = 1
        self.idx_surprised = 2
        self.idx_angry = 3
        self.idx_disgusted = 4
        self.idx_fearful = 5
        self.idx_sad = 6
        self.idx_neutral = 7
        
        # Matriz de conhecimento prévio (não aprendível)
        self.prior_matrix = self._create_prior_matrix_8classes()
        
        # Matriz de aprendizado residual (aprendível)
        self.residual_matrix = nn.Parameter(torch.zeros(num_classes, num_classes))
        
        # Escala global aprendível
        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.scale = 1.0
        
        # Bias por classe (algumas classes são mais conflitantes)
        self.class_bias = nn.Parameter(torch.zeros(num_classes))
        
    def _create_prior_matrix_8classes(self):
        """Cria matriz de conhecimento prévio para 8 classes"""
        prior = torch.zeros(self.num_classes, self.num_classes)
        
        # Usar os atributos definidos
        idx_happy = self.idx_happy
        idx_contempt = self.idx_contempt
        idx_surprised = self.idx_surprised
        idx_angry = self.idx_angry
        idx_disgusted = self.idx_disgusted
        idx_fearful = self.idx_fearful
        idx_sad = self.idx_sad
        idx_neutral = self.idx_neutral
        
        # CONFLITOS FORTES (peso 0.8)
        # Happy vs emoções negativas
        strong_conflicts = [
            (idx_happy, idx_angry),      # happy vs angry
            (idx_happy, idx_disgusted),  # happy vs disgusted
            (idx_happy, idx_fearful),    # happy vs fearful
            (idx_happy, idx_sad),        # happy vs sad
            
            # Surprised vs emoções negativas
            (idx_surprised, idx_disgusted),  # surprised vs disgusted
            (idx_surprised, idx_fearful),    # surprised vs fearful
            
            # Angry vs emoções positivas e neutras
            (idx_angry, idx_happy),      # angry vs happy
            (idx_angry, idx_sad),        # angry vs sad
            (idx_angry, idx_neutral),    # angry vs neutral
            
            # Disgusted vs emoções positivas e neutras
            (idx_disgusted, idx_happy),  # disgusted vs happy
            (idx_disgusted, idx_surprised),  # disgusted vs surprised
            (idx_disgusted, idx_neutral),  # disgusted vs neutral
            
            # Fearful vs emoções positivas e neutras
            (idx_fearful, idx_happy),    # fearful vs happy
            (idx_fearful, idx_surprised),  # fearful vs surprised
            (idx_fearful, idx_neutral),  # fearful vs neutral
            
            # Sad vs emoções positivas e neutras
            (idx_sad, idx_happy),        # sad vs happy
            (idx_sad, idx_angry),        # sad vs angry
            (idx_sad, idx_neutral),      # sad vs neutral
        ]
        
        # CONFLITOS MODERADOS (peso 0.4)
        # Contempt vs outras emoções
        moderate_conflicts = [
            # Contempt conflicts
            (idx_contempt, idx_happy),    # contempt vs happy
            (idx_contempt, idx_angry),    # contempt vs angry
            (idx_contempt, idx_disgusted), # contempt vs disgusted
            (idx_contempt, idx_sad),      # contempt vs sad
            (idx_contempt, idx_neutral),  # contempt vs neutral
            
            # Neutral vs todas as outras (conflito moderado)
            (idx_neutral, idx_happy),     # neutral vs happy
            (idx_neutral, idx_surprised), # neutral vs surprised
            (idx_neutral, idx_angry),     # neutral vs angry
            (idx_neutral, idx_disgusted), # neutral vs disgusted
            (idx_neutral, idx_fearful),   # neutral vs fearful
            (idx_neutral, idx_sad),       # neutral vs sad
            
            # Pares adicionais que podem ter conflito
            (idx_angry, idx_disgusted),   # angry vs disgusted
            (idx_disgusted, idx_fearful), # disgusted vs fearful
            (idx_fearful, idx_sad),       # fearful vs sad
        ]
        
        # Aplicar conflitos fortes
        for i, j in strong_conflicts:
            if i < self.num_classes and j < self.num_classes:
                prior[i, j] = 0.8
                prior[j, i] = 0.8
        
        # Aplicar conflitos moderados (não sobrescrever se já for forte)
        for i, j in moderate_conflicts:
            if i < self.num_classes and j < self.num_classes:
                if prior[i, j] == 0:  # Só aplicar se não for já um conflito forte
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
    
    def forward(self, logits, targets, reduction='mean'):
        # Obter matriz de conflito
        conflict_weights = self.get_conflict_matrix()
        
        # Focal Loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = torch.exp(-bce_loss)
        focal_term = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * focal_term * bce_loss
        
        if reduction == 'mean':
            focal_loss = focal_loss.mean()
        elif reduction == 'sum':
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
        consistency_loss = consistency_loss.sum(dim=(1,2))
        
        if reduction == 'mean':
            consistency_loss = consistency_loss.mean()
        elif reduction == 'sum':
            consistency_loss = consistency_loss.sum()
        
        # Regularização para suavidade
        smoothness_loss = torch.mean(torch.abs(self.residual_matrix - self.residual_matrix.t()))
        
        # Loss total
        total_loss = focal_loss + consistency_loss + 0.001 * smoothness_loss
        
        return total_loss
    
    def analyze_conflicts(self, threshold=0.3):
        """Analisa os conflitos aprendidos"""
        with torch.no_grad():
            matrix = self.get_conflict_matrix().cpu().numpy()
            
            #print("\n=== Análise de Conflitos (8 Classes) ===")
            #print(f"Ordem: {self.class_names}")
            
            # Top 10 conflitos
            conflicts = []
            for i in range(self.num_classes):
                for j in range(i+1, self.num_classes):
                    conflicts.append((i, j, matrix[i, j]))
            
            conflicts.sort(key=lambda x: x[2], reverse=True)
            
            return matrix

class KnowledgeGuidedConsistencyLoss14Classes(nn.Module):
    def __init__(self, num_classes=14, gamma=2.0, alpha=0.25, 
                 prior_strength=0.5, learnable_scale=True):
        """
        Ordem para 14 classes:
        0: happy, 1: contempt, 2: elated, 3: hopeful, 4: surprised,
        5: proud, 6: loved, 7: angry, 8: astonished, 9: disgusted,
        10: fearful, 11: sad, 12: fatigued, 13: neutral
        """
        super(KnowledgeGuidedConsistencyLoss14Classes, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.num_classes = num_classes
        self.prior_strength = prior_strength
        
        # Verificar se num_classes está correto
        if num_classes != 14:
            print(f"⚠️ Aviso: Esta loss foi projetada para 14 classes, mas num_classes={num_classes}")
        
        # Nomes das classes
        self.class_names = [
            'happy', 'contempt', 'elated', 'hopeful', 'surprised',
            'proud', 'loved', 'angry', 'astonished', 'disgusted',
            'fearful', 'sad', 'fatigued', 'neutral'
        ]
        
        # Índices das emoções
        self.idx_happy = 0
        self.idx_contempt = 1
        self.idx_elated = 2
        self.idx_hopeful = 3
        self.idx_surprised = 4
        self.idx_proud = 5
        self.idx_loved = 6
        self.idx_angry = 7
        self.idx_astonished = 8
        self.idx_disgusted = 9
        self.idx_fearful = 10
        self.idx_sad = 11
        self.idx_fatigued = 12
        self.idx_neutral = 13
        
        # Matriz de conhecimento prévio
        self.prior_matrix = self._create_prior_matrix_14classes()
        
        # Matriz de aprendizado residual
        self.residual_matrix = nn.Parameter(torch.zeros(num_classes, num_classes))
        
        # Escala global aprendível
        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.scale = 1.0
        
        # Bias por classe
        self.class_bias = nn.Parameter(torch.zeros(num_classes))
        
    def _create_prior_matrix_14classes(self):
        """Cria matriz de conhecimento prévio para 14 classes"""
        prior = torch.zeros(self.num_classes, self.num_classes)
        
        # Usar os atributos definidos no __init__
        idx_happy = self.idx_happy
        idx_contempt = self.idx_contempt
        idx_elated = self.idx_elated
        idx_hopeful = self.idx_hopeful
        idx_surprised = self.idx_surprised
        idx_proud = self.idx_proud
        idx_loved = self.idx_loved
        idx_angry = self.idx_angry
        idx_astonished = self.idx_astonished
        idx_disgusted = self.idx_disgusted
        idx_fearful = self.idx_fearful
        idx_sad = self.idx_sad
        idx_fatigued = self.idx_fatigued
        idx_neutral = self.idx_neutral
        
        # CONFLITOS FORTES (peso 0.8)
        # Emoções positivas vs negativas
        strong_conflicts = [
            # Happy e similares vs negativas
            (idx_happy, idx_angry),
            (idx_happy, idx_disgusted),
            (idx_happy, idx_fearful),
            (idx_happy, idx_sad),
            
            # Elated vs negativas
            (idx_elated, idx_angry),
            (idx_elated, idx_disgusted),
            (idx_elated, idx_sad),
            
            # Hopeful vs negativas
            (idx_hopeful, idx_angry),
            (idx_hopeful, idx_disgusted),
            (idx_hopeful, idx_fearful),
            
            # Surprised vs disgusted/fearful
            (idx_surprised, idx_disgusted),
            (idx_surprised, idx_fearful),
            
            # Proud vs negativas
            (idx_proud, idx_angry),
            (idx_proud, idx_disgusted),
            (idx_proud, idx_sad),
            
            # Loved vs negativas
            (idx_loved, idx_angry),
            (idx_loved, idx_disgusted),
            (idx_loved, idx_fearful),
            
            # Angry vs positivas e neutras
            (idx_angry, idx_happy),
            (idx_angry, idx_elated),
            (idx_angry, idx_hopeful),
            (idx_angry, idx_loved),
            (idx_angry, idx_neutral),
            
            # Disgusted vs positivas e neutras
            (idx_disgusted, idx_happy),
            (idx_disgusted, idx_elated),
            (idx_disgusted, idx_hopeful),
            (idx_disgusted, idx_surprised),
            (idx_disgusted, idx_neutral),
            
            # Fearful vs positivas e neutras
            (idx_fearful, idx_happy),
            (idx_fearful, idx_hopeful),
            (idx_fearful, idx_surprised),
            (idx_fearful, idx_loved),
            (idx_fearful, idx_neutral),
            
            # Sad vs positivas e neutras
            (idx_sad, idx_happy),
            (idx_sad, idx_elated),
            (idx_sad, idx_hopeful),
            (idx_sad, idx_loved),
            (idx_sad, idx_neutral),
            
            # Fatigued vs positivas
            (idx_fatigued, idx_happy),
            (idx_fatigued, idx_elated),
            (idx_fatigued, idx_hopeful),
        ]
        
        # CONFLITOS MODERADOS (peso 0.4)
        moderate_conflicts = [
            # Contempt vs outras
            (idx_contempt, idx_happy),
            (idx_contempt, idx_elated),
            (idx_contempt, idx_hopeful),
            (idx_contempt, idx_angry),
            (idx_contempt, idx_disgusted),
            (idx_contempt, idx_sad),
            (idx_contempt, idx_neutral),
            
            # Astonished vs outras
            (idx_astonished, idx_angry),
            (idx_astonished, idx_disgusted),
            (idx_astonished, idx_fearful),
            (idx_astonished, idx_sad),
            (idx_astonished, idx_neutral),
            
            # Neutral vs todas as outras (conflito moderado)
            (idx_neutral, idx_happy),
            (idx_neutral, idx_elated),
            (idx_neutral, idx_hopeful),
            (idx_neutral, idx_surprised),
            (idx_neutral, idx_proud),
            (idx_neutral, idx_loved),
            (idx_neutral, idx_angry),
            (idx_neutral, idx_astonished),
            (idx_neutral, idx_disgusted),
            (idx_neutral, idx_fearful),
            (idx_neutral, idx_sad),
            (idx_neutral, idx_fatigued),
            
            # Pares adicionais
            (idx_surprised, idx_astonished),  # tipos diferentes de surpresa
            (idx_angry, idx_disgusted),
            (idx_disgusted, idx_fearful),
            (idx_fearful, idx_sad),
            (idx_sad, idx_fatigued),
        ]
        
        # Aplicar conflitos fortes
        for i, j in strong_conflicts:
            if i < self.num_classes and j < self.num_classes:
                prior[i, j] = 0.8
                prior[j, i] = 0.8
        
        # Aplicar conflitos moderados
        for i, j in moderate_conflicts:
            if i < self.num_classes and j < self.num_classes:
                if prior[i, j] == 0:
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
        residual = torch.tanh(residual) * 0.5
        
        # Combinação com prior
        combined = self.prior_strength * self.prior_matrix.to(residual.device)
        combined += (1 - self.prior_strength) * (residual + 0.5)
        
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
    
    def forward(self, logits, targets, reduction='mean'):
        # Obter matriz de conflito
        conflict_weights = self.get_conflict_matrix()
        
        # Focal Loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = torch.exp(-bce_loss)
        focal_term = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * focal_term * bce_loss
        
        if reduction == 'mean':
            focal_loss = focal_loss.mean()
        elif reduction == 'sum':
            focal_loss = focal_loss.sum()
        
        # Consistency Loss
        probs = torch.sigmoid(logits)
        
        # Calcular produto pareado
        probs_expanded = probs.unsqueeze(2)
        probs_t = probs.unsqueeze(1)
        joint_probs = probs_expanded * probs_t
        
        # Usar apenas triângulo superior
        mask = torch.triu(torch.ones_like(conflict_weights), diagonal=1)
        consistency_loss = (joint_probs * conflict_weights.unsqueeze(0) * mask.unsqueeze(0))
        consistency_loss = consistency_loss.sum(dim=(1,2))
        
        if reduction == 'mean':
            consistency_loss = consistency_loss.mean()
        elif reduction == 'sum':
            consistency_loss = consistency_loss.sum()
        
        # Regularização para suavidade
        smoothness_loss = torch.mean(torch.abs(self.residual_matrix - self.residual_matrix.t()))
        
        # Loss total
        total_loss = focal_loss + consistency_loss + 0.001 * smoothness_loss
        
        return total_loss
    
    def analyze_conflicts(self, threshold=0.3):
        """Analisa os conflitos aprendidos"""
        with torch.no_grad():
            matrix = self.get_conflict_matrix().cpu().numpy()
                        
            # Top 15 conflitos
            conflicts = []
            for i in range(self.num_classes):
                for j in range(i+1, self.num_classes):
                    conflicts.append((i, j, matrix[i, j]))
            
            conflicts.sort(key=lambda x: x[2], reverse=True)
            return matrix
     
class GenericConsistencyLoss(nn.Module):
    """Versão genérica sem conhecimento prévio específico"""
    def __init__(self, num_classes=8, gamma=2.0, alpha=0.25, 
                 prior_strength=0.0, learnable_scale=True, reduction='mean'):
        super(GenericConsistencyLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.num_classes = num_classes
        self.prior_strength = prior_strength
        self.reduction = reduction
        
        # Matriz de aprendizado
        self.residual_matrix = nn.Parameter(torch.zeros(num_classes, num_classes))
        
        # Escala global aprendível
        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.scale = 1.0
        
        # Bias por classe
        self.class_bias = nn.Parameter(torch.zeros(num_classes))
        
        # Inicializar com pequenos valores aleatórios
        with torch.no_grad():
            self.residual_matrix.normal_(0, 0.01)
    
    def get_conflict_matrix(self):
        """Matriz de conflito aprendida do zero"""
        # Matriz simétrica
        residual = (self.residual_matrix + self.residual_matrix.t()) / 2
        
        # Aplicar ativação suave
        residual = torch.tanh(residual)
        
        # Aplicar escala e bias
        scaled = self.scale * residual
        
        # Adicionar bias por classe
        bias_matrix = self.class_bias.view(-1, 1) + self.class_bias.view(1, -1)
        scaled = scaled + 0.1 * bias_matrix
        
        # Garantir não-negatividade
        conflict_weights = F.softplus(scaled)
        
        # Normalizar
        conflict_weights = conflict_weights / (conflict_weights.max() + 1e-8)
        
        return conflict_weights
    
    def forward(self, logits, targets, reduction=None):
        if reduction is None:
            reduction = self.reduction
            
        # Obter matriz de conflito
        conflict_weights = self.get_conflict_matrix()
        
        # Focal Loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = torch.exp(-bce_loss)
        focal_term = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * focal_term * bce_loss
        
        if reduction == 'mean':
            focal_loss = focal_loss.mean()
        elif reduction == 'sum':
            focal_loss = focal_loss.sum()
        
        # Consistency Loss
        probs = torch.sigmoid(logits)
        
        # Calcular produto pareado
        probs_expanded = probs.unsqueeze(2)
        probs_t = probs.unsqueeze(1)
        joint_probs = probs_expanded * probs_t
        
        # Usar apenas triângulo superior
        mask = torch.triu(torch.ones_like(conflict_weights), diagonal=1)
        consistency_loss = (joint_probs * conflict_weights.unsqueeze(0) * mask.unsqueeze(0))
        consistency_loss = consistency_loss.sum(dim=(1,2))
        
        if reduction == 'mean':
            consistency_loss = consistency_loss.mean()
        elif reduction == 'sum':
            consistency_loss = consistency_loss.sum()
        
        # Regularização para suavidade
        smoothness_loss = torch.mean(torch.abs(self.residual_matrix - self.residual_matrix.t()))
        
        # Loss total
        total_loss = focal_loss + consistency_loss + 0.001 * smoothness_loss
        
        return total_loss