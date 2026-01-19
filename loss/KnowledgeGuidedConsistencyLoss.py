import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys
import matplotlib.pyplot as plt
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
class EnhancedKnowledgeConsistencyLoss(nn.Module):
    """
    Combina a Abordagem Revisada com Gradientes Melhores e 
    a Abordagem com Conhecimento Prévio Incorporado.
    
    Funciona para 8 e 14 classes com a ordem correta de emoções.
    """
    def __init__(self, num_classes=8, gamma=2.0, alpha=0.25, 
                 prior_strength=0.7, conflict_weight=1.0,
                 sparsity_weight=0.01, symmetry_weight=0.001,
                 learnable_scale=True, reduction='mean',
                 gradient_boost_factor=5.0):
        """
        Args:
            num_classes: 8 ou 14 classes
            gamma: Fator de foco para Focal Loss
            alpha: Fator de balanceamento para Focal Loss
            prior_strength: Quanto o conhecimento prévio influencia (0-1)
            conflict_weight: Peso da penalidade por contradição
            sparsity_weight: Peso da regularização de esparsidade
            symmetry_weight: Peso da regularização de simetria
            learnable_scale: Se a escala global é aprendível
            reduction: 'mean' ou 'sum'
            gradient_boost_factor: Fator para reforçar gradientes da matriz
        """
        super(EnhancedKnowledgeConsistencyLoss, self).__init__()
        
        self.gamma = gamma
        self.alpha = alpha
        self.num_classes = num_classes
        self.prior_strength = prior_strength
        self.conflict_weight = conflict_weight
        self.sparsity_weight = sparsity_weight
        self.symmetry_weight = symmetry_weight
        self.reduction = reduction
        self.gradient_boost_factor = gradient_boost_factor
        
        # Definir ordem das emoções baseada no número de classes
        if num_classes == 8:
            self.class_names = [
                'happy', 'contempt', 'surprised', 'angry',
                'disgusted', 'fearful', 'sad', 'neutral'
            ]
        elif num_classes == 14:
            self.class_names = [
                'happy', 'contempt', 'elated', 'hopeful', 'surprised',
                'proud', 'loved', 'angry', 'astonished', 'disgusted',
                'fearful', 'sad', 'fatigued', 'neutral'
            ]
        else:
            raise ValueError(f"Número de classes não suportado: {num_classes}. Use 8 ou 14.")
        
        # Índices das emoções
        self._setup_indices()
        
        # 1. MATRIZ DE CONHECIMENTO PRÉVIO (não aprendível)
        self.prior_matrix = self._create_prior_matrix()
        
        # 2. MATRIZ RESIDUAL APRENDÍVEL (com gradientes reforçados)
        # Inicializar com valores pequenos para facilitar aprendizado
        self.residual_matrix = nn.Parameter(torch.zeros(num_classes, num_classes))
        self._initialize_residual_matrix()
        
        # 3. ESCALA GLOBAL APRENDÍVEL
        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.scale = 1.0
        
        # 4. BIAS POR CLASSE APRENDÍVEL
        self.class_bias = nn.Parameter(torch.zeros(num_classes))
        
        # 5. CONTADOR PARA MONITORAMENTO
        self.update_counter = 0
        self.conflict_history = []
        
        print(f"EnhancedKnowledgeConsistencyLoss inicializada para {num_classes} classes")
        print(f"Classes: {self.class_names}")
    
    def _setup_indices(self):
        """Configura índices das emoções baseado no número de classes"""
        if self.num_classes == 8:
            self.idx = {
                'happy': 0, 'contempt': 1, 'surprised': 2, 'angry': 3,
                'disgusted': 4, 'fearful': 5, 'sad': 6, 'neutral': 7
            }
        else:  # 14 classes
            self.idx = {
                'happy': 0, 'contempt': 1, 'elated': 2, 'hopeful': 3,
                'surprised': 4, 'proud': 5, 'loved': 6, 'angry': 7,
                'astonished': 8, 'disgusted': 9, 'fearful': 10,
                'sad': 11, 'fatigued': 12, 'neutral': 13
            }
    
    def _create_prior_matrix(self):
        """
        Cria matriz de conhecimento prévio baseada em psicologia emocional
        com conflitos fortes, moderados e fracos.
        """
        prior = torch.zeros(self.num_classes, self.num_classes)
        
        if self.num_classes == 8:
            # CONFLITOS FORTES (peso 0.9) - Muito improváveis de co-ocorrer
            strong_conflicts = [
                # Happy vs emoções negativas fortes
                (self.idx['happy'], self.idx['angry']),
                (self.idx['happy'], self.idx['disgusted']),
                (self.idx['happy'], self.idx['fearful']),
                (self.idx['happy'], self.idx['sad']),
                
                # Surprised vs emoções negativas
                (self.idx['surprised'], self.idx['disgusted']),
                (self.idx['surprised'], self.idx['fearful']),
                
                # Angry vs happy/surprised/neutral
                (self.idx['angry'], self.idx['happy']),
                (self.idx['angry'], self.idx['surprised']),
                (self.idx['angry'], self.idx['neutral']),
                
                # Disgusted vs happy/surprised/neutral
                (self.idx['disgusted'], self.idx['happy']),
                (self.idx['disgusted'], self.idx['surprised']),
                (self.idx['disgusted'], self.idx['neutral']),
                
                # Fearful vs happy/surprised/neutral
                (self.idx['fearful'], self.idx['happy']),
                (self.idx['fearful'], self.idx['surprised']),
                (self.idx['fearful'], self.idx['neutral']),
                
                # Sad vs happy/neutral
                (self.idx['sad'], self.idx['happy']),
                (self.idx['sad'], self.idx['neutral']),
            ]
            
            # CONFLITOS MODERADOS (peso 0.6) - Improváveis de co-ocorrer
            moderate_conflicts = [
                # Contempt vs outras emoções
                (self.idx['contempt'], self.idx['happy']),
                (self.idx['contempt'], self.idx['angry']),
                (self.idx['contempt'], self.idx['disgusted']),
                (self.idx['contempt'], self.idx['sad']),
                (self.idx['contempt'], self.idx['neutral']),
                
                # Neutral vs todas as outras (moderado)
                (self.idx['neutral'], self.idx['happy']),
                (self.idx['neutral'], self.idx['surprised']),
                (self.idx['neutral'], self.idx['angry']),
                (self.idx['neutral'], self.idx['disgusted']),
                (self.idx['neutral'], self.idx['fearful']),
                (self.idx['neutral'], self.idx['sad']),
                
                # Pares adicionais
                (self.idx['angry'], self.idx['disgusted']),
                (self.idx['disgusted'], self.idx['fearful']),
                (self.idx['fearful'], self.idx['sad']),
                (self.idx['sad'], self.idx['angry']),
            ]
            
        else:  # 14 classes
            # CONFLITOS FORTES (peso 0.9)
            strong_conflicts = [
                # Emoções positivas vs negativas fortes
                (self.idx['happy'], self.idx['angry']),
                (self.idx['happy'], self.idx['disgusted']),
                (self.idx['happy'], self.idx['fearful']),
                (self.idx['happy'], self.idx['sad']),
                
                (self.idx['elated'], self.idx['angry']),
                (self.idx['elated'], self.idx['disgusted']),
                (self.idx['elated'], self.idx['sad']),
                
                (self.idx['hopeful'], self.idx['angry']),
                (self.idx['hopeful'], self.idx['disgusted']),
                (self.idx['hopeful'], self.idx['fearful']),
                
                (self.idx['loved'], self.idx['angry']),
                (self.idx['loved'], self.idx['disgusted']),
                (self.idx['loved'], self.idx['fearful']),
                
                # Surprised vs disgusted/fearful
                (self.idx['surprised'], self.idx['disgusted']),
                (self.idx['surprised'], self.idx['fearful']),
                
                # Astonished vs emoções negativas
                (self.idx['astonished'], self.idx['angry']),
                (self.idx['astonished'], self.idx['disgusted']),
                (self.idx['astonished'], self.idx['fearful']),
                
                # Emoções negativas vs positivas/neutras
                (self.idx['angry'], self.idx['happy']),
                (self.idx['angry'], self.idx['elated']),
                (self.idx['angry'], self.idx['hopeful']),
                (self.idx['angry'], self.idx['loved']),
                (self.idx['angry'], self.idx['neutral']),
                
                (self.idx['disgusted'], self.idx['happy']),
                (self.idx['disgusted'], self.idx['elated']),
                (self.idx['disgusted'], self.idx['hopeful']),
                (self.idx['disgusted'], self.idx['neutral']),
                
                (self.idx['fearful'], self.idx['happy']),
                (self.idx['fearful'], self.idx['hopeful']),
                (self.idx['fearful'], self.idx['loved']),
                (self.idx['fearful'], self.idx['neutral']),
                
                (self.idx['sad'], self.idx['happy']),
                (self.idx['sad'], self.idx['elated']),
                (self.idx['sad'], self.idx['hopeful']),
                (self.idx['sad'], self.idx['loved']),
                (self.idx['sad'], self.idx['neutral']),
                
                (self.idx['fatigued'], self.idx['happy']),
                (self.idx['fatigued'], self.idx['elated']),
                (self.idx['fatigued'], self.idx['hopeful']),
            ]
            
            # CONFLITOS MODERADOS (peso 0.6)
            moderate_conflicts = [
                # Contempt vs outras
                (self.idx['contempt'], self.idx['happy']),
                (self.idx['contempt'], self.idx['elated']),
                (self.idx['contempt'], self.idx['hopeful']),
                (self.idx['contempt'], self.idx['angry']),
                (self.idx['contempt'], self.idx['disgusted']),
                (self.idx['contempt'], self.idx['sad']),
                (self.idx['contempt'], self.idx['neutral']),
                
                # Proud vs negativas
                (self.idx['proud'], self.idx['angry']),
                (self.idx['proud'], self.idx['disgusted']),
                (self.idx['proud'], self.idx['fearful']),
                (self.idx['proud'], self.idx['sad']),
                
                # Neutral vs todas (moderado)
                (self.idx['neutral'], self.idx['happy']),
                (self.idx['neutral'], self.idx['elated']),
                (self.idx['neutral'], self.idx['hopeful']),
                (self.idx['neutral'], self.idx['surprised']),
                (self.idx['neutral'], self.idx['proud']),
                (self.idx['neutral'], self.idx['loved']),
                (self.idx['neutral'], self.idx['angry']),
                (self.idx['neutral'], self.idx['astonished']),
                (self.idx['neutral'], self.idx['disgusted']),
                (self.idx['neutral'], self.idx['fearful']),
                (self.idx['neutral'], self.idx['sad']),
                (self.idx['neutral'], self.idx['fatigued']),
                
                # Pares similares
                (self.idx['surprised'], self.idx['astonished']),
                (self.idx['angry'], self.idx['disgusted']),
                (self.idx['disgusted'], self.idx['fearful']),
                (self.idx['fearful'], self.idx['sad']),
                (self.idx['sad'], self.idx['fatigued']),
                (self.idx['elated'], self.idx['hopeful']),
            ]
        
        # Aplicar conflitos fortes
        for i, j in strong_conflicts:
            prior[i, j] = 0.9
            prior[j, i] = 0.9
        
        # Aplicar conflitos moderados (não sobrescrever se já for forte)
        for i, j in moderate_conflicts:
            if prior[i, j] == 0:
                prior[i, j] = 0.6
                prior[j, i] = 0.6
        
        # Conflitos fracos (peso 0.3) para todas as outras combinações
        for i in range(self.num_classes):
            for j in range(i+1, self.num_classes):
                if prior[i, j] == 0:
                    prior[i, j] = 0.3
                    prior[j, i] = 0.3
        
        # Auto-conflito zero
        for i in range(self.num_classes):
            prior[i, i] = 0.0
        
        return prior
    
    def _initialize_residual_matrix(self):
        """Inicializa a matriz residual para facilitar aprendizado"""
        with torch.no_grad():
            # Inicializar com pequenos valores aleatórios
            self.residual_matrix.normal_(0, 0.05)
            
            # Tornar simétrica
            sym_matrix = (self.residual_matrix + self.residual_matrix.t()) / 2
            self.residual_matrix.copy_(sym_matrix)
    
    def get_conflict_matrix(self, apply_gradient_boost=True):
        """
        Combina conhecimento prévio com aprendizado residual
        com técnicas para melhorar gradientes.
        """
        # 1. Obter matriz residual e garantir simetria
        residual = self.residual_matrix
        residual_sym = (residual + residual.t()) / 2
        
        # 2. Aplicar ativação suave que preserva gradientes
        # Usamos tanh escalado para manter gradientes em toda a faixa
        residual_activated = torch.tanh(residual_sym * self.gradient_boost_factor) / 2
        
        # 3. Combinação com conhecimento prévio
        combined = (self.prior_strength * self.prior_matrix.to(residual.device) +
                   (1 - self.prior_strength) * (residual_activated + 0.5))
        
        # 4. Aplicar escala global
        scaled = torch.sigmoid(self.scale) * combined
        
        # 5. Adicionar bias por classe (ajusta propensão ao conflito por emoção)
        bias_matrix = self.class_bias.view(-1, 1) + self.class_bias.view(1, -1)
        scaled = scaled + 0.1 * torch.tanh(bias_matrix)
        
        # 6. Garantir não-negatividade com softplus (melhor para gradientes que ReLU)
        conflict_weights = F.softplus(scaled)
        
        # 7. Normalizar para evitar explosão
        max_val = conflict_weights.max()
        if max_val > 0:
            conflict_weights = conflict_weights / max_val
        
        # 8. Zerar diagonal (não há conflito consigo mesmo)
        mask_eye = 1 - torch.eye(self.num_classes, device=conflict_weights.device)
        conflict_weights = conflict_weights * mask_eye
        
        return conflict_weights
    
    def forward(self, logits, targets):
        """
        Forward pass da loss combinada.
        """
        batch_size = logits.size(0)
        
        # --- PARTE 1: FOCAL LOSS ---
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = torch.exp(-bce_loss)
        focal_term = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * focal_term * bce_loss
        
        if self.reduction == 'mean':
            focal_loss = focal_loss.mean()
        elif self.reduction == 'sum':
            focal_loss = focal_loss.sum()
        
        # --- PARTE 2: CONSISTENCY LOSS COM MATRIZ APRENDIDA ---
        probs = torch.sigmoid(logits)
        conflict_weights = self.get_conflict_matrix()
        
        # Calcular todos os produtos pareados de forma eficiente
        probs_i = probs.unsqueeze(2)  # [batch, classes, 1]
        probs_j = probs.unsqueeze(1)  # [batch, 1, classes]
        joint_probs = probs_i * probs_j  # [batch, classes, classes]
        
        # Usar apenas triângulo superior para evitar dupla contagem
        mask = torch.triu(torch.ones_like(conflict_weights), diagonal=1)
        weighted_joint_probs = joint_probs * conflict_weights.unsqueeze(0) * mask.unsqueeze(0)
        
        consistency_loss = weighted_joint_probs.sum(dim=(1,2))
        
        if self.reduction == 'mean':
            consistency_loss = consistency_loss.mean()
        elif self.reduction == 'sum':
            consistency_loss = consistency_loss.sum()
        
        # --- PARTE 3: REGULARIZAÇÕES PARA ESTABILIDADE ---
        # Regularização de esparsidade (evitar que muitos pesos sejam altos)
        avg_weight = conflict_weights.mean()
        sparsity_loss = F.mse_loss(avg_weight, torch.tensor(0.4).to(avg_weight.device))
        
        # Regularização de simetria (garantir matriz simétrica)
        symmetry_loss = F.mse_loss(self.residual_matrix, self.residual_matrix.t())
        
        # Regularização de suavidade (evitar mudanças bruscas)
        smoothness_loss = torch.mean(torch.abs(self.residual_matrix - self.residual_matrix.mean()))
        
        # --- PARTE 4: LOSS TOTAL ---
        total_loss = (focal_loss + 
                     self.conflict_weight * consistency_loss +
                     self.sparsity_weight * sparsity_loss +
                     self.symmetry_weight * symmetry_loss +
                     0.0005 * smoothness_loss)
        
        # --- MONITORAMENTO ---
        self.update_counter += 1
        if self.update_counter % 100 == 0:
            with torch.no_grad():
                conflict_matrix_np = conflict_weights.cpu().detach().numpy()
                self.conflict_history.append(conflict_matrix_np.copy())
        
        return total_loss
    
    def analyze_conflict_learning(self, threshold=0.3):
        """
        Analisa o estado da aprendizagem da matriz de conflito.
        """
        with torch.no_grad():
            conflict_weights = self.get_conflict_matrix()
            matrix_np = conflict_weights.cpu().numpy()
          
            # Top conflitos
            conflicts = []
            for i in range(self.num_classes):
                for j in range(i+1, self.num_classes):
                    conflicts.append((i, j, matrix_np[i, j]))
            
            conflicts.sort(key=lambda x: x[2], reverse=True)
            
            return matrix_np
    
    def get_optimizer_params(self, base_lr, residual_lr_multiplier=5.0):
        """
        Retorna parâmetros para otimização com learning rates diferenciados.
        
        Args:
            base_lr: Learning rate base
            residual_lr_multiplier: Multiplicador para a matriz residual
            
        Returns:
            Lista de dicionários com parâmetros e learning rates
        """
        params = [
            # Parâmetros da matriz residual com LR maior
            {'params': [self.residual_matrix], 'lr': base_lr * residual_lr_multiplier},
            # Bias por classe com LR normal
            {'params': [self.class_bias], 'lr': base_lr},
        ]
        
        # Adicionar scale se for aprendível
        if isinstance(self.scale, nn.Parameter):
            params.append({'params': [self.scale], 'lr': base_lr})
        
        return params
    
    def visualize_conflict_matrix(self, title=None, save_path=None):
        """
        Visualiza a matriz de conflito atual.
        
        Returns:
            matplotlib.figure.Figure
        """
        with torch.no_grad():
            matrix = self.get_conflict_matrix()
            matrix_np = matrix.cpu().numpy() if hasattr(matrix, 'cpu') else matrix
            
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Heatmap
            im = ax.imshow(matrix_np, cmap='RdYlBu_r', vmin=0, vmax=1)
            
            # Configurar eixos
            n_classes = len(self.class_names)
            ax.set_xticks(range(n_classes))
            ax.set_yticks(range(n_classes))
            ax.set_xticklabels(self.class_names, rotation=45, ha='right', fontsize=10)
            ax.set_yticklabels(self.class_names, fontsize=10)
            
            # Adicionar grade
            ax.set_xticks(np.arange(n_classes + 1) - 0.5, minor=True)
            ax.set_yticks(np.arange(n_classes + 1) - 0.5, minor=True)
            ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
            
            # Adicionar valores
            for i in range(n_classes):
                for j in range(n_classes):
                    if i != j:  # Não mostrar diagonal
                        value = matrix_np[i, j]
                        text_color = 'white' if value > 0.5 else 'black'
                        ax.text(j, i, f'{value:.2f}',
                               ha='center', va='center',
                               color=text_color, fontsize=8)
            
            # Barra de cores
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Peso do Conflito', fontsize=12)
            
            # Título
            if title is None:
                title = f'Matriz de Conflito ({self.num_classes} classes)'
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
            
            return fig
    
    def plot_conflict_evolution(self, save_path=None):
        """
        Plota a evolução dos conflitos ao longo do treinamento.
        
        Returns:
            matplotlib.figure.Figure
        """
        if not self.conflict_history:
            return None
        
        n_epochs = len(self.conflict_history)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        # 1. Evolução da média
        mean_weights = [matrix.mean() for matrix in self.conflict_history]
        axes[0].plot(range(n_epochs), mean_weights, 'b-o', linewidth=2, markersize=4)
        axes[0].set_xlabel('Intervalo de Atualização (x100)', fontsize=11)
        axes[0].set_ylabel('Média dos Pesos', fontsize=11)
        axes[0].set_title('Evolução da Média dos Conflitos', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # 2. Evolução do número de conflitos fortes (> 0.5)
        strong_counts = [(matrix > 0.5).sum() for matrix in self.conflict_history]
        axes[1].plot(range(n_epochs), strong_counts, 'r-o', linewidth=2, markersize=4)
        axes[1].set_xlabel('Intervalo de Atualização (x100)', fontsize=11)
        axes[1].set_ylabel('Número de Conflitos Fortes', fontsize=11)
        axes[1].set_title('Evolução dos Conflitos Fortes', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # 3. Matriz inicial vs final
        cmap = plt.cm.RdYlBu_r
        im1 = axes[2].imshow(self.conflict_history[0], cmap=cmap, vmin=0, vmax=1)
        axes[2].set_title('Matriz Inicial', fontsize=12, fontweight='bold')
        axes[2].set_xticks([])
        axes[2].set_yticks([])
        
        im2 = axes[3].imshow(self.conflict_history[-1], cmap=cmap, vmin=0, vmax=1)
        axes[3].set_title(f'Matriz Final ({n_epochs} atualizações)', fontsize=12, fontweight='bold')
        axes[3].set_xticks([])
        axes[3].set_yticks([])
        
        plt.suptitle('Evolução da Matriz de Conflito durante o Treinamento', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
        
        return fig


# Função auxiliar para criar a loss no script de treinamento
def create_enhanced_consistency_loss(args, device):
    """
    Cria a EnhancedKnowledgeConsistencyLoss com base nos argumentos.
    
    Args:
        args: Argumentos do parser
        device: Dispositivo (cuda/cpu)
    
    Returns:
        Tuple: (criterion, optimizer_params)
    """
    print(f"\n{'='*60}")
    print("CONFIGURANDO ENHANCED KNOWLEDGE CONSISTENCY LOSS")
    print(f"{'='*60}")
    
    # Verificar número de classes
    if args.numberOfClasses not in [8, 14]:
        print(f"⚠️  Aviso: Número de classes não otimizado: {args.numberOfClasses}")
        print("A loss foi projetada para 8 ou 14 classes.")
    
    # Criar loss
    criterion = EnhancedKnowledgeConsistencyLoss(
        num_classes=args.numberOfClasses,
        gamma=2.0,
        alpha=0.25,
        prior_strength=getattr(args, 'prior_strength', 0.7),
        conflict_weight=getattr(args, 'conflict_weight', 1.0),
        sparsity_weight=getattr(args, 'sparsity_weight', 0.01),
        symmetry_weight=getattr(args, 'symmetry_weight', 0.001),
        learnable_scale=True,
        reduction='mean',
        gradient_boost_factor=getattr(args, 'gradient_boost', 5.0)
    ).to(device)
    
    # Obter parâmetros para otimização com LR diferenciada
    base_lr = args.learningRate
    optimizer_params = criterion.get_optimizer_params(
        base_lr=base_lr,
        residual_lr_multiplier=getattr(args, 'residual_lr_multiplier', 5.0)
    )
    
    print(f"✓ Loss configurada para {args.numberOfClasses} classes")
    print(f"✓ Prior strength: {criterion.prior_strength}")
    print(f"✓ Gradient boost factor: {criterion.gradient_boost_factor}")
    print(f"✓ Learning rate base: {base_lr}")
    print(f"✓ Residual matrix LR: {base_lr * getattr(args, 'residual_lr_multiplier', 5.0)}")
    print(f"{'='*60}\n")
    
    return criterion, optimizer_params