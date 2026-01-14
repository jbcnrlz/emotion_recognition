import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class KnowledgeGuidedConsistencyLoss(nn.Module):
    def __init__(self, num_classes=8, gamma=2.0, alpha=0.25, 
                 prior_strength=0.5, learnable_scale=True):
        """
        Versão automática que escolhe a matriz correta baseada em num_classes
        """
        super(KnowledgeGuidedConsistencyLoss, self).__init__()
        
        self.num_classes = num_classes
        
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
        return self.loss_module(logits, targets)
    
    def get_conflict_matrix(self):
        return self.loss_module.get_conflict_matrix()
    
    def analyze_conflicts(self, threshold=0.3):
        if hasattr(self.loss_module, 'analyze_conflicts'):
            return self.loss_module.analyze_conflicts(threshold)
        else:
            with torch.no_grad():
                matrix = self.get_conflict_matrix().cpu().numpy()
                print(f"\nMatriz de conflito ({self.num_classes} classes):")
                print(matrix)
                return matrix    


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
        
    def _create_prior_matrix_8classes(self):
        """Cria matriz de conhecimento prévio para 8 classes"""
        prior = torch.zeros(self.num_classes, self.num_classes)
        
        # CONFLITOS FORTES (peso 0.8)
        # Happy vs emoções negativas
        strong_conflicts = [
            (self.idx_happy, self.idx_angry),      # happy vs angry
            (self.idx_happy, self.idx_disgusted),  # happy vs disgusted
            (self.idx_happy, self.idx_fearful),    # happy vs fearful
            (self.idx_happy, self.idx_sad),        # happy vs sad
            
            # Surprised vs emoções negativas
            (self.idx_surprised, self.idx_disgusted),  # surprised vs disgusted
            (self.idx_surprised, self.idx_fearful),    # surprised vs fearful
            
            # Angry vs emoções positivas e neutras
            (self.idx_angry, self.idx_happy),      # angry vs happy
            (self.idx_angry, self.idx_sad),        # angry vs sad
            (self.idx_angry, self.idx_neutral),    # angry vs neutral
            
            # Disgusted vs emoções positivas e neutras
            (self.idx_disgusted, self.idx_happy),  # disgusted vs happy
            (self.idx_disgusted, self.idx_surprised),  # disgusted vs surprised
            (self.idx_disgusted, self.idx_neutral),  # disgusted vs neutral
            
            # Fearful vs emoções positivas e neutras
            (self.idx_fearful, self.idx_happy),    # fearful vs happy
            (self.idx_fearful, self.idx_surprised),  # fearful vs surprised
            (self.idx_fearful, self.idx_neutral),  # fearful vs neutral
            
            # Sad vs emoções positivas e neutras
            (self.idx_sad, self.idx_happy),        # sad vs happy
            (self.idx_sad, self.idx_angry),        # sad vs angry
            (self.idx_sad, self.idx_neutral),      # sad vs neutral
        ]
        
        # CONFLITOS MODERADOS (peso 0.4)
        # Contempt vs outras emoções
        moderate_conflicts = [
            # Contempt conflicts
            (self.idx_contempt, self.idx_happy),    # contempt vs happy
            (self.idx_contempt, self.idx_angry),    # contempt vs angry
            (self.idx_contempt, self.idx_disgusted), # contempt vs disgusted
            (self.idx_contempt, self.idx_sad),      # contempt vs sad
            (self.idx_contempt, self.idx_neutral),  # contempt vs neutral
            
            # Neutral vs todas as outras (conflito moderado)
            (self.idx_neutral, self.idx_happy),     # neutral vs happy
            (self.idx_neutral, self.idx_surprised), # neutral vs surprised
            (self.idx_neutral, self.idx_angry),     # neutral vs angry
            (self.idx_neutral, self.idx_disgusted), # neutral vs disgusted
            (self.idx_neutral, self.idx_fearful),   # neutral vs fearful
            (self.idx_neutral, self.idx_sad),       # neutral vs sad
            
            # Pares adicionais que podem ter conflito
            (self.idx_angry, self.idx_disgusted),   # angry vs disgusted
            (self.idx_disgusted, self.idx_fearful), # disgusted vs fearful
            (self.idx_fearful, self.idx_sad),       # fearful vs sad
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
    
    def forward(self, logits, targets):
        # Obter matriz de conflito
        conflict_weights = self.get_conflict_matrix()
        
        # Focal Loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = torch.exp(-bce_loss)
        focal_term = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.prior_strength) * (1 - targets)
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
    
    def analyze_conflicts(self, threshold=0.3):
        """Analisa os conflitos aprendidos"""
        with torch.no_grad():
            matrix = self.get_conflict_matrix().cpu().numpy()
            
            print("\n=== Análise de Conflitos (8 Classes) ===")
            print(f"Ordem: {self.class_names}")
            
            # Top 10 conflitos
            conflicts = []
            for i in range(self.num_classes):
                for j in range(i+1, self.num_classes):
                    conflicts.append((i, j, matrix[i, j]))
            
            conflicts.sort(key=lambda x: x[2], reverse=True)
            
            print("\nTop 10 conflitos:")
            for i, j, weight in conflicts[:10]:
                print(f"  {self.class_names[i]} vs {self.class_names[j]}: {weight:.3f}")
            
            # Conflitos por emoção
            print("\nConflitos por emoção (média):")
            for idx, name in enumerate(self.class_names):
                # Calcular média dos conflitos com outras emoções
                other_conflicts = []
                for j in range(self.num_classes):
                    if j != idx:
                        other_conflicts.append(matrix[idx, j])
                
                avg_conflict = np.mean(other_conflicts)
                print(f"  {name}: {avg_conflict:.3f}")
            
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
        
        # Nomes das classes
        self.class_names = [
            'happy', 'contempt', 'elated', 'hopeful', 'surprised',
            'proud', 'loved', 'angry', 'astonished', 'disgusted',
            'fearful', 'sad', 'fatigued', 'neutral'
        ]
        
        # Índices das emoções
        self.idx = {
            'happy': 0, 'contempt': 1, 'elated': 2, 'hopeful': 3,
            'surprised': 4, 'proud': 5, 'loved': 6, 'angry': 7,
            'astonished': 8, 'disgusted': 9, 'fearful': 10,
            'sad': 11, 'fatigued': 12, 'neutral': 13
        }
        
    def _create_prior_matrix_14classes(self):
        """Cria matriz de conhecimento prévio para 14 classes"""
        prior = torch.zeros(self.num_classes, self.num_classes)
        
        # CONFLITOS FORTES (peso 0.8)
        # Emoções positivas vs negativas
        strong_conflicts = [
            # Happy e similares vs negativas
            (self.idx['happy'], self.idx['angry']),
            (self.idx['happy'], self.idx['disgusted']),
            (self.idx['happy'], self.idx['fearful']),
            (self.idx['happy'], self.idx['sad']),
            
            # Elated vs negativas
            (self.idx['elated'], self.idx['angry']),
            (self.idx['elated'], self.idx['disgusted']),
            (self.idx['elated'], self.idx['sad']),
            
            # Hopeful vs negativas
            (self.idx['hopeful'], self.idx['angry']),
            (self.idx['hopeful'], self.idx['disgusted']),
            (self.idx['hopeful'], self.idx['fearful']),
            
            # Surprised vs disgusted/fearful
            (self.idx['surprised'], self.idx['disgusted']),
            (self.idx['surprised'], self.idx['fearful']),
            
            # Proud vs negativas
            (self.idx['proud'], self.idx['angry']),
            (self.idx['proud'], self.idx['disgusted']),
            (self.idx['proud'], self.idx['sad']),
            
            # Loved vs negativas
            (self.idx['loved'], self.idx['angry']),
            (self.idx['loved'], self.idx['disgusted']),
            (self.idx['loved'], self.idx['fearful']),
            
            # Angry vs positivas e neutras
            (self.idx['angry'], self.idx['happy']),
            (self.idx['angry'], self.idx['elated']),
            (self.idx['angry'], self.idx['hopeful']),
            (self.idx['angry'], self.idx['loved']),
            (self.idx['angry'], self.idx['neutral']),
            
            # Disgusted vs positivas e neutras
            (self.idx['disgusted'], self.idx['happy']),
            (self.idx['disgusted'], self.idx['elated']),
            (self.idx['disgusted'], self.idx['hopeful']),
            (self.idx['disgusted'], self.idx['surprised']),
            (self.idx['disgusted'], self.idx['neutral']),
            
            # Fearful vs positivas e neutras
            (self.idx['fearful'], self.idx['happy']),
            (self.idx['fearful'], self.idx['hopeful']),
            (self.idx['fearful'], self.idx['surprised']),
            (self.idx['fearful'], self.idx['loved']),
            (self.idx['fearful'], self.idx['neutral']),
            
            # Sad vs positivas e neutras
            (self.idx['sad'], self.idx['happy']),
            (self.idx['sad'], self.idx['elated']),
            (self.idx['sad'], self.idx['hopeful']),
            (self.idx['sad'], self.idx['loved']),
            (self.idx['sad'], self.idx['neutral']),
            
            # Fatigued vs positivas
            (self.idx['fatigued'], self.idx['happy']),
            (self.idx['fatigued'], self.idx['elated']),
            (self.idx['fatigued'], self.idx['hopeful']),
        ]
        
        # CONFLITOS MODERADOS (peso 0.4)
        moderate_conflicts = [
            # Contempt vs outras
            (self.idx['contempt'], self.idx['happy']),
            (self.idx['contempt'], self.idx['elated']),
            (self.idx['contempt'], self.idx['hopeful']),
            (self.idx['contempt'], self.idx['angry']),
            (self.idx['contempt'], self.idx['disgusted']),
            (self.idx['contempt'], self.idx['sad']),
            (self.idx['contempt'], self.idx['neutral']),
            
            # Astonished vs outras
            (self.idx['astonished'], self.idx['angry']),
            (self.idx['astonished'], self.idx['disgusted']),
            (self.idx['astonished'], self.idx['fearful']),
            (self.idx['astonished'], self.idx['sad']),
            (self.idx['astonished'], self.idx['neutral']),
            
            # Neutral vs todas as outras (conflito moderado)
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
            
            # Pares adicionais
            (self.idx['surprised'], self.idx['astonished']),  # tipos diferentes de surpresa
            (self.idx['angry'], self.idx['disgusted']),
            (self.idx['disgusted'], self.idx['fearful']),
            (self.idx['fearful'], self.idx['sad']),
            (self.idx['sad'], self.idx['fatigued']),
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
        
        # Calcular produto pareado
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
    
    def analyze_conflicts(self, threshold=0.3):
        """Analisa os conflitos aprendidos"""
        with torch.no_grad():
            matrix = self.get_conflict_matrix().cpu().numpy()
            
            print("\n=== Análise de Conflitos (14 Classes) ===")
            print(f"Ordem: {self.class_names}")
            
            # Top 15 conflitos
            conflicts = []
            for i in range(self.num_classes):
                for j in range(i+1, self.num_classes):
                    conflicts.append((i, j, matrix[i, j]))
            
            conflicts.sort(key=lambda x: x[2], reverse=True)
            
            print("\nTop 15 conflitos:")
            for i, j, weight in conflicts[:15]:
                print(f"  {self.class_names[i]:10s} vs {self.class_names[j]:10s}: {weight:.3f}")
            
            return matrix
        
class GenericConsistencyLoss(nn.Module):
    """Versão genérica sem conhecimento prévio específico"""
    def __init__(self, num_classes=8, gamma=2.0, alpha=0.25, 
                 prior_strength=0.0, learnable_scale=True):
        super(GenericConsistencyLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.num_classes = num_classes
        self.prior_strength = prior_strength
        
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
        
        # Calcular produto pareado
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