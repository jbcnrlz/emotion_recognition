import torch
import torch.nn as nn
from torchvision.models import resnet50
from torch.nn import Parameter
import torch.nn.functional as F

class CosFace(nn.Module):
    """Implementação do CosFace (Large Margin Cosine Loss)"""
    
    def __init__(self, in_features, out_features, s=30.0, m=0.35):
        super(CosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output
    
    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'

class ResNetCosFace(nn.Module):
    def __init__(self, num_classes, embedding_size=512, pretrained=True):
        super(ResNetCosFace, self).__init__()
        
        # Carrega a ResNet pré-treinada
        self.backbone = resnet50(pretrained=pretrained)
        
        # Remove a camada fully connected original
        self.backbone.fc = nn.Identity()
        
        # Adiciona uma nova cabeça para extrair embeddings
        self.embedding = nn.Sequential(
            nn.Linear(2048, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.ReLU(inplace=True)
        )
        
        # Camada CosFace
        self.cosface = CosFace(embedding_size, num_classes)
        
    def forward(self, x, labels=None):
        # Extrai features
        features = self.backbone(x)
        embeddings = self.embedding(features)
        
        # Se não houver labels, retorna apenas os embeddings
        if labels is None:
            return embeddings
            
        # Calcula os logits do CosFace
        logits = self.cosface(embeddings, labels)
        
        return logits, embeddings

# Função de perda combinada (pode ser usada para treinamento)
def cosface_loss(logits, labels):
    return F.cross_entropy(logits, labels)