import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

class CosFace(nn.Module):
    """Implementação do CosFace (Large Margin Cosine Loss)"""
    
    def __init__(self, in_features, out_features, s=30.0, m=0.35):
        """
        Args:
            in_features (int): tamanho dos features de entrada (dimensionalidade do embedding)
            out_features (int): tamanho dos features de saída (número de classes)
            s (float): fator de escala (default: 30.0)
            m (float): margem de cosine (default: 0.35)
        """
        super(CosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        
        # Pesos da camada fully connected
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, input, label):
        """
        Args:
            input (torch.Tensor): features de entrada, shape [batch_size, in_features]
            label (torch.LongTensor): rótulos de classe, shape [batch_size]
        Returns:
            torch.Tensor: valor da perda
        """
        # Calcular cosseno dos ângulos entre os embeddings e os pesos das classes
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        
        # Adicionar a margem m ao cosseno do ângulo da classe correta
        phi = cosine - self.m
        
        # Converter labels para one-hot encoding
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        
        # Aplicar a margem apenas aos exemplos corretos
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        # Escalar os logits
        output *= self.s
        
        # Calcular a perda de cross-entropy
        loss = F.cross_entropy(output, label)
        
        return loss
    
    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'