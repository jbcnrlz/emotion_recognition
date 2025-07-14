import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # Peso para a classe positiva
        self.gamma = gamma  # Foco em exemplos difíceis

    def forward(self, inputs, targets):
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)  # Probabilidade da classe correta
        loss = self.alpha * (1 - pt)**self.gamma * BCE_loss
        return loss.mean()
