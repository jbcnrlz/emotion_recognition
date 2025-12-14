import cv2, torch
import matplotlib.pyplot as plt
from matplotlib import cm
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        # Registrar hooks
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
            
        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_backward_hook(backward_hook))
    
    def __call__(self, x, target_class=None):
        # Ativar gradientes se não estiverem ativos
        if not x.requires_grad:
            x = x.clone().requires_grad_(True)
            
        # Forward pass
        probs, distribution_params, va = self.model(x)
        
        # Se target_class não for especificado, use a classe com maior probabilidade
        if target_class is None:
            target_class = probs.argmax(dim=1)
        
        # Zerar gradientes
        self.model.zero_grad()
        
        # Criar one-hot para a classe alvo
        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(1, target_class.unsqueeze(1), 1.0)
        
        # Calcular gradientes
        probs.backward(gradient=one_hot, retain_graph=True)
        
        # Obter pesos dos gradientes (média sobre dimensões espaciais)
        if self.gradients is not None:
            weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        else:
            # Se não houver gradientes, criar um tensor de zeros
            weights = torch.zeros(1, self.activations.shape[1], 1, 1, device=x.device)
        
        # Calcular Grad-CAM
        if self.activations is not None:
            cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
            cam = F.relu(cam)  # Aplicar ReLU
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
        else:
            cam = torch.zeros(1, 1, x.shape[2]//32, x.shape[3]//32, device=x.device)
        
        return cam, probs, va
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()