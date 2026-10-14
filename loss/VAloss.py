from torch.nn import functional as F
import torch.distributions as dist
import torch
from torch import nn

class CCCLoss(nn.Module):
    def __init__(self):
        super(CCCLoss, self).__init__()
        
    def forward(self, pred, target):
        # Calcula CCC para cada dimensão (V e A separadamente) e retorna a média
        mean_pred = torch.mean(pred, dim=0)
        mean_target = torch.mean(target, dim=0)
        
        var_pred = torch.var(pred, dim=0)
        var_target = torch.var(target, dim=0)
        
        cov = torch.mean((pred - mean_pred) * (target - mean_target), dim=0)
        
        ccc = (2 * cov) / (var_pred + var_target + (mean_pred - mean_target)**2 + 1e-8)
        
        # Como queremos minimizar a loss, retornamos 1 - CCC
        return 1.0 - torch.mean(ccc)

def regularized_gmm_loss(y_pred, y_true, components, n_components, alpha=0.1):
    mse_loss = F.mse_loss(y_pred, y_true)
    weights = F.softmax(components.view(components.size(0), n_components, -1)[:, :, 0], dim=1)
    cov_params = components.view(components.size(0), n_components, -1)[:, :, 3:]
    weight_reg = -torch.mean(torch.log(weights + 1e-6))
    det_reg = 0.0
    for i in range(n_components):
        sx = torch.exp(cov_params[:, i, 0])
        sy = torch.exp(cov_params[:, i, 1])
        rho = torch.tanh(cov_params[:, i, 2])
        det = sx**2 * sy**2 * (1 - rho**2)
        det_reg += torch.mean(1 / (det + 1e-6))
    return mse_loss + alpha * (weight_reg + det_reg)

def elbo_loss(output, target, model, kl_weight=1.0, sigma=1.0):
    likelihood = dist.Normal(output, sigma)
    log_likelihood = likelihood.log_prob(target).sum()
    kl_divergence = model.kl_divergence()
    elbo = log_likelihood - kl_weight * kl_divergence
    return -elbo
