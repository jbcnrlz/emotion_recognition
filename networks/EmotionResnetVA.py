from torchvision import models
import torch
import torch.nn as nn
import torch.distributions as dist
from torch.optim import Adam

class BayesianLinearVI(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Parâmetros variacionais para os pesos
        self.w_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
        self.w_rho = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
        
        # Parâmetros variacionais para os vieses
        self.b_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
        self.b_rho = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
        
        # Prior
        self.w_prior = dist.Normal(0, 1)
        self.b_prior = dist.Normal(0, 1)
        
        # Escala mínima para evitar NaN
        self.min_scale = 1e-6
    
    def forward(self, x):
        # Amostrar pesos e vieses da distribuição variacional
        w_epsilon = torch.randn_like(self.w_mu)
        w_scale = torch.log1p(torch.exp(self.w_rho)) + self.min_scale
        w = self.w_mu + w_epsilon * w_scale
        
        b_epsilon = torch.randn_like(self.b_mu)
        b_scale = torch.log1p(torch.exp(self.b_rho)) + self.min_scale
        b = self.b_mu + b_epsilon * b_scale
        
        # Calcular log prob das amostras sob o prior e a distribuição variacional
        self.w_post = dist.Normal(self.w_mu, w_scale)
        self.b_post = dist.Normal(self.b_mu, b_scale)
        
        self.kl_div = (dist.kl_divergence(self.w_post, self.w_prior).sum() +
                      dist.kl_divergence(self.b_post, self.b_prior).sum())
        
        return nn.functional.linear(x, w, b)

class BayesianNetworkVI(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = BayesianLinearVI(input_dim, hidden_dim)
        self.linear2 = BayesianLinearVI(hidden_dim, output_dim)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def kl_divergence(self):
        return self.linear1.kl_div + self.linear2.kl_div

class ResnetWithBayesianHead(nn.Module):
    def __init__(self,classes,pretrained=None,resnetModel=18,softmax=False):
        super(ResnetWithBayesianHead,self).__init__()
        if resnetModel == 18:
            self.innerResnetModel = models.resnet18(weights=pretrained)
        elif resnetModel == 34:
            self.innerResnetModel = models.resnet34(weights=pretrained)
        elif resnetModel == 50:
            self.innerResnetModel = models.resnet50(weights=pretrained)
        elif resnetModel == 101:
            self.innerResnetModel = models.resnet101(weights=pretrained)
        elif resnetModel == 152:
            self.innerResnetModel = models.resnet152(weights=pretrained)
        else:
            raise ValueError("Invalid ResNet model specified.")        
        self.innerResnetModel.fc = nn.Linear(self.innerResnetModel.fc.in_features, classes,bias=False)

        self.bayesianHead = BayesianNetworkVI(classes, 4, 2)
        self.softmax = None
        if (softmax):
            self.softmax = nn.Softmax(dim=1)


    def forward(self, x):        
        distributions = self.innerResnetModel(x)
        if self.softmax is not None:
            distributions = self.softmax(distributions)
        va = self.bayesianHead(distributions)
        return distributions, va