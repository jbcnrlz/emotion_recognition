from torchvision import models
import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F
import os
import numpy as np
from matplotlib import pyplot as plt
class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialSelfAttention, self).__init__()
        self.in_channels = in_channels
        
        # Camadas para Q, K, V - agora operando nos canais
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Fator de escala
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x, mask=None):
        """
        Input:
            x: feature map com shape (batch, channels, height, width)
            mask: optional, com shape (batch, height*width, height*width)
        """
        batch_size, C, H, W = x.shape
        
        # Gerar queries, keys e values
        query = self.query_conv(x).view(batch_size, -1, H*W).permute(0, 2, 1)  # (B, N, C')
        key = self.key_conv(x).view(batch_size, -1, H*W)  # (B, C', N)
        value = self.value_conv(x).view(batch_size, -1, H*W)  # (B, C, N)
        
        # Calcular matriz de atenção
        attention = torch.bmm(query, key)  # (B, N, N)
        attention = attention / (self.in_channels ** 0.5)
        
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-1e20"))
        
        attention = F.softmax(attention, dim=-1)
        
        attention_map_softmax = F.softmax(attention, dim=-1) 

        # Aplicar atenção aos values
        out = torch.bmm(value, attention.permute(0, 2, 1))  # (B, C, N)
        out = out.view(batch_size, C, H, W)
        
        # Conexão residual
        out = self.gamma * out + x
        
        visual_attention_map = attention_map_softmax.sum(dim=1).view(batch_size, H, W)
        visual_attention_map = visual_attention_map / visual_attention_map.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

        return out, visual_attention_map

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
    
class ResnetWithBayesianGMMHead(nn.Module):
    def __init__(self,classes,pretrained=None,resnetModel=18):
        super(ResnetWithBayesianGMMHead,self).__init__()
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
        
        out_features = self.innerResnetModel.fc.in_features
        self.innerResnetModel.fc = nn.Identity()
        self.gmm_head = nn.Sequential(
            nn.Linear(out_features, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, classes * 6),  # 6 parâmetros: peso, μx, μy, σx, σy, ρ
        )

        self.probabilities = nn.Linear(classes * 6,classes)  # Saída para os parâmetros da GMM


        self.bayesianHead = BayesianNetworkVI(classes, 4, 2)


    def forward(self, x):        
        distributions = self.innerResnetModel(x)
        distributions = self.gmm_head(distributions)
        probs = self.probabilities(distributions)
        va = self.bayesianHead(probs)
        return probs, distributions, va
    

class BottleneckWithAttention(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckWithAttention, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.attention = SpatialSelfAttention(planes)  # Atenção após a primeira conv
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out, _ = self.attention(out)  # Camada de atenção adicionada
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out  # Retorna o mapa de atenção junto com a saída

class ResNet50WithAttentionGMM(nn.Module):
    def __init__(self, num_classes=1000,pretrained=None,bottleneck='both'):
        super(ResNet50WithAttentionGMM, self).__init__()
        
        # Usar a arquitetura padrão mas com nossos bottlenecks modificados
        self.model = models.resnet50(weights=None)
        if pretrained is not None:
            print("Loading pretrained weights for ResNet50...")
            #self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        # Substituir os bottlenecks no layer2 e layer3
        self.attention_maps = []
        self._attention_hooks = []  # Lista para armazenar os hooks de atenção
        self._replace_bottlenecks(bottleneck)
        
        # Modificar camada final
        out_features = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.gmm_head = nn.Sequential(
            nn.Linear(out_features, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, num_classes * 6),  # 6 parâmetros: peso, μx, μy, σx, σy, ρ
        )

        self.probabilities = nn.Linear(num_classes * 6,num_classes)  # Saída para os parâmetros da GMM
        self.bayesianHead = BayesianNetworkVI(num_classes, 4, 2)
    
    def _replace_bottlenecks(self,btn):
        # Substituir os bottlenecks no layer2
        if btn == 'first' or btn == 'both':
            block = self.model.layer2[0]
            new_block = BottleneckWithAttention(
                block.conv1.in_channels,
                block.conv1.out_channels,
                block.stride,
                block.downsample
            )
            self.model.layer2[i] = new_block
            def hook_fn(module, input, output):
                self.attention_maps.append(output[1]) 
            self._attention_hooks.append(new_block.register_forward_hook(hook_fn))
        if btn == 'second' or btn == 'both':
            # Substituir os bottlenecks no layer3
            block = self.model.layer3[0]
            new_block = BottleneckWithAttention(
                block.conv1.in_channels,
                block.conv1.out_channels,
                block.stride,
                block.downsample
            )
            self.model.layer3[i] = new_block
            def hook_fn(module, input, output):
                self.attention_maps.append(output[1])
            self._attention_hooks.append(new_block.register_forward_hook(hook_fn))
    
    def forward(self, x):
        self.attention_maps = []  # Limpar os mapas de atenção antes de cada forward
        distributions = self.model(x)
        distributions = self.gmm_head(distributions)
        probs = self.probabilities(distributions)
        va = self.bayesianHead(probs)
        return probs, distributions, va
