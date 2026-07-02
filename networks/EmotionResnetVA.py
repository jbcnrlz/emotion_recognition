from torchvision import models
import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F
import os
import numpy as np
from matplotlib import pyplot as plt
from networks.iresnet import iresnet50
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
        
        self.visual_attention_map = None  # Mapa de atenção visual

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
        self.visual_attention_map = visual_attention_map / visual_attention_map.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

        return out

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
        self.activation = nn.Tanh()
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return self.activation(x)
    
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
        
        out = self.attention(out)  # Camada de atenção adicionada
        
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

class ResNet50WithAttentionLikelihood(nn.Module):
    def __init__(self, num_classes=1000, pretrained=None, bottleneck='both', bayesianHeadType='VA', output_dim=2):
        super(ResNet50WithAttentionLikelihood, self).__init__()
        self.num_classes = num_classes
        self.output_dim = output_dim  # Dimensão da saída (ex: 2 para valence/arousal)
        
        # Usar a arquitetura padrão mas com nossos bottlenecks modificados
        self.model = models.resnet50(weights=None)
        if pretrained is not None:
            print("Loading pretrained weights for ResNet50...")
            #self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        
        # Substituir os bottlenecks no layer2 e layer3
        self.attention_maps = []
        self._attention_hooks = []  # Lista para armazenar os hooks de atenção
        if bottleneck in ['first', 'second', 'both']:
            self._replace_bottlenecks(bottleneck)
        else:            
            self.model.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
                nn.AvgPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                SpatialSelfAttention(64)
            )
            def hook_fn(module, input, output):
                self.attention_maps.append(module[-1].visual_attention_map) 
            self._attention_hooks.append(self.model.conv1.register_forward_hook(hook_fn))

        # Modificar camada final
        out_features = self.model.fc.in_features
        self.model.fc = nn.Identity()
        
        # Cabeça para parâmetros da distribuição gaussiana
        # Para cada classe: média (output_dim) + covariância (output_dim * (output_dim + 1) / 2)
        self.likelihood_head = nn.Sequential(
            nn.Linear(out_features, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes * (output_dim + output_dim * (output_dim + 1) // 2))
        )

        self.probabilities = nn.Linear(num_classes * (output_dim + output_dim * (output_dim + 1) // 2), num_classes)
        
        if bayesianHeadType == 'VA':
            self.bayesianHead = BayesianNetworkVI(num_classes, 4, 2)
        elif bayesianHeadType == 'VAD':
            self.bayesianHead = BayesianNetworkVI(num_classes, 6, 3)
    
    def _replace_bottlenecks(self, btn):
        # (Mantido igual ao original)
        if btn == 'first' or btn == 'both':
            block = self.model.layer2[0]
            new_block = BottleneckWithAttention(
                block.conv1.in_channels,
                block.conv1.out_channels,
                block.stride,
                block.downsample
            )
            self.model.layer2[0] = new_block
            def hook_fn(module, input, output):
                self.attention_maps.append(output[1]) 
            self._attention_hooks.append(new_block.register_forward_hook(hook_fn))
        if btn == 'second' or btn == 'both':
            block = self.model.layer3[0]
            new_block = BottleneckWithAttention(
                block.conv1.in_channels,
                block.conv1.out_channels,
                block.stride,
                block.downsample
            )
            self.model.layer3[0] = new_block
            def hook_fn(module, input, output):
                self.attention_maps.append(output[1])
            self._attention_hooks.append(new_block.register_forward_hook(hook_fn))
    
    def forward(self, x):
        self.attention_maps = []  # Limpar os mapas de atenção antes de cada forward
        features = self.model(x)
        distribution_params = self.likelihood_head(features)
        probs = self.probabilities(distribution_params)
        va = self.bayesianHead(probs)
        return probs, distribution_params, va
    
    def get_distribution(self, x):
        """Método para obter a distribuição gaussiana para inferência"""
        with torch.no_grad():
            features = self.model(x)
            distribution_params = self.likelihood_head(features)
            
            # Separar médias e matrizes de covariância
            batch_size = distribution_params.shape[0]
            total_params = self.num_classes * (self.output_dim + self.output_dim * (self.output_dim + 1) // 2)
            
            # Reformatar os parâmetros
            params_reshaped = distribution_params.view(batch_size, self.num_classes, -1)
            
            # Extrair médias (primeiros output_dim elementos)
            means = params_reshaped[:, :, :self.output_dim]
            
            # Extrair elementos da matriz de covariância (triangular inferior)
            cov_params = params_reshaped[:, :, self.output_dim:]
            
            # Construir matrizes de covariância positivas definidas
            cov_matrices = []
            for i in range(self.num_classes):
                L = torch.zeros(batch_size, self.output_dim, self.output_dim, device=x.device)
                tril_indices = torch.tril_indices(self.output_dim, self.output_dim)
                L[:, tril_indices[0], tril_indices[1]] = cov_params[:, i, :]
                
                # Garantir que a diagonal seja positiva (usando softplus)
                diag_indices = torch.arange(self.output_dim)
                L[:, diag_indices, diag_indices] = torch.nn.functional.softplus(L[:, diag_indices, diag_indices])
                
                # Covariância = L * L^T (garante ser positiva definida)
                cov_matrix = torch.bmm(L, L.transpose(1, 2))
                cov_matrices.append(cov_matrix)
            
            cov_matrices = torch.stack(cov_matrices, dim=1)
            
            return means, cov_matrices
    
    def log_likelihood(self, x, targets):
        """Calcula a log-likelihood dos targets dados os parâmetros da distribuição"""
        means, cov_matrices = self.get_distribution(x)
        batch_size = x.shape[0]
        
        log_likelihoods = []
        for i in range(self.num_classes):
            # Criar distribuição multivariada normal para cada classe
            dist = torch.distributions.MultivariateNormal(
                loc=means[:, i, :],
                covariance_matrix=cov_matrices[:, i, :, :]
            )
            
            # Calcular log-likelihood para os targets
            log_prob = dist.log_prob(targets)
            log_likelihoods.append(log_prob)
        
        log_likelihoods = torch.stack(log_likelihoods, dim=1)
        return log_likelihoods

class ResNet50WithAttentionGMM(nn.Module):
    def __init__(self, num_classes=1000,pretrained=None,bottleneck='both',bayesianHeadType='VA'):
        super(ResNet50WithAttentionGMM, self).__init__()
        
        # Usar a arquitetura padrão mas com nossos bottlenecks modificados
        self.model = models.resnet50(weights=None)
        if pretrained is not None:
            print("Loading pretrained weights for ResNet50...")
            #self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        # Substituir os bottlenecks no layer2 e layer3
        self.attention_maps = []
        self._attention_hooks = []  # Lista para armazenar os hooks de atenção
        if bottleneck in ['first', 'second', 'both']:
            self._replace_bottlenecks(bottleneck)
        
        else:            
            self.model.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
                nn.AvgPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                SpatialSelfAttention(64)
            )
            def hook_fn(module, input, output):
                self.attention_maps.append(module[-1].visual_attention_map) 
            self._attention_hooks.append(self.model.conv1.register_forward_hook(hook_fn))


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
        if bayesianHeadType == 'VA':
            self.bayesianHead = BayesianNetworkVI(num_classes, 4, 2)
        elif bayesianHeadType == 'VAD':
            self.bayesianHead = BayesianNetworkVI(num_classes, 6, 3)
    
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
            self.model.layer2[0] = new_block
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
            self.model.layer3[0] = new_block
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

class ResNet50WithAttentionLikelihoodNoVA(nn.Module):
    def __init__(self, num_classes=1000, pretrained=None, bottleneck='both'):
        super(ResNet50WithAttentionLikelihoodNoVA, self).__init__()
        self.num_classes = num_classes
        
        # Usar a arquitetura padrão mas com nossos bottlenecks modificados
        self.model = models.resnet50(weights=None)
        if pretrained is not None:
            print("Loading pretrained weights for ResNet50...")
            #self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        
        # Substituir os bottlenecks no layer2 e layer3
        self.attention_maps = []
        self._attention_hooks = []  # Lista para armazenar os hooks de atenção
        if bottleneck in ['first', 'second', 'both']:
            self._replace_bottlenecks(bottleneck)
        else:            
            self.model.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
                nn.AvgPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                SpatialSelfAttention(64)
            )
            def hook_fn(module, input, output):
                self.attention_maps.append(module[-1].visual_attention_map) 
            self._attention_hooks.append(self.model.conv1.register_forward_hook(hook_fn))

        # Modificar camada final
        out_features = self.model.fc.in_features
        self.model.fc = nn.Identity()
        
        # Cabeça para parâmetros da distribuição gaussiana
        # Para cada classe: média (output_dim) + covariância (output_dim * (output_dim + 1) / 2)
        self.likelihood_head = nn.Sequential(
            nn.Linear(out_features, 256),
            nn.BatchNorm1d(256),
            nn.Sigmoid(),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Sigmoid(),
            nn.Linear(128, num_classes)
        )
    
    def _replace_bottlenecks(self, btn):
        # (Mantido igual ao original)
        if btn == 'first' or btn == 'both':
            block = self.model.layer2[0]
            new_block = BottleneckWithAttention(
                block.conv1.in_channels,
                block.conv1.out_channels,
                block.stride,
                block.downsample
            )
            self.model.layer2[0] = new_block
            def hook_fn(module, input, output):
                self.attention_maps.append(output[1]) 
            self._attention_hooks.append(new_block.register_forward_hook(hook_fn))
        if btn == 'second' or btn == 'both':
            block = self.model.layer3[0]
            new_block = BottleneckWithAttention(
                block.conv1.in_channels,
                block.conv1.out_channels,
                block.stride,
                block.downsample
            )
            self.model.layer3[0] = new_block
            def hook_fn(module, input, output):
                self.attention_maps.append(output[1])
            self._attention_hooks.append(new_block.register_forward_hook(hook_fn))
    
    def forward(self, x):
        self.attention_maps = []  # Limpar os mapas de atenção antes de cada forward
        features = self.model(x)
        distribution_params = self.likelihood_head(features)
        return distribution_params

class RegionAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(RegionAttentionFusion, self).__init__()
        
        # Self-Attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        # Cross-Attention
        self.norm2_q = nn.LayerNorm(embed_dim) # Norm para a Query (global_feat)
        self.norm2_kv = nn.LayerNorm(embed_dim) # Norm para Key/Value (patches)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        # Feed Forward final
        self.norm3 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(), # Pode testar nn.GELU() aqui no futuro, costuma ir melhor em transformers!
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )

        # Norm final opcional (ajuda a estabilizar as predições de VAD)
        self.norm_final = nn.LayerNorm(embed_dim)

    def forward(self, patches, global_feat):
        # 1. Pre-LN Self-Attention nas micro-regiões
        normed_patches = self.norm1(patches)
        attn_patches, _ = self.self_attn(normed_patches, normed_patches, normed_patches)
        patches = patches + attn_patches # Conexão residual limpa!
        
        # 2. Pre-LN Cross-Attention
        normed_global = self.norm2_q(global_feat)
        normed_patches_kv = self.norm2_kv(patches)
        
        # O global_feat (Query) atende aos patches (Key/Value)
        fused_feat, cross_attn_maps = self.cross_attn(
            query=normed_global, 
            key=normed_patches_kv, 
            value=normed_patches_kv
        )
        global_feat = global_feat + fused_feat # Conexão residual limpa!
        
        # 3. Pre-LN FFN
        normed_fused = self.norm3(global_feat)
        global_feat = global_feat + self.ffn(normed_fused) # Conexão residual limpa!
        
        # Aplica norm final e remove a dimensão extra da sequência
        out = self.norm_final(global_feat)
        
        return out.squeeze(1), cross_attn_maps

class ResNet50WithCrossAttention(nn.Module):
    def __init__(self, num_classes=8, pretrained=None, bottleneck='both', num_sectors=7):
        super(ResNet50WithCrossAttention, self).__init__()
        self.num_classes = num_classes
        self.num_sectors = num_sectors # Por padrão, saída da layer4 é 7x7
        
        self.model = None
        if pretrained is not None:
            print("Loading pretrained weights for ResNet50...")
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            self.model = models.resnet50(weights=None)
        
        self.attention_maps = []
        self._attention_hooks = []
        
        # (Seu código original de substituição de bottlenecks mantido)
        if bottleneck in ['first', 'second', 'both']:
            self._replace_bottlenecks(bottleneck)
        else:            
            pass # (Mantive enxuto, adicione seu Sequential original aqui se precisar)

        out_features = self.model.fc.in_features # 2048 para ResNet50
        
        # Remover Average Pooling e FC originais, pois vamos gerenciar o espaço manualmente
        self.model.avgpool = nn.Identity()
        self.model.fc = nn.Identity()
        
        # --- NOVO: Módulo de Atenção Espacial ---
        self.num_patches = self.num_sectors * self.num_sectors
        
        # Positional Encoding para que a rede saiba onde cada patch está (Canto do olho vs Boca)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, out_features) * 0.02)
        self.global_token = nn.Parameter(torch.randn(1, 1, out_features) * 0.02)
        
        # Módulo de Fusão
        self.attention_fusion = RegionAttentionFusion(embed_dim=out_features)

        # Cabeça para parâmetros da distribuição
        self.likelihood_head = nn.Sequential(
            nn.Linear(out_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(), # Troquei o primeiro Sigmoid por ReLU (geralmente melhor para features intermediárias)
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def _replace_bottlenecks(self, btn):
        # (Seu método mantido inalterado)
        pass 
    
    def forward(self, x):
        self.attention_maps = [] 
        B = x.size(0)
        
        # Passar pela ResNet até a layer4 (antes do avgpool que removemos)
        # x shape inicial: [B, 3, 224, 224]
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x) 
        # Saída layer4 shape: [B, 2048, 7, 7]
        
        # 1. Transformar o Grid Espacial em N x N setores (Patches)
        # [B, 2048, 7, 7] -> [B, 2048, 49] -> [B, 49, 2048]
        patches = x.flatten(2).transpose(1, 2) 
        
        # 2. Adicionar Positional Encoding aos patches
        patches = patches + self.pos_embedding
        
        # 3. Expandir o token global para o batch size
        global_feat = self.global_token.expand(B, -1, -1)
        
        # 4. Passar pelo nosso novo módulo de Atenção
        # fused_features shape: [B, 2048]
        # cross_attn_maps shape: [B, 1, 49] (Pode ser usado para visualizar onde a rede olhou!)
        fused_features, cross_attn_maps = self.attention_fusion(patches, global_feat)
        
        # Salvar o mapa de atenção final para visualização se desejar
        self.attention_maps.append(cross_attn_maps.view(B,self.num_sectors, self.num_sectors))

        # 5. Passar para a cabeça de predição
        distribution_params = self.likelihood_head(fused_features)
        
        return distribution_params

class DynamicRegionAttention(nn.Module):
    """
    Substitui a antiga RegionAttentionFusion para evitar "Attention Collapse".
    Usa Multi-Head Attention com uma Query dinâmica extraída do próprio rosto.
    """
    def __init__(self, embed_dim=512, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        
        # Multi-Head Attention nativo (otimizado em C++)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-Forward Network para não linearidade adicional e estabilidade
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, patches):
        """
        patches: Tensores de features de entrada shape [Batch, 49, 512]
        """
        # 1. Query Dinâmica: Em vez de um token fixo (que causa vício), a Query 
        # agora é a média das características de toda a imagem daquela iteração específica.
        # Shape: [Batch, 1, 512]
        dynamic_query = patches.mean(dim=1, keepdim=True)
        
        # 2. Multi-Head Attention
        # A rede vai "perguntar" aos patches quais regiões importam baseando-se no resumo global.
        # attn_weights shape: [Batch, 1, 49]
        attn_output, attn_weights = self.multihead_attn(
            query=dynamic_query,
            key=patches,
            value=patches
        )
        
        # 3. Bloco Residual + Layer Normalization
        fused_features = self.norm1(dynamic_query + attn_output)
        
        # 4. FFN + Bloco Residual + Layer Normalization
        fused_features = self.norm2(fused_features + self.ffn(fused_features))
        
        # O squeeze remove a dimensão extra do sequence_length (passando de [B, 1, 512] para [B, 512])
        return fused_features.squeeze(1), attn_weights


class Glint360kResNetWithCrossAttention(nn.Module):
    def __init__(self, num_classes=8, pretrained_path=None, num_sectors=7):
        super(Glint360kResNetWithCrossAttention, self).__init__()
        self.num_classes = num_classes
        self.num_sectors = num_sectors
        
        # 1. Instancia o backbone oficial do InsightFace
        self.backbone = iresnet50()
        
        # 2. Carregamento dos pesos do Glint360K
        if pretrained_path is not None:
            print(f"Carregando pesos Glint360K de: {pretrained_path}...")
            state_dict = torch.load(pretrained_path, map_location='cpu')
            
            # Limpa as chaves caso os pesos tenham sido salvos com nn.DataParallel ('module.')
            clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            self.backbone.load_state_dict(clean_state_dict, strict=False)
            print("Pesos do backbone carregados com sucesso!")

        # Logo após carregar o modelo no __init__
        self.set_backbone_freeze(freeze=True)  # Congela o backbone por padrão

        out_features = 512 
        self.num_patches = self.num_sectors * self.num_sectors
        
        # --- NOVO: Adaptive Pooling ---
        # Isso garante que a saída da convolução seja sempre redimensionada
        # para o grid exato (ex: 7x7) que o mecanismo de atenção espera,
        # independentemente do tamanho da imagem de entrada (112, 224, 256, etc).
        self.adaptive_pool = nn.AdaptiveAvgPool2d((self.num_sectors, self.num_sectors))
        # ------------------------------
        
        # --- Módulo de Atenção Espacial ---
        # Positional Encoding mantido para a rede saber a posição de cada feature
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, out_features) * 0.02)
        
        # 3. Nova Atenção Dinâmica
        self.attention_fusion = DynamicRegionAttention(embed_dim=out_features, num_heads=4)

        self.attention_maps = []

        # Cabeça de predição ajustada para receber 512 canais iniciais
        self.likelihood_head = nn.Sequential(
            nn.Linear(out_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.7),  # Dropout mais agressivo para evitar overfitting
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, num_classes)
        )
        
    def set_backbone_freeze(self, freeze=True):
        """
        Congela ou descongelar todos os parâmetros do backbone IResNet.
        As camadas de atenção e likelihood_head continuarão treinando.
        """
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
        
        status = "CONGELADO" if freeze else "DESCONGELADO"
        print(f"[*] Backbone Glint360K está agora: {status}")
    
    def forward(self, x):
        self.attention_maps = []
        B = x.size(0)
        
        # CRÍTICO: Certifique-se que o rosto está pré-alinhado (RetinaFace/MTCNN)!
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.prelu(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x) 
        
        # --- NOVO: Aplicar o Adaptive Pooling ---
        # Transforma o mapa de características gerado pela layer4
        # rigidamente em [B, 512, num_sectors, num_sectors]
        x = self.adaptive_pool(x) 
        # ----------------------------------------
        
        # Agora a saída é GARANTIDA de ser shape: [B, 512, 7, 7]
        
        # 1. Transformar o Grid em Setores (Patches)
        # [B, 512, 7, 7] -> [B, 512, 49] -> [B, 49, 512]
        patches = x.flatten(2).transpose(1, 2) 
        
        # 2. Somar o Positional Encoding
        # Agora `patches` sempre terá dimensão 49, casando perfeitamente com `self.pos_embedding`
        patches = patches + self.pos_embedding
        
        # 3. Fusão com a Nova Atenção Dinâmica
        # fused_features shape: [B, 512]
        # cross_attn_maps shape: [B, 1, 49]
        fused_features, cross_attn_maps = self.attention_fusion(patches)
        
        # 4. Salvar os mapas de atenção, redimensionando de volta para o Grid
        self.attention_maps.append(cross_attn_maps.view(B, self.num_sectors, self.num_sectors))
        
        # 5. Predição Final
        distribution_params = self.likelihood_head(fused_features)
        
        return distribution_params