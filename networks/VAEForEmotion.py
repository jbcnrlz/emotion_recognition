import torch.nn as nn
from diffusers.models.vae import DiagonalGaussianDistribution
from networks.unet.unet_parts import *

def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class PAM_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.init_weights()

    def init_weights(self):
        kaiming_init(self.query_conv)
        kaiming_init(self.key_conv)
        kaiming_init(self.value_conv)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

class VAEOurEmotionAttention(nn.Module):

    def __init__(self,n_channels,n_classes=8) -> None:
        super(VAEOurEmotionAttention,self).__init__()

        self.encoder = nn.Sequential(
            inconv(n_channels, 64),
            down(64, 128),
            down(128, 256),
            down(256, 512),
            down(512, 1024),
            down(1024, 1024),            
            down(1024, 1024),            
            down(1024, 1024),
            #down(1024, 1024),
            nn.MaxPool2d(kernel_size=2,stride=1)
        )

        self.selfAttn = PAM_Module(1024)

        self.featuresLearnedMu = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=True)
        )
        self.featuresLearnedSigma = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(            
            upVAE(1024, 1024),            
            upVAE(1024, 1024),            
            upVAE(1024, 1024),
            upVAE(1024, 512),
            upVAE(512, 256),
            upVAE(256, 128),
            upVAE(128, 64),
            upVAE(64, n_channels),
            outconv(n_channels, n_channels)
        )

        self.valenceEst = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024,1)
        )

        
        self.classfier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024,n_classes),
            nn.LogSoftmax(dim=1)
        )

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()


    def forward(self, x):
        fe = self.encoder(x)
        fe = self.selfAttn(fe)
        fe = fe.view(fe.size(0),-1)
        mu = self.featuresLearnedMu(fe)
        sigma = torch.exp(self.featuresLearnedSigma(fe))
        z = mu + sigma * self.N.sample(mu.shape)
        valenceValue = self.valenceEst(z)
        z_class = z.clone()
        classification = self.classfier(z_class)
        z_dec = z.reshape([-1,1024,1,1])
        decoded = self.decoder(z_dec)
        return valenceValue, classification, decoded, z


class VAEOurEmotion(nn.Module):

    def __init__(self,n_channels,n_classes=8) -> None:
        super(VAEOurEmotion,self).__init__()

        self.encoder = nn.Sequential(
            inconv(n_channels, 64),
            down(64, 128),
            down(128, 256),
            down(256, 512),
            down(512, 1024),
            down(1024, 1024),            
            down(1024, 1024),            
            down(1024, 1024),
            #down(1024, 1024),
            nn.MaxPool2d(kernel_size=2,stride=1)
        )

        self.featuresLearnedMu = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=True)
        )
        self.featuresLearnedSigma = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(            
            upVAE(1024, 1024),            
            upVAE(1024, 1024),            
            upVAE(1024, 1024),
            upVAE(1024, 512),
            upVAE(512, 256),
            upVAE(256, 128),
            upVAE(128, 64),
            upVAE(64, n_channels),
            outconv(n_channels, n_channels)
        )

        self.valenceEst = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024,1)
        )

        
        self.classfier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024,n_classes),
            nn.LogSoftmax(dim=1)
        )

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()


    def forward(self, x):
        fe = self.encoder(x)
        fe = fe.view(fe.size(0),-1)
        mu = self.featuresLearnedMu(fe)
        sigma = torch.exp(self.featuresLearnedSigma(fe))
        z = mu + sigma * self.N.sample(mu.shape)
        valenceValue = self.valenceEst(z)
        z_class = z.clone()
        classification = self.classfier(z_class)
        z_dec = z.reshape([-1,1024,1,1])
        decoded = self.decoder(z_dec)
        return valenceValue, classification, decoded, z


class VAEForEmotion(nn.Module):

    def __init__(self,vaeAutoEncoder) -> None:        
        super(VAEForEmotion,self).__init__()

        self.encoder = vaeAutoEncoder.encoder
        self.decoder = vaeAutoEncoder.decoder
        self.quant_conv = vaeAutoEncoder.quant_conv
        self.post_quant_conv = vaeAutoEncoder.post_quant_conv
        self.softmax = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 8),
            nn.Softmax(dim=1)
        )


    def forward(self, x):
        x = self.encoder(x)
        moments = self.quant_conv(x)
        dgd = DiagonalGaussianDistribution(moments)
        x = self.post_quant_conv(dgd.mode())
        x = self.decoder(x)
        return x, dgd.mode(), dgd.var, self.softmax(dgd.mode())
        
