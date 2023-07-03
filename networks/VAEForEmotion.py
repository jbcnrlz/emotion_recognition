import torch.nn as nn
from diffusers.models.vae import DiagonalGaussianDistribution
from networks.unet.unet_parts import *

class VAEOurEmotion(nn.Module):

    def __init__(self,n_channels) -> None:
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
            nn.Linear(1024,1)
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
        z = z.reshape([-1,1024,1,1])
        decoded = self.decoder(z)
        return valenceValue, decoded, z


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
        
