import torch.nn as nn
from diffusers.models.vae import DiagonalGaussianDistribution

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
        
