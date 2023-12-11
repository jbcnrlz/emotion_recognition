from torchvision.models import vit_b_16
from torch import nn
import torch

class ViTEmotionHeadClassifier(nn.Module):
    def __init__(self,classes,pretrained=None) -> None:        
        super(ViTEmotionHeadClassifier,self).__init__()              

        self.backbone = vit_b_16(image_size=256)
        self.backbone.heads.head = nn.Linear(768, classes,bias=True)
        


    def forward(self, x):        
        classLogits = self.backbone(x)
        return classLogits

