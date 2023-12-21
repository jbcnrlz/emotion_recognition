from torchvision.models import vit_b_32
from torch import nn
import torch

class ViTEmotionHeadClassifier(nn.Module):
    def __init__(self,classes,pretrained=None) -> None:        
        super(ViTEmotionHeadClassifier,self).__init__()              

        self.backbone = vit_b_32(image_size=224,weights='DEFAULT')
        self.backbone.heads.head = nn.Linear(768, classes,bias=True)
        


    def forward(self, x):        
        classLogits = self.backbone(x)
        return classLogits

