from torchvision import models
from torch import nn

class ResnetEmotionHead(nn.Module):
    def __init__(self,classes,resnetModel,pretrained=False) -> None:        
        super(ResnetEmotionHead,self).__init__()
        if (resnetModel == 'resnet18'):
            self.innerResnetModel = models.resnet18(pretrained=pretrained)
        elif (resnetModel == 'resnet50'):
            self.innerResnetModel = models.resnet50(pretrained=pretrained)

        self.softmax = nn.Linear(1000, classes)

    def forward(self, x):
        feats = self.innerResnetModel(x)
        classification = self.softmax(feats)
        return feats, classification