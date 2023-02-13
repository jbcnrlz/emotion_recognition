from torchvision import models
from torch import nn

class ResnetEmotionHead(nn.Module):
    def __init__(self,classes,resnetModel,pretrained=False) -> None:        
        super(ResnetEmotionHead,self).__init__()
        if (resnetModel == 'resnet18'):
            self.innerResnetModel = models.resnet18(pretrained=pretrained)
        elif (resnetModel == 'resnet50'):
            self.innerResnetModel = models.resnet50(pretrained=pretrained)

        modules=list(self.innerResnetModel.children())[:-1]
        self.innerResnetModel=nn.Sequential(*modules)

        self.softmax = nn.Linear(512, classes)

    def forward(self, x):
        feats = self.innerResnetModel(x).view((-1,512))
        classification = self.softmax(feats)
        return feats, classification