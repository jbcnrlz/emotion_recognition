from torchvision import models
from torch import nn
from attentionModule import *
class ResnetEmotionHead(nn.Module):
    def __init__(self,classes,resnetModel,pretrained=False,vaGuidance=False) -> None:        
        super(ResnetEmotionHead,self).__init__()
        if (resnetModel == 'resnet18'):
            self.innerResnetModel = models.resnet18(pretrained=pretrained)
        elif (resnetModel == 'resnet50'):
            self.innerResnetModel = models.resnet50(pretrained=pretrained)

        modules=list(self.innerResnetModel.children())[:-1]
        self.innerResnetModel=nn.Sequential(*modules)
        self.vaModule = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 2,bias=False)
        )
        '''
        if vaGuidance:
            self.softmax = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(512, classes + 2,bias=False)
            )        
        else:
            self.softmax = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(512, classes,bias=False)
            )        
        '''
    def forward(self, x):
        feats = self.innerResnetModel(x).view((-1,512))
        va = self.vaModule(feats)
        return feats, va

class ResnetEmotionHeadClassifierAttention(nn.Module):
    def __init__(self,classes,resnetModel,pretrained=False) -> None:        
        super(ResnetEmotionHeadClassifierAttention,self).__init__()
        if (resnetModel == 'resnet18'):
            self.innerResnetModel = models.resnet18(pretrained=pretrained)
        elif (resnetModel == 'resnet50'):
            self.innerResnetModel = models.resnet50(pretrained=pretrained)

        modules=list(self.innerResnetModel.children())[:-1]
        self.innerResnetModel=nn.Sequential(*modules)

        self.selfAttentionMoule = nn.MultiheadAttention(512,3)

        self.softmax = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, classes,bias=False),
            nn.Softmax(dim=1)
        )
        '''
        if vaGuidance:
            self.softmax = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(512, classes + 2,bias=False)
            )        
        else:
            self.softmax = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(512, classes,bias=False)
            )        
        '''
    def forward(self, x):
        feats = self.innerResnetModel(x).view((-1,512))
        va = self.softmax(feats)
        return feats, va

class ResnetEmotionHeadClassifier(nn.Module):
    def __init__(self,classes,resnetModel,pretrained=False,vaGuidance=False) -> None:        
        super(ResnetEmotionHeadClassifier,self).__init__()
        if (resnetModel == 'resnet18'):
            self.innerResnetModel = models.resnet18(pretrained=pretrained)
        elif (resnetModel == 'resnet50'):
            self.innerResnetModel = models.resnet50(pretrained=pretrained)

        modules=list(self.innerResnetModel.children())[:-1]
        self.innerResnetModel=nn.Sequential(*modules)
        self.softmax = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, classes,bias=False),
            nn.Softmax(dim=1)
        )
        '''
        if vaGuidance:
            self.softmax = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(512, classes + 2,bias=False)
            )        
        else:
            self.softmax = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(512, classes,bias=False)
            )        
        '''
    def forward(self, x):
        feats = self.innerResnetModel(x).view((-1,512))
        va = self.softmax(feats)
        return feats, va