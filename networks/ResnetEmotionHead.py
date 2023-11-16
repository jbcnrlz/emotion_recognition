from torchvision import models
from torch import nn
from networks.attentionModule import FeatureEnhanceNoCross
import torch
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
    def __init__(self,classes,resnetModel,pretrained=None) -> None:        
        super(ResnetEmotionHeadClassifierAttention,self).__init__()
        if (resnetModel == 'resnet18'):
            self.innerResnetModel = models.resnet18(weights=None)
            if (pretrained is not None):
                checkpoint = torch.load('/home/joaocardia/Projects/emotion_recognition/DAN/models/resnet18_msceleb.pth')
                self.innerResnetModel.load_state_dict(checkpoint['state_dict'],strict=True)
        elif (resnetModel == 'resnet50'):
            self.innerResnetModel = models.resnet50(pretrained=pretrained)

        modules=list(self.innerResnetModel.children())
        beforAttention = modules[:-2]
        self.innerResnetModel=nn.Sequential(*beforAttention)

        self.selfAttentionMoule = FeatureEnhanceNoCross(512)

        self.afterAttention = nn.Sequential(*modules[-2:-1])

        self.softmax = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, classes,bias=False),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        feats = self.innerResnetModel(x)
        feats = self.selfAttentionMoule(feats)
        feats = self.afterAttention(feats)
        feats = feats.view((-1,256))
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