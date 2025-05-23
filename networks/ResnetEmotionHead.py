from torchvision import models
from torch import nn
from networks.attentionModule import FeatureEnhanceNoCross, FeatureEnhanceWindow
from torch.nn import functional as F
import torch, os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

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

class ResnetEmotionHeadDANImplementation(nn.Module):
    def __init__(self,classes,pretrained=None) -> None:        
        super(ResnetEmotionHeadDANImplementation,self).__init__()
        self.innerResnetModel = models.resnet18(weights=None)
        if (pretrained is not None):
            checkpoint = torch.load('/home/joaocardia/Projects/emotion_recognition/DAN/models/resnet18_msceleb.pth')
            self.innerResnetModel.load_state_dict(checkpoint['state_dict'],strict=True)
        
        modules=list(self.innerResnetModel.children())
        beforAttention = modules[:-2]
        self.innerResnetModel=nn.Sequential(*beforAttention)

        self.selfAttentionMoule = CrossAttentionHead()

        self.softmax = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, classes,bias=False)
        )

        self.batchNorm = nn.BatchNorm1d(classes)

    def forward(self, x):        
        feats = self.innerResnetModel(x)
        att = self.selfAttentionMoule(feats)
        va = self.softmax(att)
        va = self.batchNorm(va)
        return feats, va, att

class SmallClassified(nn.Module):
    def __init__(self,classes) -> None:
        super(SmallClassified,self).__init__()
        self.softmax = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, classes,bias=False)
        )

    def forward(self, x):
        return self.softmax(x)

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
        #self.batchNorm = nn.BatchNorm2d(512)
        self.selfAttentionMoule = FeatureEnhanceWindow(512,512)
        #self.selfAttentionMoule = FeatureEnhanceNoCross(512,512)
        #self.selfAttentionMoule = FeatureEnhanceRGB(512,512)

        self.afterAttention = nn.Sequential(*modules[-2:-1])

        self.softmax = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, classes,bias=False),
            nn.BatchNorm1d(num_features=512),
            nn.Sigmoid()
        )

    def warmUP(self):
        self.innerResnetModel.requires_grad = False
        self.afterAttention.requires_grad = False
        self.softmax.requires_grad = False

    def afterWarmUp(self):
        self.innerResnetModel.requires_grad = True
        self.afterAttention.requires_grad = True
        self.softmax.requires_grad = True

    def forward(self, x):        
        feats = self.innerResnetModel(x)
        att = self.selfAttentionMoule(feats)
        feats = self.afterAttention(att)
        #att = None
        #feats = self.afterAttention(feats)
        feats = feats.view((-1,512))
        va = self.softmax(feats)
        return feats, va, att

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