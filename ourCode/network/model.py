from torch import nn
import torch.nn.functional as F, torch

class TimeSeriesLearning(nn.Module):
    def __init__(self,classes,in_channels=4):
        super(TimeSeriesLearning,self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.linearFeatures = nn.Sequential(
            nn.Linear(179776, 2048)
        )

        self.lstm = nn.LSTM(2048,1024)

        self.featureTimeLearning = nn.Linear(1024, classes)

    def forward(self,x):
        outfeatures = self.features(x)
        outfeatures = self.linearFeatures(outfeatures.view(len(x), -1))
        lstmout, _ = self.lstm(outfeatures.view(len(outfeatures),1,-1))
        tag_space = self.featureTimeLearning(lstmout.view(len(lstmout), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores, tag_space

class TimeSeriesLearningSkip(nn.Module):
    def __init__(self,in_channels=3,sequenceSize=40):
        super(TimeSeriesLearningSkip,self).__init__()

        self.sequenceSize = sequenceSize
        self.inChannels = in_channels

        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.b2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.linearFeatures = nn.Sequential(
            nn.Linear(2304, 1024)
        )

        self.lstmending = nn.LSTM(1024,1024)
        self.lstmmiddle = nn.LSTM(2304, 1024)
        self.lstmtop = nn.LSTM(30976, 1024)

        self.featureTimeLearning = nn.Linear(3072, 2,bias=False)

    def forward(self,x):
        if self.sequenceSize > 0:
            x = x.reshape((self.sequenceSize*x.shape[0],self.inChannels,x.shape[-2],x.shape[-1]))
        outfeatures1 = self.b1(x)
        outfeatures2 = self.b2(outfeatures1)
        outfeatures = outfeatures2.view(len(outfeatures2),1,-1)
        outfeatures = self.linearFeatures(outfeatures)

        outfeatures1 = outfeatures1.view(len(outfeatures1),1,-1)
        outfeatures2 = outfeatures2.view(len(outfeatures2), 1, -1)

        outfeatures1, _ = self.lstmtop(outfeatures1)
        outfeatures2, _ = self.lstmmiddle(outfeatures2)
        outfeatures, _ = self.lstmending(outfeatures.view(len(outfeatures),1,-1))

        catFeature = torch.cat((outfeatures1,outfeatures2,outfeatures),2)
        return self.featureTimeLearning(catFeature)