import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from DAN.networks.dan import DAN
from torch import nn

class DANVA(DAN):

    def convertToVA(self):
        self.fc = nn.Linear(512, 2)
        self.bn = nn.BatchNorm1d(2)