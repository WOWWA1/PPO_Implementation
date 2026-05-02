import torch 
import numpy as np
import torch.nn as nn
class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,2)
        )
    
    def forward(self,input):
        output = self.model(input)
        return output

class ValueNet(nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )
    
    def forward(self,input):
        output = self.model(input)
        return output



    




     