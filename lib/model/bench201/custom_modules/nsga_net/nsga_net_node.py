import torch
from torch.nn import Module, Linear, Sequential
from torch import nn

class ResidualNode(Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size):
        super(ResidualNode, self).__init__()

        self.model = Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, 
                                          stride=1, padding=(kernel_size-1)/2),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(inplace=True))

    def forward(self, x):
        return self.model(x)

class PreactResidualNode(Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size):
        super(PreactResidualNode, self).__init__()

        self.model = Sequential(nn.BatchNorm2d(in_channels),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(in_channels, 
                                          out_channels, 
                                          kernel_size,
                                          stride=1, 
                                          padding=(kernel_size-1)/2))
    
    def forward(self, x):
        return self.model(x)
        
class DenseNode(Module):
    k = 4   # Growth rate multiplier fixed at 4

    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size):
        super(DenseNode, self).__init__()

        self.t = out_channels
        self.model = Sequential(nn.BatchNorm2d(in_channels),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(in_channels, 
                                          out_channels=self.t*self.k, 
                                          kernel_size=1),
                                nn.BatchNorm2d(self.t*self.k),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(in_channels=self.t*self.k,
                                          out_channels=self.t,
                                          kernel_size=kernel_size,
                                          stride=1,
                                          padding=1))

    
    def forward(self, x):
        return self.model(x)