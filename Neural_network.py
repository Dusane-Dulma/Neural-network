
##first pytorch model


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm

class LinearRegressionModel(nn.Module): # nn.module foundation libary
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,
                                                requires_grad=True,  # set random weight values
                                                dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, 
                                             requires_grad=True,    #set random bias values
                                             dtype=torch.float))
        def forward(self, x:torch.Tensor) -> torch.Tensor:
            return self.weights * x + self.bias # linear progression formula
