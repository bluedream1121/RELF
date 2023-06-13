from typing import Tuple

from e2cnn.nn import *
from e2cnn.group import *

import torch
import torch.nn as nn
import numpy as np

import datetime

from scipy import stats


class ExpCNN(torch.nn.Module):
    
    def __init__(self, n_channels, n_classes,
                 fix_param: bool = False,
                 deltaorth: bool = False
                 ):
        
        super(ExpCNN, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        self.fix_param = fix_param
        
        layers = []
        
        self.LAYER = 0
        channels = n_channels
        
        # 28 px
        # Convolutional Layer 1
        
        self.LAYER += 1
        # l, channels = self.layer_builder(channels, 16, 7, 0)
        l, channels = self.layer_builder(channels, 16, 7, 1)
        layers += l
        
        # Convolutional Layer 2
        self.LAYER += 1
        l, channels = self.layer_builder(channels, 24, 5, 2, 2)
        layers += l
        
        # 14 px
        # Convolutional Layer 3
        self.LAYER += 1
        l, channels = self.layer_builder(channels, 32, 5, 2)
        layers += l
        
        # Convolutional Layer 4
        self.LAYER += 1
        l, channels = self.layer_builder(channels, 32, 5, 2, 2)
        layers += l
        
        # 7 px
        
        # Convolutional Layer 5
        self.LAYER += 1
        l, channels = self.layer_builder(channels, 48, 5, 2)
        layers += l
        
        # Convolutional Layer 6
        self.LAYER += 1
        l, channels = self.layer_builder(channels, 64, 5, 0, None, True)
        layers += l
        
        # Adaptive Pooling
        mpl = nn.AdaptiveAvgPool2d(1)
        layers.append(mpl)
        
        # 1 px
        
        # c = 64
        
        self.layers = torch.nn.ModuleList(layers)
        
        # Fully Connected
        
        self.fully_net = nn.Sequential(
            nn.Linear(channels, 64),
            nn.BatchNorm1d(64),
            nn.ELU(inplace=True),
            nn.Linear(64, n_classes),
        )
    
        if deltaorth:
            for name, module in self.named_modules():
                if isinstance(module, nn.Conv2d):
                    # delta orthogonal intialization for the Pytorch's 1x1 Conv
                    o, i, w, h = module.weight.shape
                    if o >= i:
                        module.weight.data.fill_(0.)
                        module.weight.data[:, :, w // 2, h // 2] = torch.tensor(
                            stats.ortho_group.rvs(max(i, o))[:o, :i])
                    else:
                        torch.nn.init.xavier_uniform_(module.weight.data, gain=torch.nn.init.calculate_gain('sigmoid'))

        print("MODEL TOPOLOGY:")
        for i, (name, mod) in enumerate(self.named_modules()):
            params = sum([p.numel() for p in mod.parameters() if p.requires_grad])
            if isinstance(mod, nn.Conv2d):
                print(f"\t{i: <3} - {name: <70} | {params: <8} | {mod.weight.shape[1]: <4}- {mod.weight.shape[0]: <4}")
            else:
                print(f"\t{i: <3} - {name: <70} | {params: <8} |")
        tot_param = sum([p.numel() for p in self.parameters() if p.requires_grad])
        print("Total number of parameters:", tot_param)

    def forward(self, input):
        x = input
        for layer in self.layers:
            x = layer(x)
        
        x = self.fully_net(x.reshape(x.shape[0], -1))
        
        return x
    
    def layer_builder(self, channels, C: int, s: int, padding: int = 0, pooling: int = None,
                            orientation_pooling: bool = False):
        
        if self.fix_param and not orientation_pooling and self.LAYER > 1:
        # if self.fix_param and self.LAYER > 1:
            # to keep number of parameters more or less constant when changing groups
            # (more precisely, we try to keep them close to the number of parameters in the original SFCNN)
            t = 1 / 16
            C = int(round(C / np.sqrt(t)))
        
        layers = []
        
        cl = nn.Conv2d(channels, C, s, padding=padding, bias=False)
        layers.append(cl)
        
        bn = nn.BatchNorm2d(C)
        layers.append(bn)
        
        nnl = nn.ELU(inplace=True)
        layers.append(nnl)
        
        if pooling is not None:
            pl = nn.MaxPool2d(pooling)
            layers.append(pl)
        
        return layers, C
