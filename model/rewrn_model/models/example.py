from e2cnn.nn import *
from e2cnn.group import *

import torch
import torch.nn as nn
import numpy as np


class E2SFCNN(torch.nn.Module):
    
    def __init__(self,
                 n_channels: int,
                 n_classes: int,
                 N: int = 16,
                 restrict: int = -1,
                 ):
        r"""
        
        Args:
            n_channels: number of channels in the input
            n_classes: number of output classes
            N: number of rotations of the equivariance group
            restrict: number of initial convolutional layers which are also flip equivariant.
                      After these layers, we restrict to only rotation equivariance.
                      By default (-1) the restriciton is never done so the model is flip and rotations equivariant.
                      If set to 0 the model is only rotation equivariant from the beginning
        """
        
        super(E2SFCNN, self).__init__()
        
        assert N > 1
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.N = N
        self.restrict = restrict
        
        gc = FlipRot2dOnR2(N)
        
        self.gc = gc
        
        self.LAYER = 0

        if self.restrict == self.LAYER:
            gc, _, _ = self.gc.restrict((None, N))
        
        r1 = FieldType(gc, [gc.trivial_repr] * n_channels)
        
        eq_layers = []
        
        # 28 px
        # Convolutional Layer 1

        self.LAYER += 1
        eq_layers += self.build_layers(r1, 24, 9, 0, None)
        
        # Convolutional Layer 2
        self.LAYER += 1
        eq_layers += self.build_layers(eq_layers[-1].out_type, 32, 7, 3, 2)
        
        # 14 px
        # Convolutional Layer 3
        self.LAYER += 1
        eq_layers += self.build_layers(eq_layers[-1].out_type, 36, 7, 3, None)

        # Convolutional Layer 4
        self.LAYER += 1
        eq_layers += self.build_layers(eq_layers[-1].out_type, 36, 7, 3, 2)

        # 7 px

        # Convolutional Layer 5
        self.LAYER += 1
        eq_layers += self.build_layers(eq_layers[-1].out_type, 64, 7, 3)

        # Convolutional Layer 6
        self.LAYER += 1
        eq_layers += self.build_layers(eq_layers[-1].out_type, 96, 5, 0, None, True)

        # Adaptive Pooling
        mpl = PointwiseAdaptiveMaxPool(eq_layers[-1].out_type, 1)
        eq_layers.append(mpl)

        # 1 px
        
        # c = 96
        c = eq_layers[-1].out_type.size
        
        self.in_repr = eq_layers[0].in_type
        self.eq_layers = SequentialModule(*eq_layers)

        # Fully Connected

        self.fully_net = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(c, 96),
            nn.BatchNorm1d(96),
            nn.ELU(inplace=True),
    
            nn.Dropout(p=0.3),
            nn.Linear(96, 96),
            nn.BatchNorm1d(96),
            nn.ELU(inplace=True),
            
            nn.Dropout(p=0.3),
            nn.Linear(96, n_classes),
        )
    
    def forward(self, input):
        x = GeometricTensor(input, self.in_repr)
        
        features = self.eq_layers(x)

        features = features.tensor.reshape(x.tensor.shape[0], -1)
        
        out = self.fully_net(features)
        
        return out
        
    def build_layers(self,
                     r1: FieldType,
                     C: int,
                     s: int,
                     padding: int = 0,
                     pooling: int = None,
                     orientation_pooling: bool = False,
        ):

        gc = r1.isometries

        layers = []

        r2 = FieldType(gc, [gc.representations['regular']] * C)

        cl = R2Conv(r1,
                    r2,
                    s,
                    padding=padding
        )
        layers.append(cl)

        if self.restrict == self.LAYER:
            layers.append(RestrictionModule(layers[-1].out_type, (None, self.N)))
            layers.append(DisentangleModule(layers[-1].out_type))

        bn = InnerBatchNorm(layers[-1].out_type)
        layers.append(bn)

        if orientation_pooling:
            pl = CapsulePool(layers[-1].out_type)
            layers.append(pl)

        if pooling is not None:
            pl = PointwiseMaxPool(layers[-1].out_type, pooling)
            layers.append(pl)

        nnl = ReLU(layers[-1].out_type, inplace=True)
        layers.append(nnl)

        return layers

