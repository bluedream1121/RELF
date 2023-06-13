from e2cnn.nn import *
from e2cnn.gspaces import *

import torch
import torch.nn as nn
import numpy as np


class E2SFCNN(torch.nn.Module):
    
    def __init__(self, n_channels, n_classes,
                 N=None,
                 restrict: int = -1,
                 fix_param: bool = False,
                 fco: float = 0.8,
                 p_drop_fully: float = 0.3,
                 J: int = 0,
                 sigma: float = 0.6,
                 sgsize: int = None,
                 flip: bool = True,
                 ):
        
        super(E2SFCNN, self).__init__()

        if N is None:
            N = 16
        
        assert N > 1
        
        self.n_channels = n_channels
        self.n_classes = n_classes

        # build the group O(2) or D_N depending on the number N of rotations specified
        if N > 1:
            self.gspace = FlipRot2dOnR2(N)
        elif N == 1:
            self.gspace = Flip2dOnR2()
        else:
            raise ValueError(N)

        # if flips are not required, immediately restrict to the SO(2) or C_N subgroup
        if not flip:
            if N != 1:
                sg = (None, N)
            else:
                sg = 1
            self.gspace, _, _ = self.gspace.restrict(sg)
            
        # id of the subgroup if group restriction is applied through the network
        if sgsize is not None:
            self.sgid = sgsize
        else:
            self.sgid = N
            
        if flip and N != 1:
            self.sgid = (None, self.sgid)

        if fco is not None and fco > 0.:
            fco *= np.pi
        frequencies_cutoff = fco
        
        eq_layers = []

        LAYER = 0

        def build_layers(r1: FieldType, C: int, s: int, padding: int = 0, pooling: int = None, orientantion_pooling: bool = False):
            
            gspace = r1.gspace
            
            if fix_param:
                # to keep number of parameters more or less constant when changing groups
                # (more precisely, we try to keep them close to the number of parameters in the original SFCNN)
                C /= np.sqrt(gspace.fibergroup.order()/16)
            C = int(C)
            
            layers = []

            r2 = FieldType(gspace, [gspace.representations['regular']] * C)
    
            cl = R2Conv(r1, r2, s,
                        frequencies_cutoff=frequencies_cutoff,
                        padding=padding,
                        sigma=sigma,
                        maximum_offset=J)
            layers.append(cl)

            if restrict == LAYER:
                layers.append(RestrictionModule(layers[-1].out_type, self.sgid))
                layers.append(DisentangleModule(layers[-1].out_type))
            
            bn = InnerBatchNorm(layers[-1].out_type)
            layers.append(bn)
            
            if orientantion_pooling:
                pl = GroupPooling(layers[-1].out_type)
                layers.append(pl)

            if pooling is not None:
                pl = PointwiseMaxPool(layers[-1].out_type, pooling)
                layers.append(pl)
                
            nnl = ELU(layers[-1].out_type, inplace=True)
            layers.append(nnl)

            return layers
        
        if restrict == LAYER:
            self.gspace, _, _ = self.gspace.restrict(self.sgid)
        
        r1 = FieldType(self.gspace, [self.gspace.trivial_repr] * n_channels)
        
        # 28 px
        # Convolutional Layer 1
        
        LAYER += 1
        #TODO no padding here? with such a large filter???
        eq_layers += build_layers(r1, 24, 9, 0, None)
        
        # Convolutional Layer 2
        LAYER += 1
        eq_layers += build_layers(eq_layers[-1].out_type, 32, 7, 3, 2)

        # TODO this number is right iff we used padding in the first layer!
        # 14 px
        # Convolutional Layer 3
        LAYER += 1
        eq_layers += build_layers(eq_layers[-1].out_type, 36, 7, 3, None)

        # Convolutional Layer 4
        LAYER += 1
        eq_layers += build_layers(eq_layers[-1].out_type, 36, 7, 3, 2)

        # 7 px

        # Convolutional Layer 5
        LAYER += 1
        eq_layers += build_layers(eq_layers[-1].out_type, 64, 7, 3)

        # Convolutional Layer 6
        LAYER += 1
        eq_layers += build_layers(eq_layers[-1].out_type, 96, 5, 0, None, True)

        # Adaptive Pooling
        mpl = PointwiseAdaptiveMaxPool(eq_layers[-1].out_type, 1)
        eq_layers.append(mpl)

        # 1 px
        
        # c = 96
        c = eq_layers[-1].out_type.size
        
        self.in_repr = eq_layers[0].in_type
        self.eq_layers = torch.nn.ModuleList(eq_layers)

        # Fully Connected

        self.fully_net = nn.Sequential(
            nn.Dropout(p=p_drop_fully),
            nn.Linear(c, 96),
            nn.BatchNorm1d(96),
            nn.ELU(inplace=True),
    
            nn.Dropout(p=p_drop_fully),
            nn.Linear(96, 96),
            nn.BatchNorm1d(96),
            nn.ELU(inplace=True),
            
            nn.Dropout(p=p_drop_fully),
            nn.Linear(96, n_classes),
        )
    
    def forward(self, input):
        x = GeometricTensor(input, self.in_repr)
        
        for layer in self.eq_layers:
            x = layer(x)
        
        x = self.fully_net(x.tensor.reshape(x.tensor.shape[0], -1))
        
        return x

    def features(self, input):
        x = GeometricTensor(input, self.in_repr)

        # layer_taken = [2, 4, 6, 11, 13, 16, 18, 20] ## all changed (channel expansion, resolution drop)
        layer_taken = [4, 11, 18]   ## before resolution drop

        out = []
        for idx, layer in enumerate(self.eq_layers):
            x = layer(x)
            if idx in layer_taken:
                out.append(x)

        return out[0], out[1], out[2]

