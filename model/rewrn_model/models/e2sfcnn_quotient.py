from e2cnn.nn import *
from e2cnn.gspaces import *

import torch
import torch.nn as nn
import numpy as np

class E2SFCNN_QUOT(torch.nn.Module):
    
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
        
        super(E2SFCNN_QUOT, self).__init__()

        if N is None:
            N = 16
        
        assert N > 1
        
        self.N = N
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
        self.frequencies_cutoff = fco
        self.sigma = sigma
        self.J = J
        self.fix_param = fix_param
        self.restrict = restrict
        
        eq_layers = []

        self.LAYER = 0
        
        if restrict == self.LAYER:
            gspace, _, _ = self.gspace.restrict(self.sgid)
        
        r1 = FieldType(gspace, [gspace.trivial_repr] * n_channels)
        
        # 28 px
        # Convolutional Layer 1

        self.LAYER += 1
        eq_layers += self.build_layer_quotient(r1, 24, 9, 0, None)
        
        # Convolutional Layer 2
        self.LAYER += 1
        eq_layers += self.build_layer_quotient(eq_layers[-1].out_type, 32, 7, 3, 2)
        
        # 14 px
        # Convolutional Layer 3
        self.LAYER += 1
        eq_layers += self.build_layer_quotient(eq_layers[-1].out_type, 36, 7, 3, None)

        # Convolutional Layer 4
        self.LAYER += 1
        eq_layers += self.build_layer_quotient(eq_layers[-1].out_type, 36, 7, 3, 2)

        # 7 px
        # Convolutional Layer 5
        self.LAYER += 1
        eq_layers += self.build_layer_quotient(eq_layers[-1].out_type, 64, 7, 3)

        # Convolutional Layer 6
        self.LAYER += 1
        eq_layers += self.build_layer_quotient(eq_layers[-1].out_type, 96, 5, 0, None, True)

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
    
    def forward(self, input: torch.tensor):
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

    def build_quotient_feature_type(self, gspace):
    
        assert gspace.fibergroup.order() > 0
        if isinstance(gspace, FlipRot2dOnR2):
            n = int(gspace.fibergroup.order() / 2)
            repr = [gspace.regular_repr] * 5
            for i in [0, round(n / 4), round(n / 2)]:
                repr += [gspace.quotient_repr((int(i), 1))] * 2
            repr += [gspace.quotient_repr((None, int(n / 2)))] * 2
            repr += [gspace.trivial_repr] * int(gspace.fibergroup.order() / 4)
        elif isinstance(gspace, Rot2dOnR2):
            n = gspace.fibergroup.order()
            repr = [gspace.regular_repr] * 5
            repr += [gspace.quotient_repr(int(round(n / 2)))] * 2
            repr += [gspace.quotient_repr(int(round(n / 4)))] * 2
            repr += [gspace.trivial_repr] * int(gspace.fibergroup.order() / 4)
        else:
            repr = [gspace.regular_repr]
        
        return repr

    def build_layer_quotient(self, r1: FieldType, C: int, s: int, padding: int = 0, pooling: int = None,
                             orientation_pooling: bool = False):
    
        gspace = r1.gspace
    
        if self.fix_param and not orientation_pooling and self.LAYER > 1:
            # to keep number of parameters more or less constant when changing groups
            # (more precisely, we try to keep them close to the number of parameters in the original SFCNN)
            t = gspace.fibergroup.order() / 16
            C = C / np.sqrt(t)
    
        layers = []
    
        repr = self.build_quotient_feature_type(gspace)
        
        C /= sum([r.size for r in repr]) / gspace.fibergroup.order()
    
        C = int(round(C))
    
        r2 = FieldType(gspace, repr * C).sorted()
    
        cl = R2Conv(r1, r2, s,
                    frequencies_cutoff=self.frequencies_cutoff,
                    padding=padding,
                    sigma=self.sigma,
                    maximum_offset=self.J)
        layers.append(cl)
    
        if self.restrict == self.LAYER:
            layers.append(RestrictionModule(layers[-1].out_type, self.sgid))
            layers.append(DisentangleModule(layers[-1].out_type))
    
        if orientation_pooling:
            pl = GroupPooling(layers[-1].out_type)
            layers.append(pl)
    
        bn = InnerBatchNorm(layers[-1].out_type)
        layers.append(bn)
        nnl = ELU(layers[-1].out_type, inplace=True)
        layers.append(nnl)
    
        if pooling is not None:
            pl = PointwiseMaxPool(layers[-1].out_type, pooling)
            layers.append(pl)
    
        return layers

