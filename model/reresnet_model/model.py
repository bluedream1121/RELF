from .models.backbones.re_resnet import ReResNet
from e2cnn.nn import *

import torch
import torch.nn.functional as F


class ReResNet_wrapper(torch.nn.Module):
    def __init__(self, G=8, depth=18, channels=64, strides="1,2,2,2", dilations="1,1,1,1"):
        super(ReResNet_wrapper, self).__init__()
        
        strides = [int(i) for i in strides.split(',')]
        dilations = [int(i) for i in dilations.split(',')]

        model = ReResNet(depth=depth,                  
                 in_channels=3,
                 stem_channels=channels,
                 base_channels=channels,
                 strides=strides,
                 dilations=dilations)  ## depth is the number of resnet.

        model.eval()
        
        self.model = model
        self.layer_idx = 4    ## forwarding layer index. default=4
        self.G = G

        self.in_type = model.in_type
        self.out_type = getattr(model, model.res_layers[self.layer_idx-1]).out_type


    def get_in_type(self):
        return self.in_type

    def forward(self, x):
        return self.multiscale_forward(x)
        
    def multiscale_forward(self, x):
        model = self.model
        feat = []
        x = model.conv1(x)
        x = model.norm1(x)
        x = model.relu(x)

        feat.append(x) ## 1/2 size
        B, _, H, W = x.shape ## first feature shape
        
        for i, layer_name in enumerate(model.res_layers[:self.layer_idx]): ## remove 1/32 size feature
            res_layer = getattr(model, layer_name)
            x = res_layer(x)
            feat.append(x) ## 1/4, 1/8, 1/16 sizes

        ## output feature aggregation last feature
        output = feat[-1]
        _, CG, _, _ = output.shape ## final channel size is 16 times of first feature.
        output = F.interpolate(output.tensor, size=(H, W), mode='bilinear', align_corners=True).reshape(B, -1, self.G, H, W)

        for ff in reversed(feat[:-1]):
            ff = F.interpolate(ff.tensor, size=(H, W), mode='bilinear', align_corners=True) ## spatial size align.
            ff = ff.reshape(B, -1, self.G, H, W)
            output = torch.cat([output, ff], dim=1)

        output = output.reshape(B, -1, H, W)

        return output

    def singlescale_forward(self, x):
        model = self.model
        # x = GeometricTensor(x, model.in_type)
        x = model.conv1(x)
        x = model.norm1(x)
        x = model.relu(x)
        # x = model.maxpool(x)

        for i, layer_name in enumerate(model.res_layers[:self.layer_idx]):
            res_layer = getattr(model, layer_name)
            x = res_layer(x)

        return x.tensor


