import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import feature_extractor
from .kornia_tools.utils import custom_pyrdown

class KeyNet(nn.Module):
    '''
    Key.Net model definition
    '''
    def __init__(self, keynet_conf):
        super(KeyNet, self).__init__()

        num_filters = keynet_conf['num_filters']
        self.num_levels = keynet_conf['num_levels']
        kernel_size = keynet_conf['kernel_size']
        padding = kernel_size // 2

        self.feature_extractor = feature_extractor()
        self.last_conv = nn.Sequential(nn.Conv2d(in_channels=num_filters*self.num_levels,
                                                 out_channels=1, kernel_size=kernel_size, padding=padding),
                                       nn.ReLU(inplace=True))

    def forward(self, x):
        """
        x - input image
        """
        shape_im = x.shape
        for i in range(self.num_levels):
            if i == 0:
                feats = self.feature_extractor(x)
            else:
                x = custom_pyrdown(x, factor=1.2)
                feats_i = self.feature_extractor(x)
                feats_i = F.interpolate(feats_i, size=(shape_im[2], shape_im[3]), mode='bilinear')
                feats = torch.cat([feats, feats_i], dim=1)

        scores = self.last_conv(feats)
        return scores
