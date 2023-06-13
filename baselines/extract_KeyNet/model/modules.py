import torch
import torch.nn as nn
import kornia


class feature_extractor(nn.Module):
    '''
        It loads both, the handcrafted and learnable blocks
    '''
    def __init__(self):
        super(feature_extractor, self).__init__()

        self.hc_block = handcrafted_block()
        self.lb_block = learnable_block()

    def forward(self, x):
        x_hc = self.hc_block(x)
        x_lb = self.lb_block(x_hc)
        return x_lb


class handcrafted_block(nn.Module):
    '''
        It defines the handcrafted filters within the Key.Net handcrafted block
    '''
    def __init__(self):
        super(handcrafted_block, self).__init__()

    def forward(self, x):

        sobel = kornia.filters.spatial_gradient(x)
        dx, dy = sobel[:, :, 0, :, :], sobel[:, :, 1, :, :]

        sobel_dx = kornia.filters.spatial_gradient(dx)
        dxx, dxy = sobel_dx[:, :, 0, :, :], sobel_dx[:, :, 1, :, :]

        sobel_dy = kornia.filters.spatial_gradient(dy)
        dyy = sobel_dy[:, :, 1, :, :]

        hc_feats = torch.cat([dx, dy, dx**2., dy**2., dx*dy, dxy, dxy**2., dxx, dyy, dxx*dyy], dim=1)

        return hc_feats


class learnable_block(nn.Module):
    '''
        It defines the learnable blocks within the Key.Net
    '''
    def __init__(self, in_channels=10):
        super(learnable_block, self).__init__()

        self.conv0 = conv_blck(in_channels)
        self.conv1 = conv_blck()
        self.conv2 = conv_blck()

    def forward(self, x):
        x = self.conv2(self.conv1(self.conv0(x)))
        return x


def conv_blck(in_channels=8, out_channels=8, kernel_size=5,
              stride=1, padding=2, dilation=1):
    '''
    Default learnable convolutional block.
    '''
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU(inplace=True))


class NonMaxSuppression(torch.nn.Module):
    '''
        NonMaxSuppression class
    '''
    def __init__(self, thr=0.0, nms_size=5):
        nn.Module.__init__(self)
        padding = nms_size // 2
        self.max_filter = torch.nn.MaxPool2d(kernel_size=nms_size, stride=1, padding=padding)
        self.thr = thr

    def forward(self, scores):

        # local maxima
        maxima = (scores == self.max_filter(scores))

        # remove low peaks
        maxima *= (scores > self.thr)

        return maxima.nonzero().t()[2:4]