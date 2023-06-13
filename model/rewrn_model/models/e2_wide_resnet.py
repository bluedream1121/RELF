import os.path
from typing import Tuple

import torch.nn.functional as F

from .utils import *

__all__ = [
    'e2wrn10_8R',    
    'e2wrn16_8R_stl',
    'e2wrn16_8_stl',
    'e2wrn28_10',
    'e2wrn28_7',
    'e2wrn28_10R',
    'e2wrn28_7R',
    'e2wrn28_10_gpool',
]

model_paths = {
    'e2wrn10_8R': os.path.join(STORE_PATH, 'e2wrn10-8R.model'),    
    'e2wrn16_8R_stl': os.path.join(STORE_PATH, 'e2wrn16-8R_stl.model'),
    'e2wrn16_8_stl': os.path.join(STORE_PATH, 'e2wrn16-8_stl.model'),
    'e2wrn28_10': os.path.join(STORE_PATH, 'e2wrn28-10.model'),
    'e2wrn28_7': os.path.join(STORE_PATH, 'e2wrn28-7.model'),
    'e2wrn28_10R': os.path.join(STORE_PATH, 'e2wrn28-10R.model'),
    'e2wrn28_7R': os.path.join(STORE_PATH, 'e2wrn28-7R.model'),
    'e2wrn28_10_gpool': os.path.join(STORE_PATH, 'e2wrn28_10_gpool.model'),
}

########################################################################################################################
# Code adapted from:
# https://github.com/meliketoy/wide-resnet.pytorch/blob/master/networks/wide_resnet.py
# which has the following MIT License:
########################################################################################################################
# MIT License
#
# Copyright (c) 2018 Bumsoo Kim
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
########################################################################################################################
import torch


class WideBasic(nn.EquivariantModule):
    
    def __init__(self,
                 in_fiber: nn.FieldType,
                 inner_fiber: nn.FieldType,
                 dropout_rate, stride=1,
                 out_fiber: nn.FieldType = None,
                 F: float = 1.,
                 sigma: float = 0.45,
                 ):
        super(WideBasic, self).__init__()
        
        if out_fiber is None:
            out_fiber = in_fiber
        
        self.in_type = in_fiber
        inner_class = inner_fiber
        self.out_type = out_fiber
        
        if isinstance(in_fiber.gspace, gspaces.FlipRot2dOnR2):
            rotations = in_fiber.gspace.fibergroup.rotation_order
        elif isinstance(in_fiber.gspace, gspaces.Rot2dOnR2):
            rotations = in_fiber.gspace.fibergroup.order()
        else:
            rotations = 0
        
        if rotations in [0, 2, 4]:
            conv = conv3x3
        else:
            conv = conv5x5
        
        self.bn1 = nn.InnerBatchNorm(self.in_type)
        self.relu1 = nn.ReLU(self.in_type, inplace=True)
        self.conv1 = conv(self.in_type, inner_class, sigma=sigma, F=F, initialize=False)
        
        self.bn2 = nn.InnerBatchNorm(inner_class)
        self.relu2 = nn.ReLU(inner_class, inplace=True)
        
        self.dropout = nn.PointwiseDropout(inner_class, p=dropout_rate)
        
        self.conv2 = conv(inner_class, self.out_type, stride=stride, sigma=sigma, F=F, initialize=False)
        
        self.shortcut = None
        if stride != 1 or self.in_type != self.out_type:
            self.shortcut = conv1x1(self.in_type, self.out_type, stride=stride, bias=False, sigma=sigma, F=F, initialize=False)
            # if rotations in [0, 2, 4]:
            #     self.shortcut = conv1x1(self.in_type, self.out_type, stride=stride, bias=False, sigma=sigma, F=F, initialize=False)
            # else:
            #     self.shortcut = conv3x3(self.in_type, self.out_type, stride=stride, bias=False, sigma=sigma, F=F, initialize=False)
    
    def forward(self, x):
        x_n = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(x_n)))
        out = self.dropout(out)
        out = self.conv2(out)
        
        if self.shortcut is not None:
            out += self.shortcut(x_n)
        else:
            out += x
        
        return out
    
    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        if self.shortcut is not None:
            return self.shortcut.evaluate_output_shape(input_shape)
        else:
            return input_shape


class Wide_ResNet(torch.nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes=100,
                 N: int = 8,
                 r: int = 1,
                 f: bool = True,
                 main_fiber: str = "regular",
                 inner_fiber: str = "regular",
                 F: float = 1.,
                 sigma: float = 0.45,
                 deltaorth: bool = False,
                 fixparams: bool = True,
                 initial_stride: int = 1,
                 conv2triv: bool = True,
                 ):
        super(Wide_ResNet, self).__init__()
        
        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = int((depth - 4) / 6)
        k = widen_factor
        
        print(f'| Wide-Resnet {depth}x{k} ({CHANNELS_CONSTANT * 100}%)')
        
        nStages = [16, 16 * k, 32 * k, 64 * k]
        
        self.distributed = False
        self._fixparams = fixparams
        self.conv2triv = conv2triv
        
        self._layer = 0
        self._N = N
        
        # if the model is [F]lip equivariant
        self._f = f
        
        # level of [R]estriction:
        #   r < 0 : never do restriction, i.e. initial group (either D8 or C8) preserved for the whole network
        #   r = 0 : do restriction before first layer, i.e. initial group doesn't have rotation equivariance (C1 or D1)
        #   r > 0 : restrict after every block, i.e. start with 8 rotations, then restrict to 4 and finally 1
        self._r = r
        
        self._F = F
        self._sigma = sigma
        
        if self._f:
            self.gspace = gspaces.FlipRot2dOnR2(N)
            print("Successfully init reflection+rotation group.\n")

            print(" [Warning] Do not use reflection+rotation.")
            # raise NotImplementedError
        else:
            self.gspace = gspaces.Rot2dOnR2(N)
            print("Successfully init rotation group.\n") 


        if self._r == 0:
            id = (0, 1) if self._f else 1
            self.gspace, _, _ = self.gspace.restrict(id)
        
        r1 = nn.FieldType(self.gspace, [self.gspace.trivial_repr] * 3)
        self.in_type = r1
        
        # r2 = FIBERS[main_fiber](self.gspace, nStages[0], fixparams=self._fixparams)
        r2 = FIBERS[main_fiber](self.gspace, nStages[0], fixparams=True)
        self._in_type = r2
        
        self.conv1 = conv5x5(r1, r2, sigma=sigma, F=F, initialize=False)
        self.layer1 = self._wide_layer(WideBasic, nStages[1], n, dropout_rate, stride=initial_stride,
                                       main_fiber=main_fiber,
                                       inner_fiber=inner_fiber)
        if self._r > 0:
            id = (0, 4) if self._f else 4
            self.restrict1 = self._restrict_layer(id)
        else:
            self.restrict1 = lambda x: x
        
        self.layer2 = self._wide_layer(WideBasic, nStages[2], n, dropout_rate, stride=2,
                                       main_fiber=main_fiber,
                                       inner_fiber=inner_fiber)
        if self._r > 1:
            id = (0, 1) if self._f else 1
            self.restrict2 = self._restrict_layer(id)
        else:
            self.restrict2 = lambda x: x
        
        if self.conv2triv:
            out_fiber = "trivial"
        else:
            out_fiber = None
            
        self.layer3 = self._wide_layer(WideBasic, nStages[3], n, dropout_rate, stride=2,
                                       main_fiber=main_fiber,
                                       inner_fiber=inner_fiber,
                                       out_fiber=out_fiber
                                       )
        
        self.bn1 = nn.InnerBatchNorm(self.layer3.out_type, momentum=0.9)
        if self.conv2triv:
            self.relu = nn.ReLU(self.bn1.out_type, inplace=True)
        else:
            self.mp = nn.GroupPooling(self.layer3.out_type)
            self.relu = nn.ReLU(self.mp.out_type, inplace=True)
            
        self.linear = torch.nn.Linear(self.relu.out_type.size, num_classes)
        
        for name, module in self.named_modules():
            if isinstance(module, nn.R2Conv):
                if deltaorth:
                    init.deltaorthonormal_init(module.weights.data, module.basisexpansion)
                else:
                    init.generalized_he_init(module.weights.data, module.basisexpansion)
            elif isinstance(module, torch.nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, torch.nn.Linear):
                module.bias.data.zero_()
        
        # print("MODEL TOPOLOGY:")
        # for i, (name, mod) in enumerate(self.named_modules()):
        #     print(f"\t{i} - {name}")
        # for i, (name, mod) in enumerate(self.named_modules()):
        #     params = sum([p.numel() for p in mod.parameters() if p.requires_grad])
        #     if isinstance(mod, nn.EquivariantModule) and isinstance(mod.in_type, nn.FieldType) and isinstance(mod.out_type,
        #                                                                                                 nn.FieldType):
        #         print(f"\t{i: <3} - {name: <70} | {params: <8} | {mod.in_type.size: <4}- {mod.out_type.size: <4}")
        #     else:
        #         print(f"\t{i: <3} - {name: <70} | {params: <8} |")
        tot_param = sum([p.numel() for p in self.parameters() if p.requires_grad])
        print("Total number of parameters:", tot_param)


    def _restrict_layer(self, subgroup_id):
        layers = list()
        layers.append(nn.RestrictionModule(self._in_type, subgroup_id))
        layers.append(nn.DisentangleModule(layers[-1].out_type))
        self._in_type = layers[-1].out_type
        self.gspace = self._in_type.gspace
        
        restrict_layer = nn.SequentialModule(*layers)
        return restrict_layer
    
    def _wide_layer(self, block, planes: int, num_blocks: int, dropout_rate: float, stride: int,
                    main_fiber: str = "regular",
                    inner_fiber: str = "regular",
                    out_fiber: str = None,
                    ):
        
        self._layer += 1
        print("start building", self._layer)
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        main_type = FIBERS[main_fiber](self.gspace, planes, fixparams=self._fixparams)
        inner_class = FIBERS[inner_fiber](self.gspace, planes, fixparams=self._fixparams)
        if out_fiber is None:
            out_fiber = main_fiber
        out_type = FIBERS[out_fiber](self.gspace, planes, fixparams=self._fixparams)
        
        for b, stride in enumerate(strides):
            if b == num_blocks - 1:
                out_f = out_type
            else:
                out_f = main_type
            layers.append(
                block(self._in_type, inner_class, dropout_rate, stride, out_fiber=out_f, sigma=self._sigma, F=self._F))
            self._in_type = out_f
        print("built", self._layer)
        return nn.SequentialModule(*layers)
    
    def features(self, x):
        
        x = nn.GeometricTensor(x, self.in_type)
        
        out = self.conv1(x)
        
        x1 = self.layer1(out)
        
        if self.distributed:
            x1.tensor = x1.tensor.cuda(1)
        
        x2 = self.layer2(self.restrict1(x1))
        
        if self.distributed:
            x2.tensor = x2.tensor.cuda(2)
        
        x3 = self.layer3(self.restrict2(x2))
        # out = self.relu(self.mp(self.bn1(out)))
        
        return x1, x2, x3
    
    def forward(self, x):
        
        x = nn.GeometricTensor(x, self.in_type)
        
        out = self.conv1(x)
        out = self.layer1(out)
        
        if self.distributed:
            out.tensor = out.tensor.cuda(1)
        
        out = self.layer2(self.restrict1(out))
        
        if self.distributed:
            out.tensor = out.tensor.cuda(2)
        
        out = self.layer3(self.restrict2(out))
        
        if self.distributed:
            out.tensor = out.tensor.cuda(3)
        
        out = self.bn1(out)
        if not self.conv2triv:
            out = self.mp(out)
        out = self.relu(out)
        
        out = out.tensor
        
        b, c, w, h = out.shape
        out = F.avg_pool2d(out, (w, h))
        
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return out
    
    def distribute(self):
        
        self.distributed = True
        
        self.conv1 = self.conv1.cuda(0)
        self.layer1 = self.layer1.cuda(0)
        
        if self._r:
            self.restrict1 = self.restrict1.cuda(1)
        self.layer2 = self.layer2.cuda(1)
        
        if self._r:
            self.restrict2 = self.restrict2.cuda(2)
        self.layer3 = self.layer3.cuda(2)
        
        self.relu = self.relu.cuda(3)
        self.bn1 = self.bn1.cuda(3)
        # self.mp = self.mp.cuda(3)
        self.avgpool = self.avgpool.cuda(3)
        self.linear = self.linear.cuda(3)
        
        return self


def e2wrn10_8R(pretrained=False, **kwargs):
    """Constructs a Wide ResNet 16-8 model with initial stride of 2 as mentioned here:
    https://github.com/uoguelph-mlrg/Cutout/issues/2

    Args:
        pretrained (bool): If True, returns a model pre-trained on STL10
    """
    model = Wide_ResNet(10, 8, 0, f=False, initial_stride=2, **kwargs)  ## No dropout
    if pretrained:
        model.load_state_dict(model_paths['e2wrn10_8R'])
    return model

def e2wrn16_8R_stl(pretrained=False, **kwargs):
    """Constructs a Wide ResNet 16-8 model with initial stride of 2 as mentioned here:
    https://github.com/uoguelph-mlrg/Cutout/issues/2

    Args:
        pretrained (bool): If True, returns a model pre-trained on STL10
    """
    model = Wide_ResNet(16, 8, 0, f=False, initial_stride=2, **kwargs)
    if pretrained:
        model.load_state_dict(model_paths['e2wrn16_8R_stl'])
    return model

def e2wrn16_8_stl(pretrained=False, **kwargs):
    """Constructs a Wide ResNet 16-8 model with initial stride of 2 as mentioned here:
    https://github.com/uoguelph-mlrg/Cutout/issues/2

    Args:
        pretrained (bool): If True, returns a model pre-trained on STL10
    """
    model = Wide_ResNet(16, 8, 0.3, f=True, initial_stride=2, **kwargs)
    if pretrained:
        model.load_state_dict(model_paths['e2wrn16_8_stl'])
    return model


def e2wrn28_10(pretrained=False, **kwargs):
    """Constructs a Wide ResNet 28-10 model

    Args:
        pretrained (bool): If True, returns a model pre-trained
    """
    model = Wide_ResNet(28, 10, 0.3, f=True, initial_stride=1, **kwargs)
    if pretrained:
        model.load_state_dict(model_paths['e2wrn28_10'])
    return model


def e2wrn28_7(pretrained=False, **kwargs):
    """Constructs a Wide ResNet 28-7 model

    Args:
        pretrained (bool): If True, returns a model pre-trained
    """
    model = Wide_ResNet(28, 7, 0.3, f=True, initial_stride=1, **kwargs)
    if pretrained:
        model.load_state_dict(model_paths['e2wrn28_7'])
    return model



def e2wrn28_10R(pretrained=False, **kwargs):
    """Constructs a Wide ResNet 28-10 model.
    This model is only [R]otation equivariant (no flips equivariance)

    Args:
        pretrained (bool): If True, returns a model pre-trained on Cifar100
    """
    model = Wide_ResNet(28, 10, 0.3, f=False, initial_stride=1, **kwargs)
    if pretrained:
        model.load_state_dict(model_paths['e2wrn28_10R'])
    return model


def e2wrn28_7R(pretrained=False, **kwargs):
    """Constructs a Wide ResNet 28-10 model.
    This model is only [R]otation equivariant (no flips equivariance)

    Args:
        pretrained (bool): If True, returns a model pre-trained on Cifar100
    """
    model = Wide_ResNet(28, 7, 0.3, f=False, initial_stride=1, **kwargs)
    if pretrained:
        model.load_state_dict(model_paths['e2wrn28_7R'])
    return model


def e2wrn28_10_gpool(pretrained=False, **kwargs):
    """Constructs a Wide ResNet 28-10 model with group pooling in the end

    Args:
        pretrained (bool): If True, returns a model pre-trained
    """
    model = Wide_ResNet(28, 10, 0.3, f=True, initial_stride=1, conv2triv=False, **kwargs)
    if pretrained:
        model.load_state_dict(model_paths['e2wrn28_10'])
    return model
