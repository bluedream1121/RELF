
from e2cnn import nn
from e2cnn import group
from e2cnn import gspaces
from e2cnn.nn import init

import math
import numpy as np

STORE_PATH = "./models/stored/"

CHANNELS_CONSTANT = 1


def _get_fco(fco):
    if fco > 0.:
        fco *= np.pi
    return fco


def conv7x7(in_type: nn.FieldType, out_type: nn.FieldType, stride=1, padding=3, dilation=1, bias=False, sigma=None, F=1., initialize=True):
    """7x7 convolution with padding"""
    fco = _get_fco(F)
    return nn.R2Conv(in_type, out_type, 7,
                     stride=stride,
                     padding=padding,
                     dilation=dilation,
                     bias=bias,
                     sigma=sigma,
                     frequencies_cutoff=fco,
                     initialize=initialize
                     )


def conv5x5(in_type: nn.FieldType, out_type: nn.FieldType, stride=1, padding=2, dilation=1, bias=False, sigma=None, F=1., initialize=True):
    """5x5 convolution with padding"""
    fco = _get_fco(F)
    return nn.R2Conv(in_type, out_type, 5,
                     stride=stride,
                     padding=padding,
                     dilation=dilation,
                     bias=bias,
                     sigma=sigma,
                     frequencies_cutoff=fco,
                     initialize=initialize
                     )


def conv3x3(in_type: nn.FieldType, out_type: nn.FieldType, padding=1, stride=1, dilation=1, bias=False, sigma=None, F=1., initialize=True):
    """3x3 convolution with padding"""
    fco = _get_fco(F)
    return nn.R2Conv(in_type, out_type, 3,
                     stride=stride,
                     padding=padding,
                     dilation=dilation,
                     bias=bias,
                     sigma=sigma,
                     frequencies_cutoff=fco,
                     initialize=initialize
                     )


def conv1x1(in_type: nn.FieldType, out_type: nn.FieldType, padding=0, stride=1, dilation=1, bias=False, sigma=None, F=1., initialize=True):
    """1x1 convolution"""
    fco = _get_fco(F)
    return nn.R2Conv(in_type, out_type, 1,
                     stride=stride,
                     padding=padding,
                     dilation=dilation,
                     bias=bias,
                     sigma=sigma,
                     frequencies_cutoff=fco,
                     initialize=initialize
                     )


def regular_fiber(gspace: gspaces.GeneralOnR2, planes: int, fixparams: bool = True):
    """ build a regular fiber with the specified number of channels"""
    assert gspace.fibergroup.order() > 0
    N = gspace.fibergroup.order()
    planes = planes / N
    if fixparams:
        planes *= math.sqrt(N * CHANNELS_CONSTANT)
    planes = int(planes)
    
    return nn.FieldType(gspace, [gspace.regular_repr] * planes)


def quotient_fiber(gspace: gspaces.GeneralOnR2, planes: int, fixparams: bool = True):
    """ build a quotient fiber with the specified number of channels"""
    N = gspace.fibergroup.order()
    assert N > 0
    if isinstance(gspace, gspaces.FlipRot2dOnR2):
        n = N/2
        subgroups = []
        for axis in [0, round(n/4), round(n/2)]:
            subgroups.append((int(axis), 1))
    elif isinstance(gspace, gspaces.Rot2dOnR2):
        assert N % 4 == 0
        # subgroups = [int(round(N/2)), int(round(N/4))]
        subgroups = [2, 4]
    elif isinstance(gspace, gspaces.Flip2dOnR2):
        subgroups = [2]
    else:
        raise ValueError(f"Space {gspace} not supported")
    
    rs = [gspace.quotient_repr(subgroup) for subgroup in subgroups]
    size = sum([r.size for r in rs])
    planes = planes / size
    if fixparams:
        planes *= math.sqrt(N * CHANNELS_CONSTANT)
    planes = int(planes)
    return nn.FieldType(gspace, rs * planes).sorted()


def trivial_fiber(gspace: gspaces.GeneralOnR2, planes: int, fixparams: bool = True):
    """ build a trivial fiber with the specified number of channels"""

    if fixparams:
        planes *= math.sqrt(gspace.fibergroup.order() * CHANNELS_CONSTANT)
    planes = int(planes)
    return nn.FieldType(gspace, [gspace.trivial_repr] * planes)


def mixed_fiber(gspace: gspaces.GeneralOnR2, planes: int, ratio: float, fixparams: bool = True):

    N = gspace.fibergroup.order()
    assert N > 0
    if isinstance(gspace, gspaces.FlipRot2dOnR2):
        subgroup = (0, 1)
    elif isinstance(gspace, gspaces.Flip2dOnR2):
        subgroup = 1
    else:
        raise ValueError(f"Space {gspace} not supported")
    
    qr = gspace.quotient_repr(subgroup)
    rr = gspace.regular_repr
    
    planes = planes / rr.size
    
    if fixparams:
        planes *= math.sqrt(N * CHANNELS_CONSTANT)
    
    r_planes = int(planes * ratio)
    q_planes = int(2*planes * (1-ratio))
    
    return nn.FieldType(gspace, [rr] * r_planes + [qr] * q_planes).sorted()


def mixed1_fiber(gspace: gspaces.GeneralOnR2, planes: int, fixparams: bool = True):
    return mixed_fiber(gspace=gspace, planes=planes, ratio=0.5, fixparams=fixparams)


def mixed2_fiber(gspace: gspaces.GeneralOnR2, planes: int, fixparams: bool = True):
    return mixed_fiber(gspace=gspace, planes=planes, ratio=0.25, fixparams=fixparams)


FIBERS = {
    "trivial": trivial_fiber,
    "quotient": quotient_fiber,
    "regular": regular_fiber,
    "mixed1": mixed1_fiber,
    "mixed2": mixed2_fiber,
}

