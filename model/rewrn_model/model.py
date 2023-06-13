from .models import *
from e2cnn.nn import *

import easydict, torch
import torch.nn.functional as F

class E2CNN_wrapper(torch.nn.Module):
    def __init__(self, args):
        super(E2CNN_wrapper, self).__init__()

        config = {'model': args.model, 'flip': False, 'sgsize': None, 'fixparams': False, 'N': args.num_group,
                'F': 0.8, 'sigma': 0.6, 'J': 0, 'restrict': -1, 'deltaorth': False, 'antialias': 0.0}
        config = easydict.EasyDict(config)

        model = build_model(config, 3, 10) ## input channel, output dummy  num of classes.
        model.eval()

        self.in_type = model.in_repr if 'E2SFCNN' in args.model else model.in_type
        self.model = model
        self.G = args.num_group

    def get_in_type(self):
        return self.in_type
    
    def forward(self, x):
        return self.multiscale_forward(x)

    def multiscale_forward(self, x):
        a, b, c = self.model.features(x.tensor)  

        B, _, H, W = a.shape ## first feature shape
        _, CG, _, _ = c.shape ## last feature shape
        output = F.interpolate(c.tensor, size=(H, W), mode='bilinear', align_corners=True).reshape(B, -1, self.G, H, W)

        for ff in [b, a]:
            ff = F.interpolate(ff.tensor, size=(H, W), mode='bilinear', align_corners=True) ## spatial size align.
            ff = ff.reshape(B, -1, self.G, H, W)
            output = torch.cat([output, ff], dim=1)

        output = output.reshape(B, -1, H, W)

        return output
    
    def singlescale_forward(self, x):
        raise NotImplementedError

def build_model(config, n_inputs, n_outputs):
    # SFCNN VARIANTS
    if config.model == 'E2SFCNN':
        model = E2SFCNN(n_inputs, n_outputs, restrict=config.restrict, N=config.N, fco=config.F, J=config.J,
                        sigma=config.sigma, fix_param=config.fixparams, sgsize=config.sgsize, flip=config.flip)
    elif config.model == 'E2SFCNN_QUOT':
        model = E2SFCNN_QUOT(n_inputs, n_outputs, restrict=config.restrict, N=config.N, fco=config.F, J=config.J,
                             sigma=config.sigma, sgsize=config.sgsize, flip=config.flip)
    elif config.model == 'EXP':
        model = ExpE2SFCNN(n_inputs, n_outputs, layer_type=config.type, restrict=config.restrict, N=config.N,
                           fix_param=config.fixparams, fco=config.F, J=config.J, sigma=config.sigma,
                           deltaorth=config.deltaorth, antialias=config.antialias, sgsize=config.sgsize,
                           flip=config.flip)
    elif config.model == 'CNN':
        model = ExpCNN(n_inputs, n_outputs, fix_param=config.fixparams, deltaorth=config.deltaorth)
    elif config.model == "wrn16_8_stl":
        model = wrn16_8_stl(num_classes=n_outputs, deltaorth=config.deltaorth)
    elif config.model == "e2wrn16_8_stl":
        model = e2wrn16_8_stl(N=config.N, r=config.restrict, num_classes=n_outputs, sigma=config.sigma, F=config.F,
                               deltaorth=config.deltaorth, fixparams=config.fixparams)
    elif config.model == "e2wrn28_10":
        model = e2wrn28_10(N=config.N, r=config.restrict, num_classes=n_outputs, sigma=config.sigma, F=config.F,
                            deltaorth=config.deltaorth, fixparams=config.fixparams)
    elif config.model == "e2wrn28_7":
        model = e2wrn28_7(N=config.N, r=config.restrict, num_classes=n_outputs, sigma=config.sigma, F=config.F,
                           deltaorth=config.deltaorth, fixparams=config.fixparams)
    elif config.model == "e2wrn28_10R":
        model = e2wrn28_10R(N=config.N, r=config.restrict, num_classes=n_outputs, sigma=config.sigma, F=config.F,
                             deltaorth=config.deltaorth, fixparams=config.fixparams)
    elif config.model == "e2wrn28_7R":
        model = e2wrn28_7R(N=config.N, r=config.restrict, num_classes=n_outputs, sigma=config.sigma, F=config.F,
                            deltaorth=config.deltaorth, fixparams=config.fixparams)
    elif config.model == "e2wrn10_8R":
        model = e2wrn10_8R(N=config.N, r=config.restrict, num_classes=n_outputs, sigma=config.sigma, F=config.F,
                            deltaorth=config.deltaorth, fixparams=config.fixparams)
    elif config.model == "e2wrn16_8R_stl":
        model = e2wrn16_8R_stl(N=config.N, r=config.restrict, num_classes=n_outputs, sigma=config.sigma, F=config.F,
                            deltaorth=config.deltaorth, fixparams=config.fixparams)                          
    else:
        raise ValueError("Model selected ({}) not recognized!".format(config.model))
    
    return model
