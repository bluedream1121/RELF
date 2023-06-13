from e2cnn.gspaces import *
from e2cnn.nn import *
from e2cnn.group import *
from e2cnn.nn import init

import torch
import torch.nn as nn
import numpy as np


class ExpE2SFCNN(torch.nn.Module):
    
    def __init__(self, n_channels, n_classes,
                 layer_type: str = "regular",
                 N=None,
                 restrict: int = -1,
                 fix_param: bool = False,
                 fco: float = 0.8,
                 J: int = 0,
                 sigma: float = 0.6,
                 deltaorth: bool = False,
                 antialias: float = 0.,
                 sgsize: int = None,
                 flip: bool = True,
                 ):
        
        super(ExpE2SFCNN, self).__init__()
        
        if N is None:
            N = 16
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # build the group O(2) or D_N depending on the number N of rotations specified
        if N > 1:
            self.gspace = FlipRot2dOnR2(N)
        elif N <= 0:
            if restrict < -1:
                axis = 0.
                restrict = -1
            else:
                axis = np.pi / 2
            self.gspace = FlipRot2dOnR2(-1, maximum_frequency=-N, axis=axis)
            N = -1
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

        if isinstance(fco, int) or isinstance(fco, float):
            fco *= np.pi
        self.frequencies_cutoff = fco
        self.sigma = sigma
        self.N = N
        self.restrict = restrict
        self.fix_param = fix_param
        self.J = J
        
        self.antialias = antialias
        
        # id of the subgroup if group restriction is applied through the network
        if sgsize is not None:
            self.sgid = sgsize
        elif N > 1 or N < 0:
            self.sgid = N
        else:
            self.sgid = 1
        
        if flip and N != 1:
            self.sgid = (None, self.sgid)

        eq_layers = []
        
        self.LAYER = 0
        
        r1 = FieldType(self.gspace, [self.gspace.trivial_repr] * n_channels)
        
        if layer_type == "regular":
            layer_builder = self.build_layer_regular
        elif layer_type == "quotient":
            layer_builder = self.build_layer_quotient
        elif layer_type == "gated_norm":
            layer_builder = self.build_layer_gated_normpool
        elif layer_type == "gated_norm_shared":
            layer_builder = self.build_layer_gated_normpool_shared
        elif layer_type == "gated_conv":
            layer_builder = self.build_layer_conv2triv
        elif layer_type == "gated_conv_shared":
            layer_builder = self.build_layer_gated_conv2triv_shared
        elif layer_type == "hnet_conv":
            layer_builder = self.build_layer_hnet_conv2triv
        elif layer_type == "hnet_norm":
            layer_builder = self.build_layer_hnet_normpool
        elif layer_type == "inducedhnet_conv":
            layer_builder = self.build_layer_inducedhnet_conv2triv
        elif layer_type == "inducedgated_norm":
            layer_builder = self.build_layer_inducedgated_normpool
        elif layer_type == "inducedgated_conv":
            layer_builder = self.build_layer_inducedgated_conv2triv
        elif layer_type == "sharednorm":
            layer_builder = self.build_layer_sharednorm
        elif layer_type == "sharednorm2":
            layer_builder = self.build_layer_sharednorm2
        elif layer_type == "vectorfield":
            layer_builder = self.build_layer_vectorfield
        elif layer_type == "regvector":
            layer_builder = self.build_layer_regvector
        elif layer_type == "scalarfield":
            layer_builder = self.build_layer_scalarfield
        elif layer_type == "trivial":
            layer_builder = self.build_layer_trivial
        elif layer_type == "squash":
            layer_builder = self.build_layer_squash
        elif layer_type == "realhnet":
            layer_builder = self.build_layer_realhnet
        elif layer_type == "realhnet2":
            layer_builder = self.build_layer_realhnet2
        elif layer_type == "debug":
            layer_builder = self.build_layer_debug
        else:
            raise ValueError(f"Error! Layer type {layer_type} not recognized!")
        
        # 28 px
        # Convolutional Layer 1
        
        self.LAYER += 1
        eq_layers += layer_builder(r1, 16, 7, 1)
        
        # 24 px
        # Convolutional Layer 2
        self.LAYER += 1
        eq_layers += layer_builder(eq_layers[-1].out_type, 24, 5, 2, 2)
        
        # 12 px
        # Convolutional Layer 3
        self.LAYER += 1
        eq_layers += layer_builder(eq_layers[-1].out_type, 32, 5, 2)
        
        # 12 px
        # Convolutional Layer 4
        self.LAYER += 1
        eq_layers += layer_builder(eq_layers[-1].out_type, 32, 5, 2, 2)
        
        # 6 px
        # Convolutional Layer 5
        self.LAYER += 1
        eq_layers += layer_builder(eq_layers[-1].out_type, 48, 5, 2)
        
        # 6 px
        # Convolutional Layer 6
        self.LAYER += 1
        eq_layers += layer_builder(eq_layers[-1].out_type, 64, 5, 0, None, True)
        
        # Adaptive Pooling
        mpl = PointwiseAdaptiveMaxPool(eq_layers[-1].out_type, 1)
        eq_layers.append(mpl)
        
        # 1 px
        
        # c = 64
        c = eq_layers[-1].out_type.size
        
        self.in_repr = eq_layers[0].in_type
        self.eq_layers = torch.nn.ModuleList(eq_layers)
        
        # Fully Connected
        
        self.fully_net = nn.Sequential(
            nn.Linear(c, 64),
            nn.BatchNorm1d(64),
            nn.ELU(inplace=True),
            
            nn.Linear(64, n_classes),
        )
        
        if deltaorth:
            for name, module in self.named_modules():
                if isinstance(module, R2Conv):
                    print(name)
                    init.deltaorthonormal_init(module.weights.data, module.basisexpansion)
        
        print("MODEL Architecture:")
        for i, (name, mod) in enumerate(self.named_modules()):
            params = sum([p.numel() for p in mod.parameters() if p.requires_grad])
            if isinstance(mod, EquivariantModule) and isinstance(mod.in_type, FieldType) and isinstance(mod.out_type,
                                                                                                        FieldType):
                print(f"\t{i: <3} - {name: <70} | {params: <8} | {mod.in_type.size: <4}- {mod.out_type.size: <4}")
            else:
                print(f"\t{i: <3} - {name: <70} | {params: <8} |")
        tot_param = sum([p.numel() for p in self.parameters() if p.requires_grad])
        print("Total number of parameters:", tot_param)
    
    def forward(self, input: torch.tensor):
        x = GeometricTensor(input, self.in_repr)
        
        for layer in self.eq_layers:
            x = layer(x)
        
        x = self.fully_net(x.tensor.reshape(x.tensor.shape[0], -1))
        
        return x
    
    def build_layer_regular(self, r1: FieldType, C: int, s: int, padding: int = 0, pooling: int = None,
                            invariant_map: bool = False):
        
        gc = r1.gspace
        
        if self.fix_param and not invariant_map and self.LAYER > 1:
            # if self.fix_param and self.LAYER > 1:
            # to keep number of parameters more or less constant when changing groups
            # (more precisely, we try to keep them close to the number of parameters in the original SFCNN)
            t = gc.fibergroup.order() / 16
            C = int(round(C / np.sqrt(t)))
        
        layers = []
        
        r2 = FieldType(gc, [gc.representations['regular']] * C)
        
        cl = R2Conv(r1, r2, s,
                    padding=padding,
                    frequencies_cutoff=self.frequencies_cutoff,
                    sigma=self.sigma,
                    maximum_offset=self.J,
                    )
        
        layers.append(cl)
        
        if self.restrict == self.LAYER:
            layers.append(RestrictionModule(layers[-1].out_type, self.sgid))
            layers.append(DisentangleModule(layers[-1].out_type))
        
        if invariant_map:
            pl = GroupPooling(layers[-1].out_type)
            layers.append(pl)
        
        bn = InnerBatchNorm(layers[-1].out_type)
        layers.append(bn)
        nnl = ELU(layers[-1].out_type)
        layers.append(nnl)
        
        if pooling is not None:
            if self.antialias > 0.:
                pl = PointwiseMaxPoolAntialiased(layers[-1].out_type, stride=pooling, sigma=self.antialias)
            elif self.antialias < 0.:
                pl = PointwiseAvgPoolAntialiased(layers[-1].out_type, stride=pooling, sigma=-self.antialias)
            else:
                pl = PointwiseMaxPool(layers[-1].out_type, pooling)
            layers.append(pl)
        
        return layers

    def build_layer_quotient(self, r1: FieldType, C: int, s: int, padding: int = 0, pooling: int = None,
                             invariant_map: bool = False):
    
        gc = r1.gspace
    
        if self.fix_param and not invariant_map and self.LAYER > 1:
            # to keep number of parameters more or less constant when changing groups
            # (more precisely, we try to keep them close to the number of parameters in the original SFCNN)
            t = gc.fibergroup.order() / 16
            C = C / np.sqrt(t)
    
        layers = []
    
        assert gc.fibergroup.order() > 0
        if isinstance(gc, FlipRot2dOnR2):
            n = int(gc.fibergroup.order() / 2)
            repr = [gc.regular_repr] * 5
            for i in [0, round(n / 4), round(n / 2)]:
                repr += [gc.quotient_repr((int(i), 1))] * 2
            if n % 2 == 0:
                repr += [gc.quotient_repr((None, 2))] * 2
            else:
                raise ValueError()
            repr += [gc.trivial_repr] * int(gc.fibergroup.order() / 4)
        elif isinstance(gc, Rot2dOnR2):
            n = gc.fibergroup.order()
            repr = [gc.regular_repr] * 5
            if n % 2 == 0:
                repr += [gc.quotient_repr(2)] * 2
            else:
                raise ValueError()
        
            if n % 4 == 0:
                repr += [gc.quotient_repr(4)] * 2
            elif n % 3 == 0:
                repr += [gc.quotient_repr(3)] * 2
            else:
                raise ValueError()
        
            repr += [gc.trivial_repr] * int(gc.fibergroup.order() / 4)
        else:
            repr = [gc.regular_repr]
    
        C /= sum([r.size for r in repr]) / gc.fibergroup.order()
    
        C = int(round(C))
    
        r2 = FieldType(gc, repr * C).sorted()
    
        cl = R2Conv(r1, r2, s,
                    frequencies_cutoff=self.frequencies_cutoff,
                    padding=padding,
                    sigma=self.sigma,
                    maximum_offset=self.J)
        layers.append(cl)
    
        if self.restrict == self.LAYER:
            layers.append(RestrictionModule(layers[-1].out_type, self.sgid))
            layers.append(DisentangleModule(layers[-1].out_type))
    
        if invariant_map:
            pl = GroupPooling(layers[-1].out_type)
            layers.append(pl)
    
        bn = InnerBatchNorm(layers[-1].out_type)
        layers.append(bn)
        nnl = ELU(layers[-1].out_type)
        layers.append(nnl)
    
        if pooling is not None:
            pl = PointwiseMaxPool(layers[-1].out_type, pooling)
            layers.append(pl)
    
        return layers

    def build_layer_vectorfield(self, r1: FieldType, C: int, s: int, padding: int = 0, pooling: int = None,
                                invariant_map: bool = False):
        
        # C_N equivariant model with vector-field non-linearity.
        
        assert isinstance(r1.gspace.fibergroup, CyclicGroup)
        
        gc = r1.gspace
    
        if self.fix_param and not invariant_map and self.LAYER > 1:
            # to keep number of parameters more or less constant when changing groups
            # (more precisely, we try to keep them close to the number of parameters in the original SFCNN)
            # t = gc.fibergroup.order() / 16
            t = gc.fibergroup.order() / 16
            C /= np.sqrt(t)
    
        C *= np.sqrt(gc.fibergroup.order() / 2)
        C = int(round(C))
    
        layers = []
    
        r2 = FieldType(gc, [gc.representations['regular']] * C)
    
        cl = R2Conv(r1, r2, s,
                    padding=padding,
                    frequencies_cutoff=self.frequencies_cutoff,
                    sigma=self.sigma,
                    maximum_offset=self.J
                    )
        layers.append(cl)
    
        if self.restrict == self.LAYER:
            layers.append(RestrictionModule(layers[-1].out_type, self.sgid))
            layers.append(DisentangleModule(layers[-1].out_type))
    
        if invariant_map:
            pl = GroupPooling(layers[-1].out_type)
            layers.append(pl)
        
            bn = InnerBatchNorm(layers[-1].out_type)
            layers.append(bn)
        
            nnl = ELU(layers[-1].out_type)
            layers.append(nnl)
            if pooling is not None:
                pl = PointwiseMaxPool(layers[-1].out_type, pooling)
                layers.append(pl)
        else:
            bn = InnerBatchNorm(layers[-1].out_type)
            layers.append(bn)
            nnl = VectorFieldNonLinearity(layers[-1].out_type)
            layers.append(nnl)
        
            if pooling is not None:
                pl = NormMaxPool(layers[-1].out_type, pooling)
                layers.append(pl)
    
        return layers

    def build_layer_scalarfield(self, r1: FieldType, C: int, s: int, padding: int = 0, pooling: int = None,
                                invariant_map: bool = False):
        
        # equivariant model using group pooling at each layer
        
        assert r1.gspace.fibergroup.order() > 0
    
        gc = r1.gspace
    
        if self.fix_param and not invariant_map and self.LAYER > 1:
            # to keep number of parameters more or less constant when changing groups
            # (more precisely, we try to keep them close to the number of parameters in the original SFCNN)
            # t = gc.fibergroup.order() / 16
            t = gc.fibergroup.order() / 16
            C /= np.sqrt(t)
    
        C *= np.sqrt(gc.fibergroup.order())
        C = int(round(C))
    
        layers = []
    
        r2 = FieldType(gc, [gc.representations['regular']] * C)
    
        cl = R2Conv(r1, r2, s,
                    padding=padding,
                    frequencies_cutoff=self.frequencies_cutoff,
                    sigma=self.sigma,
                    maximum_offset=self.J
                    )
        layers.append(cl)
    
        if self.restrict == self.LAYER:
            layers.append(RestrictionModule(layers[-1].out_type, self.sgid))
            layers.append(DisentangleModule(layers[-1].out_type))
    
        if invariant_map:
            pl = GroupPooling(layers[-1].out_type)
            layers.append(pl)
            bn = InnerBatchNorm(layers[-1].out_type)
            layers.append(bn)
            nnl = ELU(layers[-1].out_type)
            layers.append(nnl)
        else:
            bn = InnerBatchNorm(layers[-1].out_type)
            layers.append(bn)
            nnl = GroupPooling(layers[-1].out_type)
            layers.append(nnl)
            nnl = ELU(layers[-1].out_type)
            layers.append(nnl)
    
        if pooling is not None:
            pl = PointwiseMaxPool(layers[-1].out_type, pooling)
            layers.append(pl)
    
        return layers

    def build_layer_regvector(self, r1: FieldType, C: int, s: int, padding: int = 0, pooling: int = None,
                              invariant_map: bool = False):
    
        RATIO = 0.5
    
        gc = r1.gspace
    
        if self.fix_param and not invariant_map and self.LAYER > 1:
            # to keep number of parameters more or less constant when changing groups
            # (more precisely, we try to keep them close to the number of parameters in the original SFCNN)
            t = (RATIO * gc.fibergroup.order() + (1 - RATIO) * 2) / 16
            C /= np.sqrt(t)
    
        R = int(round(C * RATIO))
        V = int(round(C * (1 - RATIO)))
        C = V + R
    
        print(C, V, R)
    
        layers = []
    
        r2 = FieldType(gc, [gc.representations['regular']] * C)
    
        cl = R2Conv(r1, r2, s,
                    padding=padding,
                    frequencies_cutoff=self.frequencies_cutoff,
                    sigma=self.sigma,
                    maximum_offset=self.J
                    )
        layers.append(cl)
    
        if self.restrict == self.LAYER:
            layers.append(RestrictionModule(layers[-1].out_type, self.sgid))
            layers.append(DisentangleModule(layers[-1].out_type))
    
        if invariant_map:
            pl = GroupPooling(layers[-1].out_type)
            layers.append(pl)
        
            bn = InnerBatchNorm(layers[-1].out_type)
            layers.append(bn)
        
            nnl = ELU(layers[-1].out_type, inplace=True)
            layers.append(nnl)
            if pooling is not None:
                pl = PointwiseMaxPool(layers[-1].out_type, pooling)
                layers.append(pl)
        else:
            bn = InnerBatchNorm(layers[-1].out_type)
            layers.append(bn)
        
            r = layers[-1].out_type
            assert R + V == len(r)
            labels = ["regular"] * R + ["vector"] * V
            r3 = r.group_by_labels(labels)
            regular = r3["regular"]
            vector = r3["vector"]
        
            modules = []
            modules += [(ELU(regular), "regular")]
            modules += [(VectorFieldNonLinearity(vector), "vector")]
            bn = MultipleModule(layers[-1].out_type, labels, modules)
            layers.append(bn)
        
            if pooling is not None:
                r4 = layers[-1].out_type
                labels = ["vector" if r.irreducible else "regular" for r in r4]
                r4 = r4.group_by_labels(labels)
                regular = r4["regular"]
                vector = r4["vector"]
            
                modules = []
                modules += [(PointwiseMaxPool(regular, pooling), "regular")]
                modules += [(NormMaxPool(vector, pooling), "vector")]
                pl = MultipleModule(layers[-1].out_type, labels, modules)
                layers.append(pl)
    
        return layers

    def build_layer_hnet_normpool(self, r1: FieldType, C: int, s: int, padding: int = 0, pooling: int = None,
                                  invariant_map: bool = False, function: str = "n_relu"):
        
        # NORM POOL in the end
        
        gc = r1.gspace
        
        irreps = []
        for n, irr in gc.fibergroup.irreps.items():
            if n != gc.trivial_repr.name:
                irreps += [irr] * int(irr.size // irr.sum_of_squares_constituents)
        irreps = list(irreps)
        
        if self.fix_param and not invariant_map and self.LAYER > 1:
            # to keep number of parameters more or less constant when changing groups
            # (more precisely, we try to keep them close to the number of parameters in the original SFCNN)
            r_in = FieldType(gc, [gc.trivial_repr] + irreps)
            r_out = FieldType(gc, [gc.trivial_repr] + irreps)
            
            tmp_cl = R2Conv(r_in, r_out, s,
                            frequencies_cutoff=self.frequencies_cutoff,
                            padding=padding,
                            sigma=self.sigma,
                            maximum_offset=self.J)
            
            t = tmp_cl.basisexpansion.dimension()
            
            t /= 16 * s ** 2 * 3 / 4
            
            C = int(round(C / np.sqrt(t)))
        
        elif invariant_map:
            # in order to preserve the same number of output channels
            size = sum(int(irr.size // irr.sum_of_squares_constituents) for irr in gc.fibergroup.irreps.values())
            C = int(round(C / size))
        
        layers = []
        
        irreps = list(irreps)
        
        trivials = FieldType(gc, [gc.trivial_repr] * C)
        if len(irreps) > 0:
            others = FieldType(gc, irreps * C).sorted()
            r2 = trivials + others
        else:
            others = []
            r2 = trivials
        
        cl = R2Conv(r1, r2, s,
                    frequencies_cutoff=self.frequencies_cutoff,
                    padding=padding,
                    sigma=self.sigma,
                    maximum_offset=self.J)
        layers.append(cl)
        
        if self.restrict == self.LAYER:
            layers.append(RestrictionModule(layers[-1].out_type, self.sgid))
            layers.append(DisentangleModule(layers[-1].out_type))
        
        r3 = layers[-1].out_type
        labels = ["trivial" if r.is_trivial() else "others" for r in r3]
        r3 = r3.group_by_labels(labels)
        trivials = r3["trivial"]
        others = r3["others"]
        
        for r in trivials:
            r.supported_nonlinearities.add("pointwise")
        
        modules = []
        modules += [(InnerBatchNorm(trivials), "trivial")]
        if len(irreps) > 0:
            modules += [(NormBatchNorm(others), "others")]
        bn = MultipleModule(layers[-1].out_type, labels, modules)
        layers.append(bn)
        
        labels = ["trivial" if r.is_trivial() else "others" for r in layers[-1].out_type]
        
        modules = []
        modules += [(ELU(trivials), "trivial")]
        if len(irreps) > 0:
            modules += [(NormNonLinearity(others, function=function), "others")]
        
        nnl = MultipleModule(layers[-1].out_type, labels, modules)
        layers.append(nnl)
        
        if invariant_map:
            modules = [
                (IdentityModule(trivials), "trivial"),
                (NormPool(others), "others")
            ]
            pl = MultipleModule(layers[-1].out_type, labels, modules)
            layers.append(pl)
            
            if pooling is not None:
                pl = PointwiseMaxPool(layers[-1].out_type, pooling)
                layers.append(pl)
        elif pooling is not None:
            modules = []
            modules += [(PointwiseMaxPool(trivials, pooling), "trivial")]
            if len(irreps) > 0:
                modules += [(NormMaxPool(others, pooling), "others")]
            
            pl = MultipleModule(layers[-1].out_type, labels, modules)
            layers.append(pl)
        
        return layers

    def build_layer_squash(self, r1: FieldType, C: int, s: int, padding: int = 0, pooling: int = None,
                           invariant_map: bool = False):
        return self.build_layer_hnet_conv2triv(r1, C, s, padding, pooling, invariant_map, function="squash",
                                               norm_bias=False)

    def build_layer_hnet_conv2triv(self, r1: FieldType, C: int, s: int, padding: int = 0, pooling: int = None,
                                   invariant_map: bool = False, function="n_relu", norm_bias=True):
    
        # conv2triv in the end
        # convolve to frequency-zero scalar field
        # if the group is O(2) (or D_N), convolve also to the sign-flip irreps and then takes the absolute value
        # of those fields
        
        gc = r1.gspace
    
        if self.fix_param and not invariant_map and self.LAYER > 1:
            # to keep number of parameters more or less constant when changing groups
            # (more precisely, we try to keep them close to the number of parameters in the original SFCNN)
            irreps = []
            for n, irr in gc.fibergroup.irreps.items():
                irreps += [irr] * int(irr.size // irr.sum_of_squares_constituents)
            r = FieldType(gc, irreps)
            tmp_cl = R2Conv(r, r, s,
                            frequencies_cutoff=self.frequencies_cutoff,
                            padding=padding,
                            sigma=self.sigma,
                            maximum_offset=self.J)
        
            t = tmp_cl.basisexpansion.dimension()
        
            t /= 16 * s ** 2 * 3 / 4
        
            C = int(round(C / np.sqrt(t)))
    
        layers = []
    
        irreps = []
        for n, irr in gc.fibergroup.irreps.items():
            if n != gc.trivial_repr.name:
                irreps += [irr] * int(irr.size // irr.sum_of_squares_constituents)
    
        if len(irreps) > 0 and not invariant_map:
            trivials = FieldType(gc, [gc.trivial_repr] * C)
            others = FieldType(gc, irreps * C).sorted()
            r2 = trivials + others
        elif invariant_map:
            zero_freq = [irr for irr in gc.fibergroup.irreps.values() if irr.attributes['frequency'] == 0]
            irreps = [irr for irr in zero_freq if not irr.is_trivial()]
        
            if len(irreps) > 0:
                t = sum(irr.size for irr in zero_freq)
                t_i = sum(irr.size for irr in irreps)
                others = FieldType(gc, irreps * int(C * t_i / t)).sorted()
                trivials = FieldType(gc, [gc.trivial_repr] * int(C / t))
                r2 = trivials + others
            else:
                trivials = FieldType(gc, [gc.trivial_repr] * C)
                irreps = []
                others = None
                r2 = trivials
        else:
            trivials = FieldType(gc, [gc.trivial_repr] * C)
            irreps = []
            others = None
            r2 = trivials
    
        cl = R2Conv(r1, r2, s,
                    frequencies_cutoff=self.frequencies_cutoff,
                    padding=padding,
                    sigma=self.sigma,
                    maximum_offset=self.J)
        layers.append(cl)
    
        if self.restrict == self.LAYER:
            layers.append(RestrictionModule(layers[-1].out_type, self.sgid))
            layers.append(DisentangleModule(layers[-1].out_type))
    
        if len(irreps) > 0:
            r3 = layers[-1].out_type
            labels = ["trivial" if r.is_trivial() else "others" for r in r3]
            r3 = r3.group_by_labels(labels)
            trivials = r3["trivial"]
            others = r3["others"]
        else:
            trivials = layers[-1].out_type
            others = None
    
        assert all([r.is_trivial() for r in trivials])
    
        for r in trivials:
            r.supported_nonlinearities.add("pointwise")
    
        if len(irreps) > 0:
            modules = []
            modules += [(InnerBatchNorm(trivials), "trivial")]
            modules += [(NormBatchNorm(others), "others")]
            bn = MultipleModule(layers[-1].out_type, labels, modules)
            layers.append(bn)
        else:
            bn = InnerBatchNorm(layers[-1].out_type)
            layers.append(bn)
    
        if len(irreps) > 0:
            labels = ["trivial" if r.is_trivial() else "others" for r in layers[-1].out_type]
        
            modules = []
            modules += [(ELU(trivials), "trivial")]
            modules += [(NormNonLinearity(others, function=function, bias=norm_bias), "others")]
        
            nnl = MultipleModule(layers[-1].out_type, labels, modules)
            layers.append(nnl)
    
        else:
            nnl = ELU(layers[-1].out_type)
            layers.append(nnl)
    
        if invariant_map and others is not None:
            modules = [
                (IdentityModule(trivials), "trivial"),
                (NormPool(others), "others")
            ]
            pl = MultipleModule(layers[-1].out_type, labels, modules)
            layers.append(pl)
        
            if pooling is not None:
                pl = PointwiseMaxPool(layers[-1].out_type, pooling)
                layers.append(pl)
    
        elif len(irreps) == 0 and pooling is not None:
            pl = PointwiseMaxPool(layers[-1].out_type, pooling)
            layers.append(pl)
        elif pooling is not None:
            modules = []
            modules += [(PointwiseMaxPool(trivials, pooling), "trivial")]
            modules += [(NormMaxPool(others, pooling), "others")]
        
            pl = MultipleModule(layers[-1].out_type, labels, modules)
            layers.append(pl)
    
        return layers

    def build_layer_sharednorm(self, r1: FieldType, C: int, s: int, padding: int = 0, pooling: int = None,
                               invariant_map: bool = False):
        
        # shared Norm-ReLU on irreps, ELU on trivial + Norm-Pooling"
        
        gc = r1.gspace
        
        irreps = []
        for n, irr in gc.fibergroup.irreps.items():
            if n != gc.trivial_repr.name:
                irreps += [irr] * int(irr.size // irr.sum_of_squares_constituents)
        irreps = list(irreps)
        
        if self.fix_param and not invariant_map and self.LAYER > 1:
            # to keep number of parameters more or less constant when changing groups
            # (more precisely, we try to keep them close to the number of parameters in the original SFCNN)
            r_in = FieldType(gc, [gc.trivial_repr] + irreps)
            r_out = FieldType(gc, [gc.trivial_repr] + irreps)
            
            tmp_cl = R2Conv(r_in, r_out, s,
                            frequencies_cutoff=self.frequencies_cutoff,
                            padding=padding,
                            sigma=self.sigma,
                            maximum_offset=self.J)
            
            t = tmp_cl.basisexpansion.dimension()
            
            t /= 16 * s ** 2 * 3 / 4
            
            C = int(round(C / np.sqrt(t)))
        
        elif invariant_map:
            
            # in order to preserve the same number of output channels
            if len(irreps) > 0:
                # there is a trivial field and a big field with all other frequencies
                size = 2
            else:
                # there is only a trivial field
                size = 1
            C = int(round(C / size))
        
        layers = []
        
        irreps_field = directsum(list(irreps), name="irreps")
        
        trivials = FieldType(gc, [gc.trivial_repr] * C)
        
        if len(irreps) > 0:
            others = FieldType(gc, [irreps_field] * C).sorted()
            r2 = trivials + others
        else:
            others = []
            r2 = trivials
            
        cl = R2Conv(r1, r2, s,
                    frequencies_cutoff=self.frequencies_cutoff,
                    padding=padding,
                    sigma=self.sigma,
                    maximum_offset=self.J)
        layers.append(cl)
        
        if self.restrict == self.LAYER:
            layers.append(RestrictionModule(layers[-1].out_type, self.sgid))
            layers.append(DisentangleModule(layers[-1].out_type))
        
        r3 = layers[-1].out_type
        labels = ["trivial" if r.is_trivial() else "others" for r in r3]
        r3 = r3.group_by_labels(labels)
        trivials = r3["trivial"]
        others = r3["others"]
        
        for r in trivials:
            r.supported_nonlinearities.add("pointwise")
        
        modules = []
        modules += [(InnerBatchNorm(trivials), "trivial")]
        if len(irreps) > 0:
            modules += [(NormBatchNorm(others), "others")]
        bn = MultipleModule(layers[-1].out_type, labels, modules)
        layers.append(bn)
        
        labels = ["trivial" if r.is_trivial() else "others" for r in layers[-1].out_type]
        
        modules = []
        modules += [(ELU(trivials), "trivial")]
        if len(irreps) > 0:
            modules += [(NormNonLinearity(others), "others")]
        
        nnl = MultipleModule(layers[-1].out_type, labels, modules)
        layers.append(nnl)
        
        if invariant_map:
            modules = [
                (IdentityModule(trivials), "trivial"),
                (NormPool(others), "others")
            ]
            pl = MultipleModule(layers[-1].out_type, labels, modules)
            layers.append(pl)
            
            if pooling is not None:
                pl = PointwiseMaxPool(layers[-1].out_type, pooling)
                layers.append(pl)
        elif pooling is not None:
            modules = []
            modules += [(PointwiseMaxPool(trivials, pooling), "trivial")]
            if len(irreps) > 0:
                modules += [(NormMaxPool(others, pooling), "others")]
            
            pl = MultipleModule(layers[-1].out_type, labels, modules)
            layers.append(pl)
        
        return layers
    
    def build_layer_sharednorm2(self, r1: FieldType, C: int, s: int, padding: int = 0, pooling: int = None,
                                invariant_map: bool = False):

        # shared Norm-ReLU on all irreps + Norm-Pooling
        
        gc = r1.gspace
        irreps = []
        for n, irr in gc.fibergroup.irreps.items():
            irreps += [irr] * int(irr.size // irr.sum_of_squares_constituents)
        irreps = list(irreps)
        
        if self.fix_param and not invariant_map and self.LAYER > 1:
            # to keep number of parameters more or less constant when changing groups
            # (more precisely, we try to keep them close to the number of parameters in the original SFCNN)
            r = FieldType(gc, irreps)
            tmp_cl = R2Conv(r, r, s,
                            frequencies_cutoff=self.frequencies_cutoff,
                            padding=padding,
                            sigma=self.sigma,
                            maximum_offset=self.J)
            
            t = tmp_cl.basisexpansion.dimension()
            
            t /= 16 * s ** 2 / 2
            
            C = int(round(C / np.sqrt(t)))
        
        layers = []
        irreps = sorted(irreps, key=lambda i: i.size)
        irreps_field = directsum(irreps, name="irreps")
        
        r2 = FieldType(gc, [irreps_field] * C)
        
        cl = R2Conv(r1, r2, s,
                    frequencies_cutoff=self.frequencies_cutoff,
                    padding=padding,
                    sigma=self.sigma,
                    maximum_offset=self.J)
        layers.append(cl)
        
        if self.restrict == self.LAYER:
            layers.append(RestrictionModule(layers[-1].out_type, self.sgid))
            layers.append(DisentangleModule(layers[-1].out_type))
        
        bn = GNormBatchNorm(layers[-1].out_type)
        layers.append(bn)
        
        nnl = NormNonLinearity(layers[-1].out_type)
        layers.append(nnl)
        
        if invariant_map:
            pl = NormPool(layers[-1].out_type)
            layers.append(pl)
            
            if pooling is not None:
                pl = PointwiseMaxPool(layers[-1].out_type, pooling)
                layers.append(pl)
        elif pooling is not None:
            pl = NormMaxPool(layers[-1].out_type, pooling)
            layers.append(pl)
        
        return layers

    def build_layer_realhnet(self, r1: FieldType, C: int, s: int, padding: int = 0, pooling: int = None,
                             invariant_map: bool = False):
    
        gc = r1.gspace
    
        assert gc.fibergroup.order() < 0
        ################################################################################################################
        ############# NOTICE ###########################################################################################
        # The number of parameters is preserved as if Hnet3 is used.
        # This model, however, has a little less parameters as a more constrained kernel space is used
        ################################################################################################################
    
        if self.fix_param and self.LAYER > 1 and not invariant_map:
            # to keep number of parameters more or less constant when changing groups
            # (more precisely, we try to keep them close to the number of parameters in the original SFCNN)
        
            irreps = [irr for irr in gc.fibergroup.irreps.values()]
            tmp_repr = FieldType(gc, irreps)
        
            tmp_cl = R2Conv(tmp_repr, tmp_repr, s,
                            frequencies_cutoff=self.frequencies_cutoff,
                            padding=padding,
                            sigma=self.sigma,
                            maximum_offset=self.J,
                            )
        
            t = tmp_cl.basisexpansion.dimension()
        
            t /= 16 * s ** 2 * 3 / 4
            C = int(round(C / np.sqrt(t)))
    
        layers = []
    
        irreps = []
        for n, irr in gc.fibergroup.irreps.items():
            if n != gc.trivial_repr.name:
                irreps += [irr] * int(irr.size // irr.sum_of_squares_constituents)
    
        if len(irreps) > 0 and not invariant_map:
            trivials = FieldType(gc, [gc.trivial_repr] * C)
            others = FieldType(gc, irreps * C).sorted()
            r2 = trivials + others
        elif invariant_map:
            zero_freq = [irr for irr in gc.fibergroup.irreps.values() if irr.attributes['frequency'] == 0]
            irreps = [irr for irr in zero_freq if not irr.is_trivial()]
        
            if len(irreps) > 0:
                t = sum(irr.size for irr in zero_freq)
                t_i = sum(irr.size for irr in irreps)
                others = FieldType(gc, irreps * int(C * t_i / t)).sorted()
                trivials = FieldType(gc, [gc.trivial_repr] * int(C / t))
                r2 = trivials + others
            else:
                trivials = FieldType(gc, [gc.trivial_repr] * C)
                irreps = []
                others = None
                r2 = trivials
        else:
            trivials = FieldType(gc, [gc.trivial_repr] * C)
            irreps = []
            others = None
            r2 = trivials
    
        hnet_basis_filter = lambda d: d["s"] == 1 if "s" in d else True
    
        cl = R2Conv(r1, r2, s,
                    frequencies_cutoff=self.frequencies_cutoff,
                    padding=padding,
                    sigma=self.sigma,
                    maximum_offset=self.J,
                    basis_filter=hnet_basis_filter,
                    )
        layers.append(cl)
    
        if self.restrict == self.LAYER:
            layers.append(RestrictionModule(layers[-1].out_type, self.sgid))
            layers.append(DisentangleModule(layers[-1].out_type))
    
        if len(irreps) > 0:
            r3 = layers[-1].out_type
            labels = ["trivial" if r.is_trivial() else "others" for r in r3]
            r3 = r3.group_by_labels(labels)
            trivials = r3["trivial"]
            others = r3["others"]
        else:
            trivials = layers[-1].out_type
            others = None
    
        assert all([r.is_trivial() for r in trivials])
    
        for r in trivials:
            r.supported_nonlinearities.add("pointwise")
    
        if len(irreps) > 0:
            modules = []
            modules += [(InnerBatchNorm(trivials), "trivial")]
            modules += [(NormBatchNorm(others), "others")]
            bn = MultipleModule(layers[-1].out_type, labels, modules)
            layers.append(bn)
        else:
            bn = InnerBatchNorm(layers[-1].out_type)
            layers.append(bn)
    
        if len(irreps) > 0:
            labels = ["trivial" if r.is_trivial() else "others" for r in layers[-1].out_type]
        
            modules = []
            modules += [(ELU(trivials), "trivial")]
            modules += [(NormNonLinearity(others), "others")]
        
            nnl = MultipleModule(layers[-1].out_type, labels, modules)
            layers.append(nnl)
    
        else:
            nnl = ELU(layers[-1].out_type)
            layers.append(nnl)
    
        if invariant_map and others is not None:
            modules = [
                (IdentityModule(trivials), "trivial"),
                (NormPool(others), "others")
            ]
            pl = MultipleModule(layers[-1].out_type, labels, modules)
            layers.append(pl)
        
            if pooling is not None:
                pl = PointwiseMaxPool(layers[-1].out_type, pooling)
                layers.append(pl)
    
        elif len(irreps) == 0 and pooling is not None:
            pl = PointwiseMaxPool(layers[-1].out_type, pooling)
            layers.append(pl)
        elif pooling is not None:
            modules = []
            modules += [(PointwiseMaxPool(trivials, pooling), "trivial")]
            modules += [(NormMaxPool(others, pooling), "others")]
        
            pl = MultipleModule(layers[-1].out_type, labels, modules)
            layers.append(pl)
    
        return layers

    def build_layer_realhnet2(self, r1: FieldType, C: int, s: int, padding: int = 0, pooling: int = None,
                              invariant_map: bool = False):
    
        gc = r1.gspace
    
        assert gc.fibergroup.order() < 0
    
        hnet_basis_filter = lambda d: d["s"] == 1 if "s" in d else True
    
        ################################################################################################################
        ############# NOTICE ###########################################################################################
        # WRT "realhnet", here we scale up the model to match the number of parameters.
        ################################################################################################################
    
        if self.fix_param and self.LAYER > 1 and not invariant_map:
            # to keep number of parameters more or less constant when changing groups
            # (more precisely, we try to keep them close to the number of parameters in the original SFCNN)
        
            irreps = [irr for irr in gc.fibergroup.irreps.values()]
            tmp_repr = FieldType(gc, irreps)
        
            tmp_cl = R2Conv(tmp_repr, tmp_repr, s,
                            frequencies_cutoff=self.frequencies_cutoff,
                            padding=padding,
                            sigma=self.sigma,
                            maximum_offset=self.J,
                            basis_filter=hnet_basis_filter,
                            )
        
            t = tmp_cl.basisexpansion.dimension()
        
            t /= 16 * s ** 2 * 3 / 4
        
            C = int(round(C / np.sqrt(t)))
    
        layers = []
    
        irreps = []
        for n, irr in gc.fibergroup.irreps.items():
            if n != gc.trivial_repr.name:
                irreps += [irr] * int(irr.size // irr.sum_of_squares_constituents)
    
        if len(irreps) > 0 and not invariant_map:
            trivials = FieldType(gc, [gc.trivial_repr] * C)
            others = FieldType(gc, irreps * C).sorted()
            r2 = trivials + others
        elif invariant_map:
            zero_freq = [irr for irr in gc.fibergroup.irreps.values() if irr.attributes['frequency'] == 0]
            irreps = [irr for irr in zero_freq if not irr.is_trivial()]
        
            if len(irreps) > 0:
                t = sum(irr.size for irr in zero_freq)
                t_i = sum(irr.size for irr in irreps)
                others = FieldType(gc, irreps * int(C * t_i / t)).sorted()
                trivials = FieldType(gc, [gc.trivial_repr] * int(C / t))
                r2 = trivials + others
            else:
                trivials = FieldType(gc, [gc.trivial_repr] * C)
                irreps = []
                others = None
                r2 = trivials
        else:
            trivials = FieldType(gc, [gc.trivial_repr] * C)
            irreps = []
            others = None
            r2 = trivials
    
        print(r2.size)
    
        cl = R2Conv(r1, r2, s,
                    frequencies_cutoff=self.frequencies_cutoff,
                    padding=padding,
                    sigma=self.sigma,
                    maximum_offset=self.J,
                    basis_filter=hnet_basis_filter,
                    )
        layers.append(cl)
    
        if self.restrict == self.LAYER:
            layers.append(RestrictionModule(layers[-1].out_type, self.sgid))
            layers.append(DisentangleModule(layers[-1].out_type))
    
        if len(irreps) > 0:
            r3 = layers[-1].out_type
            labels = ["trivial" if r.is_trivial() else "others" for r in r3]
            r3 = r3.group_by_labels(labels)
            trivials = r3["trivial"]
            others = r3["others"]
        else:
            trivials = layers[-1].out_type
            others = None
    
        assert all([r.is_trivial() for r in trivials])
    
        for r in trivials:
            r.supported_nonlinearities.add("pointwise")
    
        if len(irreps) > 0:
            modules = []
            modules += [(InnerBatchNorm(trivials), "trivial")]
            modules += [(NormBatchNorm(others), "others")]
            bn = MultipleModule(layers[-1].out_type, labels, modules)
            layers.append(bn)
        else:
            bn = InnerBatchNorm(layers[-1].out_type)
            layers.append(bn)
    
        if len(irreps) > 0:
            labels = ["trivial" if r.is_trivial() else "others" for r in layers[-1].out_type]
        
            modules = []
            modules += [(ELU(trivials), "trivial")]
            modules += [(NormNonLinearity(others), "others")]
        
            nnl = MultipleModule(layers[-1].out_type, labels, modules)
            layers.append(nnl)
    
        else:
            nnl = ELU(layers[-1].out_type)
            layers.append(nnl)
    
        if invariant_map and others is not None:
            modules = [
                (IdentityModule(trivials), "trivial"),
                (NormPool(others), "others")
            ]
            pl = MultipleModule(layers[-1].out_type, labels, modules)
            layers.append(pl)
        
            if pooling is not None:
                pl = PointwiseMaxPool(layers[-1].out_type, pooling)
                layers.append(pl)
    
        elif len(irreps) == 0 and pooling is not None:
            pl = PointwiseMaxPool(layers[-1].out_type, pooling)
            layers.append(pl)
        elif pooling is not None:
            modules = []
            modules += [(PointwiseMaxPool(trivials, pooling), "trivial")]
            modules += [(NormMaxPool(others, pooling), "others")]
        
            pl = MultipleModule(layers[-1].out_type, labels, modules)
            layers.append(pl)
    
        return layers

    def build_layer_gated_normpool(self, r1: FieldType, C: int, s: int, padding: int = 0, pooling: int = None,
                                   invariant_map: bool = False):
    
        ############################################################
        # 1 gate per irrep (except trivial); ELU on trivial irreps
        ############################################################
    
        gc = r1.gspace
    
        irreps = []
        for n, irr in gc.fibergroup.irreps.items():
            if n != gc.trivial_repr.name:
                irreps += [irr] * int(irr.size // irr.sum_of_squares_constituents)
        irreps = list(irreps)
    
        # number of fields requiring a gate
        I = len(irreps)
    
        # size of all irreps (including trivial)
        S = FieldType(gc, irreps).size + 1
        M = S + I
    
        if self.fix_param and not invariant_map and self.LAYER > 1:
            # to keep number of parameters more or less constant when changing groups
            # (more precisely, we try to keep them close to the number of parameters in the original SFCNN)
            r_in = FieldType(gc, [gc.trivial_repr] * (I + 1) + irreps)
            r_out = FieldType(gc, [gc.trivial_repr] + irreps)
        
            tmp_cl = R2Conv(r_in, r_out, s,
                            frequencies_cutoff=self.frequencies_cutoff,
                            padding=padding,
                            sigma=self.sigma,
                            maximum_offset=self.J)
        
            t = tmp_cl.basisexpansion.dimension()
        
            t /= 16 * s ** 2 * 3 / 4
        
            C = int(round(C / np.sqrt(t)))
    
        elif invariant_map:
            # in order to preserve the same number of output channels
            size = sum(int(irr.size // irr.sum_of_squares_constituents) for irr in gc.fibergroup.irreps.values())
            C = int(round(C / size))
    
        layers = []
    
        # total channels =   C * S    +   1 * C * I     = C * M
        # i.e.:            (fields) + (gates for non-trivial fields)
    
        trivials = FieldType(gc, [gc.trivial_repr] * C)
        gates = FieldType(gc, [gc.trivial_repr] * C * I)
        gated = FieldType(gc, irreps * C).sorted()
        gate = gates + gated
    
        r2 = trivials + gate
    
        cl = R2Conv(r1, r2, s,
                    frequencies_cutoff=self.frequencies_cutoff,
                    padding=padding,
                    sigma=self.sigma,
                    maximum_offset=self.J)
        layers.append(cl)
    
        labels = ["trivial"] * (len(trivials) + len(gates)) + ["gated"] * len(gated)
    
        modules = [
            (InnerBatchNorm(trivials + gates), "trivial"),
            (NormBatchNorm(gated), "gated")
        ]
        bn = MultipleModule(layers[-1].out_type, labels, modules)
        layers.append(bn)
    
        labels = ["trivial"] * len(trivials) + ["gate"] * len(gate)
        modules = [
            (ELU(trivials), "trivial"),
            (GatedNonLinearity1(gate), "gate")
        ]
        nnl = MultipleModule(layers[-1].out_type, labels, modules)
        layers.append(nnl)
    
        if self.restrict == self.LAYER:
            layers.append(RestrictionModule(layers[-1].out_type, self.sgid))
            layers.append(DisentangleModule(layers[-1].out_type))
    
        r3 = layers[-1].out_type
        labels = ["trivial" if r.is_trivial() else "others" for r in r3]
        r3 = r3.group_by_labels(labels)
        trivials = r3["trivial"]
        others = r3["others"]
    
        for r in trivials:
            r.supported_nonlinearities.add("pointwise")
    
        if invariant_map:
            modules = [
                (IdentityModule(trivials), "trivial"),
                (NormPool(others), "others")
            ]
            pl = MultipleModule(layers[-1].out_type, labels, modules)
            layers.append(pl)
        
            if pooling is not None:
                pl = PointwiseMaxPool(layers[-1].out_type, pooling)
                layers.append(pl)
        elif pooling is not None:
            modules = [
                (PointwiseMaxPool(trivials, pooling), "trivial"),
                (NormMaxPool(others, pooling), "others")
            ]
            pl = MultipleModule(layers[-1].out_type, labels, modules)
            layers.append(pl)
    
        return layers

    def build_layer_gated_normpool_shared(self, r1: FieldType, C: int, s: int, padding: int = 0, pooling: int = None,
                                          invariant_map: bool = False):
    
        ###################################################################################
        # 1 gate per field containing all irreps except trivial; ELU on trivial irreps
        ###################################################################################
    
        gc = r1.gspace
    
        irreps = []
        for n, irr in gc.fibergroup.irreps.items():
            if n != gc.trivial_repr.name:
                irreps += [irr] * int(irr.size // irr.sum_of_squares_constituents)
        irreps = list(irreps)
    
        if self.fix_param and not invariant_map and self.LAYER > 1:
            # to keep number of parameters more or less constant when changing groups
            # (more precisely, we try to keep them close to the number of parameters in the original SFCNN)
            r_in = FieldType(gc, [gc.trivial_repr] * 2 + irreps)
            r_out = FieldType(gc, [gc.trivial_repr] + irreps)
        
            tmp_cl = R2Conv(r_in, r_out, s,
                            frequencies_cutoff=self.frequencies_cutoff,
                            padding=padding,
                            sigma=self.sigma,
                            maximum_offset=self.J)
        
            t = tmp_cl.basisexpansion.dimension()
        
            t /= 16 * s ** 2 * 3 / 4
        
            C = int(round(C / np.sqrt(t)))
    
        elif invariant_map:
            # in order to preserve the same number of output channels
            size = sum(int(irr.size // irr.sum_of_squares_constituents) for irr in gc.fibergroup.irreps.values())
            C = int(round(C / size))
    
        layers = []
    
        irreps_field = directsum(list(irreps), name="irreps")
    
        trivials = FieldType(gc, [gc.trivial_repr] * C)
        gates = FieldType(gc, [gc.trivial_repr] * C)
        gated = FieldType(gc, [irreps_field] * C).sorted()
        gate = gates + gated
    
        r2 = trivials + gate
    
        cl = R2Conv(r1, r2, s,
                    frequencies_cutoff=self.frequencies_cutoff,
                    padding=padding,
                    sigma=self.sigma,
                    maximum_offset=self.J)
        layers.append(cl)
    
        labels = ["trivial"] * (len(trivials) + len(gates)) + ["gated"] * len(gated)
    
        modules = [
            (InnerBatchNorm(trivials + gates), "trivial"),
            (NormBatchNorm(gated), "gated")
        ]
        bn = MultipleModule(layers[-1].out_type, labels, modules)
        layers.append(bn)
    
        labels = ["trivial"] * len(trivials) + ["gate"] * len(gate)
        modules = [
            (ELU(trivials), "trivial"),
            (GatedNonLinearity1(gate), "gate")
        ]
        nnl = MultipleModule(layers[-1].out_type, labels, modules)
        layers.append(nnl)
    
        if self.restrict == self.LAYER:
            layers.append(RestrictionModule(layers[-1].out_type, self.sgid))
            layers.append(DisentangleModule(layers[-1].out_type))
    
        r3 = layers[-1].out_type
        labels = ["trivial" if r.is_trivial() else "others" for r in r3]
        r3 = r3.group_by_labels(labels)
        trivials = r3["trivial"]
        others = r3["others"]
    
        for r in trivials:
            r.supported_nonlinearities.add("pointwise")
    
        if invariant_map:
            modules = [
                (IdentityModule(trivials), "trivial"),
                (NormPool(others), "others")
            ]
            pl = MultipleModule(layers[-1].out_type, labels, modules)
            layers.append(pl)
        
            if pooling is not None:
                pl = PointwiseMaxPool(layers[-1].out_type, pooling)
                layers.append(pl)
        elif pooling is not None:
            modules = [
                (PointwiseMaxPool(trivials, pooling), "trivial"),
                (NormMaxPool(others, pooling), "others")
            ]
            pl = MultipleModule(layers[-1].out_type, labels, modules)
            layers.append(pl)
    
        return layers

    def build_layer_conv2triv(self, r1: FieldType, C: int, s: int, padding: int = 0, pooling: int = None,
                              invariant_map: bool = False):
    
        ############################################################
        # 1 gate per irrep (except trivial); ELU on trivial irreps
        # conv 2 triv in the end
        ############################################################
    
        gc = r1.gspace
    
        irreps = []
        for n, irr in gc.fibergroup.irreps.items():
            if n != gc.trivial_repr.name:
                if not invariant_map or irr.attributes['frequency'] == 0:
                    irreps += [irr] * int(irr.size // irr.sum_of_squares_constituents)
        irreps = list(irreps)
    
        # number of fields requiring a gate
        I = len(irreps)
    
        # size of all irreps (including trivial)
        S = sum(r.size for r in irreps) + 1
        M = S + I
    
        if self.fix_param and not invariant_map and self.LAYER > 1:
            # to keep number of parameters more or less constant when changing groups
            # (more precisely, we try to keep them close to the number of parameters in the original SFCNN)
            r_in = FieldType(gc, [gc.trivial_repr] * (I + 1) + irreps)
            r_out = FieldType(gc, [gc.trivial_repr] + irreps)
        
            tmp_cl = R2Conv(r_in, r_out, s,
                            frequencies_cutoff=self.frequencies_cutoff,
                            padding=padding,
                            sigma=self.sigma,
                            maximum_offset=self.J)
        
            t = tmp_cl.basisexpansion.dimension()
        
            t /= 16 * s ** 2 * 3 / 4
        
            C = int(round(C / np.sqrt(t)))
    
        elif invariant_map:
            # in order to preserve the same number of output channels
            size = sum(int(irr.size // irr.sum_of_squares_constituents) for irr in gc.fibergroup.irreps.values() if
                       irr.attributes["frequency"] == 0)
            C = int(round(C / size))
    
        layers = []
    
        # total channels =   C * S    +   1 * C * I     = C * M
        # i.e.:            (fields) + (gates for non-trivial fields)
    
        trivials = FieldType(gc, [gc.trivial_repr] * C)
        if len(irreps) > 0:
            gates = FieldType(gc, [gc.trivial_repr] * C * I)
            gated = FieldType(gc, irreps * C).sorted()
            gate = gates + gated
        
            r2 = trivials + gate
        else:
            gates = gated = gate = []
            r2 = trivials
    
        cl = R2Conv(r1, r2, s,
                    frequencies_cutoff=self.frequencies_cutoff,
                    padding=padding,
                    sigma=self.sigma,
                    maximum_offset=self.J)
        layers.append(cl)
    
        if len(irreps) > 0:
            labels = ["trivial"] * (len(trivials) + len(gates)) + ["gated"] * len(gated)
        
            modules = [
                (InnerBatchNorm(trivials + gates), "trivial"),
                (NormBatchNorm(gated), "gated")
            ]
            bn = MultipleModule(layers[-1].out_type, labels, modules)
        else:
            bn = InnerBatchNorm(trivials)
        layers.append(bn)
    
        if len(irreps) > 0:
            labels = ["trivial"] * len(trivials) + ["gate"] * len(gate)
            modules = [
                (ELU(trivials), "trivial"),
                (GatedNonLinearity1(gate), "gate")
            ]
            nnl = MultipleModule(layers[-1].out_type, labels, modules)
        else:
            nnl = ELU(trivials)
        layers.append(nnl)
    
        if self.restrict == self.LAYER:
            layers.append(RestrictionModule(layers[-1].out_type, self.sgid))
            layers.append(DisentangleModule(layers[-1].out_type))
    
        r3 = layers[-1].out_type
        labels = ["trivial" if r.is_trivial() else "others" for r in r3]
        r3 = r3.group_by_labels(labels)
        trivials = r3["trivial"]
        if "others" in r3:
            others = r3["others"]
        else:
            others = []
    
        for r in trivials:
            r.supported_nonlinearities.add("pointwise")
    
        if invariant_map:
            if len(others) > 0:
                modules = [
                    (IdentityModule(trivials), "trivial"),
                    (NormPool(others), "others")
                ]
                pl = MultipleModule(layers[-1].out_type, labels, modules)
                layers.append(pl)
        
            if pooling is not None:
                pl = PointwiseMaxPool(layers[-1].out_type, pooling)
                layers.append(pl)
        elif pooling is not None:
            if len(others) > 0:
                modules = [
                    (PointwiseMaxPool(trivials, pooling), "trivial"),
                    (NormMaxPool(others, pooling), "others")
                ]
                pl = MultipleModule(layers[-1].out_type, labels, modules)
            else:
                pl = PointwiseMaxPool(trivials, pooling)
            layers.append(pl)
    
        return layers

    def build_layer_gated_conv2triv_shared(self, r1: FieldType, C: int, s: int, padding: int = 0, pooling: int = None,
                                           invariant_map: bool = False):
    
        ###################################################################################
        # 1 gate per field containing all irreps except trivial; ELU on trivial irreps
        # conv 2 triv in the end
        ###################################################################################
    
        gc = r1.gspace
        assert not isinstance(gc, FlipRot2dOnR2)
    
        irreps = []
        if not invariant_map:
            for n, irr in gc.fibergroup.irreps.items():
                if n != gc.trivial_repr.name:
                    irreps += [irr] * int(irr.size // irr.sum_of_squares_constituents)
        irreps = list(irreps)
    
        if self.fix_param and not invariant_map and self.LAYER > 1:
            # to keep number of parameters more or less constant when changing groups
            # (more precisely, we try to keep them close to the number of parameters in the original SFCNN)
            r_in = FieldType(gc, [gc.trivial_repr] * 2 + irreps)
            r_out = FieldType(gc, [gc.trivial_repr] + irreps)
        
            tmp_cl = R2Conv(r_in, r_out, s,
                            frequencies_cutoff=self.frequencies_cutoff,
                            padding=padding,
                            sigma=self.sigma,
                            maximum_offset=self.J)
        
            t = tmp_cl.basisexpansion.dimension()
        
            t /= 16 * s ** 2 * 3 / 4
        
            C = int(round(C / np.sqrt(t)))
    
        layers = []
    
        trivials = FieldType(gc, [gc.trivial_repr] * C)
    
        if len(irreps) > 0:
            irreps_field = directsum(list(irreps), name="irreps")
        
            gates = FieldType(gc, [gc.trivial_repr] * C)
            gated = FieldType(gc, [irreps_field] * C).sorted()
            gate = gates + gated
        
            r2 = trivials + gate
        else:
            r2 = trivials
            gate = gated = gates = []
    
        cl = R2Conv(r1, r2, s,
                    frequencies_cutoff=self.frequencies_cutoff,
                    padding=padding,
                    sigma=self.sigma,
                    maximum_offset=self.J)
        layers.append(cl)
    
        if len(irreps) > 0:
            labels = ["trivial"] * (len(trivials) + len(gates)) + ["gated"] * len(gated)
        
            modules = [
                (InnerBatchNorm(trivials + gates), "trivial"),
                (NormBatchNorm(gated), "gated")
            ]
            bn = MultipleModule(layers[-1].out_type, labels, modules)
        else:
            bn = InnerBatchNorm(trivials)
        layers.append(bn)
    
        if len(irreps) > 0:
            labels = ["trivial"] * len(trivials) + ["gate"] * len(gate)
            modules = [
                (ELU(trivials), "trivial"),
                (GatedNonLinearity1(gate), "gate")
            ]
            nnl = MultipleModule(layers[-1].out_type, labels, modules)
        else:
            nnl = ELU(trivials)
        layers.append(nnl)
    
        if self.restrict == self.LAYER:
            layers.append(RestrictionModule(layers[-1].out_type, self.sgid))
            layers.append(DisentangleModule(layers[-1].out_type))
    
        r3 = layers[-1].out_type
        labels = ["trivial" if r.is_trivial() else "others" for r in r3]
        r3 = r3.group_by_labels(labels)
        trivials = r3["trivial"]
        if "others" in r3:
            others = r3["others"]
        else:
            others = []
    
        for r in trivials:
            r.supported_nonlinearities.add("pointwise")
    
        if invariant_map:
            if len(others) > 0:
                modules = [
                    (IdentityModule(trivials), "trivial"),
                    (NormPool(others), "others")
                ]
                pl = MultipleModule(layers[-1].out_type, labels, modules)
                layers.append(pl)
        
            if pooling is not None:
                pl = PointwiseMaxPool(layers[-1].out_type, pooling)
                layers.append(pl)
        elif pooling is not None:
            if len(others) > 0:
                modules = [
                    (PointwiseMaxPool(trivials, pooling), "trivial"),
                    (NormMaxPool(others, pooling), "others")
                ]
                pl = MultipleModule(layers[-1].out_type, labels, modules)
            else:
                pl = PointwiseMaxPool(trivials, pooling)
            layers.append(pl)
    
        return layers

    def build_layer_trivial(self, r1: FieldType, C: int, s: int, padding: int = 0, pooling: int = None,
                            invariant_map: bool = False):
    
        gc = r1.gspace
    
        if self.fix_param and not invariant_map and self.LAYER > 1:
            # to keep number of parameters more or less constant
            # C *= np.sqrt(16 * 2*s)
            C *= np.sqrt(16 * s)
    
        C = int(round(C))
    
        layers = []
    
        r2 = FieldType(gc, [gc.trivial_repr] * C)
    
        cl = R2Conv(r1, r2, s,
                    padding=padding,
                    frequencies_cutoff=self.frequencies_cutoff,
                    sigma=self.sigma,
                    maximum_offset=self.J
                    )
        layers.append(cl)
    
        if self.restrict == self.LAYER:
            layers.append(RestrictionModule(layers[-1].out_type, self.sgid))
            layers.append(DisentangleModule(layers[-1].out_type))
    
        bn = InnerBatchNorm(layers[-1].out_type)
        layers.append(bn)
        nnl = ELU(layers[-1].out_type, inplace=True)
        layers.append(nnl)
    
        if pooling is not None:
            pl = PointwiseMaxPool(layers[-1].out_type, pooling)
            layers.append(pl)
    
        return layers

    def build_layer_inducedhnet_conv2triv(self, r1: FieldType, C: int, s: int, padding: int = 0, pooling: int = None,
                                          invariant_map: bool = False):
        
        # individual Norm-ReLU with bias shared within induced field,
        # conv to induced trivial (+gpool).
        # Fix total number of params
        
        gc = r1.gspace
        assert isinstance(gc, FlipRot2dOnR2)
        
        if self.fix_param and self.LAYER > 1 and not invariant_map:
            # to keep number of parameters more or less constant when changing groups
            # (more precisely, we try to keep them close to the number of parameters in the original SFCNN)
            F = max([r.attributes["frequency"] for r in gc.fibergroup.irreps.values()])
            M = 3 * s / 2
            B = F * (2 * M + 1) - M * min(M, F)
            
            t = 4 * B / 6 + 2 * min(F, M) / 6 + 1
            t *= s / 2
            t *= sum([r.size ** 2 // r.sum_of_squares_constituents for r in gc.fibergroup.irreps.values()])
            t /= 2 * F + 1
            
            t /= 16 * s ** 2 / 2
            C = (C / np.sqrt(t))
        
        C = int(round(C))
        
        subgroup_id = (None, gc.fibergroup.rotation_order)
        sg, _, _ = gc.restrict(subgroup_id)
        
        layers = []
        
        irreps = []
        for n, irr in sg.fibergroup.irreps.items():
            if n != sg.trivial_repr.name:
                irreps += [gc.induced_repr(subgroup_id, irr)]
        
        trivials = FieldType(gc, [gc.induced_repr(subgroup_id, sg.trivial_repr)] * C)
        
        if len(irreps) > 0 and not invariant_map:
            others = FieldType(gc, irreps * C).sorted()
            r2 = trivials + others
        else:
            irreps = []
            others = None
            r2 = trivials
        
        cl = R2Conv(r1, r2, s,
                    frequencies_cutoff=self.frequencies_cutoff,
                    padding=padding,
                    sigma=self.sigma,
                    maximum_offset=self.J)
        layers.append(cl)
        
        if self.restrict == self.LAYER:
            layers.append(RestrictionModule(layers[-1].out_type, self.sgid))
            layers.append(DisentangleModule(layers[-1].out_type))
        
        def only_zero_freq(repr: Representation):
            for irr in repr.irreps:
                if repr.group.irreps[irr].attributes['frequency'] != 0:
                    return False
            return True
        
        if len(irreps) > 0:
            r3 = layers[-1].out_type
            labels = ["trivial" if only_zero_freq(r) else "others" for r in r3]
            r3 = r3.group_by_labels(labels)
            trivials = r3["trivial"]
            others = r3["others"]
        else:
            trivials = layers[-1].out_type
            others = None
        
        assert all([only_zero_freq(r) for r in trivials])
        
        if len(irreps) > 0:
            modules = []
            modules += [(InnerBatchNorm(trivials), "trivial")]
            modules += [(InducedNormBatchNorm(others), "others")]
            bn = MultipleModule(layers[-1].out_type, labels, modules)
            layers.append(bn)
        else:
            bn = InnerBatchNorm(layers[-1].out_type)
            layers.append(bn)
        
        if len(irreps) > 0:
            labels = ["trivial" if only_zero_freq(r) else "others" for r in layers[-1].out_type]
            
            modules = []
            modules += [(ELU(trivials), "trivial")]
            modules += [(InducedNormNonLinearity(others), "others")]
            
            nnl = MultipleModule(layers[-1].out_type, labels, modules)
            layers.append(nnl)
        
        else:
            nnl = ELU(layers[-1].out_type)
            layers.append(nnl)
        
        if invariant_map:
            opool = GroupPooling(layers[-1].out_type)
            layers.append(opool)
            
            if pooling is not None:
                pl = PointwiseMaxPool(layers[-1].out_type, pooling)
                layers.append(pl)
        
        elif len(irreps) == 0 and pooling is not None:
            pl = PointwiseMaxPool(layers[-1].out_type, pooling)
            layers.append(pl)
        elif pooling is not None:
            modules = []
            modules += [(PointwiseMaxPool(trivials, pooling), "trivial")]
            modules += [(NormMaxPool(others, pooling), "others")]
            
            pl = MultipleModule(layers[-1].out_type, labels, modules)
            layers.append(pl)
        
        return layers

    def build_layer_inducedgated_normpool(self, r1: FieldType, C: int, s: int, padding: int = 0, pooling: int = None, invariant_map: bool = False):
        gc = r1.gspace
        assert isinstance(gc, FlipRot2dOnR2)

        ############################################################
        # 1 gate per irrep (except trivial); ELU on trivial irreps
        ############################################################
    
        gc = r1.gspace
        
        subgroup_id = (None, gc.fibergroup.rotation_order)
        sg, _, _ = gc.restrict(subgroup_id)

        layers = []

        irreps = []
        for n, irr in sg.fibergroup.irreps.items():
            if not irr.is_trivial():
                irreps += [gc.induced_repr(subgroup_id, irr)]

        irreps = list(irreps)

        induced_trivial = gc.induced_repr(subgroup_id, sg.trivial_repr)

        # number of fields requiring a gate
        I = len(irreps)
    
        # size of all irreps (including trivial)
        S = FieldType(gc, irreps).size + induced_trivial.size
    
        if self.fix_param and not invariant_map and self.LAYER > 1:
            # to keep number of parameters more or less constant when changing groups
            # (more precisely, we try to keep them close to the number of parameters in the original SFCNN)
            r_in = FieldType(gc, [induced_trivial] * (I + 1) + irreps)
            r_out = FieldType(gc, [induced_trivial] + irreps)
        
            tmp_cl = R2Conv(r_in, r_out, s,
                            frequencies_cutoff=self.frequencies_cutoff,
                            padding=padding,
                            sigma=self.sigma,
                            maximum_offset=self.J)
        
            t = tmp_cl.basisexpansion.dimension()
        
            t /= 16 * s ** 2 * 3 / 4
        
            C = int(round(C / np.sqrt(t)))
    
        elif invariant_map:
            # in order to preserve the same number of output channels
            size = I + 1
            C = int(round(C / size))
    
        layers = []
    
        # total channels =   C * S    +   1 * C * I     = C * M
        # i.e.:            (fields) + (gates for non-trivial fields)
    
        trivials = FieldType(gc, [induced_trivial] * C)
        gates = FieldType(gc, [induced_trivial] * C * I)
        gated = FieldType(gc, irreps * C).sorted()
        gate = gates + gated
    
        r2 = trivials + gate
    
        cl = R2Conv(r1, r2, s,
                    frequencies_cutoff=self.frequencies_cutoff,
                    padding=padding,
                    sigma=self.sigma,
                    maximum_offset=self.J)
        layers.append(cl)
    
        labels = ["trivial"] * (len(trivials) + len(gates)) + ["gated"] * len(gated)
    
        modules = [
            (InnerBatchNorm(trivials + gates), "trivial"),
            (InducedNormBatchNorm(gated), "gated")
        ]
        bn = MultipleModule(layers[-1].out_type, labels, modules)
        layers.append(bn)
    
        labels = ["trivial"] * len(trivials) + ["gate"] * len(gate)
        modules = [
            (ELU(trivials), "trivial"),
            (InducedGatedNonLinearity1(gate), "gate")
        ]
        nnl = MultipleModule(layers[-1].out_type, labels, modules)
        layers.append(nnl)
    
        if self.restrict == self.LAYER:
            layers.append(RestrictionModule(layers[-1].out_type, self.sgid))
            layers.append(DisentangleModule(layers[-1].out_type))

        def only_zero_freq(repr: Representation):
            for irr in repr.irreps:
                if repr.group.irreps[irr].attributes['frequency'] != 0:
                    return False
            return True

        if len(irreps) > 0:
            r3 = layers[-1].out_type
            labels = ["trivial" if only_zero_freq(r) else "others" for r in r3]
            r3 = r3.group_by_labels(labels)
            trivials = r3["trivial"]
            others = r3["others"]
        else:
            trivials = layers[-1].out_type
            others = None

        assert all([only_zero_freq(r) for r in trivials])

        if invariant_map:
            modules = [
                (GroupPooling(trivials), "trivial"),
                (InducedNormPool(others), "others")
            ]
            pl = MultipleModule(layers[-1].out_type, labels, modules)
            layers.append(pl)
        
            if pooling is not None:
                pl = PointwiseMaxPool(layers[-1].out_type, pooling)
                layers.append(pl)
        elif pooling is not None:
            modules = [
                (PointwiseMaxPool(trivials, pooling), "trivial"),
                (NormMaxPool(others, pooling), "others")
            ]
            pl = MultipleModule(layers[-1].out_type, labels, modules)
            layers.append(pl)
    
        return layers

    def build_layer_inducedgated_conv2triv(self, r1: FieldType, C: int, s: int, padding: int = 0, pooling: int = None,
                                           invariant_map: bool = False):
    
        ############################################################
        # 1 gate per irrep (except trivial); ELU on trivial irreps
        # conv 2 triv in the end
        ############################################################
    
        gc = r1.gspace
        assert isinstance(gc, FlipRot2dOnR2)
        
        subgroup_id = (None, gc.fibergroup.rotation_order)
        sg, _, _ = gc.restrict(subgroup_id)

        layers = []

        irreps = []
        if not invariant_map:
            for n, irr in sg.fibergroup.irreps.items():
                if not irr.is_trivial():
                    irreps += [gc.induced_repr(subgroup_id, irr)]

        induced_trivial = gc.induced_repr(subgroup_id, sg.trivial_repr)

        # number of fields requiring a gate
        I = len(irreps)

        # size of all irreps (including trivial)
        S = sum(r.size for r in irreps) + induced_trivial.size

        if self.fix_param and not invariant_map and self.LAYER > 1:
            # to keep number of parameters more or less constant when changing groups
            # (more precisely, we try to keep them close to the number of parameters in the original SFCNN)
            r_in = FieldType(gc, [induced_trivial] * (I + 1) + irreps)
            r_out = FieldType(gc, [induced_trivial] + irreps)
        
            tmp_cl = R2Conv(r_in, r_out, s,
                            frequencies_cutoff=self.frequencies_cutoff,
                            padding=padding,
                            sigma=self.sigma,
                            maximum_offset=self.J)
        
            t = tmp_cl.basisexpansion.dimension()
        
            t /= 16 * s ** 2 * 3 / 4
        
            C = int(round(C / np.sqrt(t)))
    
        layers = []
    
        # total channels =   C * S    +   1 * C * I     = C * M
        # i.e.:            (fields) + (gates for non-trivial fields)
    
        trivials = FieldType(gc, [induced_trivial] * C)
        if len(irreps) > 0:
            gates = FieldType(gc, [induced_trivial] * C * I)
            gated = FieldType(gc, irreps * C).sorted()
            gate = gates + gated
        
            r2 = trivials + gate
        else:
            gates = gated = gate = []
            r2 = trivials
    
        cl = R2Conv(r1, r2, s,
                    frequencies_cutoff=self.frequencies_cutoff,
                    padding=padding,
                    sigma=self.sigma,
                    maximum_offset=self.J)
        layers.append(cl)
    
        if len(irreps) > 0:
            labels = ["trivial"] * (len(trivials) + len(gates)) + ["gated"] * len(gated)
        
            modules = [
                (InnerBatchNorm(trivials + gates), "trivial"),
                (InducedNormBatchNorm(gated), "gated")
            ]
            bn = MultipleModule(layers[-1].out_type, labels, modules)
        else:
            bn = InnerBatchNorm(trivials)
            
        layers.append(bn)
    
        if len(irreps) > 0:
            labels = ["trivial"] * len(trivials) + ["gate"] * len(gate)
            modules = [
                (ELU(trivials), "trivial"),
                (InducedGatedNonLinearity1(gate), "gate")
            ]
            nnl = MultipleModule(layers[-1].out_type, labels, modules)
        else:
            nnl = ELU(trivials)
        layers.append(nnl)
    
        if self.restrict == self.LAYER:
            layers.append(RestrictionModule(layers[-1].out_type, self.sgid))
            layers.append(DisentangleModule(layers[-1].out_type))
            
        def only_zero_freq(repr: Representation):
            for irr in repr.irreps:
                if repr.group.irreps[irr].attributes['frequency'] != 0:
                    return False
            return True

        if len(irreps) > 0:
            r3 = layers[-1].out_type
            labels = ["trivial" if only_zero_freq(r) else "others" for r in r3]
            r3 = r3.group_by_labels(labels)
            trivials = r3["trivial"]
            others = r3["others"]
        else:
            trivials = layers[-1].out_type
            others = []

        assert all([only_zero_freq(r) for r in trivials])

        if invariant_map:
            
            pl = GroupPooling(trivials)
            layers.append(pl)
        
            if pooling is not None:
                pl = PointwiseMaxPool(layers[-1].out_type, pooling)
                layers.append(pl)
                
        elif pooling is not None:
            if len(others) > 0:
                modules = [
                    (PointwiseMaxPool(trivials, pooling), "trivial"),
                    (NormMaxPool(others, pooling), "others")
                ]
                pl = MultipleModule(layers[-1].out_type, labels, modules)
            else:
                pl = PointwiseMaxPool(trivials, pooling)
            layers.append(pl)
    
        return layers
    
    def build_layer_debug(self, r1: FieldType, C: int, s: int, padding: int = 0, pooling: int = None,
                          invariant_map: bool = False):
        
        gc = r1.gspace
        
        C /= 4
        
        if self.fix_param and not invariant_map and self.LAYER > 1:
            t = gc.fibergroup.order() / 16
            C = C / np.sqrt(t)
        
        C = int(round(C))
        
        layers = []
        
        r2 = FieldType(gc, [gc.representations['regular']] * C)
        
        cl = R2Conv(r1, r2, s,
                    padding=padding,
                    frequencies_cutoff=self.frequencies_cutoff,
                    sigma=self.sigma,
                    maximum_offset=self.J
                    )
        layers.append(cl)
        
        if self.restrict == self.LAYER:
            layers.append(RestrictionModule(layers[-1].out_type, self.sgid))
            layers.append(DisentangleModule(layers[-1].out_type))
        
        if invariant_map:
            pl = GroupPooling(layers[-1].out_type)
            layers.append(pl)
        
        nnl = ReLU(layers[-1].out_type, inplace=False)
        layers.append(nnl)
        
        if pooling is not None:
            pl = PointwiseMaxPool(layers[-1].out_type, pooling)
            layers.append(pl)
        
        return layers

