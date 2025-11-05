# -*- coding: utf-8 -*-
# Flows.py : file containing flow base class
# Author: Juan Maroñas and  Ollie Hamelijnck
# Modification: by Zhidi Lin
# more complex version compared to 'utils_flow_demo.py'

# Libraries
import sys
sys.path.append('../../')
import numpy
import numpy as np
import matplotlib.pyplot as plt
import typing
import torch
import torch.nn as nn
from torch.nn.functional import softplus
import gpytorch
from gpytorch.utils.transforms import inv_softplus
import src.utils as cg
cg.reset_seed(0)

def SAL(num_blocks):
    block_array = []
    for nb in range(num_blocks):
        a_aff, b_aff = 1.0, 0.0
        a_sal, b_sal = 0.0, 1.0

        init_affine = {'init_a': a_aff, 'init_b': b_aff, 'set_restrictions': False}
        init_sinh_arcsinh = {'init_a': a_sal, 'init_b': b_sal, 'add_init_f0': False, 'set_restrictions': False}
        block = [('sinh_arcsinh', init_sinh_arcsinh), ('affine', init_affine)]
        block_array.extend(block)
    return block_array

def TANH(num_blocks):
    block_array = []
    for nb in range(num_blocks):
        a, b, c, d = 0.0, 1.0, 0.0, 1.0
        init_tanh = {'init_a': a, 'init_b': b, 'init_c': c, 'init_d': d, 'add_init_f0': False, 'set_restrictions': False}
        block = [('tanh', init_tanh)]
        block_array.extend(block)
    return block_array

def SATANH(num_blocks):
    block_array = []
    for nb in range(num_blocks):
        a, b, c, d = 0.0, 1.0, 0.0, 1.0
        a_sal, b_sal = 0.0, 1.0

        init_sinh_arcsinh = {'init_a': a_sal, 'init_b': b_sal, 'add_init_f0': False, 'set_restrictions': False}
        init_tanh = {'init_a': a, 'init_b': b, 'init_c': c, 'init_d': d, 'add_init_f0': False, 'set_restrictions': False}
        block = [('sinh_arcsinh', init_sinh_arcsinh), ('tanh', init_tanh)]
        block_array.extend(block)
    return block_array

# ====== BoxCox FLow Generator ====== #
## BoxCox + affine
def BoxCoxL(num_blocks,**kwargs):
    set_res=False
    addf0=False
    init_random=True
    constraint=None

    block_array = []
    for nb in range(num_blocks):
        if init_random:
            a_aff,b_aff = numpy.random.randn(2)
            init_lam    = numpy.random.randn(1)+1. # center it around 1.0 which is the one that makes the flow the identity
            constraint  = None
        else:
            a_aff,b_aff = 1.0,0.0
            init_lam    = 5.0 # if applied the constraint below makes the flow the identity
            #def constraint(lam):
            #    lam = 2*torch.sigmoid(0.3*lam-1.5)+0.05
            #    if lam == 0:
            #        lam = lam + 1e-11
            #        return lam

        init_bc = {'init_lam': init_lam, 'add_init_f0': addf0 , 'constraint': constraint}
        init_affine = {'init_a':a_aff, 'init_b': b_aff, 'set_restrictions': set_res}

        block = [ ('boxcox',init_bc), ('affine',init_affine)  ]
        block_array.extend(block) # note that we append instead of extend because we will add each block per add_fow step

    return block_array




#####################  ###################  ########################

def initialize_flows(flow_specs):
    """ Initializes the flows applied on the prior . Flow_specs is a list with an instance of the flow used per output GP.
    """
    G_matrix = []
    for idx, fl in enumerate(flow_specs):
        G_matrix.append(fl)
    G_matrix = nn.ModuleList(G_matrix)
    return G_matrix

# DEFINE SOME UTILITIES
def instance_flow(flow_list, is_composite=True):
    ''' From these flows only Box-Cox, sinh-arcsinh and affine return to the identity '''
    FL = []
    for flow_name in flow_list:
        flow_name, init_values = flow_name
        if flow_name == 'arcsinh':
            fl = ArcsinhFlow(**init_values)
        elif flow_name == 'inverse_arcsinh':
            fl = InverseArchsinhFlow(**init_values)
        elif flow_name == 'sinh':
            fl = SinhFlow(**init_values)
        elif flow_name == 'normalCDF':
            fl = NormalCDFFlow(**init_values)
        elif flow_name == 'exp':
            fl = ExpFlow()
        elif flow_name == 'affine':
            fl = AffineFlow(**init_values)
        elif flow_name == 'boxcox':
            fl = BoxCoxFlow(**init_values)
        elif flow_name == 'inverseboxcox' or flow_name == 'inverse_boxcox':
            fl = InverseBoxCoxFlow(**init_values)
        elif flow_name == 'sinh_arcsinh':
            fl = Sinh_ArcsinhFlow(**init_values)
        elif flow_name == 'inverse_sinh_arcsinh':
            fl = Inverse_Sinh_ArcsinhFlow(**init_values)
        elif flow_name == 'identity':
            fl = IdentityFlow()
        elif flow_name == 'tanh':
            fl = TanhFlow(**init_values)
        elif flow_name == 'log_exp':
            fl = LogExpFlow(**init_values)
        elif flow_name == 'step_flow':
            fl = StepFlow(**init_values)
        elif flow_name == 'tukey_left':
            fl = TukeyLeftFlow(**init_values)
        elif flow_name == 'tukey_right':
            fl = TukeyRightFlow(**init_values)
        else:
            raise ValueError("Unkown flow identifier {}".format(flow_name))

        FL.append(fl)

    if is_composite:
        return CompositeFlow(FL)
    return FL


class Flow(nn.Module):
    """ General Flow Class.
        All flows should inherit and overwrite this method
    """

    def __init__(self) -> None:
        super(Flow, self).__init__()

    def forward(self, f0: torch.tensor, X: torch.tensor = None) -> torch.tensor:
        raise NotImplementedError("Not Implemented")

    def forward_initializer(self, X):
        # just return 0 if it is not needed
        raise NotImplementedError("Not Implemented")


class CompositeFlow(Flow):
    def __init__(self, flow_arr: list) -> None:
        """
            Args:
                flow_arr: is an array of flows. The first element is the first flow applied.
        """
        super(CompositeFlow, self).__init__()
        self.flow_arr = nn.ModuleList(flow_arr)

    def forward(self, f: torch.tensor, X: torch.tensor = None) -> torch.tensor:
        for flow in self.flow_arr:
            f = flow.forward(f, X)
        return f

    def forward_initializer(self, X: torch.tensor):
        loss = 0.0
        for flow in self.flow_arr:
            loss += flow.forward_initializer(X)
        return loss


class InverseFlow(Flow):
    def __init__(self, flow: Flow) -> None:
        super(InverseFlow, self).__init__()

        self.flow = flow

    def forward(self, f: torch.tensor) -> torch.tensor:
        return self.flow.inverse(f)

    def inverse(self, f: torch.tensor) -> torch.tensor:
        return self.flow.forward(f)


class LogExpFlow(Flow):
    """
        Positive Forcing Flow from Copula Process paper
        fk = ∑ ak log (exp(b(f + c)) + 1)
    """

    def __init__(self, init_a: np.ndarray, init_b: np.ndarray, init_c: np.ndarray) -> None:
        """
            Args:
                init_a: k array
                init_b: k array
                init_c: k array
        """
        super(LogExpFlow, self).__init__()

        self.K = init_a.shape[0]

        self.a = nn.Parameter(torch.tensor(init_a, dtype=cg.dtype))
        self.b = nn.Parameter(torch.tensor(init_b, dtype=cg.dtype))
        self.c = nn.Parameter(torch.tensor(init_c, dtype=cg.dtype))

    def forward(self, f0: torch.tensor, X: torch.tensor = None) -> torch.tensor:
        res = 0.0

        for k in range(self.K):
            a = softplus(self.a[k])
            b = softplus(self.b[k])
            c = self.c[k]

            res += a * torch.log(torch.exp(b * (f0 + c)) + 1)

        return res


class SoftminusFlow(Flow):
    def __init__(self, set_restrictions=False) -> None:
        """
            Args:
                init_a: k array
                init_b: k array
                init_c: k array
        """
        super(SoftminusFlow, self).__init__()
        self.softplus = torch.nn.Softplus()
        self.set_restrictions = False

    def forward(self, f0: torch.tensor, X: torch.tensor = None) -> torch.tensor:
        # soft minus function
        return gpytorch.utils.transforms.inv_softplus(f0 + 1e-8)

    def inverse(self, f: torch.tensor) -> torch.tensor:
        # soft plus
        return self.softplus(f + 1e-8)


class SoftplusFlow(Flow):
    def __init__(self, set_restrictions=False) -> None:
        """
            Args:
                init_a: k array
                init_b: k array
                init_c: k array
        """
        super(SoftplusFlow, self).__init__()
        self.softplus = torch.nn.Softplus(beta=1)
        self.set_restrictions = False

    def forward(self, f0: torch.tensor, X: torch.tensor = None) -> torch.tensor:
        # soft plus
        return self.softplus(f0)

    def inverse(self, f: torch.tensor) -> torch.tensor:
        # soft minus function
        return gpytorch.utils.transforms.inv_softplus(f + 1e-8)


class ExpFlow(Flow):
    """ Exponential Flow
          fk = exp(f0)
    """

    def __init__(self) -> None:
        super(ExpFlow, self).__init__()

    def forward(self, f0: torch.tensor, X: torch.tensor = None) -> torch.tensor:
        return torch.exp(f0)

    def inverse(self, f: torch.tensor) -> torch.tensor:
        return torch.log(f)


class IdentityFlow(Flow):
    """ Identity Flow
           fk = f0
    """

    def __init__(self) -> None:
        super(IdentityFlow, self).__init__()

    def forward(self, f0: torch.tensor, X: torch.tensor = None) -> torch.tensor:
        return f0

    def inverse(self, f: torch.tensor) -> torch.tensor:
        return f


class AffineFlow(Flow):
    def __init__(self, init_a: float, init_b: float, set_restrictions: bool) -> None:
        ''' Affine Flow
            fk = a*f0+b
            * recovers the identity for a = 1 b = 0
            * a has to be strictly possitive to ensure invertibility if this flow is used in a linear
            combination, i.e with the step flow
            Args:
                a                (float) :->: Initial value for the slope
                b                (float) :->: Initial value for the bias
                set_restrictions (bool)  :->: If true then a >= 0 using  a = softplus(a)
        '''
        super(AffineFlow, self).__init__()
        self.a = nn.Parameter(torch.tensor(init_a, dtype=cg.dtype))
        self.b = nn.Parameter(torch.tensor(init_b, dtype=cg.dtype))

        self.set_restrictions = set_restrictions

    def forward(self, f0: torch.tensor, X: torch.tensor = None) -> torch.tensor:
        a = self.a
        if self.set_restrictions:
            a = softplus(a)
        b = self.b
        return a * f0 + b


class TranslationFlow(Flow):
    def __init__(self, init_b: float) -> None:
        super(TranslationFlow, self).__init__()
        self.b = nn.Parameter(torch.tensor(init_b, dtype=cg.dtype))

    def forward(self, f0: torch.tensor,  X: torch.tensor = None) -> torch.tensor:
        b = self.b
        return f0 + b

    def inverse(self, f: torch.tensor) -> torch.tensor:
        b = self.b
        return (f - b)


class BoxCoxFlow(Flow):
    def __init__(self, init_lam: float, add_init_f0: bool, constraint: typing.Callable = None) -> None:
        ''' Box Cox Flow
            fk = (sgn(f0)|f0|^lam - 1)/lam
            * lam > 0 to ensure invertibility. We restrict it to a range where the box cox is well behaved
            * Recovers linear function for lam = 1 although it has a negative bias of -1
                f(x,lam=1) = x-1
            Args:
                 init_value  (float) :->: Initial value for lam, will be passed through a sigmoid
                 add_init_f0 (bool)  :->: if true then fk = f0 + a + b*arcsinh((f0-c)/d)
                 constraint (callable) : if None then resort to default transormation. Else use passed callable.
        '''
        super(BoxCoxFlow, self).__init__()
        self.lam = nn.Parameter(torch.tensor(init_lam, dtype=cg.dtype))
        self.add_init_f0 = add_init_f0
        self.constraint = constraint

    def transform_param(self):
        ''' Parameter restricted to a friendly/stable range
        '''
        if self.constraint is None:
            lam = self.lam
            if lam == 0:
                lam = lam + 1e-11
        else:
            lam = self.constraint(self.lam)

        assert lam != 0, "Invalid value for Box Cox parameter. This flow is not defined for values of lam == 0"

        return lam

    def forward(self, f0: torch.tensor, X: torch.tensor = None) -> torch.tensor:
        lam = self.transform_param()
        sgn = torch.sign(f0)

        pos = sgn * f0
        fk = (sgn * torch.pow(pos, lam) - 1) / lam
        if self.add_init_f0:
            return fk + f0
        return fk

    def inverse(self, f: torch.tensor) -> torch.tensor:
        pass
        # lam = self.transform_param()
        # return torch.pow( f*lam+1, 1./lam )


class InverseBoxCoxFlow(BoxCoxFlow):
    def __init__(self, init_lam: float, add_init_f0: bool, constraint: typing.Callable = None) -> None:
        '''Inverse Box Cox Flow
            fk = (sgn(lam*f0+1)|lam*f0+1|^(1/lam)
            * lam > 0 to ensure invertibility. We restrict it to a range where the box cox is well behaved
            * Recovers linear function for lam = 1 although it has a possitive bias of +1
                f(x,lam=1) = x+1
            Args:
                 init_value  (float) :->: Initial value for lam, will be passed through a sigmoid
                 add_init_f0 (bool)  :->: if true then fk = f0 + a + b*arcsinh((f0-c)/d)
                 constraint (callable) : if None then resort to default transormation. Else use passed callable.
        '''
        super(InverseBoxCoxFlow, self).__init__(init_lam, add_init_f0, constraint)

    def forward(self, f0: torch.tensor, X: torch.tensor = None) -> torch.tensor:
        lam = self.transform_param()
        aux = lam * f0 + 1
        sgn = torch.sign(aux)

        pos = sgn * aux
        fk = (sgn * torch.pow(pos, 1. / lam))
        if self.add_init_f0:
            return fk + f0
        return fk


class TukeyFlow(Flow):
    def __init__(self, init_g: float, init_h: float, add_init_f0: bool) -> None:
        """
            Tukey Transfomation:
                T(f) = (1/g) (exp(gf) - 1)exp(g f^2/2)
            where
                 g!= 0
                 h >= 0
        """

        super(TukeyFlow, self).__init__()
        self.g = nn.Parameter(torch.tensor(init_g, dtype=cg.dtype))
        self.h = nn.Parameter(torch.tensor(init_h, dtype=cg.dtype))

        self.add_init_f0 = add_init_f0

    def get_g_h(self):
        g = self.g
        h = softplus(self.h)
        return g, h

    def forward(self, f: torch.tensor, X: torch.tensor = None) -> torch.tensor:
        g, h = self.get_g_h()
        return (1 / g) * (torch.exp(g * f) - 1) * torch.exp((h * (f ** 2)) / 2)


class TukeyRightFlow(TukeyFlow):
    """
        Forces g to be positive, implies a right skewed distribution
    """

    def get_g_h(self):
        g = softplus(self.g)
        h = softplus(self.h)
        return g, h


class TukeyLeftFlow(TukeyFlow):
    """
        Forces g to be negative and implies a left skewed distribution
    """

    def get_g_h(self):
        g = -softplus(self.g)
        h = softplus(self.h)
        return g, h


class ArcsinhFlow(Flow):
    def __init__(self, init_a: float, init_b: float, init_c: float, init_d: float, add_init_f0: bool,
                 set_restrictions: bool) -> None:
        ''' Arcsinh Flow
          fk = a + b*arcsinh((f-c)/d)
        * Smooth flow, but does not recover the identity
        * b and d has to be strictly possitive/negative to ensure invertibility when using this flow in a
          linear combination.
        * b and d has to be strictly possitive to recover the Johnson’sSU-distribution

            Args:
                   init_a           (float) :->: initial value for a
                   init_b           (float) :->: initial value for b
                   init_c           (float) :->: initial value for c
                   init_d           (float) :->: initial value for d
                   set_restrictions (bool)  :->: if true then b,d > 0 with b = softplus(b)
                   add_init_f0      (bool)  :->: if true then fk = f0 + a + b*arcsinh((f0-c)/d)
                                                 If true then set_restrictions = True
        '''
        super(ArcsinhFlow, self).__init__()
        self.a = nn.Parameter(torch.tensor(init_a, dtype=cg.dtype))
        self.b = nn.Parameter(torch.tensor(init_b, dtype=cg.dtype))
        self.c = nn.Parameter(torch.tensor(init_c, dtype=cg.dtype))
        self.d = nn.Parameter(torch.tensor(init_d, dtype=cg.dtype))
        if add_init_f0:
            set_restrictions = True
        self.set_restrictions = set_restrictions
        self.add_init_f0 = add_init_f0

    def asinh(self, f: torch.tensor) -> torch.tensor:
        return torch.log(f + (f ** 2 + 1) ** (0.5))

    def forward(self, f0: torch.tensor, X: torch.tensor = None) -> torch.tensor:
        a = self.a
        c = self.c
        d = self.d
        b = self.b
        if self.set_restrictions:
            d = softplus(d)
            b = softplus(b)
        fk = a + b * self.asinh((f0 - c) / d)

        if self.add_init_f0:
            return fk + f0
        return fk

    def _forward_grad(self, x):
        a = self.a
        c = self.c
        d = self.d
        b = self.b
        if self.set_restrictions:
            d = softplus(d)
            b = softplus(b)

        return b * torch.cosh(b * self.asinh(x) - a) / torch.sqrt(1 + x ** 2)

    def inverse(self, f: torch.tensor) -> torch.tensor:
        b = self.b
        d = self.d
        if self.set_restrictions:
            b = softplus(self.b)
            d = softplus(self.d)
        return self.c + d * torch.sinh((f - self.a) / b)


class InverseArchsinhFlow(ArcsinhFlow):
    def forward(self, f: torch.tensor) -> torch.tensor:
        return super(InverseArchsinhFlow, self).inverse(f)

    def inverse(self, f: torch.tensor) -> torch.tensor:
        return super(InverseArchsinhFlow, self).forward(f)


class SinhFlow(Flow):
    def __init__(self, init_a: float, init_b: float, init_c: float, init_d: float, add_init_f0: bool,
                 set_restrictions: bool) -> None:
        ''' Sinh Flow
          fk = a + b*sinh((f-c)/d)
        * Is the inverse of the arcsinh flow (for appropiate values of a,b,c,d). However as we do not use the inverse
          and only require flows in one direction this functional form might be more interesting.
        * Smooth flow, but does not recover the identity
        * b and d has to be strictly possitive/negative to ensure invertibility when using this flow in a
          linear combination.
        * b and d has to be strictly possitive to recover the Johnson’sSU-distribution

            Args:
                   init_a           (float) :->: initial value for a
                   init_b           (float) :->: initial value for b
                   init_c           (float) :->: initial value for c
                   init_d           (float) :->: initial value for d
                   set_restrictions (bool)  :->: if true then b,d > 0 with b = softplus(b)
                   add_init_f0      (bool)  :->: if true then fk = f0 + a + b*arcsinh((f0-c)/d)
                                                 If true then set_restrictions = True
        '''
        super(SinhFlow, self).__init__()
        self.a = nn.Parameter(torch.tensor(init_a, dtype=cg.dtype))
        self.b = nn.Parameter(torch.tensor(init_b, dtype=cg.dtype))
        self.c = nn.Parameter(torch.tensor(init_c, dtype=cg.dtype))
        self.d = nn.Parameter(torch.tensor(init_d, dtype=cg.dtype))
        if add_init_f0:
            set_restrictions = True
        self.set_restrictions = set_restrictions
        self.add_init_f0 = add_init_f0

    def forward(self, f0: torch.tensor, X: torch.tensor = None) -> torch.tensor:
        a = self.a
        c = self.c
        d = self.d
        b = self.b
        if self.set_restrictions:
            d = softplus(d)
            b = softplus(b)
        fk = a + b * torch.sinh((f0 - c) / d)

        if self.add_init_f0:
            return fk + f0
        return fk

    def inverse(self, f: torch.tensor) -> torch.tensor:
        pass
        # b = softplus(self.b)
        # d = softplus(self.d)
        # return self.c+d*torch.sinh((f-self.a)/b)


class TanhFlow(Flow):
    def __init__(self, init_a: float, init_b: float, init_c: float, init_d: float, add_init_f0: bool,
                 set_restrictions: bool) -> None:
        ''' Tanh Flow
         fk = a + b*tanh((f0-c)/d)
        * Smooth flow, but does not recover the identity
        *  b and d has to be strictly possitive/negative to ensure invertibility when using this flow in a
          linear combination.
        * Similar behaviour to arcsinh flow though more restrictive. Same flow changing non linearity
        * The tanhflow receives an additional option add_init_f0 specifying if fk += f0 so that the range of the transformation
          is (-inf,inf) rather than (-1,1). Should be set to false when used in the step_flow
        * This is the base function used in the WGP paper. That paper performs a linear combination of this base
          transformation. We add a bias self.a

          Args:
                init_a           (float) :->: initial value for a
                init_b           (float) :->: initial value for b
                init_c           (float) :->: initial value for c
                init_d           (float) :->: initial value for d
                set_restrictions (bool)  :->: if true then b,d > 0 with b = softplus(b)
                add_init_f0      (bool)  :->: if true then fk = f0 + a + b*tanh((f0-c)/d).
                                              If true then set_restrictions = True
        '''
        super(Flow, self).__init__()

        self.a = nn.Parameter(torch.tensor(init_a, dtype=cg.dtype))
        self.b = nn.Parameter(torch.tensor(init_b, dtype=cg.dtype))
        self.c = nn.Parameter(torch.tensor(init_c, dtype=cg.dtype))
        self.d = nn.Parameter(torch.tensor(init_d, dtype=cg.dtype))

        if add_init_f0:
            set_restrictions = True
        self.set_restrictions = set_restrictions
        self.add_init_f0 = add_init_f0

    def forward(self, f0: torch.tensor, X: torch.tensor = None) -> torch.tensor:
        # assert self.is_initialized, "This flow hasnt been initialized. Either set self.is_initialized = False or use an initializer"
        a = self.a
        c = self.c
        d = self.d
        b = self.b

        if self.set_restrictions:
            b = softplus(b)
            d = softplus(d)

        fk = a + b * torch.tanh((f0 - c) / d)
        if self.add_init_f0:
            return fk + f0

        return fk

    def inverse(self, f: torch.tensor) -> torch.tensor:
        raise NotImplementedError("Not Implemented")

    def forward_initializer(self, X):
        return 0.0


class Sinh_ArcsinhFlow(Flow):
    def __init__(self, init_a: float, init_b: float, add_init_f0: bool, set_restrictions: bool) -> None:
        ''' SinhArcsinh Flow
          fk = sinh( b*arcsinh(f) - a)
          * b has to be strictkly possitive when used in a linear combination so that function is invertible.
          * Recovers the identity function

          Args:
                 init_a           (float) :->: initial value for a. Only used if input_dependent = False. Also used by the initializer if input_dependent = True so
                                               that NNets parameters are matched to take this value.
                 init_b           (float) :->: initial value for b. Only used if input_dependent = False. Also used by the initializer if input_dependent = True so
                                               that NNets parameters are matched to take this value.
                 set_restrictions (bool)  :->: if true then b > 0 with b = softplus(b)
                 add_init_f0      (bool)  :->: if true then fk = f0 + sinh( b*arcsinh(f) - a)
                                               If true then set_restrictions = True
        '''
        super(Sinh_ArcsinhFlow, self).__init__()

        self.a = nn.Parameter(torch.tensor(init_a, dtype=cg.dtype))
        self.b = nn.Parameter(torch.tensor(init_b, dtype=cg.dtype))

        if add_init_f0:
            set_restrictions = True
        self.set_restrictions = set_restrictions
        self.add_init_f0 = add_init_f0

    def asinh(self, f: torch.tensor) -> torch.tensor:
        return torch.log(f + (f ** 2 + 1) ** (0.5))

    def forward_initializer(self, X):
        return 0.0

    def forward(self, f0: torch.tensor, X: torch.tensor = None) -> torch.tensor:
        a = self.a
        b = self.b
        if self.set_restrictions:
            b = softplus(b)
        fk = torch.sinh(b * self.asinh(f0) - a)

        if self.add_init_f0:
            return fk + f0

        return fk

    def inverse(self, f: torch.tensor) -> torch.tensor:
        a = self.a
        b = self.b
        if self.set_restrictions:
            b = softplus(b)

        return torch.sinh(1 / b * (self.asinh(f) + self.a))


class Inverse_Sinh_ArcsinhFlow(Sinh_ArcsinhFlow):
    def forward(self, f0: torch.tensor) -> torch.tensor:
        return super(Inverse_Sinh_ArcsinhFlow, self).inverse(f0)

    def inverse(self, f: torch.tensor) -> torch.tensor:
        return super(Inverse_Sinh_ArcsinhFlow, self).forward(f)


class NormalCDFFlow(Flow):
    """ Normal CDF flow
          fk = \Phi(f0)
    """

    def __init__(self, init_a: float, init_b: float, init_c: float, init_d: float, add_init_f0: bool,
                 set_restrictions: bool, is_learnable: bool) -> None:
        super(NormalCDFFlow, self).__init__()
        self.SN = torch.distributions.normal.Normal(0, 1)
        self.is_learnable = is_learnable
        if add_init_f0:
            set_restrictions = True
        self.set_restrictions = set_restrictions
        self.add_init_f0 = add_init_f0
        if self.is_learnable:
            self.a = nn.Parameter(torch.tensor(init_a, dtype=cg.dtype))
            self.b = nn.Parameter(torch.tensor(init_b, dtype=cg.dtype))
            self.c = nn.Parameter(torch.tensor(init_c, dtype=cg.dtype))
            self.d = nn.Parameter(torch.tensor(init_d, dtype=cg.dtype))

    def forward(self, f0: torch.tensor, X: torch.tensor = None) -> torch.tensor:
        a, c = 0.0, 0.0
        b, d = 1.0, 1.0
        if self.is_learnable:
            a, b, c, d = self.a, self.b, self.c, self.d
            if self.set_restrictions:
                b = softplus(b)
                d = softplus(d)

        fk = a + b * self.SN.cdf((f0 - c) / d)

        if self.add_init_f0:
            return fk + f0
        return fk


class StepFlow(Flow):
    def __init__(self, flow_arr: list, add_init_f0: bool) -> None:
        ''' Implements a linear combination of K flows (similar to WGP paper).
         fk = \sum_{i=1}^K [c_i g(f_0) + b_i] c_i>0
         * This module adds c_i when the original flow does not have it, i.e in boxcox and sinh_arcsinh. This parameters
         are learnable and can switch off elements from the linear transformation

         Args:
               flow_arr    (list) :->: contains a list where each element represent the elements of the linear combinations
                                       each of this flows must be strictly increasing function to ensure invertibility
               add_init_f0 (bool) :->:  If true then f0 is added: fk = f0 + \sum_{i=1}^K [c_i g(f_0)] c_i>0
        '''
        super(StepFlow, self).__init__()
        assert isinstance(add_init_f0, bool), "add_init_f0 must be boolean, got {}".format(type(flow_arr[-1]))

        self.add_init_f0 = add_init_f0
        self.switch_off = nn.ModuleList()

        # For those flows that don't have a bias and a scale we add a parameter so that the flow can be switched off from the
        # combination. These parameters are learnable
        n_steps = len(flow_arr)
        for step in range(n_steps):

            if (type(flow_arr[step]) == list) or (type(flow_arr[step]) == tuple):
                # flow has been passed as dictionary
                name, param_dict = flow_arr[step]
                if name != 'boxcox' and name != 'inverseboxcox':
                    set_restrictions = param_dict['set_restrictions']
            else:
                if type(flow_arr[step]) == BoxCoxFlow:
                    name = 'boxcox'
                elif type(flow_arr[step]) == Sinh_ArcsinhFlow:
                    name = 'sinh_arcsinh'
                elif type(flow_arr[step]) == TanhFlow:
                    name = 'tanh'

                set_restrictions = flow_arr[step].set_restrictions

            assert name != 'step_flow', "cannot combine step flow with step flow"
            assert name == 'boxcox' or name == 'inverseboxcox' or set_restrictions, "set_restrictions must be True. Got false for flow {}".format(
                name)

            rg = self.__requires_switch_off(name)
            self.switch_off.append(switch_off(rg, n_steps))

        # set_restrictions True so that we ensure that all the individual flows are stricly increasing/decreasing
        if (type(flow_arr[0]) == list) or (type(flow_arr[step]) == tuple):
            # flow has been passed as dictionary
            self.flow_arr = nn.ModuleList(instance_flow(flow_arr, is_composite=False))
        else:
            self.flow_arr = nn.ModuleList(flow_arr)

    def __requires_switch_off(self, name):
        if name == 'boxcox' or name == 'sinh_arcsinh' or name == 'inverseboxcox':
            return True
        return False

    def forward(self, f0: torch.tensor, X: torch.tensor = None) -> torch.tensor:
        fk = 0.0
        for sw_off, flow in zip(self.switch_off, self.flow_arr):
            a, b = sw_off()
            fk += a * flow.forward(f0, X=X) + b
        if self.add_init_f0:
            fk += f0
        return fk

    def forward_initializer(self, X):
        out = 0.0
        for sw_off, flow in zip(self.switch_off, self.flow_arr):
            a, b = sw_off()  # . NO need to ponderate by a or b. In this method we just one the parameters to match those from the initializer
            if isinstance(a, nn.Parameter) or isinstance(b, nn.Parameter):
                raise NotImplementedError(
                    "Double check the input dependent case in case these are input dependent parameters. ")
            out += flow.forward_initializer(X)
        return out

    def inverse(self, f: torch.tensor) -> torch.tensor:
        raise NotImplementedError("Not Implemented")


class switch_off(nn.Module):
    '''This is used by the step flow to switch off steps
       Have to create a separate model to properly register and keep track
    '''

    def __init__(self, is_trainable: bool, n_steps: int):
        super(switch_off, self).__init__()
        self.is_trainable = is_trainable
        if self.is_trainable:
            # the weight a is initialized to 1/number_steps so that if we perform a linear combination of lots of flows we do not easily saturate
            a = torch.tensor(1.0 / float(n_steps), dtype=cg.dtype)
            a = inv_softplus(
                a)  # as we apply the softplus we need to first compute the softminus so that at initialization the scale parameter a = 1/n_steps
            self.a = nn.Parameter(a)
            self.b = nn.Parameter(torch.tensor(0.0, dtype=cg.dtype))

    def forward(self):
        a, b = 1.0, 0.0
        if self.is_trainable:
            a = softplus(self.a)
            b = self.b
        return a, b
