# -*- coding: utf-8 -*-
# utils_flow_demo.py : file containing flow base class (simplified version)
# Author: Juan Maroñas
# Modification: by Zhidi Lin

import sys
import numpy
import torch
import torch.nn as nn
from torch.nn.functional import softplus
import gpytorch
from gpytorch.utils.transforms import inv_softplus
# custom
import utils as cg

def initialize_flows(flow_specs):
    """ Initializes the flows applied on the prior . Flow_specs is a list with an instance of the flow used per output GP.
    """
    G_matrix = []
    for idx, fl in enumerate(flow_specs):
        G_matrix.append(fl)
    G_matrix = nn.ModuleList(G_matrix)
    return G_matrix

def instance_flow(flow_list, is_composite=True):
    """
     From these flows only Box-Cox, sinh-arcsinh and affine return to the identity
    """
    FL = []
    for flow_name in flow_list:

        flow_name, init_values = flow_name

        if flow_name == 'affine':
            fl = AffineFlow(**init_values)

        elif flow_name == 'sinh_arcsinh':
            fl = Sinh_ArcsinhFlow(**init_values)

        elif flow_name == 'identity':
            fl = IdentityFlow()

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


class IdentityFlow(Flow):
    """ Identity Flow
           fk = f0
    """

    def __init__(self) -> None:
        super(IdentityFlow, self).__init__()
        self.input_dependent = False

    def forward(self, f0: torch.tensor, X: torch.tensor = None) -> torch.tensor:
        return f0


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

    def forward(self, f0: torch.tensor, X: torch.tensor = None) -> torch.tensor:
        # assert self.is_initialized, "This flow hasnt been initialized. Either set self.is_initialized = False or use an initializer"

        a = self.a
        b = self.b
        if self.set_restrictions:
            b = softplus(b)
        fk = torch.sinh(b * self.asinh(f0) - a)

        if self.add_init_f0:
            return fk + f0
        return fk

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

