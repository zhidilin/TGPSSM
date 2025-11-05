import torch
import torch.nn as nn
import numpy as np

class Flow(nn.Module):
    """
    Generic class for flow functions
    """

    def __init__(self):
        super().__init__()

    def forward(self, z):
        """
        :param z: input variable, first dimension is batch dim [...,  batch_dim x input_dim]
        :return: transformed z and log of absolute determinant
        """
        raise NotImplementedError("Forward pass has not been implemented.")

    def inverse(self, z):
        raise NotImplementedError("This flow has no algebraic inverse.")



class Split(Flow):
    """
    Split features into two sets
    """

    def __init__(self, mode="channel"):
        """
        Constructor
        :param mode: Splitting mode, can be
            channel: Splits first feature dimension, usually channels, into two halfs
            channel_inv: Same as channel, but with z1 and z2 flipped
            checkerboard: Splits features using a checkerboard pattern (last feature dimension must be even)
            checkerboard_inv: Same as checkerboard, but with inverted coloring
        """
        super().__init__()
        self.mode = mode

    def forward(self, z):
        if self.mode == "channel":
            z1, z2 = z.chunk(2, dim=-1)
        elif self.mode == "channel_inv":
            z2, z1 = z.chunk(2, dim=-1)
        else:
            raise NotImplementedError("Mode " + self.mode + " is not implemented.")
        log_det = 0
        return [z1, z2], log_det

    def inverse(self, z):
        z1, z2 = z
        if self.mode == "channel":
            z = torch.cat([z1, z2], dim=-1)
        elif self.mode == "channel_inv":
            z = torch.cat([z2, z1], dim=-1)
        else:
            raise NotImplementedError("Mode " + self.mode + " is not implemented.")
        log_det = 0
        return z, log_det


class Merge(Split):
    """
    Same as Split but with forward and backward pass interchanged
    """

    def __init__(self, mode="channel"):
        super().__init__(mode)

    def forward(self, z):
        return super().inverse(z)

    def inverse(self, z):
        return super().forward(z)



class Permute(Flow):
    """
    Permutation features along the channel dimension
    """

    def __init__(self, num_channels, mode="shuffle"):
        """
        Constructor
        :param num_channel: Number of channels
        :param mode: Mode of permuting features, can be shuffle for
        random permutation or swap for interchanging upper and lower part
        """
        super().__init__()
        self.mode = mode
        self.num_channels = num_channels
        if self.mode == "shuffle":
            perm = torch.randperm(self.num_channels)
            inv_perm = torch.empty_like(perm).scatter_(
                dim=0, index=perm, src=torch.arange(self.num_channels)
            )
            self.register_buffer("perm", perm)
            self.register_buffer("inv_perm", inv_perm)

        self.index = int( np.ceil(self.num_channels / 2) )
        self.index1 = int( np.ceil( (self.num_channels + 1) / 2) )

    def forward(self, z):
        if self.mode == "shuffle":
            z = z[..., self.perm]
        elif self.mode == "swap":
            z1 = z[..., : self.index]
            z2 = z[..., self.index :]
            z = torch.cat([z2, z1], dim=-1)
        else:
            raise NotImplementedError("The mode " + self.mode + " is not implemented.")
        log_det = 0
        return z, log_det

    def inverse(self, z):
        if self.mode == "shuffle":
            z = z[..., self.inv_perm]
        elif self.mode == "swap":
            z1 = z[..., : self.index1]
            z2 = z[..., self.index1 :]
            z = torch.cat([z2, z1], dim=-1)
        else:
            raise NotImplementedError("The mode " + self.mode + " is not implemented.")
        log_det = 0
        return z, log_det

class AffineCoupling(Flow):
    """
    Affine Coupling layer as introduced RealNVP paper, see arXiv: 1605.08803
    """

    def __init__(self, param_map, scale=True, scale_map="exp"):
        """
        Constructor
        :param param_map: Maps features to shift and scale parameter (if applicable)
        :param scale: Flag whether scale shall be applied
        :param scale_map: Map to be applied to the scale parameter, can be 'exp' as in
        RealNVP or 'sigmoid' as in Glow, 'sigmoid_inv' uses multiplicative sigmoid
        scale when sampling from the model
        """
        super().__init__()
        self.add_module("param_map", param_map)
        self.scale = scale
        self.scale_map = scale_map

    def forward(self, z):
        """
        z is a list of z1 and z2; z = [z1, z2]
        z1 is left constant and affine map is applied to z2 with parameters depending
        on z1
        """
        z1, z2 = z
        param = self.param_map(z1)
        if self.scale:
            shift = param[..., 0::2]
            scale_ = param[..., 1::2]
            if self.scale_map == "exp":
                z2 = z2 * torch.exp(scale_) + shift
                log_det = torch.sum(scale_, dim=list(range(1, shift.dim())))
            elif self.scale_map == "sigmoid":
                scale = torch.sigmoid(scale_ + 2)
                z2 = z2 / scale + shift
                log_det = -torch.sum(torch.log(scale), dim=list(range(1, shift.dim())))
            elif self.scale_map == "sigmoid_inv":
                scale = torch.sigmoid(scale_ + 2)
                z2 = z2 * scale + shift
                log_det = torch.sum(torch.log(scale), dim=list(range(1, shift.dim())))
            else:
                raise NotImplementedError("This scale map is not implemented.")
        else:
            z2 += param
            log_det = 0
        return [z1, z2], log_det

    def inverse(self, z):
        z1, z2 = z
        param = self.param_map(z1)
        if self.scale:
            shift = param[..., 0::2]
            scale_ = param[..., 1::2]
            if self.scale_map == "exp":
                z2 = (z2 - shift) * torch.exp(-scale_)
                log_det = -torch.sum(scale_, dim=list(range(1, shift.dim())))
            elif self.scale_map == "sigmoid":
                scale = torch.sigmoid(scale_ + 2)
                z2 = (z2 - shift) * scale
                log_det = torch.sum(torch.log(scale), dim=list(range(1, shift.dim())))
            elif self.scale_map == "sigmoid_inv":
                scale = torch.sigmoid(scale_ + 2)
                z2 = (z2 - shift) / scale
                log_det = -torch.sum(torch.log(scale), dim=list(range(1, shift.dim())))
            else:
                raise NotImplementedError("This scale map is not implemented.")
        else:
            z2 -= param
            log_det = 0
        return [z1, z2], log_det



class AffineCouplingBlock(Flow):
    """
    Affine Coupling layer including split and merge operation
    """

    def __init__(self, param_map, scale=True, scale_map="exp", split_mode="channel"):
        """
        Constructor
        :param param_map: Maps features to shift and scale parameter (if applicable)
        :param scale: Flag whether scale shall be applied
        :param scale_map: Map to be applied to the scale parameter, can be 'exp' as in
        RealNVP or 'sigmoid' as in Glow
        :param split_mode: Splitting mode, for possible values see Split class
        """
        super().__init__()
        self.flows = nn.ModuleList([])
        # Split layer
        self.flows += [Split(split_mode)]
        # Affine coupling layer
        self.flows += [AffineCoupling(param_map, scale, scale_map)]
        # Merge layer
        self.flows += [Merge(split_mode)]

    def forward(self, z):
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for flow in self.flows:
            z, log_det = flow(z)
            log_det_tot += log_det
        return z, log_det_tot

    def inverse(self, z):
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_det_tot += log_det
        return z, log_det_tot


class MaskedAffineFlow(Flow):
    """
    RealNVP as introduced in arXiv: 1605.08803
    Masked affine flow f(z) = b * z + (1 - b) * (z * exp(s(b * z)) + t)
    class AffineHalfFlow(Flow): is MaskedAffineFlow with alternating bit mask
    NICE is AffineFlow with only shifts (volume preserving)
    """

    def __init__(self, b, t=None, s=None):
        """
        Constructor
        :param b: mask for features, i.e. tensor of same size as latent data point filled with 0s and 1s
        :param t: translation mapping, i.e. neural network, where first input dimension is batch dim,
        if None no translation is applied
        :param s: scale mapping, i.e. neural network, where first input dimension is batch dim,
        if None no scale is applied
        """
        super().__init__()
        self.b_cpu = b.view(1, *b.size())
        self.register_buffer("b", self.b_cpu)

        if s is None:
            self.s = lambda x: torch.zeros_like(x)
        else:
            self.add_module("s", s)

        if t is None:
            self.t = lambda x: torch.zeros_like(x)
        else:
            self.add_module("t", t)

    def forward(self, z):
        z_masked = self.b * z
        scale = self.s(z_masked)
        nan = torch.tensor(np.nan, dtype=z.dtype, device=z.device)
        scale = torch.where(torch.isfinite(scale), scale, nan)
        trans = self.t(z_masked)
        trans = torch.where(torch.isfinite(trans), trans, nan)
        z_ = z_masked + (1 - self.b) * (z * torch.exp(scale) + trans)
        log_det = torch.sum((1 - self.b) * scale, dim=list(range(1, self.b.dim())))
        return z_, log_det

    def inverse(self, z):
        z_masked = self.b * z
        scale = self.s(z_masked)
        nan = torch.tensor(np.nan, dtype=z.dtype, device=z.device)
        scale = torch.where(torch.isfinite(scale), scale, nan)
        trans = self.t(z_masked)
        trans = torch.where(torch.isfinite(trans), trans, nan)
        z_ = z_masked + (1 - self.b) * (z - trans) * torch.exp(-scale)
        log_det = -torch.sum((1 - self.b) * scale, dim=list(range(1, self.b.dim())))
        return z_, log_det

