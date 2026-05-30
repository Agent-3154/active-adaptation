import torch
import torch.distributions as D
from torch.distributions import constraints
from torchrl.modules.distributions.continuous import (
    FasterTransformedDistribution,
    SafeTanhTransform
)


class ScaledTanhNormal(FasterTransformedDistribution):
    """
    Different from torchrl's TanhNormal which uses upscale as
    .. math::
        loc = tanh(loc / upscale) * upscale,
    this class uses
    .. math::
        sample = tanh(sample / upscale) * upscale.
    """
    def __init__(
        self,
        loc: torch.Tensor,
        std: torch.Tensor,
        upscale: torch.Tensor,
        event_dims: int = 1
    ):
        self.loc = loc
        self.std = std
        self.upscale = upscale

        t = D.ComposeTransform([
            D.AffineTransform(loc=0.0, scale=1/self.upscale),
            SafeTanhTransform(),
            D.AffineTransform(loc=0.0, scale=self.upscale),
        ])
        base_dist = D.Independent(
            D.Normal(self.loc, self.std),
            reinterpreted_batch_ndims=event_dims,
        )
        super().__init__(base_dist, t)


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log(1.0 + torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.expm1(torch.abs(x)))


class SymlogTransform(D.Transform):
    """
    Element-wise bidirectional map (Dreamer-style symlog / symexp).

    Forward :math:`y = u \\,\\operatorname{sign}(x/u)\\,\\log(1+|x/u|)` with scale :math:`u>0`.
    Inverse is the symexp :math:`x = u\\,\\operatorname{sign}(v)\\,(e^{|v|}-1)` for :math:`v=y/u`.
    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    sign = +1

    def __init__(self, upscale: torch.Tensor | float=1.0):
        super().__init__(cache_size=1)
        self.upscale = upscale

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        return symlog(x / self.upscale) * self.upscale

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        v = y / self.upscale
        u = symexp(v)
        return u * self.upscale

    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # y = symlog(x); |dy/dx| = 1 / (1 + |x/upscale|) element-wise
        u = x / self.upscale
        return -torch.log1p(torch.abs(u))


class ScaledSymlogNormal(FasterTransformedDistribution):
    """Gaussian with symlog squashing: sample = symlog(z), z ~ Normal(loc, std)."""

    def __init__(
        self,
        loc: torch.Tensor,
        std: torch.Tensor,
        upscale: float=1.0,
        event_dims: int = 1,
    ):
        self.loc = loc
        self.std = std
        self.upscale = upscale

        t = SymlogTransform(upscale=upscale)
        base_dist = D.Independent(
            D.Normal(self.loc, self.std),
            reinterpreted_batch_ndims=event_dims,
        )
        super().__init__(base_dist, t)

