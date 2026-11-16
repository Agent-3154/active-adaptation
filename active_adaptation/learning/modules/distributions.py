# MIT License
# 
# Copyright (c) 2023 Botian Xu, Tsinghua University
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


import functools
import math
from numbers import Number
from typing import List, Tuple, Union

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import constraints
from torch.distributions.utils import lazy_property

D.Distribution.set_default_validate_args(False)


def init(module: nn.Module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def expln(x):
    """A smooth, continuous positive mapping presented in "State-Dependent Exploration for Policy Gradient Methods".

    https://people.idsia.ch/~juergen/ecml2008rueckstiess.pdf

    """
    out = torch.empty_like(x)
    idx_neg = x <= 0
    out[idx_neg] = x[idx_neg].exp()
    out[~idx_neg] = x[~idx_neg].log1p() + 1
    return out


_mappings = {
    "softplus": F.softplus,
    "exp": torch.exp,
    "relu": torch.relu,
    "expln": expln,
}


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=False, gain=0.01):
        super(DiagGaussian, self).__init__()

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.log_std = nn.Parameter(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)
        action_std = torch.broadcast_to(torch.exp(self.log_std), action_mean.shape)
        dist = D.Independent(D.Normal(action_mean, action_std), 1)
        return dist


# class SafeTanhTransform(D.TanhTransform):
#     """Safe version of TanhTransform that avoids NaNs."""

#     def _inverse(self, y: torch.Tensor):
#         eps = torch.finfo(y.dtype).eps
#         y = y.clamp(-1 + eps, 1 - eps)
#         return torch.atanh(y)


# class TanhIndependentNormal(D.TransformedDistribution):
#     arg_constraints = {"loc": constraints.real, "scale": constraints.positive}

#     def __init__(
#         self,
#         loc: torch.Tensor,
#         scale: torch.Tensor,
#         min: Union[torch.Tensor, Number] = -1.0,
#         max: Union[torch.Tensor, Number] = 1.0,
#         event_dims=1,
#     ):
#         self.min = torch.as_tensor(min, device=loc.device).broadcast_to(loc.shape)
#         self.max = torch.as_tensor(max, device=loc.device).broadcast_to(loc.shape)
#         self._loc = (self.min + self.max) / 2
#         self._scale = (self.max - self.min) / 2
#         self.eps = torch.finfo(loc.dtype).eps
#         base_dist = D.Independent(D.Normal(loc, scale), event_dims)
#         t = SafeTanhTransform()

#         super().__init__(base_dist, t)

#     def sample(self, sample_shape: torch.Size = torch.Size()):
#         return super().sample(sample_shape) * self._scale + self._loc

#     def rsample(self, sample_shape: torch.Size = torch.Size()):
#         return super().rsample(sample_shape) * self._scale + self._loc

#     def log_prob(self, value: torch.Tensor):
#         return super().log_prob(
#             ((value - self._loc) / self._scale).clamp(-1 + self.eps, 1 - self.eps)
#         )

#     @property
#     def mode(self):
#         m = self.base_dist.mode
#         return torch.tanh(m)

#     @property
#     def mean(self):
#         return self.mode

#     def entropy(self):
#         return -self.log_prob(self.rsample(self.batch_shape))
from torchrl.modules.distributions import TanhNormal, TruncatedNormal

class IndependentNormal(D.Independent):
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}

    def __init__(self, loc, scale, validate_args=None):
        scale = torch.clamp_min(scale, 1e-6)
        base_dist = D.Normal(loc, scale)
        super().__init__(base_dist, 1, validate_args=validate_args)

    @property
    def loc(self):
        return self.base_dist.loc

    @property
    def scale(self):
        return self.base_dist.scale
    
    @property
    def deterministic_sample(self):
        return self.base_dist.mean
    
    @staticmethod
    def kl(q_loc, q_scale, p_loc, p_scale):
        """Directly compute the KL divergence to save the overhead of class construction."""
        return torch.sum(
            torch.log(p_scale) - torch.log(q_scale)
            + (torch.square(q_scale) + torch.square(p_loc - q_loc)) / (2.0 * torch.square(p_scale))
            - 0.5,
            axis=-1,
        )


class SplitNormal(D.Distribution):
    """Elementwise split (two-piece) normal, a.k.a. the Fechner distribution.

    The density is a normal density whose scale differs on each side of ``loc``::

        loc - scale * exp(-skew)   for the left half
        loc + scale * exp(+skew)   for the right half

    ``skew == 0`` reproduces ``Normal(loc, scale)`` exactly; the two halves are
    weighted so the density is continuous at ``loc``. ``loc`` is therefore both
    the mode and the median, while the mean moves toward the heavy side::

        mean = loc + sqrt(2 / pi) * (scale_right - scale_left)

    The parameterization matches the split-normal PPO toy benchmark
    (``loc``, ``log_std`` and one skew per axis), so results there transfer.
    """

    arg_constraints = {}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, scale, skew, validate_args=None):
        # ``skew`` is unbounded in principle; callers are expected to bound it.
        loc, scale, skew = torch.broadcast_tensors(loc, scale, skew)
        scale = torch.clamp_min(scale, 1e-6)
        self.loc = loc
        self.scale = scale
        self.skew = skew
        super().__init__(batch_shape=loc.shape, validate_args=validate_args)

    @property
    def _log_2cosh_skew(self) -> torch.Tensor:
        # log(exp(k) + exp(-k)) == |k| + softplus(-2|k|), stable for any |k|.
        k = self.skew.abs()
        return k + F.softplus(-2.0 * k)

    @property
    def scale_left(self) -> torch.Tensor:
        return self.scale * torch.exp(-self.skew)

    @property
    def scale_right(self) -> torch.Tensor:
        return self.scale * torch.exp(self.skew)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        z = (value - self.loc) / self.scale
        side = torch.where(z >= 0, self.skew, -self.skew)
        log_norm = (
            0.5 * math.log(2.0 / math.pi)
            - self.scale.log()
            - self._log_2cosh_skew
        )
        return log_norm - 0.5 * (z * torch.exp(-side)).square()

    def entropy(self) -> torch.Tensor:
        """Exact differential entropy: ``log(sl + sr) + 0.5 log(pi/2) + 0.5``.

        Written as ``log s + log(2 cosh k) + ...``. At ``skew == 0`` this is the
        Gaussian entropy. It *increases* with ``|skew|`` at fixed ``scale``, so an
        entropy bonus applied to it would reward skewness rather than exploration.
        Use :meth:`entropy_scale_only` to keep the bonus measuring scale only.
        """
        return (
            self.scale.log()
            + self._log_2cosh_skew
            + 0.5 * math.log(math.pi / 2.0)
            + 0.5
        )

    def entropy_scale_only(self) -> torch.Tensor:
        """Entropy of ``Normal(loc, scale)``, i.e. dropping the skew term.

        Entropy is invariant to the mean, so this is a skew-neutral exploration
        measure that keeps ``entropy_coef`` meaning what it did for a Gaussian.
        """
        return self.scale.log() + 0.5 * math.log(2.0 * math.pi) + 0.5

    @property
    def mode(self) -> torch.Tensor:
        return self.loc

    @property
    def median(self) -> torch.Tensor:
        return self.loc

    @property
    def mean(self) -> torch.Tensor:
        return self.loc + math.sqrt(2.0 / math.pi) * (self.scale_right - self.scale_left)

    @property
    def variance(self) -> torch.Tensor:
        sl, sr = self.scale_left, self.scale_right
        # E[Z^2] = 1 on each side, so E[Z^2] = (sl^3 + sr^3)/(sl + sr).
        # Var = E[Z^2] - E[Z]^2 with E[Z] = sqrt(2/pi) (sr - sl).
        return (sl**3 + sr**3) / (sl + sr) - (2.0 / math.pi) * (sr - sl).square()

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        # Exact inverse-CDF (quantile) reparameterization: piecewise normal quantiles,
        # with the left piece carrying probability p = scale_left / (left + right).
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        shape = self._extended_shape(sample_shape)
        u = torch.rand(shape, device=self.loc.device, dtype=self.loc.dtype)
        p_left = torch.sigmoid(-2.0 * self.skew)
        # Both branches stay inside (0, 1) before erfinv so the untaken branch
        # cannot inject NaN gradients through torch.where.
        u_left = (u / (2.0 * p_left)).clamp(1e-7, 0.5)
        u_right = (0.5 + (u - p_left) / (2.0 * (1.0 - p_left))).clamp(0.5, 1.0 - 1e-7)
        z_left = math.sqrt(2.0) * torch.erfinv(2.0 * u_left - 1.0)
        z_right = math.sqrt(2.0) * torch.erfinv(2.0 * u_right - 1.0)
        return self.loc + torch.where(
            u < p_left,
            self.scale_left * z_left,
            self.scale_right * z_right,
        )

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        return self.rsample(sample_shape).detach()


class IndependentSplitNormal(D.Independent):
    """``SplitNormal`` over the last batch dim; mirrors :class:`IndependentNormal`."""

    arg_constraints = {}

    def __init__(self, loc, scale, skew, validate_args=None):
        scale = torch.clamp_min(scale, 1e-6)
        super().__init__(
            SplitNormal(loc, scale, skew), 1, validate_args=validate_args
        )

    @property
    def loc(self) -> torch.Tensor:
        return self.base_dist.loc

    @property
    def scale(self) -> torch.Tensor:
        return self.base_dist.scale

    @property
    def skew(self) -> torch.Tensor:
        return self.base_dist.skew

    @property
    def deterministic_sample(self) -> torch.Tensor:
        # Same convention as IndependentNormal (loc under its parameterization).
        return self.base_dist.mode

    def entropy_scale_only(self) -> torch.Tensor:
        return self.base_dist.entropy_scale_only().sum(-1)


# Gauss-Hermite nodes for deterministic entropy estimation (see
# ``SinhArcsinh.entropy``). Module-level cache: the nodes depend only on the
# order, never on the parameters, so building them once is enough.
_GH_CACHE: dict[tuple[int, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}


def _gauss_hermite(n: int, dtype: torch.dtype, device: torch.device):
    key = (n, dtype)
    if key not in _GH_CACHE:
        import numpy as np

        x, w = np.polynomial.hermite.hermgauss(n)
        nodes = torch.as_tensor(x, dtype=dtype) * math.sqrt(2.0)
        weights = torch.as_tensor(w, dtype=dtype) / math.sqrt(math.pi)
        _GH_CACHE[key] = (nodes, weights)
    nodes, weights = _GH_CACHE[key]
    return nodes.to(device), weights.to(device)


class SinhArcsinh(D.Distribution):
    """Elementwise sinh-arcsinh (Jones-Faddy) distribution.

    ``a = loc + scale * sinh(asinh(u) + skew)`` with ``u ~ N(0, 1)``, so sampling
    is a monotone transform of a standard normal and ``rsample`` is exact and
    reparameterized. ``skew == 0`` reproduces ``Normal(loc, scale)`` exactly.

    Two differences from :class:`SplitNormal` matter for PPO:

    * The density is smooth at ``loc`` (no kink), which can help SGD.
    * ``loc`` is **not** the mode. The mode sits at
      ``loc + scale * z`` where ``z`` solves ``dlogp/dz = 0``; it is found by
      bisection in :attr:`mode`. For eval rollouts this is the right centre; for
      the PPO ratio only ``log_prob`` and ``rsample`` matter.

    Entropy has no closed form; see :meth:`entropy` for the estimator and
    :meth:`entropy_scale_only` for the skew-neutral variant used by the
    entropy bonus.
    """

    arg_constraints = {}
    support = constraints.real
    has_rsample = True

    # Entropy estimator controls. GH order 32 already agrees with 8M-sample
    # Monte Carlo to ~1e-4 nats across |skew| <= 1.5 (see tests).
    _GH_ORDER = 32
    _MODE_ITERS = 60
    _MODE_BOUND = 8.0

    def __init__(self, loc, scale, skew, validate_args=None):
        loc, scale, skew = torch.broadcast_tensors(loc, scale, skew)
        scale = torch.clamp_min(scale, 1e-6)
        self.loc = loc
        self.scale = scale
        self.skew = skew
        super().__init__(batch_shape=loc.shape, validate_args=validate_args)

    def _z(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.loc) / self.scale

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        z = self._z(value)
        v = torch.asinh(z) - self.skew
        u = torch.sinh(v)
        # cosh(asinh(u)) == sqrt(1 + u^2), so the Jacobian term becomes
        # 0.5*log1p(u^2) - 0.5*log1p(z^2); this avoids cosh overflow at large v.
        return (
            -0.5 * u.square()
            + 0.5 * torch.log1p(u.square())
            - 0.5 * torch.log1p(z.square())
            - 0.5 * math.log(2.0 * math.pi)
            - self.scale.log()
        )

    def _log_prob_from_u(self, u: torch.Tensor) -> torch.Tensor:
        """log density at the sample generated by base standard normal value ``u``."""
        z = torch.sinh(torch.asinh(u) + self.skew)
        return (
            -0.5 * u.square()
            + 0.5 * torch.log1p(u.square())
            - 0.5 * torch.log1p(z.square())
            - 0.5 * math.log(2.0 * math.pi)
            - self.scale.log()
        )

    def entropy(self) -> torch.Tensor:
        """Differential entropy, estimated deterministically by Gauss-Hermite.

        ``H = -E[log p]`` under ``u ~ N(0, 1)``. Using a fixed quadrature rather
        than :meth:`sample` avoids consuming the training RNG and avoids the
        bias/variance of a Monte Carlo estimate inside the loss loop.

        Order 32 matches an 8M-sample Monte Carlo to ~1e-4 nats for
        ``|skew| <= 1.5``. Accuracy degrades for extreme ``|skew|`` because the
        transformed integrand grows steeply.
        """
        nodes, weights = _gauss_hermite(
            self._GH_ORDER, self.loc.dtype, self.loc.device
        )
        # nodes: (K,), weights: (K,) -> broadcast over batch on the last dim.
        u = nodes.reshape(-1, *([1] * self.loc.dim()))
        lp = self._log_prob_from_u(u)
        w = weights.reshape(-1, *([1] * self.loc.dim()))
        return -(w * lp).sum(dim=0)

    def entropy_scale_only(self) -> torch.Tensor:
        """Entropy of ``Normal(loc, scale)``: the skew-neutral entropy bonus.

        True entropy *increases* with ``|skew|`` at fixed ``scale`` (it reaches
        ~+0.66 nats at ``|skew| = 1.5``), so a bonus on the true entropy would
        pay the policy to skew rather than explore. See ``SplitNormal`` for the
        same reasoning.
        """
        return self.scale.log() + 0.5 * math.log(2.0 * math.pi) + 0.5

    def _dlogp_dz(self, z: torch.Tensor) -> torch.Tensor:
        v = torch.asinh(z) - self.skew
        u = torch.sinh(v)
        # d/dz of [-0.5 u^2 + 0.5 log(1+u^2) - 0.5 log(1+z^2)]
        du_dz = torch.cosh(v) / torch.sqrt(1.0 + z.square())
        return (-u + u / (1.0 + u.square())) * du_dz - z / (1.0 + z.square())

    @property
    def mode(self) -> torch.Tensor:
        """Density mode; solved by bisection on a sign-changing bracket.

        ``dlogp/dz`` goes from positive to negative as ``z`` sweeps ``R`` (the
        ``-z^3`` term dominates both tails), so ``[-B, B]`` always brackets. A
        plain Newton step from 0 stalls at the stationary point ``z=0`` for
        negative skew, hence bisection.
        """
        lo = torch.full_like(self.skew, -self._MODE_BOUND)
        hi = torch.full_like(self.skew, self._MODE_BOUND)
        for _ in range(self._MODE_ITERS):
            mid = 0.5 * (lo + hi)
            rising = self._dlogp_dz(mid) > 0
            lo = torch.where(rising, mid, lo)
            hi = torch.where(rising, hi, mid)
        return self.loc + self.scale * (0.5 * (lo + hi))

    @property
    def mean(self) -> torch.Tensor:
        """Approximate mean. No closed form; estimated with the same quadrature."""
        nodes, weights = _gauss_hermite(
            self._GH_ORDER, self.loc.dtype, self.loc.device
        )
        u = nodes.reshape(-1, *([1] * self.loc.dim()))
        w = weights.reshape(-1, *([1] * self.loc.dim()))
        return self.loc + self.scale * (w * torch.sinh(torch.asinh(u) + self.skew)).sum(0)

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        shape = self._extended_shape(sample_shape)
        u = torch.randn(shape, device=self.loc.device, dtype=self.loc.dtype)
        return self.loc + self.scale * torch.sinh(torch.asinh(u) + self.skew)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        return self.rsample(sample_shape).detach()


class IndependentSinhArcsinh(D.Independent):
    """``SinhArcsinh`` over the last batch dim; mirrors :class:`IndependentNormal`."""

    arg_constraints = {}

    def __init__(self, loc, scale, skew, validate_args=None):
        scale = torch.clamp_min(scale, 1e-6)
        super().__init__(
            SinhArcsinh(loc, scale, skew), 1, validate_args=validate_args
        )

    @property
    def loc(self) -> torch.Tensor:
        return self.base_dist.loc

    @property
    def scale(self) -> torch.Tensor:
        return self.base_dist.scale

    @property
    def skew(self) -> torch.Tensor:
        return self.base_dist.skew

    @property
    def deterministic_sample(self) -> torch.Tensor:
        # The mode, which for sinh-arcsinh is NOT loc.
        return self.base_dist.mode

    def entropy_scale_only(self) -> torch.Tensor:
        return self.base_dist.entropy_scale_only().sum(-1)


class IndependentBeta(D.Independent):
    def __init__(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        min: Union[float, torch.Tensor] = 0.0,
        max: Union[float, torch.Tensor] = 1.0,
        event_dims: int = 1,
    ):
        self.min = torch.as_tensor(min, device=alpha.device).broadcast_to(alpha.shape)
        self.max = torch.as_tensor(max, device=alpha.device).broadcast_to(alpha.shape)
        self.scale = self.max - self.min
        self.eps = torch.finfo(alpha.dtype).eps
        base_dist = D.Beta(alpha, beta)
        super().__init__(base_dist, event_dims)

    def sample(self, sample_shape: torch.Size = torch.Size()):
        return super().sample(sample_shape) * self.scale + self.min

    def rsample(self, sample_shape: torch.Size = torch.Size()):
        return super().rsample(sample_shape) * self.scale + self.min

    def log_prob(self, value: torch.Tensor):
        return super().log_prob(
            ((value - self.min) / self.scale).clamp(self.eps, 1.0 - self.eps)
        )


class IndependentNormalModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        state_dependent_std: bool = False,
        scale_mapping: str = "exp",
        scale_lb: float = 1e-4,
    ):
        super().__init__()
        self.state_dependent_std = state_dependent_std
        if self.state_dependent_std:
            self.operator = nn.Linear(input_dim, output_dim * 2)
        else:
            self.operator = nn.Linear(input_dim, output_dim)
            self.log_std = nn.Parameter(torch.zeros(output_dim))
        if isinstance(scale_mapping, str):
            self.scale_mapping = _mappings[scale_mapping]
        elif callable(self.scale_mapping):
            self.scale_mapping = scale_mapping
        else:
            raise ValueError("scale_mapping must be a string or a callable function.")
        self.scale_lb = scale_lb

    def forward(self, tensor: torch.Tensor) -> Tuple[torch.Tensor]:
        if self.state_dependent_std:
            loc, scale = self.operator(tensor).chunk(2, -1)
            scale = self.scale_mapping(scale).clamp_min(self.scale_lb)
        else:
            loc, scale = self.operator(tensor), self.scale_mapping(
                self.log_std
            ).clamp_min(self.scale_lb)
        return IndependentNormal(loc, scale)


class TanhNormalWithEntropy(TanhNormal):

    def log_prob(self, value, **kwargs):
        value_clamped = value.clamp(self.low+1e-6, self.high-1e-6)
        return super().log_prob(value + (value_clamped - value).detach(), **kwargs)

    def entropy(self):
        return -self.log_prob(self.sample())

class TruncatedNormalWithEntropy(TruncatedNormal):

    def log_prob(self, value, **kwargs):
        value_clamped = value.clamp(self.low+1e-6, self.high-1e-6)
        return super().log_prob(value + (value_clamped - value).detach(), **kwargs)

    def entropy(self):
        return -self.log_prob(self.sample())

class TanhIndependentNormalModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        scale_mapping: str = "softplus",
        state_dependent_std: bool = True,
        scale_lb: float = 1e-4,
        min: Union[torch.Tensor, Number] = -1.0,
        max: Union[torch.Tensor, Number] = 1.0,
        event_dims=1,
    ):
        super().__init__()
        self.state_dependent_std = state_dependent_std
        if self.state_dependent_std:
            self.operator = nn.Linear(input_dim, output_dim * 2)
        else:
            self.operator = nn.Linear(input_dim, output_dim)
            self.log_std = nn.Parameter(torch.zeros(output_dim))
        if isinstance(scale_mapping, str):
            self.scale_mapping = _mappings[scale_mapping]
        elif callable(self.scale_mapping):
            self.scale_mapping = scale_mapping
        else:
            raise ValueError("scale_mapping must be a string or a callable function.")
        self.scale_lb = scale_lb
        self.dist_cls = functools.partial(
            TanhNormalWithEntropy, min=min, max=max, event_dims=event_dims
        )

    def forward(self, tensor: torch.Tensor) -> Tuple[torch.Tensor]:
        if self.state_dependent_std:
            loc, scale = self.operator(tensor).chunk(2, -1)
            scale = self.scale_mapping(scale).clamp_min(self.scale_lb)
        else:
            loc = self.operator(tensor)
            scale = self.scale_mapping(self.log_std).clamp_min(self.scale_lb)
        return self.dist_cls(loc, scale)


class IndependentBetaModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        scale_mapping: str = "softplus",
        min: Union[float, torch.Tensor] = 0.0,
        max: Union[float, torch.Tensor] = 1.0,
        event_dims: int = 1,
    ):
        super().__init__()
        self.operator = nn.Linear(input_dim, output_dim * 2)
        if isinstance(scale_mapping, str):
            self.scale_mapping = _mappings[scale_mapping]
        elif callable(self.scale_mapping):
            self.scale_mapping = scale_mapping
        else:
            raise ValueError("scale_mapping must be a string or a callable function.")
        self.dist_cls = functools.partial(
            IndependentBeta, min=min, max=max, event_dims=event_dims
        )

    def forward(self, tensor: torch.Tensor) -> Tuple[torch.Tensor]:
        alpha, beta = self.operator(tensor).chunk(2, -1)
        alpha = self.scale_mapping(alpha)
        beta = self.scale_mapping(beta)
        return self.dist_cls(alpha, beta)


class MultiCategorical(D.Distribution):
    def __init__(
        self,
        logits: List[torch.Tensor] = None,
        probs: List[torch.Tensor] = None,
    ):
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )

        if probs is not None:
            self.base_dists = [D.Categorical(probs=probs_) for probs_ in probs]
        else:
            self.base_dists = [D.Categorical(logits=logits_) for logits_ in logits]

    def sample(self, sample_shape=torch.Size()):
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        return torch.stack(
            [dist.sample(sample_shape) for dist in self.base_dists], dim=-1
        )

    @lazy_property
    def logits(self):
        return [dist.logits for dist in self.base_dists]

    @lazy_property
    def probs(self):
        return [dist.probs for dist in self.base_dists]

    @property
    def mode(self):
        return torch.stack([dist.mode for dist in self.base_dists], dim=-1)

    def sample(self, sample_shape=torch.Size()):
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        return torch.stack(
            [dist.sample(sample_shape) for dist in self.base_dists], dim=-1
        )

    def log_prob(self, value):
        return torch.stack(
            [
                dist.log_prob(v)
                for dist, v in zip(self.base_dists, value.unbind(dim=-1))
            ],
            dim=-1,
        ).sum(dim=-1)

    def entropy(self):
        return torch.stack([dist.entropy() for dist in self.base_dists], dim=-1).sum(
            dim=-1
        )


class MultiCategoricalModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dims: Union[List[int], torch.Tensor],
    ):
        super().__init__()
        self.operator = nn.Linear(input_dim, sum(output_dims))
        self.output_dims = (
            output_dims.tolist()
            if isinstance(output_dims, torch.Tensor)
            else output_dims
        )

    def forward(self, tensor: torch.Tensor) -> Tuple[torch.Tensor]:
        logits = self.operator(tensor)
        logits = logits.split(self.output_dims, dim=-1)
        return MultiCategorical(logits=logits)


class MultiOneHotCategorical(D.Independent):
    def __init__(
        self,
        logits: torch.Tensor = None,
        probs: torch.Tensor = None,
        unimix: float = 0.01
    ):
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )
        
        if logits is not None:
            probs = F.softmax(logits, dim=-1)
        
        probs = probs * (1. - unimix) + unimix / probs.shape[-1]
        super().__init__(
            D.OneHotCategoricalStraightThrough(probs=probs),
            1
        )

class TwoHot(D.Distribution):
    def __init__(
        self, 
        logits: torch.Tensor, 
        low=-20.0, 
        high=20.0, 
    ):
        super().__init__(batch_shape=logits.shape[:-1])
        self.logits = logits
        self.probs = torch.softmax(logits, -1)
        self.buckets = torch.linspace(low, high, steps=logits.shape[-1]).to(logits.device)

    @property
    def mean(self):
        return torch.sum(self.probs * self.buckets, -1, keepdim=True)

    @property
    def mode(self):
        return self.mean

    def log_prob(self, x):
        # x(time, batch, 1)
        below = torch.sum((self.buckets <= x[..., None]).to(torch.int32), dim=-1) - 1
        above = len(self.buckets) - torch.sum(
            (self.buckets > x[..., None]).to(torch.int32), dim=-1
        )
        below = torch.clip(below, 0, len(self.buckets) - 1)
        above = torch.clip(above, 0, len(self.buckets) - 1)
        equal = below == above

        dist_to_below = torch.where(equal, 1, torch.abs(self.buckets[below] - x))
        dist_to_above = torch.where(equal, 1, torch.abs(self.buckets[above] - x))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
            F.one_hot(below, num_classes=len(self.buckets)) * weight_below[..., None]
            + F.one_hot(above, num_classes=len(self.buckets)) * weight_above[..., None]
        )
        log_pred = self.logits - torch.logsumexp(self.logits, -1, keepdim=True)
        target = target.squeeze(-2)

        return (target * log_pred).sum(-1)

