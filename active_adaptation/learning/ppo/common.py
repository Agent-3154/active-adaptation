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


import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModuleBase as ModBase
from torchrl.modules import ProbabilisticActor
from torchrl.data import Composite
from typing import Any, List, Tuple, Union
from collections.abc import Sequence

OBS_KEY = "policy"  # ("agents", "observation")
OBS_PRIV_KEY = "priv"
OBS_HIST_KEY = "policy_h"
ACTION_KEY = "action"  # ("agents", "action")
REWARD_KEY = ("next", "reward")  # ("agents", "reward")
# DONE_KEY = ("next", "done")
TERM_KEY = ("next", "terminated")
DONE_KEY = ("next", "done")
CMD_KEY = "command"


ACTION_DISTS = ("normal", "split_normal", "sinh_arcsinh")


def resolve_action_dist(action_dist: str) -> str:
    """Validate ``action_dist`` against the supported action distributions."""
    if action_dist not in ACTION_DISTS:
        raise ValueError(
            f"Unknown action_dist {action_dist!r}; expected one of {ACTION_DISTS}."
        )
    return action_dist


def action_dist_class(action_dist: str):
    """Distribution class for ``ProbabilisticActor(distribution_class=...)``.

    TorchRL instantiates this with keyword args taken from the actor's output
    keys (``loc``/``scale``, plus ``skew`` for the split normal), so returning the
    class itself keeps the rollout path identical to the explicit
    :func:`make_action_dist` calls in the PPO update.
    """
    from active_adaptation.learning.modules.distributions import (
        IndependentNormal,
        IndependentSinhArcsinh,
        IndependentSplitNormal,
    )

    if action_dist == "normal":
        return IndependentNormal
    if action_dist == "split_normal":
        return IndependentSplitNormal
    if action_dist == "sinh_arcsinh":
        return IndependentSinhArcsinh
    raise ValueError(
        f"Unknown action_dist {action_dist!r}; expected one of {ACTION_DISTS}."
    )


def make_action_dist(
    action_dist: str,
    loc: torch.Tensor,
    scale: torch.Tensor,
    skew: torch.Tensor | None = None,
):
    """Build the configured action distribution from actor outputs.

    Single source of truth for the distribution used at rollout and in every
    PPO update, so the Gaussian and split-normal paths can never disagree about
    which tensors feed them.
    """
    cls = action_dist_class(action_dist)
    if action_dist in ("split_normal", "sinh_arcsinh"):
        if skew is None:
            raise ValueError(
                f"action_dist={action_dist!r} requires a 'skew' tensor from the actor."
            )
        return cls(loc=loc, scale=scale, skew=skew)
    return cls(loc=loc, scale=scale)


def dist_entropy(dist, action_dist: str) -> torch.Tensor:
    """Entropy for the PPO bonus.

    For both skewed families this returns the scale-only (skew-neutral) entropy.
    The true entropy *increases* with the shape parameter at fixed scale — for
    split normal and sinh-arcsinh alike — so feeding it an entropy bonus would
    pay the policy to skew rather than explore.

    For ``sinh_arcsinh`` the true entropy has no closed form and is estimated by
    Gauss-Hermite quadrature (see ``SinhArcsinh.entropy``); use
    ``dist.entropy()`` directly if you want it for logging.
    """
    if action_dist in ("split_normal", "sinh_arcsinh"):
        return dist.entropy_scale_only()
    return dist.entropy()


def resolve_clip_param(clip_param: Any) -> Tuple[float, float]:
    """Normalize ``clip_param`` to ``(eps_neg, eps_pos)``.

    A scalar ``ε`` is treated as symmetric ``(ε, ε)``. A length-2 sequence
    ``(eps_neg, eps_pos)`` (tuple, list, or OmegaConf ``ListConfig``) clips
    the importance ratio to ``[1 - eps_neg, 1 + eps_pos]``.
    """
    if isinstance(clip_param, Sequence) and not isinstance(clip_param, (str, bytes)):
        if len(clip_param) != 2:
            raise ValueError(
                f"clip_param sequence must have length 2, got {clip_param!r}"
            )
        eps_neg, eps_pos = float(clip_param[0]), float(clip_param[1])
    else:
        eps_neg = eps_pos = float(clip_param)
    if eps_neg < 0.0 or eps_pos < 0.0:
        raise ValueError(
            f"clip_param values must be non-negative, got {(eps_neg, eps_pos)!r}"
        )
    return eps_neg, eps_pos


def ppo_clipped_loss(
    ratio: torch.Tensor, adv: torch.Tensor, clip_param: Tuple[float, float]
) -> torch.Tensor:
    """Loss to minimize; negates the clipped surrogate so gradient descent maximizes it.

    ``clip_param`` may be a scalar ``ε`` (symmetric clip to ``[1-ε, 1+ε]``) or
    a length-2 sequence ``(eps_neg, eps_pos)`` (asymmetric clip to
    ``[1-eps_neg, 1+eps_pos]``).
    """
    assert ratio.shape == adv.shape
    eps_neg, eps_pos = clip_param
    surr1 = adv * ratio
    surr2 = adv * ratio.clamp(1.0 - eps_neg, 1.0 + eps_pos)
    return -torch.min(surr1, surr2)


def spo_loss(
    ratio: torch.Tensor, adv: torch.Tensor, clip_param: Tuple[float, float]
) -> torch.Tensor:
    """
    Simple Policy Optimization Loss from https://arxiv.org/pdf/2401.16025.
    Loss to minimize; negates the SPO objective so gradient descent maximizes it.

    For asymmetric ``clip_param=(eps_neg, eps_pos)``, the quadratic penalty
    uses ``eps_pos`` when ``ratio > 1`` and ``eps_neg`` when ``ratio < 1``.
    """
    assert ratio.shape == adv.shape
    eps_neg, eps_pos = resolve_clip_param(clip_param)
    if eps_neg == eps_pos:
        eps = ratio.new_full((), eps_neg)
    else:
        eps = torch.where(ratio >= 1.0, ratio.new_full((), eps_pos), ratio.new_full((), eps_neg))
    obj = ratio * adv - adv.abs() / (2.0 * eps) * (ratio - 1.0).square()
    return -obj


def aspo_loss(
    ratio: torch.Tensor, adv: torch.Tensor, clip_param: Tuple[float, float]
) -> torch.Tensor:
    """
    Asymmetric Simple Policy Optimization Loss from https://arxiv.org/pdf/2401.16025.
    Loss to minimize; negates the ASPO objective so gradient descent maximizes it.
    """
    return torch.where(
        adv >= 0.0,
        ppo_clipped_loss(ratio, adv, clip_param),
        spo_loss(ratio, adv, clip_param),
    )


PPO_LOSS_MAP = {
    "ppo": ppo_clipped_loss,
    "spo": spo_loss,
    "aspo": aspo_loss,
}


class DtypeConversion(nn.Module):
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.dtype = dtype

    def forward(self, x: torch.Tensor):
        return x.to(self.dtype)


class ResidualFC(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, activation: nn.Module = nn.Mish
    ):
        super().__init__()
        if input_dim != output_dim:
            self.linear_skip = nn.LazyLinear(output_dim)
        else:
            self.linear_skip = nn.Identity()
        self.linear = nn.LazyLinear(output_dim)
        self.act = activation()
        self.ln = nn.LayerNorm(output_dim)

    def forward(self, x):
        x_skip = self.linear_skip(x)
        return self.ln(self.act(self.linear(x)) + x_skip)


def make_mlp(num_units, activation=nn.Mish, norm="before", dropout=0.0):
    assert norm in ("before", "after", None)
    layers = []
    for n in num_units:
        layers.append(nn.LazyLinear(n))
        if norm == "before":
            layers.append(nn.LayerNorm(n))
            layers.append(activation())
        elif norm == "after":
            layers.append(activation())
            layers.append(nn.LayerNorm(n))
        else:
            layers.append(activation())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def make_conv(
    in_channels: int,
    num_channels: List[int],
    kernel_sizes: int | List[int] = 3,
    activation=nn.GELU,
    flatten: bool = True,
    auto_dtype: bool = True,
):
    if isinstance(kernel_sizes, int):
        kernel_sizes = [kernel_sizes] * len(num_channels)
    num_channels = [in_channels] + num_channels

    layers = []
    if auto_dtype:
        layers.append(DtypeConversion(torch.float32))
    for in_channels, out_channels, k in zip(
        num_channels[:-1], num_channels[1:], kernel_sizes, strict=True
    ):
        layer = nn.Conv2d(
            in_channels, out_channels, kernel_size=k, stride=2, padding=k // 2
        )
        nn.init.orthogonal_(layer.weight, 0.02)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(activation())
    if flatten:
        layers.append(nn.Flatten())
    module = nn.Sequential(*layers)
    return FlattenBatch(module, data_dim=3)


class FlattenBatch(nn.Module):
    def __init__(self, module, data_dim: int = 1):
        super().__init__()
        self.module = module
        self.data_dim = data_dim

    def forward(self, *args: torch.Tensor):
        batch_shape = args[0].shape[: -self.data_dim]
        args_flattened = (arg.flatten(0, len(batch_shape) - 1) for arg in args)
        output_flattened = self.module(*args_flattened)
        if isinstance(output_flattened, tuple):
            output = tuple(arg.unflatten(0, batch_shape) for arg in output_flattened)
        else:
            output = output_flattened.unflatten(0, batch_shape)
        return output


def make_batch(tensordict: TensorDict, num_minibatches: int, seq_len: int = -1):
    if seq_len > 1:
        N, T = tensordict.shape
        T = (T // seq_len) * seq_len
        tensordict = tensordict[:, :T].reshape(-1, seq_len)
        perm = torch.randperm(
            (tensordict.shape[0] // num_minibatches) * num_minibatches,
            device=tensordict.device,
        ).reshape(num_minibatches, -1)
        for indices in perm:
            yield tensordict[indices]
    else:
        tensordict = tensordict.reshape(-1)
        perm = torch.randperm(
            (tensordict.shape[0] // num_minibatches) * num_minibatches,
            device=tensordict.device,
        ).reshape(num_minibatches, -1)
        for indices in perm:
            yield tensordict[indices]


class Chunk(nn.Module):
    def __init__(self, n) -> None:
        super().__init__()
        self.n = n

    def forward(self, x):
        return x.chunk(self.n, dim=-1)


class Duplicate(nn.Module):
    def __init__(self, n) -> None:
        super().__init__()
        self.n = n

    def forward(self, x):
        return tuple(x for _ in range(self.n))


class Split(nn.Module):
    def __init__(self, split_size):
        super().__init__()
        self.split_size = split_size

    def forward(self, x: torch.Tensor):
        return x.split(self.split_size, dim=-1)


class Actor(nn.Module):
    """Maps backbone features to a diagonal Gaussian policy (mean and scale).

    The final linear uses ``nn.LazyLinear``; its weight is marked ``_non_muon`` so
    optimizers such as ``MuonAdamWWrapper`` keep the output head on AdamW.

    If ``predict_std`` is false, per-action standard deviation is a single learned
    vector ``actor_std`` broadcast to the batch; ``scale_mapping`` is identity. If
    true, the linear output is split; the second half is mapped with ``torch.exp`` to
    a positive diagonal scale.

    Args:
        action_dim: Size of the action vector (output dim of the mean, or half of
            the linear output when ``predict_std`` is true).
        predict_std: If true, one linear layer outputs ``2 * action_dim`` and is
            chunked into ``loc`` and ``scale``; if false, only ``loc`` is predicted.
        predict_skew: If true, the head additionally emits a state-dependent
            elementwise ``skew`` for :class:`IndependentSplitNormal`, bound to
            ``max_skew * tanh(.)`` so it is exactly ``0`` at init (Gaussian start)
            and cannot run away. Ignored by the Gaussian path.
        max_skew: Bound on ``|skew|`` in log-scale ratio units. ``|skew| = s``
            means one side's std is ``exp(s)`` times the other's.
    """

    def __init__(
        self,
        action_dim: int,
        predict_std: bool = False,
        predict_skew: bool = False,
        max_skew: float = 1.0,
    ) -> None:
        super().__init__()
        self.predict_std = predict_std
        self.predict_skew = predict_skew
        self.max_skew = max_skew
        self.action_dim = action_dim
        n_out = action_dim * 2 if predict_std else action_dim
        if predict_skew:
            n_out += action_dim
        # The scale parameterization depends on predict_std only; predicting skew
        # adds an output chunk but must not switch how scale is produced.
        if predict_std:
            self.actor_mean = nn.LazyLinear(n_out)
            self.scale_mapping = torch.exp
        else:
            self.actor_mean = nn.LazyLinear(n_out)
            self.actor_std = nn.Parameter(torch.ones(action_dim))
            self.scale_mapping = nn.Identity()
        self.actor_mean.weight._non_muon = True

    def forward(self, features: torch.Tensor):
        """Return ``(loc, scale)``, or ``(loc, scale, skew)`` when predicting skew."""
        out = self.actor_mean(features)
        if not self.predict_std and not self.predict_skew:
            loc = out
            scale = torch.ones_like(loc) * self.actor_std
            return loc, self.scale_mapping(scale)
        # Split off the skew chunk last so enabling it leaves loc/scale layout intact.
        if self.predict_skew:
            out, skew_raw = out.split(
                [out.shape[-1] - self.action_dim, self.action_dim], dim=-1
            )
        if self.predict_std:
            loc, scale = out.chunk(2, dim=-1)
        else:
            loc = out
            scale = torch.ones_like(loc) * self.actor_std
        scale = self.scale_mapping(scale)
        if self.predict_skew:
            return loc, scale, self.max_skew * torch.tanh(skew_raw)
        return loc, scale

    @torch.no_grad()
    def init_skew_(self, init: float = 0.0, signs: torch.Tensor | None = None) -> None:
        """Set the skew head so each action starts at ``init * sign``.

        ``init=0`` reproduces :meth:`zero_skew_` (exact Gaussian start). A small
        nonzero value breaks the symmetric stationary point without biasing the
        shape toward one side more than the data supports, which is what lets a
        state-dependent head actually express sustained skew: at ``skew == 0``
        the only path to skew is a weight direction that must agree across all
        states.

        A plain ``LazyLinear`` init gives the skew rows a random weight and bias,
        so this must be called after the lazy parameters are materialized (i.e.
        after a warm-up forward) and after any general ``apply(init_)`` pass,
        which would otherwise re-randomize these rows.

        A nonzero initial skew is applied along the symmetry-invariant direction
        when the action mirror map is known. That is not because the symmetry loss
        penalizes asymmetry (it is only logged, not optimized), but because under
        ``symaug`` the mirrored batch reuses the *same* stored ``action_log_prob``;
        a mirror-inconsistent skew makes the two halves disagree about the target
        shape, which adds avoidable ratio noise early on.

        Args:
            init: Target initial skew magnitude per action, in ``tanh`` units
                before ``max_skew`` scaling. Bias is set to
                ``atanh(init)`` so that ``max_skew * tanh(bias) == init``.
            signs: Optional per-action signs (typically the action-space mirror
                signs). ``None`` means all +1.
        """
        if not self.predict_skew:
            return
        if type(self.actor_mean).__name__.startswith("Uninitialized") or type(
            self.actor_mean
        ).__name__.startswith("Lazy"):
            raise RuntimeError(
                "init_skew_() called before the head was materialized; run a "
                "warm-up forward on the actor first."
            )
        # Output layout is [loc | scale | skew] with the scale block present only
        # when predict_std is set, so the skew block always starts at the end.
        cols = slice(
            self.action_dim * 2 if self.predict_std else self.action_dim, None
        )
        self.actor_mean.weight[cols].zero_()
        if init == 0.0:
            self.actor_mean.bias[cols].zero_()
            return
        if abs(init) >= 1.0:
            raise ValueError(
                f"init_skew_ init must be in (-1, 1) since tanh is bounded, got {init}."
            )
        sign = (
            torch.ones(self.action_dim, device=self.actor_mean.bias.device)
            if signs is None
            else torch.as_tensor(
                signs, device=self.actor_mean.bias.device, dtype=self.actor_mean.bias.dtype
            )
        )
        if sign.numel() != self.action_dim:
            raise ValueError(
                f"signs must have length {self.action_dim}, got {sign.numel()}."
            )
        # The sign of `init` is folded into the sign vector, so
        # max_skew * tanh(bias) == init * sign elementwise. atanh is applied to
        # |init| because tanh is odd: atanh(-c) == -atanh(c).
        direction = sign if init > 0 else -sign
        self.actor_mean.bias[cols] = torch.atanh(torch.as_tensor(abs(init))) * direction

    def zero_skew_(self) -> None:
        """Zero the skew head so the policy starts as an exact Gaussian."""
        self.init_skew_(0.0)


def symmetry_invariant_skew_signs(act_transform, action_dim: int) -> torch.Tensor | None:
    """Per-action skew vector left unchanged by the action mirror transform.

    Skew is *signed* (it names the heavy side), so it transforms exactly like
    ``loc``: ``(M k)_i = signs[i] * k[perm[i]]``. A uniform init is therefore
    generally **not** symmetry-consistent -- under ``symaug`` the mirror loss
    would push it back toward zero, cancelling most of the benefit of starting
    off-centre.

    We find the invariant subspace by orbit averaging,
    ``k = mean_t M^t 1``, which projects onto ``{k : M k = k}``. ``M`` is a
    signed permutation, so this converges exactly in at most ``action_dim``
    steps.

    Returns:
        A unit-scale vector in the invariant subspace (sign pattern only; the
        caller rescales), or ``None`` when the invariant subspace is trivial
        (i.e. no nonzero symmetric skew exists for this action space).
    """
    if act_transform is None:
        return None
    perm = act_transform.perm
    signs = act_transform.signs.to(torch.float32)
    if perm.numel() != action_dim:
        return None

    def mirror(x: torch.Tensor) -> torch.Tensor:
        return signs.to(x.device) * x[perm.to(x.device)]

    acc = torch.zeros(action_dim, device=perm.device, dtype=torch.float32)
    cur = torch.ones_like(acc)
    for _ in range(action_dim):
        acc = acc + cur
        cur = mirror(cur)
    acc = acc / action_dim
    if acc.abs().max() < 1e-6:
        return None
    # Normalize so the largest component is 1; caller multiplies by `init`.
    return acc / acc.abs().max()


class Critic(nn.Module):
    """Value head: a single ``nn.LazyLinear`` from features to ``output_dim`` scalars.

    The linear weight is marked ``_non_muon`` so composite optimizers can exclude
    this layer from Muon and train it with AdamW.

    Args:
        output_dim: Trailing dimension of the value output (often ``1`` for V(s)).
    """

    def __init__(self, output_dim: int):
        super().__init__()
        self.critic = nn.LazyLinear(output_dim)
        self.critic.weight._non_muon = True

    def forward(self, features: torch.Tensor):
        """Return predicted value(s) with shape ``(*features.shape[:-1], output_dim)``."""
        return self.critic(features)


class GAE(nn.Module):
    def __init__(self, gamma, lmbda):
        super().__init__()
        self.register_buffer("gamma", torch.tensor(gamma))
        self.register_buffer("lmbda", torch.tensor(lmbda))
        self.gamma: torch.Tensor
        self.lmbda: torch.Tensor

    def forward(
        self,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
        next_value: torch.Tensor,
        discount: torch.Tensor | None = None,
    ):
        num_steps = terminated.shape[1]
        advantages = torch.zeros_like(reward)
        nonterm = 1 - terminated.float()  # whether to backup value
        nondone = 1 - done.float()  # whether to backup reward
        if discount is None:
            discount = torch.ones_like(nonterm)
        gae = 0
        for step in reversed(range(num_steps)):
            next_value_t = next_value[:, step] * nonterm[:, step]
            gamma_t = discount[:, step] * self.gamma
            delta = reward[:, step] + gamma_t * next_value_t - value[:, step]
            advantages[:, step] = gae = delta + (
                gamma_t * self.lmbda * nondone[:, step] * gae
            )
        returns = advantages + value
        return advantages, returns


def init_(module):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, 0.01)
        nn.init.constant_(module.bias, 0.0)


def _paired_named_tensors(
    source_module: nn.Module,
    target_module: nn.Module,
    *,
    include_buffers: bool = True,
):
    """Yield ``(name, src, tgt)`` for matching parameters (and optionally buffers).

    Raises if modules are aliased, names/shapes disagree, any pair shares storage,
    or either side still has lazy / uninitialized parameters or buffers.
    """
    if source_module is target_module:
        raise ValueError(
            "source_module and target_module are the same object; refusing to copy."
        )

    def _ensure_initialized(name: str, tensor: torch.Tensor, which: str) -> None:
        # LazyLinear / Lazy* leave UninitializedParameter/Buffer until first forward.
        if type(tensor).__name__.startswith("Uninitialized"):
            raise ValueError(
                f"{which} {name!r} is still lazy/uninitialized ({type(tensor).__name__}). "
                "Run a warm-up forward on both modules before hard_copy_/soft_copy_, "
                "or build them separately with make_*() instead of copy.deepcopy()."
            )

    src_params = list(source_module.named_parameters())
    tgt_params = list(target_module.named_parameters())
    if len(src_params) != len(tgt_params):
        raise ValueError(
            f"Parameter count mismatch: source has {len(src_params)}, "
            f"target has {len(tgt_params)}."
        )

    pairs = []
    for (src_name, src), (tgt_name, tgt) in zip(src_params, tgt_params, strict=True):
        if src_name != tgt_name:
            raise ValueError(
                f"Parameter name mismatch: source {src_name!r} vs target {tgt_name!r}. "
                "Modules must have identical named_parameters() order and names."
            )
        _ensure_initialized(src_name, src, "source parameter")
        _ensure_initialized(tgt_name, tgt, "target parameter")
        if src.shape != tgt.shape:
            raise ValueError(
                f"Parameter shape mismatch for {src_name!r}: {tuple(src.shape)} vs "
                f"{tuple(tgt.shape)}."
            )
        if src is tgt or src.data_ptr() == tgt.data_ptr():
            raise ValueError(
                f"Parameter {src_name!r} is shared between source and target "
                "(same Tensor storage). Copy would be a no-op or corrupt both; "
                "use separate make_*() construction, not a shallow alias."
            )
        pairs.append((src_name, src, tgt))

    if include_buffers:
        src_bufs = list(source_module.named_buffers())
        tgt_bufs = list(target_module.named_buffers())
        if len(src_bufs) != len(tgt_bufs):
            raise ValueError(
                f"Buffer count mismatch: source has {len(src_bufs)}, "
                f"target has {len(tgt_bufs)}."
            )
        for (src_name, src), (tgt_name, tgt) in zip(src_bufs, tgt_bufs, strict=True):
            if src_name != tgt_name:
                raise ValueError(
                    f"Buffer name mismatch: source {src_name!r} vs target {tgt_name!r}."
                )
            _ensure_initialized(src_name, src, "source buffer")
            _ensure_initialized(tgt_name, tgt, "target buffer")
            if src.shape != tgt.shape:
                raise ValueError(
                    f"Buffer shape mismatch for {src_name!r}: {tuple(src.shape)} vs "
                    f"{tuple(tgt.shape)}."
                )
            if src is tgt or (
                src.numel() > 0 and tgt.numel() > 0 and src.data_ptr() == tgt.data_ptr()
            ):
                raise ValueError(
                    f"Buffer {src_name!r} is shared between source and target."
                )
            pairs.append((src_name, src, tgt))

    if not pairs:
        raise ValueError("No parameters or buffers to copy.")

    return pairs


@torch.no_grad()
def hard_copy_(source_module: nn.Module, target_module: nn.Module):
    """In-place ``target ← source`` for all parameters and buffers.

    Validates matching names/shapes and rejects aliased / shared storages.
    """
    for _name, src, tgt in _paired_named_tensors(source_module, target_module):
        tgt.data.copy_(src.data)


@torch.no_grad()
def soft_copy_(source_module: nn.Module, target_module: nn.Module, tau: float = 0.01):
    """Polyak update: ``target ← (1 - τ) target + τ source`` (in-place).

    Floating tensors use ``lerp_``; non-floating buffers use a hard copy when
    ``τ == 1``, otherwise are left unchanged. Validates names/shapes and rejects
    shared storages (same checks as :func:`hard_copy_`).
    """
    if not (0.0 < tau <= 1.0):
        raise ValueError(f"tau must be in (0, 1], got {tau!r}")
    for _name, src, tgt in _paired_named_tensors(source_module, target_module):
        if tgt.is_floating_point() and src.is_floating_point():
            if tau == 1.0:
                tgt.data.copy_(src.data)
            else:
                tgt.data.lerp_(src.data, tau)
        elif tau == 1.0:
            tgt.data.copy_(src.data)


class L2Norm(nn.Module):

    def forward(self, x):
        return x / torch.norm(x, dim=-1, keepdim=True).clamp(1e-7)


class SimNorm(nn.Module):
    """
    Simplicial normalization.
    Adapted from https://arxiv.org/abs/2204.00616.
    """

    def __init__(self, dim: int, method="l2"):
        super().__init__()
        self.dim = dim
        if method == "softmax":
            self.f = F.softmax
        elif method == "l2":
            self.f = lambda x: x / x.norm(dim=-1, keepdim=True).clamp(1e-6)
        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor):
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = self.f(x)
        return x.view(*shp)


class ConsistentDropout(nn.Module):
    def __init__(self, p: float, return_mask: bool = True):
        super().__init__()
        self.p = p
        self.scale_factor = 1 / (1 - p)
        self.return_mask = return_mask

    def forward(self, input: torch.Tensor, mask=None):
        if mask is None:
            mask = input.data.bernoulli(self.p)
        if self.return_mask:
            return input * mask * self.scale_factor, mask
        else:
            return input * mask * self.scale_factor


class MaskWithEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.embedding)

    def forward(self, input, mask):
        output = torch.where(mask, self.embedding.expand_as(input), input.detach())
        return output


class NormalExtractor(nn.Module):
    def __init__(self, include_loc: bool = True, num_samples: int = 1):
        super().__init__()
        self.include_loc = include_loc
        self.num_sample = num_samples

    def forward(self, x: torch.Tensor):
        x_loc, x_scale = x.chunk(2, -1)
        x_sample = x_loc.unsqueeze(-2).expand(*x_loc.shape[:-1], self.num_sample, -1)
        x_sample = x_sample + torch.randn_like(x_sample) * x_scale.exp().unsqueeze(-2)
        if self.include_loc:
            x_sample = torch.cat([x_sample, x_loc.unsqueeze(-2)], dim=-2)
        return x_sample.flatten(-2), x_loc, x_scale


def normalize(x: torch.Tensor, subtract_mean: bool = False):
    if subtract_mean:
        return (x - x.mean()) / x.std().clamp(1e-7)
    else:
        return x / x.std().clamp(1e-7)


def parse_keys(spec: Composite, keys: list[str]):
    """
    Parse the keys into `mlp_keys`, `cnn_keys`, and `aux_keys`.
    Keys ending with "_" are considered auxiliary keys.

    """
    mlp_keys = []
    cnn_keys = []
    aux_keys = []

    for key in keys:
        if key in spec.keys(True, True):
            _spec = spec[key]
        if key.endswith("_"):
            aux_keys.append(key)
            continue
        if _spec.ndim == 2:
            mlp_keys.append(key)
        else:
            cnn_keys.append(key)
    return mlp_keys, cnn_keys, aux_keys
