import torch
from typing import TYPE_CHECKING, Dict, List, Tuple
from typing_extensions import override

from active_adaptation.envs.mdp.rewards.base import Reward

if TYPE_CHECKING:
    from active_adaptation.envs.env_base import EnvBase


class action_rate_l2(Reward):
    """Penalize the rate of change of the action."""

    def __init__(
        self,
        weight: float,
        key: str = "action",
        names: str | List[str] = ".*",
        enabled: bool = True,
        track_var: bool = False,
    ):
        super().__init__(weight, enabled=enabled, track_var=track_var)
        self.key = key
        self.names = names

    @override
    def _initialize(self, env: "EnvBase"):
        super()._initialize(env)
        self.action_manager = self.env.input_managers[self.key]
        self.indices, self.names = self.action_manager.find_names(self.names)
        assert self.action_manager.action_buf.shape[-1] == self.action_manager.action_dim

    def _compute(self) -> torch.Tensor:
        action_buf = self.action_manager.action_buf[:, :, self.indices]
        action_diff = action_buf[:, 0] - action_buf[:, 1]
        rew = -action_diff.square().sum(dim=-1, keepdim=True)
        return rew


class action_rate2_l2(Reward):
    """Penalize the second order rate of change of the action."""

    def __init__(
        self,
        weight: float,
        key: str = "action",
        names: str | List[str] = ".*",
        enabled: bool = True,
        track_var: bool = False,
    ):
        super().__init__(weight, enabled=enabled, track_var=track_var)
        self.key = key
        self.names = names

    @override
    def _initialize(self, env: "EnvBase"):
        super()._initialize(env)
        self.action_manager = self.env.input_managers[self.key]
        self.indices, self.names = self.action_manager.find_names(self.names)
        assert self.action_manager.action_buf.shape[-1] == self.action_manager.action_dim

    def _compute(self) -> torch.Tensor:
        action_buf = self.action_manager.action_buf[:, :, self.indices]
        action_diff = action_buf[:, 0] - 2 * action_buf[:, 1] + action_buf[:, 2]
        rew = -action_diff.square().sum(dim=-1, keepdim=True)
        return rew


class action_saturation(Reward):
    """Penalize actions that leave the allowed range.

    Soft overshoot cost: ``-(relu(low - a) + relu(a - high)).square().sum()``.
    Use a uniform ``(low, high)`` for all dims, or a ``{name_pattern: (low, high)}``
    map for per-action limits (unmatched dims are not penalized).
    """

    def __init__(
        self,
        weight: float,
        range: Tuple[float, float] | Dict[str, Tuple[float, float]] = (-1.0, 1.0),
        key: str = "action",
        enabled: bool = True,
        track_var: bool = False,
    ):
        super().__init__(weight, enabled=enabled, track_var=track_var)
        self.key = key
        self.range = dict(range)

    @override
    def _initialize(self, env: "EnvBase"):
        super()._initialize(env)
        self.action_manager = self.env.input_managers[self.key]
        action_dim = self.action_manager.action_dim
        assert self.action_manager.action_buf.shape[-1] == action_dim

        low = torch.full((action_dim,), float("-inf"), device=self.device)
        high = torch.full((action_dim,), float("inf"), device=self.device)

        if isinstance(self.range, dict):
            for pattern, (lo, hi) in self.range.items():
                indices, _ = self.action_manager.find_names(pattern)
                if not indices:
                    raise ValueError(
                        f"action_saturation: no action names matched {pattern!r} "
                        f"in {self.action_manager.names}"
                    )
                low[indices] = float(lo)
                high[indices] = float(hi)
        else:
            lo, hi = self.range
            low[:] = float(lo)
            high[:] = float(hi)

        if (low > high).any():
            raise ValueError(
                f"action_saturation: low > high for some dims "
                f"(low={low.tolist()}, high={high.tolist()})"
            )

        self.low = low
        self.high = high

    def _compute(self) -> torch.Tensor:
        action = self.action_manager.action_buf[:, 0]
        violation = (self.low - action).clamp_min(0.0) + (action - self.high).clamp_min(0.0)
        return -violation.square().sum(dim=-1, keepdim=True)


class body_angvel_penalty(Reward):
    """Penalize the angular velocity of the body."""

    def __init__(
        self,
        weight: float,
        body_names: str | List[str] = ".*",
        mask: List[float] = [1.0, 1.0, 1.0],
        enabled: bool = True,
        track_var: bool = False,
    ):
        super().__init__(weight, enabled=enabled, track_var=track_var)
        self.body_names = body_names
        self.mask = mask

    @override
    def _initialize(self, env: "EnvBase"):
        super()._initialize(env)
        self.asset = self.env.scene.entities["robot"]
        self.body_ids = self.asset.find_bodies(self.body_names)[0]
        self.body_ids = torch.tensor(self.body_ids, device=self.device)
        self.mask = torch.tensor(self.mask, device=self.device)

    def _compute(self) -> torch.Tensor:
        if self.env.backend == "isaaclab":
            body_angvel = self.asset.data.body_com_ang_vel_w[:, self.body_ids]
        elif self.env.backend == "mjlab":
            body_angvel = self.asset.data.body_link_ang_vel_w[:, self.body_ids]
        else:
            raise ValueError(f"Unsupported backend: {self.env.backend}")
        rew = -(body_angvel * self.mask).square().sum(dim=-1, keepdim=True)
        return rew.sum(1).reshape(self.num_envs, 1)
