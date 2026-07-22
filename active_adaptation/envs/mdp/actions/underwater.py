from __future__ import annotations

import torch

from typing import TYPE_CHECKING, Tuple, cast
from typing_extensions import override

from active_adaptation.utils.symmetry import SymmetryTransform

from .base import ActionV2
from tensordict import TensorDictBase

if TYPE_CHECKING:
    from active_adaptation.envs.env_base import _EnvBase
    from active_adaptation.envs.robots.underwater import UnderwaterRobotData


class UnderwaterThrottle(ActionV2):
    """Throttle action for underwater robots.

    The action directly controls per-rotor normalized throttle in ``[-1, 1]``.
    Throttle-to-thrust conversion is handled by ``UnderwaterRobot.write_data_to_sim``.
    """
    uw: "UnderwaterRobotData"

    def __init__(
        self,
        action_scaling: float = 1.0,
        alpha_range: Tuple[float, float] = (0.5, 1.0),
    ):
        super().__init__()
        self.action_scaling = float(action_scaling)
        self.alpha_range = tuple(alpha_range)

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        if not hasattr(self.asset, "data_underwater"):
            raise RuntimeError(
                "UnderwaterThrottle requires robot.data_underwater to be initialized."
            )
        self.uw = cast("UnderwaterRobotData", self.asset.data_underwater)
        self.action_dim = int(self.uw.throttle_cmd.shape[-1])
        self.action_buf = torch.zeros(self.num_envs, 4, self.action_dim, device=self.device)
        self.applied_action = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self.alpha = torch.ones(self.num_envs, 1, device=self.device)

    @override
    def reset(self, env_ids: torch.Tensor, tensordict: TensorDictBase):
        alpha = torch.empty(len(env_ids), 1, device=self.device)
        alpha.uniform_(self.alpha_range[0], self.alpha_range[1])
        self.alpha[env_ids] = alpha
        self.action_buf[env_ids] = 0.0
        self.applied_action[env_ids] = 0.0
        self.uw.throttle_cmd[env_ids] = 0.0
        self.uw.throttle[env_ids] = 0.0

    @override
    def process_action(self, action: torch.Tensor | None):
        if action is None:
            return
        self.action_buf = self.action_buf.roll(1, dims=1)
        self.action_buf[:, 0] = action

    @override
    def apply_action(self, substep: int):
        self.applied_action.lerp_(self.action_buf[:, 0], self.alpha)
        self.uw.throttle_cmd.copy_(
            torch.clamp(self.applied_action * self.action_scaling, -1.0, 1.0)
        )

    @override
    def symmetry_transform(self):
        return SymmetryTransform(
            perm=torch.arange(self.action_dim),
            signs=[1] * self.action_dim,
        )


__all__ = ["UnderwaterThrottle"]
