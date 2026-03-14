import torch

from active_adaptation.envs.mdp.rewards.base import Reward
from active_adaptation.utils.math import normalize, quat_rotate

from .command import Game


class chase_distance(Reward[Game]):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.last_distance = torch.zeros(self.num_envs, 1, device=self.device)
        self.distance_change = torch.zeros(self.num_envs, 1, device=self.device)

    def update(self):
        self.distance_change = self.command_manager.distance - self.last_distance
        self.last_distance = self.command_manager.distance

    def _compute(self) -> torch.Tensor:
        rew = torch.where(
            self.command_manager.role[:, None] == 0,
            -self.distance_change,
            self.distance_change,
        )
        return rew.reshape(self.num_envs, 1)


class chase_velocity(Reward[Game]):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset = self.command_manager.asset

    def _compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        is_active = torch.arange(self.num_envs, device=self.device) % 2 == 0
        direction = normalize(self.command_manager.target_diff[:, :2])
        velocity = self.asset.data.root_link_lin_vel_w[:, :2]
        rew = torch.sum(direction * velocity, dim=1, keepdim=True)
        rew = torch.where(rew > 0, rew.log1p(), rew)
        return rew.reshape(self.num_envs, 1), is_active.reshape(self.num_envs, 1)


class evade(Reward[Game]):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)

    def _compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        is_active = torch.arange(self.num_envs, device=self.device) % 2 == 1
        rew = 1 - torch.exp(-self.command_manager.distance * 0.5).reshape(
            self.num_envs, 1
        )
        return rew.reshape(self.num_envs, 1), is_active.reshape(self.num_envs, 1)


class target_in_sight(Reward[Game]):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset = self.command_manager.asset

    def _compute(self) -> torch.Tensor:
        forward_vec = quat_rotate(
            self.asset.data.root_link_quat_w,
            torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, 3),
        )
        diff = normalize(self.command_manager.target_diff)
        rew = torch.sum(forward_vec[:, :2] * diff[:, :2], dim=1, keepdim=True)
        rew = torch.where(self.command_manager.role[:, None] == 0, rew, -rew)
        return rew.reshape(self.num_envs, 1)


class caught_reward(Reward[Game]):
    def _compute(self) -> torch.Tensor:
        caught = self.command_manager.target_caught.float()
        return torch.where(self.command_manager.role[:, None] == 0, caught, -caught)


__all__ = [
    "chase_distance",
    "chase_velocity",
    "evade",
    "target_in_sight",
    "caught_reward",
]
