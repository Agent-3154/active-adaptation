import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

from active_adaptation.envs.mdp.base import Reward


class joint_acc_l2(Reward):
    def __init__(self, env, weight: float, enabled: bool = True, joint_names: str = ".*"):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids = self.asset.find_joints(joint_names)[0]
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)

    def compute(self) -> torch.Tensor:
        r = -self.asset.data.joint_acc[:, self.joint_ids].square().sum(dim=-1, keepdim=True)
        if hasattr(self.asset.data, "linvel_exp"):
            return r * (0.5 + 0.5 * self.asset.data.linvel_exp)
        else:
            return r


class energy_l1(Reward):
    def __init__(self, env, weight: float, enabled: bool = True, joint_names: str = ".*"):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, self.joint_names = self.asset.find_joints(joint_names)
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)

    def update(self):
        self.torques = self.asset.data.applied_torque[:, self.joint_ids]
        self.joint_vel = self.asset.data.joint_vel[:, self.joint_ids]

    def compute(self) -> torch.Tensor:
        power = (self.torques * self.joint_vel).abs()
        return -(power).sum(1, keepdim=True)


class energy_l2(Reward):
    """
    Penalize the energy of the joints. This is less commonly used than energy_l1 because it is much
    larger and therefore imposes a much stronger regularization.
    """
    def __init__(self, env, weight: float, enabled: bool = True, joint_names: str = ".*"):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, self.joint_names = self.asset.find_joints(joint_names)
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)

    def update(self):
        self.torques = self.asset.data.applied_torque[:, self.joint_ids]
        self.joint_vel = self.asset.data.joint_vel[:, self.joint_ids]

    def compute(self) -> torch.Tensor:
        power = (self.torques * self.joint_vel)
        return -(power).square().sum(1, keepdim=True)


class joint_vel_l2(Reward):
    def __init__(self, env, joint_names: str, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, _ = self.asset.find_joints(joint_names)
        self.joint_vel = torch.zeros(
            self.num_envs, 2, len(self.joint_ids), device=self.device
        )

    def post_step(self, substep):
        self.joint_vel[:, substep % 2] = self.asset.data.joint_vel[:, self.joint_ids]

    def compute(self) -> torch.Tensor:
        joint_vel = self.joint_vel.mean(1)
        return -joint_vel.square().sum(1, True)


class joint_vel_limits(Reward):
    def __init__(self, env, weight: float, enabled: bool = True, joint_names: str = ".*", factor: float = 0.8):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, self.joint_names = self.asset.find_joints(joint_names)
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)
        self.limits = torch.abs(self.asset.data.joint_vel_limits[:, self.joint_ids]) * factor
    
    def compute(self) -> torch.Tensor:
        jvel = self.asset.data.joint_vel[:, self.joint_ids]
        low, high = -self.limits, self.limits
        violation = (low - jvel).clamp_min(0) + (jvel - high).clamp_min(0)
        return - violation.sum(1, True)


class joint_torque_limits(Reward):
    def __init__(self, env, weight: float, enabled: bool = True, joint_names: str = ".*", factor: float = 0.8):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, self.joint_names = self.asset.find_joints(joint_names)
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)
        self.soft_limits = torch.abs(self.asset.data.joint_effort_limits[:, self.joint_ids]) * factor
    
    def compute(self) -> torch.Tensor:
        applied_torque = self.asset.data.applied_torque[:, self.joint_ids]
        low, high = -self.soft_limits, self.soft_limits
        violation = (low - applied_torque).clamp_min(0) + (applied_torque - high).clamp_min(0)
        return - violation.sum(1, True)


class joint_deviation_l1(Reward):
    def __init__(self, env, weight: float, enabled: bool = True, joint_names: str=".*"):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, self.joint_names = self.asset.find_joints(joint_names)
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)
        self.default_joint_pos = self.asset.data.default_joint_pos[:, self.joint_ids].clone()
    
    def update(self):
        self.joint_pos = self.asset.data.joint_pos[:, self.joint_ids]
    
    def compute(self) -> torch.Tensor:
        deviation = self.joint_pos - self.default_joint_pos
        return - deviation.abs().sum(1, True)


class joint_deviation_l2(Reward):
    def __init__(self, env, weight: float, enabled: bool = True, joint_names: str=".*"):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, self.joint_names = self.asset.find_joints(joint_names)
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)
        self.default_joint_pos = self.asset.data.default_joint_pos[:, self.joint_ids].clone()
    
    def update(self):
        self.joint_pos = self.asset.data.joint_pos[:, self.joint_ids]
    
    def compute(self) -> torch.Tensor:
        deviation = self.joint_pos - self.default_joint_pos
        return - deviation.square().sum(1, True)


class joint_deviation_cum(Reward):
    """
    Penalize the cumulative deviation of the joints.
    """
    def __init__(self, env, weight: float, enabled: bool = True, joint_names: str=".*"):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, self.joint_names = self.asset.find_joints(joint_names)
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)
        self.default_joint_pos = self.asset.data.default_joint_pos[:, self.joint_ids].clone()
        self.cum_deviation = torch.zeros(self.num_envs, len(self.joint_ids), device=self.device)
        self.cum_thres = 0.15
    
    def reset(self, env_ids: torch.Tensor):
        self.cum_deviation[env_ids] = 0.0
    
    def update(self):
        self.joint_pos = self.asset.data.joint_pos[:, self.joint_ids]
        deviation = torch.abs(self.joint_pos - self.default_joint_pos)
        self.cum_deviation = torch.where(deviation > self.cum_thres, self.cum_deviation + self.cum_thres, 0.)
    
    def compute(self) -> torch.Tensor:
        return - self.cum_deviation.sum(1, True)

