import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

from .base import Reward


class joint_acc_l2(Reward):
    def __init__(self, env, weight: float, joint_names: str = ".*"):
        super().__init__(env, weight)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.joint_ids = self.asset.find_joints(joint_names)[0]
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)
    
    def update(self):
        self.joint_acc = self.asset.data.joint_acc

    def _compute(self) -> torch.Tensor:
        r = -self.joint_acc[:, self.joint_ids].square().sum(dim=-1, keepdim=True)
        return r


class energy_l1(Reward):
    def __init__(self, env, weight: float, joint_names: str = ".*"):
        super().__init__(env, weight)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.joint_ids, self.joint_names = self.asset.find_joints(joint_names)
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)
        if self.env.backend in ("isaac", "mujoco"):
            self.get_torques = lambda: self.asset.data.applied_torque[:, self.joint_ids]
        elif self.env.backend == "mjlab":
            self.get_torques = lambda: self.asset.data.actuator_force[:, self.joint_ids]

    def update(self):
        self.torques = self.get_torques()
        self.joint_vel = self.asset.data.joint_vel[:, self.joint_ids]

    def _compute(self) -> torch.Tensor:
        power = (self.torques * self.joint_vel).abs()
        return -(power).sum(1, keepdim=True)


class energy_l2(Reward):
    """
    Penalize the energy of the joints. This is less commonly used than energy_l1 because it is much
    larger and therefore imposes a much stronger regularization.
    """
    def __init__(self, env, weight: float, joint_names: str = ".*"):
        super().__init__(env, weight)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.joint_ids, self.joint_names = self.asset.find_joints(joint_names)
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)

    def update(self):
        self.torques = self.asset.data.applied_torque[:, self.joint_ids]
        self.joint_vel = self.asset.data.joint_vel[:, self.joint_ids]

    def _compute(self) -> torch.Tensor:
        power = (self.torques * self.joint_vel)
        return -(power).square().sum(1, keepdim=True)


class joint_vel_l2(Reward):
    def __init__(self, env, weight: float, joint_names: str=".*"):
        super().__init__(env, weight)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.joint_ids, _ = self.asset.find_joints(joint_names)
        self.joint_vel = torch.zeros(
            self.num_envs, 2, len(self.joint_ids), device=self.device
        )

    def post_step(self, substep):
        self.joint_vel[:, substep % 2] = self.asset.data.joint_vel[:, self.joint_ids]

    def _compute(self) -> torch.Tensor:
        joint_vel = self.joint_vel.mean(1)
        return -joint_vel.square().sum(1, True)


class joint_vel_limits(Reward):
    def __init__(self, env, weight: float, joint_names: str = ".*", factor: float = 0.8):
        super().__init__(env, weight)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.joint_ids, self.joint_names = self.asset.find_joints(joint_names)
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)
        self.limits = torch.abs(self.asset.data.joint_vel_limits[:, self.joint_ids]) * factor
        self.update()
    
    def update(self):
        self.jvel = self.asset.data.joint_vel[:, self.joint_ids]

    def _compute(self) -> torch.Tensor:
        low, high = -self.limits, self.limits
        violation = (low - self.jvel).clamp_min(0) + (self.jvel - high).clamp_min(0)
        discount = torch.exp(- violation * 0.25).prod(1, True)
        self.env.discount.mul_(discount)
        rew = - violation.sum(1, True)
        return rew


class joint_tau_limits(Reward):
    def __init__(self, env, weight: float, joint_names: str = ".*", factor: float = 0.8):
        super().__init__(env, weight)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.joint_ids, self.joint_names = self.asset.find_joints(joint_names)
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)
        self.soft_limits = torch.abs(self.asset.data.joint_effort_limits[:, self.joint_ids]) * factor
        self.update()
    
    def update(self):
        self.applied_torque = self.asset.data.applied_torque[:, self.joint_ids]
    
    def _compute(self) -> torch.Tensor:
        low, high = -self.soft_limits, self.soft_limits
        violation = (low - self.applied_torque).clamp_min(0) + (self.applied_torque - high).clamp_min(0)
        discount = torch.exp(- violation * 0.25).prod(1, True)
        self.env.discount.mul_(discount)
        rew = - violation.sum(1, True)
        return rew


class joint_torque_disc(Reward):
    def __init__(self, env, weight: float, joint_names: str = ".*"):
        super().__init__(env, weight)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.joint_ids, self.joint_names = self.asset.find_joints(joint_names)
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)
        
        self.applied_torques = []
        self.computed_torques = []
        self.projected_joint_forces = []
    
    def update(self):
        self.applied_torque = self.asset.data.applied_torque[:, self.joint_ids]
        self.computed_torque = self.asset.data.computed_torque[:, self.joint_ids]
        self.projected_joint_force = self.asset.root_physx_view.get_dof_projected_joint_forces()[:, self.joint_ids]
        self.applied_torques.append(self.applied_torque)
        self.computed_torques.append(self.computed_torque)
        self.projected_joint_forces.append(self.projected_joint_force)
        if self.env.timestamp == 990:
            applied_torques = torch.stack(self.applied_torques).cpu()
            computed_torques = torch.stack(self.computed_torques).cpu()
            projected_joint_forces = torch.stack(self.projected_joint_forces).cpu()
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(4, 4, figsize=(10, 10), sharex=True, sharey=True)
            axes = axes.flatten()
            for i, name in enumerate(self.asset.joint_names):
                ax = axes[i]
                ax.plot(applied_torques[:, 0, i], label="applied")
                # ax.plot(computed_torques[:, 0, i], label="computed")
                ax.plot(projected_joint_forces[:, 0, i], label="projected")
                ax.set_ylim(-80, 80)
                ax.legend()
            plt.show()
    
    def _compute(self) -> torch.Tensor:
        # print((projected_joint_forces[:, self.joint_ids] - self.computed_torque).abs())
        discrepancy = (self.projected_joint_force - self.applied_torque).abs()
        return - discrepancy.sum(1, True)


class joint_deviation_l1(Reward):
    def __init__(self, env, weight: float, joint_names: str=".*"):
        super().__init__(env, weight)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.joint_ids, self.joint_names = self.asset.find_joints(joint_names)
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)
        self.default_joint_pos = self.asset.data.default_joint_pos[:, self.joint_ids].clone()
    
    def update(self):
        self.joint_pos = self.asset.data.joint_pos
    
    def _compute(self) -> torch.Tensor:
        deviation = self.joint_pos[:, self.joint_ids] - self.default_joint_pos
        return - deviation.abs().sum(1, True)


class joint_deviation_l2(Reward):
    def __init__(self, env, weight: float, joint_names: str=".*"):
        super().__init__(env, weight)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.joint_ids, self.joint_names = self.asset.find_joints(joint_names)
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)
        self.default_joint_pos = self.asset.data.default_joint_pos[:, self.joint_ids].clone()
    
    def update(self):
        self.joint_pos = self.asset.data.joint_pos
    
    def _compute(self) -> torch.Tensor:
        deviation = self.joint_pos[:, self.joint_ids] - self.default_joint_pos
        return - deviation.square().sum(1, True)


class joint_deviation_cum(Reward):
    """
    Penalize the cumulative deviation of the joints.
    """
    def __init__(self, env, weight: float, joint_names: str=".*"):
        super().__init__(env, weight)
        self.asset: Articulation = self.env.scene.articulations["robot"]
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
    
    def _compute(self) -> torch.Tensor:
        return - self.cum_deviation.sum(1, True)


class joint_torques_l2(Reward):
    def __init__(self, env, weight: float, joint_names: str = ".*"):
        super().__init__(env, weight)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.joint_ids = self.asset.find_joints(joint_names)[0]
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)
        if self.env.backend in ("isaac", "mujoco"):
            self.get_torques = lambda: self.asset.data.applied_torque[:, self.joint_ids]
        elif self.env.backend == "mjlab":
            self.get_torques = lambda: self.asset.data.actuator_force[:, self.joint_ids]
    
    def update(self):
        self.applied_torque = self.get_torques()

    def _compute(self) -> torch.Tensor:
        return -self.applied_torque.square().sum(1, keepdim=True)
