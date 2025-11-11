import torch
import einops

from active_adaptation.envs.mdp import Reward
from active_adaptation.utils.math import wrap_to_pi
from .impedance import Impedance
from .impedance_manip import ImpedanceCommandManager


class impedance_pos(Reward[Impedance]):
    def __init__(self, env, weight: float):
        super().__init__(env, weight)
        self.asset = self.command_manager.asset

    def compute(self) -> torch.Tensor:
        target_pos_xy = self.command_manager.surr_pos_target[:, :, :2]
        current_pos_xy = self.asset.data.root_pos_w[:, :2].unsqueeze(1)
        diff = target_pos_xy - current_pos_xy
        error_l2 = diff.square().sum(dim=-1, keepdim=True)
        r = (- error_l2 / 0.25).exp().mean(1)
        return r.reshape(self.num_envs, 1)


class impedance_vel(Reward[Impedance]):
    def __init__(self, env, weight: float):
        super().__init__(env, weight)
        self.asset = self.command_manager.asset

    def compute(self) -> torch.Tensor:
        target_vel_xy = self.command_manager.surr_lin_vel_target[:, :, :2]
        diff = target_vel_xy - self.asset.data.root_lin_vel_w[:, :2].unsqueeze(1)
        error_l2 = diff.square().sum(dim=-1, keepdim=True)
        r = ((- error_l2 / 0.25).exp() - 0.25 * error_l2).mean(1)
        return r.reshape(self.num_envs, 1)


class impedance_acc(Reward):
    def __init__(self, env, weight: float):
        super().__init__(env, weight)
        self.command_manager: Impedance = self.env.command_manager
        self.asset = self.command_manager.asset

    def compute(self) -> torch.Tensor:
        lin_acc_w = self.asset.data.body_acc_w[:, 0, :2]
        error_l2 = (self.command_manager.ref_lin_acc_w[:, 0, :2] - lin_acc_w).square().sum(1, True)
        return torch.exp(- error_l2 / 2.0)


class impedance_yaw_pos(Reward[Impedance]):
    def __init__(self, env, weight: float):
        super().__init__(env, weight)
        self.asset = self.command_manager.asset
        
    def compute(self) -> torch.Tensor:
        target_yaw = self.command_manager.surr_yaw_target 
        diff = target_yaw - self.asset.data.heading_w.reshape(-1, 1, 1)
        diff = wrap_to_pi(diff)
        error_l2 = diff.square()
        r = torch.exp(-error_l2 / 0.25).mean(1)
        return r


class impedance_yaw_vel(Reward[Impedance]):
    def __init__(self, env, weight: float):
        super().__init__(env, weight)
        self.asset = self.command_manager.asset

    def compute(self) -> torch.Tensor:
        target_yaw_vel = self.command_manager.surr_yaw_vel_target
        current_yaw_vel = self.asset.data.root_ang_vel_w[:, 2:3].unsqueeze(1)
        diff = target_yaw_vel - current_yaw_vel
        error_l2 = diff.square()
        r = ((- error_l2 / 0.25).exp() - 0.25 * error_l2).mean(1)
        return r.reshape(self.num_envs, 1)


class impedance_pos_error(Reward[Impedance]):
    def __init__(self, env, weight: float):
        super().__init__(env, weight)
        self.impedance: Impedance = self.env.command_manager

    def compute(self) -> torch.Tensor:
        diff = self.impedance.ref_pos_w[:, [*self.impedance.surr_steps, -1]] - self.impedance.get_pos_w().unsqueeze(1)
        error_l2 = diff[:, :, :2].square().sum(dim=-1, keepdim=True)
        return error_l2.mean(1)


class impedance_vel_error(Reward[Impedance]):
    def __init__(self, env, weight: float):
        super().__init__(env, weight)
        self.impedance: Impedance = self.env.command_manager

    def compute(self) -> torch.Tensor:
        diff = self.impedance.ref_lin_vel_w[:, [*self.impedance.surr_steps, -1]] - self.impedance.get_lin_vel_w().unsqueeze(1)
        error_l2 = diff[:, :, :2].square().sum(dim=-1, keepdim=True)
        return error_l2.mean(1)


# class impedance_acc_error(Reward[Impedance]):
#     def __init__(self, env, weight: float):
#         super().__init__(env, weight)
#         self.impedance: Impedance = self.env.command_manager

#     def compute(self) -> torch.Tensor:
#         diff = self.impedance.ref_lin_acc_w[:, 0] - self.impedance.asset.data.body_acc_w[:, 0, :3]
#         error_l2 = diff[:, :2].square().sum(dim=-1, keepdim=True)
#         return error_l2

