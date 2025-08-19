import torch
import warp as wp

from active_adaptation.envs.mdp.base import Command, Reward
from active_adaptation.utils.math import wrap_to_pi, quat_from_euler_xyz
from active_adaptation.utils.symmetry import SymmetryTransform


class SiriusDemoCommand(Command):
    def __init__(self, env, teleop: bool = False) -> None:
        super().__init__(env, teleop)
        
        with torch.device(self.device):
            self.cmd_lin_vel_w = torch.zeros(self.num_envs, 3)
            self.cmd_rpy_w = torch.zeros(self.num_envs, 3)
            self.cmd_contact = torch.zeros(self.num_envs, 4)
            self.cmd_time = torch.zeros(self.num_envs, 1)
            self.cmd_duration = torch.zeros(self.num_envs, 1)
        
        if self.env.sim.has_gui() and self.env.backend == "isaac":
            from isaaclab.markers import RED_ARROW_X_MARKER_CFG, VisualizationMarkers
            self.frame_marker = VisualizationMarkers(
                RED_ARROW_X_MARKER_CFG.replace(
                    prim_path="/Visuals/Command/frame",
                )
            )
            self.frame_marker.set_visibility(True)
    
    def reset(self, env_ids: torch.Tensor):
        self.cmd_rpy_w[env_ids, 2] = - self.asset.data.heading_w[env_ids]

    @property
    def command(self):
        cmd_yaw_b = wrap_to_pi(self.cmd_rpy_w[:, 2] - self.asset.data.heading_w)
        return cmd_yaw_b.reshape(self.num_envs, -1)
    
    def symmetry_transforms(self):
        return SymmetryTransform.cat([
            SymmetryTransform(perm=torch.arange(1), signs=torch.tensor([-1])), # flip yaw
        ])
    
    @property
    def command_mode(self):
        return torch.zeros(self.num_envs, 1, dtype=torch.int32, device=self.device)

    @property
    def yaw_error(self):
        return wrap_to_pi(self.cmd_rpy_w[:, 2] - self.asset.data.heading_w)
    
    def update(self):
        c1 = self.env.episode_length_buf % 50 == 0
        c2 = self.yaw_error.square() < 0.2
        c = (c1 & c2).reshape(self.num_envs)
        self.cmd_rpy_w[c, 2] = - self.asset.data.heading_w[c]
    
    def debug_draw(self):
        if self.env.sim.has_gui() and self.env.backend == "isaac":
            quat = quat_from_euler_xyz(*self.cmd_rpy_w.unbind(1))
            self.frame_marker.visualize(
                translations=self.asset.data.root_pos_w + torch.tensor([0.0, 0.0, 0.2], device=self.device),
                orientations=quat,
                scales=torch.tensor([4.0, 1.0, 0.1]).expand(self.num_envs, 3),
            )


class sirius_yaw(Reward[SiriusDemoCommand]):
    def compute(self) -> torch.Tensor:
        error = self.command_manager.yaw_error
        return torch.cos(error).reshape(self.num_envs, -1)

