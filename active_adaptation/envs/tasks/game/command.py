import torch

from active_adaptation.envs.mdp.commands.base import Command
from active_adaptation.utils.math import quat_rotate_inverse, sample_quat_yaw
from active_adaptation.utils.symmetry import SymmetryTransform


class Game(Command):
    def __init__(self, env, catch_radius: float = 0.8) -> None:
        super().__init__(env)
        self.catch_radius = catch_radius

        with torch.device(self.device):
            self.role = torch.arange(self.num_envs) % 2
            self.target_caught_time = torch.zeros(self.num_envs, 1)
            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=bool)

        if self.env.sim.has_gui() and self.env.backend == "isaac":
            from isaaclab.markers import RED_ARROW_X_MARKER_CFG, VisualizationMarkers

            self.frame_marker = VisualizationMarkers(
                RED_ARROW_X_MARKER_CFG.replace(
                    prim_path="/Visuals/Command/frame",
                )
            )
            self.frame_marker.set_visibility(True)
        self.update()

    @property
    def command(self):
        arange = torch.arange(self.num_envs, device=self.device)
        return torch.cat(
            [
                quat_rotate_inverse(self.asset.data.root_link_quat_w, self.target_diff),
                quat_rotate_inverse(
                    self.asset.data.root_link_quat_w, self.target_lin_vel_w
                ),
                (arange % 2 == 0).reshape(self.num_envs, 1),
                (arange % 2 == 1).reshape(self.num_envs, 1),
            ],
            dim=-1,
        )

    @property
    def command_mode(self):
        return self.role.reshape(self.num_envs, 1)

    def symmetry_transform(self):
        return SymmetryTransform(
            perm=torch.arange(8),
            signs=torch.tensor([1, -1, 1, 1, -1, 1, 1, 1]),
        )

    def sample_init(self, env_ids: torch.Tensor) -> torch.Tensor:
        chase = env_ids % 2 == 0
        init_root_state = self.init_root_state[env_ids]
        origins = self.env.scene.get_spawn_origins(env_ids)
        init_pos_even = origins[chase]
        offset = torch.zeros_like(init_pos_even)
        offset[:, 0].uniform_(3.0, 4.0).mul_(
            torch.randn(offset.shape[0], device=self.device).sign()
        )
        init_pos_odd = init_pos_even + offset
        init_root_state[chase, :3] += init_pos_even
        init_root_state[~chase, :3] += init_pos_odd
        init_root_state[:, 3:7] = sample_quat_yaw(len(env_ids), device=self.device)
        return init_root_state

    def reset(self, env_ids: torch.Tensor):
        self.target_caught_time[env_ids] = 0.0
        return super().reset(env_ids)

    def update(self):
        self.target_pos_w = torch.stack(
            [
                self.asset.data.root_pos_w[1::2],
                self.asset.data.root_pos_w[::2],
            ],
            1,
        ).reshape(self.num_envs, 3)
        self.target_lin_vel_w = torch.cat(
            [
                self.asset.data.root_link_lin_vel_w[1::2],
                self.asset.data.root_link_lin_vel_w[::2],
            ],
            1,
        ).reshape(self.num_envs, 3)
        self.target_diff = self.target_pos_w - self.asset.data.root_pos_w
        self.distance = self.target_diff[:, :2].norm(dim=-1, keepdim=True)
        self.target_caught = self.distance < 0.8
        self.target_caught_time = torch.where(
            self.target_caught,
            self.target_caught_time + self.env.step_dt,
            torch.zeros_like(self.target_caught_time),
        )

    def debug_draw(self):
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w[::2],
            self.target_diff[::2],
            color=(1, 0, 0, 1),
        )
        if hasattr(self, "frame_marker"):
            self.frame_marker.visualize(
                self.asset.data.root_pos_w[::2]
                + torch.tensor([0.0, 0.0, 0.2], device=self.device),
                self.asset.data.root_link_quat_w[::2],
                scales=torch.tensor([[4.0, 1.0, 0.1]], device=self.device).expand(
                    self.num_envs // 2, 3
                ),
            )


__all__ = ["Game"]
