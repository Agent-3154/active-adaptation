import torch
import warp as wp

from active_adaptation.envs.mdp.base import Command, Reward
from active_adaptation.utils.math import quat_from_euler_xyz, wrap_to_pi
from active_adaptation.utils.symmetry import SymmetryTransform


@wp.kernel
def sample_command(
    heading_w: wp.array(dtype=wp.float32),
    cmd_lin_vel_w: wp.array(dtype=wp.vec3),
    cmd_lin_vel_b: wp.array(dtype=wp.vec3),
    use_lin_vel_w: wp.array(dtype=wp.bool),
    cmd_rpy_w: wp.array(dtype=wp.vec3),
    cmd_ang_vel_w: wp.array(dtype=wp.vec3),
    sample: wp.array(dtype=wp.bool),
    next_mode: wp.array(dtype=wp.int32),
    seed: wp.int32,
):
    tid = wp.tid()
    seed_ = wp.rand_init(seed, tid)
    if sample[tid]:
        if next_mode[tid] == 0:
            cmd_lin_vel_w[tid] = wp.vec3()
            cmd_lin_vel_b[tid] = wp.vec3(wp.randf(seed_, 0.5, 1.0), 0.0, 0.0)
            use_lin_vel_w[tid] = False
            cmd_rpy_w[tid] = wp.vec3(0.0, 0.0, heading_w[tid])
            cmd_ang_vel_w[tid].z = wp.randf(seed_, wp.PI / 4.0, wp.PI / 2.0)
        if next_mode[tid] == 1:
            cmd_lin_vel_w[tid] = wp.vec3(1.0, 0.0, 0.0)
            cmd_lin_vel_b[tid] = wp.vec3()
            cmd_rpy_w[tid] = wp.vec3(0.0, 0.0, heading_w[tid])
            cmd_ang_vel_w[tid] = wp.vec3(0.0, 0.0, 0.0)
            use_lin_vel_w[tid] = True


class SiriusDemoCommand(Command):
    def __init__(self, env, teleop: bool = False) -> None:
        super().__init__(env, teleop)

        with torch.device(self.device):
            self.cmd_lin_vel_w = torch.zeros(self.num_envs, 3)
            self.cmd_lin_vel_b = torch.zeros(self.num_envs, 3)
            self.use_lin_vel_w = torch.zeros(self.num_envs, 1, dtype=bool)
            self.cmd_ang_vel_w = torch.zeros(self.num_envs, 3)
            self.cmd_rpy_w = torch.zeros(self.num_envs, 3)
            self.cmd_contact = torch.zeros(self.num_envs, 4)
            self.cmd_time = torch.zeros(self.num_envs, 1)
            self.cmd_duration = torch.zeros(self.num_envs, 1)
            self.cmd_mode = torch.zeros(self.num_envs, 1, dtype=torch.int32)

        if self.env.sim.has_gui() and self.env.backend == "isaac":
            from isaaclab.markers import RED_ARROW_X_MARKER_CFG, VisualizationMarkers

            self.frame_marker = VisualizationMarkers(
                RED_ARROW_X_MARKER_CFG.replace(
                    prim_path="/Visuals/Command/frame",
                )
            )
            self.frame_marker.set_visibility(True)
        self.seed = wp.rand_init(0)

    def reset(self, env_ids: torch.Tensor):
        self.cmd_rpy_w[env_ids, 2] = -self.asset.data.heading_w[env_ids]

    @property
    def command(self):
        cmd_rpy_b = self.cmd_rpy_w.clone()
        cmd_rpy_b[:, 2] = wrap_to_pi(cmd_rpy_b[:, 2] - self.asset.data.heading_w)
        return torch.cat(
            [
                self.cmd_lin_vel_b,
                self.cmd_ang_vel_w,
                cmd_rpy_b,
            ],
            dim=1,
        )

    def symmetry_transforms(self):
        return SymmetryTransform.cat(
            [
                SymmetryTransform(
                    perm=torch.arange(3), signs=torch.tensor([1, -1, 1])
                ),  # flip y
                SymmetryTransform(
                    perm=torch.arange(3), signs=torch.tensor([-1, 1, -1])
                ),  # flip roll and yaw
                SymmetryTransform(
                    perm=torch.arange(3), signs=torch.tensor([-1, 1, -1])
                ),  # flip yaw
            ]
        )

    @property
    def command_mode(self):
        return torch.zeros(self.num_envs, 1, dtype=torch.int32, device=self.device)

    @property
    def yaw_error(self):
        return wrap_to_pi(self.cmd_rpy_w[:, 2] - self.asset.data.heading_w)

    def update(self):
        c1 = self.env.episode_length_buf % 50 == 0
        c2 = torch.rand(self.num_envs, device=self.device) < 0.5
        resample = c1 & c2
        next_mode = torch.randint(
            0, 2, (self.num_envs,), dtype=torch.int32, device=self.device
        )

        wp.launch(
            sample_command,
            dim=self.num_envs,
            inputs=[
                wp.from_torch(self.asset.data.heading_w, return_ctype=True),
                wp.from_torch(self.cmd_lin_vel_w, return_ctype=True),
                wp.from_torch(self.cmd_lin_vel_b, return_ctype=True),
                wp.from_torch(self.use_lin_vel_w, return_ctype=True),
                wp.from_torch(self.cmd_rpy_w, return_ctype=True),
                wp.from_torch(self.cmd_ang_vel_w, return_ctype=True),
                wp.from_torch(resample, return_ctype=True),
                wp.from_torch(next_mode),
                self.seed,
            ],
            device=self.device.type,
        )
        self.cmd_rpy_w = self.cmd_rpy_w + self.cmd_ang_vel_w * self.env.step_dt

    def debug_draw(self):
        if self.env.sim.has_gui() and self.env.backend == "isaac":
            quat = quat_from_euler_xyz(*self.cmd_rpy_w.unbind(1))
            self.frame_marker.visualize(
                translations=self.asset.data.root_pos_w
                + torch.tensor([0.0, 0.0, 0.2], device=self.device),
                orientations=quat,
                scales=torch.tensor([4.0, 1.0, 0.1]).expand(self.num_envs, 3),
            )
            self.env.debug_draw.vector(
                self.asset.data.root_pos_w + torch.tensor([0.0, 0.0, 0.2], device=self.device),
                self.cmd_lin_vel_w
            )


class sirius_yaw(Reward[SiriusDemoCommand]):
    def compute(self) -> torch.Tensor:
        error = self.command_manager.yaw_error
        return torch.cos(error).reshape(self.num_envs, -1)


class sirius_lin_vel_xy(Reward[SiriusDemoCommand]):
    def compute(self) -> torch.Tensor:
        target_lin_vel_xy = torch.where(
            self.command_manager.use_lin_vel_w,
            self.command_manager.cmd_lin_vel_w[:, :2],
            self.command_manager.cmd_lin_vel_b[:, :2],
        )
        current_lin_vel_xy = torch.where(
            self.command_manager.use_lin_vel_w,
            self.command_manager.asset.data.root_lin_vel_w[:, :2],
            self.command_manager.asset.data.root_lin_vel_b[:, :2],
        )
        error = target_lin_vel_xy - current_lin_vel_xy
        error = error.square().sum(1, True)
        return torch.exp(-error / 0.25)


class sirius_ang_vel_z(Reward[SiriusDemoCommand]):
    def compute(self) -> torch.Tensor:
        error = (
            self.command_manager.cmd_ang_vel_w[:, 2]
            - self.command_manager.asset.data.root_ang_vel_b[:, 2]
        )
        error = error.square().reshape(self.num_envs, 1)
        return torch.exp(-error / 0.25)


class sirius_base_height(Reward[SiriusDemoCommand]):
    def compute(self) -> torch.Tensor:
        error = 0.45 - self.command_manager.asset.data.root_pos_w[:, 2]
        error = error.square().reshape(self.num_envs, 1)
        rew = torch.exp(-error / 0.1)
        return rew
