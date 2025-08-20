import torch
import warp as wp

from active_adaptation.envs.mdp.base import Command, Reward
from active_adaptation.utils.math import (
    quat_rotate,
    quat_rotate_inverse,
    quat_from_euler_xyz,
    wrap_to_pi,
)
from active_adaptation.utils.symmetry import SymmetryTransform


PRE_JUMP_TIME = 0.5
POST_JUMP_TIME = 0.5


@wp.kernel
def sample_command(
    heading_w: wp.array(dtype=wp.float32),
    cmd_lin_vel_w: wp.array(dtype=wp.vec3),
    cmd_lin_vel_b: wp.array(dtype=wp.vec3),
    use_lin_vel_w: wp.array(dtype=wp.bool),
    cmd_rpy_w: wp.array(dtype=wp.vec3),
    cmd_ang_vel_w: wp.array(dtype=wp.vec3),
    sample: wp.array(dtype=wp.bool),
    mode: wp.array(dtype=wp.int32),
    next_mode: wp.array(dtype=wp.int32),
    cmd_time: wp.array(dtype=wp.float32),
    cmd_duration: wp.array(dtype=wp.float32),
    seed: wp.int32,
):
    tid = wp.tid()
    seed_ = wp.rand_init(seed, tid)
    if sample[tid]:
        if next_mode[tid] == 0:
            cmd_lin_vel_w[tid] = wp.vec3()
            cmd_lin_vel_b[tid] = wp.vec3(wp.randf(seed_, 0.4, 1.2), wp.randf(seed_, -0.5, 0.5), 0.0)
            use_lin_vel_w[tid] = False
            cmd_rpy_w[tid] = wp.vec3(0.0, 0.0, heading_w[tid])
            yaw_rate = wp.randf(seed_, wp.PI / 4.0, wp.PI / 2.0)
            cmd_ang_vel_w[tid].z = yaw_rate * wp.sign(wp.randn(seed_))
            cmd_duration[tid] = wp.randf(seed_, 1.0, 3.0)
        if next_mode[tid] == 1:
            cmd_lin_vel_w[tid] = wp.vec3(1.0, 0.0, 0.0)
            cmd_lin_vel_b[tid] = wp.vec3()
            cmd_rpy_w[tid] = wp.vec3(0.0, 0.0, heading_w[tid])
            cmd_ang_vel_w[tid] = wp.vec3(0.0, 0.0, 0.0)
            use_lin_vel_w[tid] = True
            cmd_duration[tid] = 1.7
        cmd_time[tid] = 0.0  # reset time
        mode[tid] = next_mode[tid]


@wp.kernel
def step_command(
    cmd_ang_vel_w: wp.array(dtype=wp.vec3),
    cmd_rpy_w: wp.array(dtype=wp.vec3),
    cmd_contact: wp.array(dtype=wp.vec4),
    cmd_height: wp.array(dtype=wp.float32),
    mode: wp.array(dtype=wp.int32),
    cmd_time: wp.array(dtype=wp.float32),
    cmd_duration: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    time = cmd_time[tid]
    if mode[tid] == 0:
        cmd_height[tid] = 0.45
        cmd_contact[tid] = wp.vec4(0.0, 0.0, 0.0, 0.0)
    elif mode[tid] == 1:  # jump
        if time < PRE_JUMP_TIME :
            cmd_height[tid] = 0.40
            cmd_contact[tid] = wp.vec4(0.0, 0.0, 0.0, 0.0)
        elif time < PRE_JUMP_TIME + 0.2:
            cmd_height[tid] = 0.40 + (time - PRE_JUMP_TIME)
            cmd_contact[tid] = wp.vec4(0.0, 0.0, 0.0, 0.0)
        elif time < cmd_duration[tid] - POST_JUMP_TIME:
            cmd_height[tid] = 0.60
            cmd_contact[tid] = - wp.vec4(1.0, 1.0, 1.0, 1.0)
        else:
            cmd_contact[tid] = wp.vec4(0.0, 0.0, 0.0, 0.0)
            cmd_height[tid] = 0.45
    cmd_time[tid] += 0.02
    cmd_rpy_w[tid] += cmd_ang_vel_w[tid] * 0.02


class SiriusDemoCommand(Command):
    def __init__(self, env, teleop: bool = False) -> None:
        super().__init__(env, teleop)

        with torch.device(self.device):
            self.cmd_lin_vel_w = torch.zeros(self.num_envs, 3)
            self.cmd_lin_vel_b = torch.zeros(self.num_envs, 3)
            self.use_lin_vel_w = torch.zeros(self.num_envs, 1, dtype=bool)
            self.cmd_height = torch.zeros(self.num_envs, 1)
            self.cmd_ang_vel_w = torch.zeros(self.num_envs, 3)
            self.cmd_rpy_w = torch.zeros(self.num_envs, 3)
            self.cmd_contact = torch.zeros(self.num_envs, 4)
            self.cmd_time = torch.zeros(self.num_envs, 1)
            self.cmd_duration = torch.zeros(self.num_envs, 1)
            self.cmd_mode = torch.zeros(self.num_envs, dtype=torch.int32)

            self.transition_prob = torch.tensor([
                [0.2, 0.8],
                [1.0, 0.0],
            ], device=self.device)

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
        resample = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        resample[env_ids] = True
        next_mode = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
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
                wp.from_torch(self.cmd_mode, return_ctype=True),
                wp.from_torch(next_mode, return_ctype=True),
                wp.from_torch(self.cmd_time, return_ctype=True),
                wp.from_torch(self.cmd_duration, return_ctype=True),
                self.seed,
            ],
            device=self.device.type,
        )
        
    @property
    def command(self):
        cmd_rpy_b = self.cmd_rpy_w.clone()
        cmd_rpy_b[:, 2] = wrap_to_pi(cmd_rpy_b[:, 2] - self.asset.data.heading_w)
        return torch.cat(
            [
                self.obs_cmd_lin_vel_b,
                self.cmd_ang_vel_w,
                cmd_rpy_b,
                torch.where(self.cmd_mode[:, None] == 1, self.cmd_time, torch.zeros_like(self.cmd_time)),
                torch.where(self.cmd_mode[:, None] == 1, self.cmd_duration - self.cmd_time, torch.zeros_like(self.cmd_time)),
                torch.nn.functional.one_hot(self.cmd_mode.long(), num_classes=2),
                self.cmd_contact,
            ],
            dim=1,
        )

    def symmetry_transforms(self):
        return SymmetryTransform.cat(
            [
                SymmetryTransform(perm=torch.arange(3), signs=torch.tensor([1, -1, 1])),  # flip y
                SymmetryTransform(perm=torch.arange(3), signs=torch.tensor([-1, 1, -1])),  # flip roll and yaw
                SymmetryTransform(perm=torch.arange(3), signs=torch.tensor([-1, 1, -1])),  # flip yaw,
                SymmetryTransform(perm=torch.arange(2), signs=torch.ones(2)), # phase: do nothing
                SymmetryTransform(perm=torch.arange(2), signs=torch.ones(2)), # cmd_mode: do nothing
                SymmetryTransform(perm=torch.tensor([2, 3, 0, 1]), signs=torch.ones(4)) # cmd_contact: flip left and right
            ]
        )

    @property
    def command_mode(self):
        return torch.zeros(self.num_envs, 1, dtype=torch.int32, device=self.device)

    @property
    def yaw_error(self):
        return wrap_to_pi(self.cmd_rpy_w[:, 2] - self.asset.data.heading_w)

    def update(self):
        c1 = self.env.episode_length_buf % 25 == 0
        c2 = torch.rand(self.num_envs, device=self.device) < 0.5
        c3 = (self.cmd_time > self.cmd_duration).squeeze(1)
        resample = (c1 & c2) | c3
        next_mode_prob = self.transition_prob[self.cmd_mode.long()]
        next_mode = next_mode_prob.multinomial(1, replacement=True).squeeze(-1)

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
                wp.from_torch(self.cmd_mode, return_ctype=True),
                wp.from_torch(next_mode, return_ctype=True),
                wp.from_torch(self.cmd_time, return_ctype=True),
                wp.from_torch(self.cmd_duration, return_ctype=True),
                self.env.timestamp,
            ],
            device=self.device.type,
        )
        wp.launch(
            step_command,
            dim=self.num_envs,
            inputs=[
                wp.from_torch(self.cmd_ang_vel_w, return_ctype=True),
                wp.from_torch(self.cmd_rpy_w, return_ctype=True),
                wp.from_torch(self.cmd_contact, return_ctype=True),
                wp.from_torch(self.cmd_height, return_ctype=True),
                wp.from_torch(self.cmd_mode, return_ctype=True),
                wp.from_torch(self.cmd_time, return_ctype=True),
                wp.from_torch(self.cmd_duration, return_ctype=True),
            ],
            device=self.device.type,
        )

    @property
    def obs_cmd_lin_vel_b(self):
        return torch.where(
            self.use_lin_vel_w,
            quat_rotate_inverse(self.asset.data.root_quat_w, self.cmd_lin_vel_w),
            self.cmd_lin_vel_b,
        )
    
    @property
    def des_cmd_lin_vel_w(self):
        return torch.where(
            self.use_lin_vel_w,
            self.cmd_lin_vel_w,
            quat_rotate(self.asset.data.root_quat_w, self.cmd_lin_vel_b),
        )

    def debug_draw(self):
        if self.env.sim.has_gui() and self.env.backend == "isaac":
            quat = quat_from_euler_xyz(*self.cmd_rpy_w.unbind(1))
            translations = self.asset.data.root_pos_w.clone()
            translations[:, 2:3] = self.cmd_height
            self.frame_marker.visualize(
                translations=translations + torch.tensor([0.0, 0.0, 0.2], device=self.device),
                orientations=quat,
                scales=torch.tensor([4.0, 1.0, 0.1]).expand(self.num_envs, 3),
            )
            self.env.debug_draw.vector(
                self.asset.data.root_pos_w
                + torch.tensor([0.0, 0.0, 0.2], device=self.device),
                self.des_cmd_lin_vel_w,
            )


class sirius_yaw(Reward[SiriusDemoCommand]):
    def compute(self) -> torch.Tensor:
        error = self.command_manager.yaw_error
        return torch.cos(error).reshape(self.num_envs, -1)


class sirius_lin_vel_xy(Reward[SiriusDemoCommand]):
    def compute(self) -> torch.Tensor:
        target_lin_vel_xy = self.command_manager.des_cmd_lin_vel_w[:, :2]
        current_lin_vel_xy = self.command_manager.asset.data.root_lin_vel_w[:, :2]
        error_l2 = (target_lin_vel_xy - current_lin_vel_xy).square().sum(1, True)
        return torch.exp(-error_l2 / 0.25) * (
            -self.command_manager.asset.data.projected_gravity_b[:, 2:3]
        )


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
        root_height = self.command_manager.asset.data.root_pos_w[:, 2:3]
        error = self.command_manager.cmd_height - root_height
        error = error.square().reshape(self.num_envs, 1)
        rew = torch.exp(-error / 0.1)
        return rew


class sirius_contact(Reward[SiriusDemoCommand]):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.contact_forces = self.env.scene["contact_forces"]
        self.foot_ids = self.contact_forces.find_bodies(".*_FOOT")[0]

    def compute(self) -> torch.Tensor:
        contact_forces = self.contact_forces.data.net_forces_w[:, self.foot_ids]
        in_contact = contact_forces.norm(dim=-1) > 0.2
        rew = (in_contact * self.command_manager.cmd_contact).sum(1, True)
        return torch.exp(rew)

