import torch
import warp as wp

from active_adaptation.envs.mdp.base import Command, Reward, Observation
from active_adaptation.utils.math import (
    quat_rotate,
    quat_rotate_inverse,
    quat_mul,
    sample_quat_yaw,
    quat_from_euler_xyz,
    euler_from_quat,
    wrap_to_pi,
    yaw_quat,
)
from active_adaptation.utils.symmetry import SymmetryTransform, joint_space_symmetry


PRE_JUMP_TIME = 0.8
POST_JUMP_TIME = 0.8


@wp.kernel(enable_backward=False)
def sample_command(
    quat_w: wp.array(dtype=wp.quatf),
    heading_w: wp.array(dtype=wp.float32),
    lin_vel_w: wp.array(dtype=wp.vec3),
    cmd_lin_vel_w: wp.array(dtype=wp.vec3),
    cmd_lin_vel_b: wp.array(dtype=wp.vec3),
    use_lin_vel_w: wp.array(dtype=wp.bool),
    des_rpy_w: wp.array(dtype=wp.vec3),
    cmd_rpy_w: wp.array(dtype=wp.vec3),
    cmd_ang_vel_w: wp.array(dtype=wp.vec3),
    yaw_stiffness: wp.array(dtype=wp.float32),
    use_yaw_stiffness: wp.array(dtype=wp.bool),
    sample: wp.array(dtype=wp.bool),
    mode: wp.array(dtype=wp.int32),
    next_mode: wp.array(dtype=wp.int32),
    cmd_time: wp.array(dtype=wp.float32),
    cmd_duration: wp.array(dtype=wp.float32),
    cmd_jump_turn: wp.array(dtype=wp.float32),
    seed: wp.int32,
    homogeneous: bool
):
    lin_vel_prob = 0.9
    yaw_stiffness_prob = 0.7

    tid = wp.tid()
    if homogeneous:
        seed_ = wp.rand_init(seed)
    else:
        seed_ = wp.rand_init(seed, tid)
    if sample[tid]:
        if next_mode[tid] == 0:
            # body frame linear velocity command
            cmd_lin_vel_w[tid] = wp.vec3(0.0, 0.0, 0.0)
            use_lin_vel_w[tid] = False
            has_lin_vel = wp.randf(seed_) < lin_vel_prob
            if has_lin_vel:
                cmd_lin_vel_b[tid] = wp.vec3(
                    wp.randf(seed_, 0.3, 1.6) * wp.sign(wp.randn(seed_)),
                    wp.randf(seed_, -0.8, 0.8), 0.0)
            else:
                cmd_lin_vel_b[tid] = wp.vec3(0.0, 0.0, 0.0)
            # yaw command
            use_yaw_stiffness[tid] = wp.randf(seed_) < yaw_stiffness_prob
            if use_yaw_stiffness[tid]:
                yaw_stiffness[tid] = wp.randf(seed_, 0.5, 1.0)
                cmd_ang_vel_w[tid].z = 0.0 # computed in `step_command`
                des_rpy_w[tid] = wp.vec3(0.0, 0.0, wp.randf(seed_) * wp.PI * 2.0)
            else:
                yaw_rate_sample = wp.randf(seed_, wp.PI / 4.0, wp.PI / 2.0)
                yaw_stiffness[tid] = 0.0
                cmd_ang_vel_w[tid].z = yaw_rate_sample * wp.sign(wp.randn(seed_))
                des_rpy_w[tid] = wp.vec3(0.0, 0.0, heading_w[tid])
            cmd_duration[tid] = wp.randf(seed_, 1.0, 3.0)
        if next_mode[tid] == 1:
            cmd_lin_vel_b[tid] = wp.cw_mul(cmd_lin_vel_b[tid], wp.vec3(1.0, 0.0, 0.0))
            x_vel = cmd_lin_vel_b[tid].x
            x_vel = (wp.abs(x_vel) + wp.randf(seed_, 0.1, 0.4)) * wp.sign(x_vel)
            cmd_lin_vel_w[tid] = wp.quat_rotate(quat_w[tid], cmd_lin_vel_b[tid])
            use_lin_vel_w[tid] = True
            # cmd_lin_vel_b will be updated by `step_command`
            turn = wp.randf(seed_) < 0.0
            if turn:
                air_time = wp.randf(seed_, 0.6, 0.9) # more time to turn
                cmd_jump_turn[tid] = wp.PI
                des_rpy_w[tid] = wp.vec3(0.0, 0.0, heading_w[tid] + wp.PI)
            else:
                air_time = wp.randf(seed_, 1.0, 1.0)
                cmd_jump_turn[tid] = 0.0
                des_rpy_w[tid] = wp.vec3(0.0, 0.0, heading_w[tid])
            cmd_ang_vel_w[tid] = wp.vec3(0.0, 0.0, 0.0)
            use_yaw_stiffness[tid] = False
            cmd_duration[tid] = air_time + PRE_JUMP_TIME + POST_JUMP_TIME
        cmd_time[tid] = 0.0  # reset time
        mode[tid] = next_mode[tid]


@wp.kernel(enable_backward=False)
def step_command(
    heading_w: wp.array(dtype=wp.float32),
    cmd_ang_vel_w: wp.array(dtype=wp.vec3),
    cmd_lin_vel_w: wp.array(dtype=wp.vec3),
    cum_hip_deviation: wp.array(dtype=wp.vec4),
    in_contact: wp.array(dtype=wp.bool),
    swing_thres: wp.array(dtype=wp.float32),
    des_rpy_w: wp.array(dtype=wp.vec3),
    cmd_rpy_w: wp.array(dtype=wp.vec3),
    yaw_stiffness: wp.array(dtype=wp.float32),
    use_yaw_stiffness: wp.array(dtype=wp.bool),
    cmd_contact: wp.array(dtype=wp.vec4),
    cmd_height: wp.array(dtype=wp.float32),
    mode: wp.array(dtype=wp.int32),
    cmd_time: wp.array(dtype=wp.float32),
    cmd_duration: wp.array(dtype=wp.float32),
    cmd_in_air: wp.array(dtype=wp.bool),
    cmd_jump_turn: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    time = cmd_time[tid]
    if mode[tid] == 0:
        cmd_height[tid] = 0.45
        # cmd_contact[tid] = wp.vec4(0.0, 0.0, 0.0, 0.0)
        for i in range(4):
            cond = (cum_hip_deviation[tid][i] > swing_thres[tid]) and in_contact[tid]
            cmd_contact[tid][i] = wp.where(cond, -1.0, 0.0)
        if use_yaw_stiffness[tid]:
            yaw_error = (des_rpy_w[tid].z - heading_w[tid])
            yaw_error = (yaw_error + wp.PI) % (2.0 * wp.PI) - wp.PI
            cmd_ang_vel_w[tid].z = wp.clamp(yaw_stiffness[tid] * yaw_error, -2.0, 2.0)
        cmd_in_air[tid] = False
    elif mode[tid] == 1:  # jump
        air_time = cmd_duration[tid] - PRE_JUMP_TIME - POST_JUMP_TIME
        cmd_contact[tid] = wp.vec4(0.0, 0.0, 0.0, 0.0)
        if time < PRE_JUMP_TIME :
            cmd_height[tid] = 0.40
            cmd_ang_vel_w[tid].z = 0.0
        elif time < PRE_JUMP_TIME + 0.3:
            cmd_height[tid] = 0.40 + 0.8 * (time - PRE_JUMP_TIME)
            cmd_lin_vel_w[tid] = wp.vec3(0.0, 0.0, (time - PRE_JUMP_TIME) * 8.0)
            cmd_ang_vel_w[tid].z = cmd_jump_turn[tid] / air_time
            cmd_in_air[tid] = True
        elif time < PRE_JUMP_TIME + 0.3 + 0.3:
            cmd_height[tid] = 0.64
            cmd_ang_vel_w[tid].z = cmd_jump_turn[tid] / air_time
            cmd_in_air[tid] = True
        elif time < cmd_duration[tid] - POST_JUMP_TIME:
            cmd_height[tid] = 0.64
            cmd_in_air[tid] = True
        else:
            cmd_height[tid] = 0.45
            cmd_ang_vel_w[tid].z = 0.0
            cmd_in_air[tid] = False
    cmd_rpy_w[tid] += cmd_ang_vel_w[tid] * 0.02
    cmd_time[tid] += 0.02


class SiriusDemoCommand(Command):
    def __init__(self, env, transition_prob, homogeneous: bool = False, teleop: bool = False) -> None:
        super().__init__(env, teleop)
        self.contact_sensor = self.env.scene["contact_forces"]
        self.homogeneous = homogeneous
        self.joint_ids, self.joint_names = self.asset.find_joints(".*HAA")
        self.default_hip_jpos = self.asset.data.default_joint_pos[:, self.joint_ids]
        self.wheel_ids_contact = self.contact_sensor.find_bodies(".*_FOOT")[0]

        with torch.device(self.device):
            self.cmd_lin_vel_w = torch.zeros(self.num_envs, 3)
            self.cmd_lin_vel_b = torch.zeros(self.num_envs, 3)
            self.use_lin_vel_w = torch.zeros(self.num_envs, 1, dtype=bool)
            self.cmd_height = torch.zeros(self.num_envs, 1)
            self.cmd_ang_vel_w = torch.zeros(self.num_envs, 3)
            self.des_rpy_w = torch.zeros(self.num_envs, 3)
            self.cmd_rpy_w = torch.zeros(self.num_envs, 3)
            self.yaw_stiffness = torch.zeros(self.num_envs, 1)
            self.use_yaw_stiffness = torch.zeros(self.num_envs, 1, dtype=bool)
            self.cmd_contact = torch.zeros(self.num_envs, 4)
            self.cmd_time = torch.zeros(self.num_envs, 1)
            self.cmd_duration = torch.zeros(self.num_envs, 1)
            self.cmd_mode = torch.zeros(self.num_envs, dtype=torch.int32)
            self.in_air = torch.zeros(self.num_envs, 1, dtype=bool)
            self.cmd_jump_turn = torch.zeros(self.num_envs, 1)
            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=bool)

            self.cum_hip_deviation = torch.zeros(self.num_envs, 4)
            self.swing_thres = torch.zeros(self.num_envs, 1)
            self.swing_thres.uniform_(0.5, 0.8)

            self.transition_prob = torch.tensor(transition_prob, device=self.device)
            self.transition_prob = self.transition_prob / self.transition_prob.sum(1, True)

        if self.env.sim.has_gui() and self.env.backend == "isaac":
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg, ISAAC_NUCLEUS_DIR, sim_utils, BLUE_ARROW_X_MARKER_CFG

            marker_cfg = VisualizationMarkersCfg(
                prim_path="/World/Visuals/testMarkers",
                markers={
                    "red_arrow": sim_utils.UsdFileCfg(
                        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                        scale=(1.0, 0.1, 0.1),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    ),
                    "blue_arrow": sim_utils.UsdFileCfg(
                        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                        scale=(1.0, 0.1, 0.1),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                    )
                }
            )
            self.frame_marker = VisualizationMarkers(marker_cfg)
            self.frame_marker.set_visibility(True)
        
        if self.teleop:
            from active_adaptation.utils.gamepad import Gamepad
            self.gamepad = Gamepad()
            
        self.seed = wp.rand_init(0)

    def reset(self, env_ids: torch.Tensor):
        resample = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        resample[env_ids] = True
        next_mode = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        wp.launch(
            sample_command,
            dim=self.num_envs,
            inputs=[
                wp.from_torch(self.asset.data.root_quat_w.roll(-1, dims=1), return_ctype=True),
                wp.from_torch(self.asset.data.heading_w, return_ctype=True),
                wp.from_torch(self.asset.data.root_lin_vel_w, return_ctype=True),
                wp.from_torch(self.cmd_lin_vel_w, return_ctype=True),
                wp.from_torch(self.cmd_lin_vel_b, return_ctype=True),
                wp.from_torch(self.use_lin_vel_w, return_ctype=True),
                wp.from_torch(self.des_rpy_w, return_ctype=True),
                wp.from_torch(self.cmd_rpy_w, return_ctype=True),
                wp.from_torch(self.cmd_ang_vel_w, return_ctype=True),
                wp.from_torch(self.yaw_stiffness, return_ctype=True),
                wp.from_torch(self.use_yaw_stiffness, return_ctype=True),
                wp.from_torch(resample, return_ctype=True),
                wp.from_torch(self.cmd_mode, return_ctype=True),
                wp.from_torch(next_mode, return_ctype=True),
                wp.from_torch(self.cmd_time, return_ctype=True),
                wp.from_torch(self.cmd_duration, return_ctype=True),
                wp.from_torch(self.cmd_jump_turn, return_ctype=True),
                self.seed,
                self.homogeneous,
            ],
            device=self.device.type,
        )
        self.cmd_rpy_w[env_ids, 2] = self.asset.data.heading_w[env_ids]
        self.cum_hip_deviation[env_ids] = 0.0
    
    def sample_init(self, env_ids: torch.Tensor) -> torch.Tensor:
        """
        Called before `reset` to sample initial state for the next episodes.
        This can be used for implementing curriculum learning.
        """
        init_root_state = self.init_root_state[env_ids]
        if self.terrain_type == "plane":
            pos_w = self.env.scene.env_origins[env_ids]
        else:
            idx = torch.randint(0, len(self._origins), (len(env_ids),), device=self.device)
            pos_w = self._origins[idx]
        quat = sample_quat_yaw(len(env_ids), device=self.device)
        if self.homogeneous:
            quat = quat[0:1].expand(len(env_ids), -1)
        init_root_state[:, :3] += pos_w
        init_root_state[:, 3:7] = quat_mul(init_root_state[:, 3:7], quat)
        return init_root_state
        
    @property
    def command(self):
        cmd_rpy_b = self.cmd_rpy_w.clone()
        cmd_rpy_b[:, 2] = wrap_to_pi(cmd_rpy_b[:, 2] - self.asset.data.heading_w)
        result = torch.cat(
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
        return result

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
        return self.cmd_mode.reshape(self.num_envs, 1)

    @property
    def euler_error(self):
        euler = euler_from_quat(self.asset.data.root_quat_w)
        error = wrap_to_pi(self.cmd_rpy_w - euler)
        return error

    def update(self):
        heading_wp = wp.from_torch(self.asset.data.heading_w, return_ctype=True)
        error = (self.asset.data.joint_pos[:, self.joint_ids] - self.default_hip_jpos).abs()
        self.cum_hip_deviation = torch.where(error < 0.1, 0., self.cum_hip_deviation + error * self.env.step_dt)
        in_contact = self.contact_sensor.data.current_contact_time[:, self.wheel_ids_contact] > 0.0

        if self.teleop:
            self.gamepad.update()
            self.cmd_lin_vel_b[:, 0] = self.gamepad.lxy[1] * 1.2
            self.cmd_lin_vel_b[:, 1] = self.gamepad.lxy[0]
            self.cmd_ang_vel_w[:, 2] = -self.gamepad.rxy[0]
            self.use_yaw_stiffness[:] = False
            
            if (self.gamepad.buttons["A"]):
                air_time = 0.8
                self.cmd_mode[:] = 1
                self.cmd_time[:] = 0.0
                self.cmd_duration[:] = air_time + PRE_JUMP_TIME + POST_JUMP_TIME
                self.cmd_jump_turn[:] = torch.pi
                self.cmd_lin_vel_w[:] = quat_rotate(
                    self.asset.data.root_quat_w,
                    self.cmd_lin_vel_b * torch.tensor([1.0, 0.0, 0.0], device=self.device)
                )
                self.use_lin_vel_w[:] = True
                self.des_rpy_w[:, 2] = self.asset.data.heading_w + self.cmd_jump_turn
            
            jump_end = (self.cmd_mode == 1) & (self.cmd_time > self.cmd_duration).squeeze(1)
            self.use_lin_vel_w[jump_end] = False
            self.cmd_mode[jump_end] = 0
            self.cmd_time[jump_end] = 0.0
            self.cmd_duration[jump_end] = 0.0
        else:
            c1 = self.env.episode_length_buf % 25 == 0
            if self.homogeneous:
                c2 = torch.rand(1, device=self.device) < 0.5
            else:
                c2 = torch.rand(self.num_envs, device=self.device) < 0.5
            c3 = (self.cmd_time > self.cmd_duration).squeeze(1)
            resample = (c1 & c2 & c3) | c3
            next_mode_prob = self.transition_prob[self.cmd_mode.long()]
            next_mode = next_mode_prob.multinomial(1, replacement=True).squeeze(-1)
            if self.homogeneous:
                next_mode = next_mode[0].expand_as(next_mode)
            wp.launch(
                sample_command,
                dim=self.num_envs,
                inputs=[
                    wp.from_torch(self.asset.data.root_quat_w.roll(-1, dims=1), return_ctype=True),
                    heading_wp,
                    wp.from_torch(self.asset.data.root_lin_vel_w, return_ctype=True),
                    wp.from_torch(self.cmd_lin_vel_w, return_ctype=True),
                    wp.from_torch(self.cmd_lin_vel_b, return_ctype=True),
                    wp.from_torch(self.use_lin_vel_w, return_ctype=True),
                    wp.from_torch(self.des_rpy_w, return_ctype=True),
                    wp.from_torch(self.cmd_rpy_w, return_ctype=True),
                    wp.from_torch(self.cmd_ang_vel_w, return_ctype=True),
                    wp.from_torch(self.yaw_stiffness, return_ctype=True),
                    wp.from_torch(self.use_yaw_stiffness, return_ctype=True),
                    wp.from_torch(resample, return_ctype=True),
                    wp.from_torch(self.cmd_mode, return_ctype=True),
                    wp.from_torch(next_mode, return_ctype=True),
                    wp.from_torch(self.cmd_time, return_ctype=True),
                    wp.from_torch(self.cmd_duration, return_ctype=True),
                    wp.from_torch(self.cmd_jump_turn, return_ctype=True),
                    self.env.timestamp,
                    self.homogeneous,
                ],
                device=self.device.type,
            )
        wp.launch(
            step_command,
            dim=self.num_envs,
            inputs=[
                heading_wp,
                wp.from_torch(self.cmd_ang_vel_w, return_ctype=True),
                wp.from_torch(self.cmd_lin_vel_w, return_ctype=True),
                wp.from_torch(self.cum_hip_deviation, return_ctype=True),
                wp.from_torch(in_contact, return_ctype=True),
                wp.from_torch(self.swing_thres, return_ctype=True),
                wp.from_torch(self.des_rpy_w, return_ctype=True),
                wp.from_torch(self.cmd_rpy_w, return_ctype=True),
                wp.from_torch(self.yaw_stiffness, return_ctype=True),
                wp.from_torch(self.use_yaw_stiffness, return_ctype=True),
                wp.from_torch(self.cmd_contact, return_ctype=True),
                wp.from_torch(self.cmd_height, return_ctype=True),
                wp.from_torch(self.cmd_mode, return_ctype=True),
                wp.from_torch(self.cmd_time, return_ctype=True),
                wp.from_torch(self.cmd_duration, return_ctype=True),
                wp.from_torch(self.in_air, return_ctype=True),
                wp.from_torch(self.cmd_jump_turn, return_ctype=True),
            ],
            device=self.device.type,
        )
        self.is_standing_env = self.obs_cmd_lin_vel_b.norm(dim=-1, keepdim=True) < 0.1

    @property
    def obs_cmd_lin_vel_b(self):
        quat = yaw_quat(self.asset.data.root_quat_w)
        return torch.where(
            self.use_lin_vel_w,
            quat_rotate_inverse(quat, self.cmd_lin_vel_w),
            self.cmd_lin_vel_b,
        )
    
    @property
    def des_cmd_lin_vel_w(self):
        quat = yaw_quat(self.asset.data.root_quat_w)
        return torch.where(
            self.use_lin_vel_w,
            self.cmd_lin_vel_w,
            quat_rotate(quat, self.cmd_lin_vel_b),
        )

    def debug_draw(self):
        if self.env.sim.has_gui() and self.env.backend == "isaac":
            translations = torch.cat([self.asset.data.root_pos_w, self.asset.data.root_pos_w]) 
            translations[:self.num_envs, 2:3] = self.cmd_height
            orientations = torch.cat([
                quat_from_euler_xyz(*self.cmd_rpy_w.unbind(1)),
                self.asset.data.root_quat_w,
            ])
            self.frame_marker.visualize(
                translations=translations + torch.tensor([0., 0., 0.2], device=self.device),
                orientations=orientations,
                scales=torch.tensor([4.0, 1.0, 0.1]).expand(2 * self.num_envs, 3),
                marker_indices=[0] * self.num_envs + [1] * self.num_envs
            )
            self.env.debug_draw.vector(
                self.asset.data.root_pos_w
                + torch.tensor([0.0, 0.0, 0.2], device=self.device),
                self.des_cmd_lin_vel_w,
            )
            self.env.debug_draw.vector(
                self.asset.data.root_pos_w
                + torch.tensor([0.0, 0.0, 0.2], device=self.device),
                self.asset.data.root_lin_vel_w * torch.tensor([1., 1., 0.], device=self.device),
                color=(1., 1., 1., 1.)
            )


class sirius_joint_deviation(Observation[SiriusDemoCommand]):
    def __init__(self, env):
        super().__init__(env)
        self.asset = self.command_manager.asset
        self.joint_ids = self.asset.find_joints(".*HAA")[0]
        self.default_jpos = self.asset.data.default_joint_pos[:, self.joint_ids]
        self.cum_error = torch.zeros(self.num_envs, len(self.joint_ids), device=self.device)
    
    def reset(self, env_ids: torch.Tensor):
        self.cum_error[env_ids] = 0.0
    
    def update(self):
        error = (self.asset.data.joint_pos[:, self.joint_ids] - self.default_jpos).abs()
        self.cum_error = torch.where(error < 0.1, 0., self.cum_error + error)
    
    def compute(self) -> torch.Tensor:
        return self.cum_error
    
    def symmetry_transforms(self):
        return SymmetryTransform(perm=torch.tensor([2, 3, 0, 1]), signs=torch.ones(4))


class sirius_yaw(Reward[SiriusDemoCommand]):
    def compute(self) -> torch.Tensor:
        rew = torch.cos(self.command_manager.euler_error[:, 2])
        return rew.reshape(self.num_envs, -1)


class sirius_pitch(Reward[SiriusDemoCommand]):
    def compute(self) -> torch.Tensor:
        error = self.command_manager.euler_error[:, 1].square()
        rew = torch.exp(-error) - error
        return rew.reshape(self.num_envs, -1)


class sirius_lin_vel_xy(Reward[SiriusDemoCommand]):
    def compute(self) -> torch.Tensor:
        target_lin_vel_xy = self.command_manager.des_cmd_lin_vel_w[:, :2]
        current_lin_vel_xy = self.command_manager.asset.data.root_lin_vel_w[:, :2]
        error_l2 = (target_lin_vel_xy - current_lin_vel_xy).square().sum(1, True)
        return torch.exp(-error_l2 / 0.25)


class sirius_lin_vel_z(Reward[SiriusDemoCommand]):
    def compute(self) -> torch.Tensor:
        is_active = (self.command_manager.cmd_mode[:, None] == 1)
        target_lin_vel_z = self.command_manager.des_cmd_lin_vel_w[:, 2]
        current_lin_vel_z = self.command_manager.asset.data.root_lin_vel_w[:, 2]
        error_l2 = (target_lin_vel_z - current_lin_vel_z).square()
        return torch.exp(-error_l2 / 0.2).reshape(self.num_envs, 1), is_active


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
        error = (self.command_manager.cmd_height - root_height).clamp_min(0.0)
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
        self.env.discount.mul_(torch.exp(0.25 * rew.clamp_max(0.0)))
        return rew.reshape(self.num_envs, 1)


class sirius_jump_landing(Reward[SiriusDemoCommand]):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset = self.command_manager.asset
        self.is_landing = torch.zeros(self.num_envs, 1, dtype=bool, device=self.device)
        self.last_in_air = torch.zeros(self.num_envs, 1, dtype=bool, device=self.device)

    def reset(self, env_ids: torch.Tensor):
        self.is_landing[env_ids] = False
        self.last_in_air[env_ids] = False
    
    def update(self):
        self.is_landing = (self.last_in_air & ~self.command_manager.in_air)
        self.last_in_air = self.command_manager.in_air.clone()

    def compute(self) -> torch.Tensor:
        landing_yaw = self.asset.data.heading_w
        desired_yaw = self.command_manager.des_rpy_w[:, 2]
        rew = torch.pi/2 - wrap_to_pi(desired_yaw - landing_yaw).abs()
        return rew.reshape(self.num_envs, 1), self.is_landing.reshape(self.num_envs, 1)


class sirius_jump_behave(Reward[SiriusDemoCommand]):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset = self.command_manager.asset
        self.joint_ids = self.asset.find_joints(".*HAA")[0]
        self.default_jpos = self.asset.data.default_joint_pos[:, self.joint_ids]
    
    def update(self):
        self.joint_pos = self.asset.data.joint_pos[:, self.joint_ids]

    def compute(self) -> torch.Tensor:
        is_active = (self.command_manager.cmd_mode[:, None] == 1) & (~self.command_manager.in_air)
        rew_dev = - torch.abs(self.joint_pos - self.default_jpos).sum(1, True)
        return rew_dev, is_active


class sirius_land_behave(Reward[SiriusDemoCommand]):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset = self.command_manager.asset
        self.joint_ids = self.asset.find_joints(".*(HFE|KFE)")[0]
        self.default_jpos = self.asset.data.default_joint_pos[:, self.joint_ids]
    
    def update(self):
        self.joint_pos = self.asset.data.joint_pos[:, self.joint_ids]

    def compute(self) -> torch.Tensor:
        is_landing = (self.command_manager.cmd_time > self.command_manager.cmd_duration - POST_JUMP_TIME)
        is_active = (self.command_manager.cmd_mode[:, None] == 1) & is_landing
        rew_dev = - torch.square(self.joint_pos - self.default_jpos).sum(1, True)
        return rew_dev, is_active


class sirius_walk_behave(Reward[SiriusDemoCommand]):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset = self.command_manager.asset
        self.body_ids, self.body_names = self.asset.find_bodies(".*_hip")
    
    def compute(self) -> torch.Tensor:
        is_active = self.command_manager.cmd_mode[:, None] == 0
        rew_hip_dev = - self.command_manager.cum_hip_deviation.square().sum(1, True)
        rew_roll_dev = - self.command_manager.euler_error[:, 0:1].abs()
        return rew_roll_dev, is_active

    def debug_draw(self):
        body_pos_w = self.asset.data.body_pos_w[:, self.body_ids]
        vec = torch.zeros_like(body_pos_w)
        vec[:, :, 2] = self.command_manager.cum_hip_deviation
        self.env.debug_draw.vector(body_pos_w, vec, size=2.0, color=(1., 0., 0., 1.))


class sirius_takeoff(Reward[SiriusDemoCommand]):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset = self.command_manager.asset
        self.non_contact = torch.zeros(self.num_envs, 1, dtype=bool, device=self.device)
        self.takeoff = torch.zeros(self.num_envs, 1, dtype=bool, device=self.device)
    
    def reset(self, env_ids: torch.Tensor):
        self.non_contact[env_ids] = False
        self.takeoff[env_ids] = False

    def update(self):
        non_contact = ~self.command_manager.cmd_contact.any(dim=1, keepdim=True)
        self.takeoff = self.non_contact & ~non_contact
        self.non_contact = non_contact

    def compute(self) -> torch.Tensor:
        is_active = (self.command_manager.cmd_mode[:, None] == 1) & (self.command_manager.cmd_time > PRE_JUMP_TIME)
        rew = self.takeoff.float()
        return rew, is_active
