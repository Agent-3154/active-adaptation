from typing import Optional, Tuple
import torch
import einops
import warp as wp
from active_adaptation.envs.mdp.base import Command
from active_adaptation.utils.math import (
    quat_rotate,
    quat_rotate_inverse,
    yaw_quat,
    clamp_norm,
    wrap_to_pi,
    euler_from_quat,
    quat_from_euler_xyz,
    yaw_rotate,
)

from tensordict import TensorClass
from pathlib import Path
from active_adaptation.envs.mdp.utils.forces import ConstantForce, SpringForce, ImpulseForce
from active_adaptation.utils.symmetry import SymmetryTransform


import active_adaptation
if active_adaptation.get_backend() == "isaac":
    from isaaclab.markers import (
        BLUE_ARROW_X_MARKER_CFG,
        FRAME_MARKER_CFG,
        VisualizationMarkers,
        VisualizationMarkersCfg,
        sim_utils
    )


def saturate(x: torch.Tensor, a: float):
    norm = x.norm(dim=-1, keepdim=True)
    return (x /norm.clamp_min(1e-6)) * torch.log1p(norm / a) * a


class ImpedanceCommand(TensorClass):
    setpoint: torch.Tensor
    setpoint_eef: torch.Tensor
    kp_base: torch.Tensor # [*, 2] for lin and ang
    kd_base: torch.Tensor # [*, 2] for lin and ang
    kp_eef: torch.Tensor
    kd_eef: torch.Tensor
    virtual_mass_base: torch.Tensor
    virtual_mass_eef: torch.Tensor
    mode: torch.Tensor # 0: world; 1: set lin vel
    set_lin_vel: torch.Tensor
    # set_ang_vel: torch.Tensor
    transmission: torch.Tensor # whether `eef_spring_force` is transmitted to the base

    @classmethod
    def zero(cls, num_envs: int, device: torch.device):
        return cls(
            setpoint=torch.zeros(num_envs, 6, device=device),
            setpoint_eef=torch.zeros(num_envs, 3, device=device),
            kp_base=torch.zeros(num_envs, 2, device=device),
            kd_base=torch.zeros(num_envs, 2, device=device),
            kp_eef=torch.zeros(num_envs, 1, device=device),
            kd_eef=torch.zeros(num_envs, 1, device=device),
            virtual_mass_base=torch.ones(num_envs, 1, device=device),
            virtual_mass_eef=torch.ones(num_envs, 1, device=device),
            mode=torch.zeros(num_envs, dtype=torch.long, device=device),
            set_lin_vel=torch.zeros(num_envs, 3, device=device),
            transmission=torch.zeros(num_envs, 1, device=device),
            batch_size=[num_envs],
        )


def make_point_marker(name: str, color: Tuple[float, float, float]):
    marker = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path=f"/Visuals/Command/{name}",
            markers={
                name: sim_utils.SphereCfg(
                    radius=0.03,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
            }
        )
    )
    marker.set_visibility(True)
    return marker

def make_frame_marker(name: str):
    marker = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path=f"/Visuals/Command/{name}",
            markers={
                name: sim_utils.UsdFileCfg(
                    usd_path=f"/home/btx0424/lab50/frame_prim.usd",
                    scale=(0.1, 0.1, 0.1),
                ),
            }
        )
    )
    marker.set_visibility(True)
    return marker


class State(TensorClass):
    pos_w: torch.Tensor
    vel_w: torch.Tensor
    acc_w: torch.Tensor

    @classmethod
    def zeros(cls, shape: torch.Size):
        return cls(
            pos_w=torch.zeros(*shape, 3),
            vel_w=torch.zeros(*shape, 3),
            acc_w=torch.zeros(*shape, 3),
            batch_size=shape,
        )

    def integrate(self, dt: float, mask: torch.Tensor, vel_clamp: Optional[float]=None):
        self.vel_w.add_(self.acc_w * dt * mask)
        if vel_clamp is not None:
            self.vel_w = clamp_norm(self.vel_w, 0.0, vel_clamp)
        self.pos_w.add_(self.vel_w * dt * mask)
    
    def roll(self, steps: int, dims: int=1):
        return State(
            pos_w=self.pos_w.roll(steps, dims=dims),
            vel_w=self.vel_w.roll(steps, dims=dims),
            acc_w=self.acc_w.roll(steps, dims=dims),
            batch_size=self.batch_size,
        )


@wp.kernel(enable_backward=False)
def sample_command_world(
    # input
    mask: wp.array(dtype=wp.bool),
    root_link_pos_w: wp.array(dtype=wp.vec3),
    root_link_rpy_w: wp.array(dtype=wp.vec3),
    seed: wp.int32,
    # output
    setpoint_pos_w: wp.array(dtype=wp.vec3),
    setpoint_rpy_w: wp.array(dtype=wp.vec3),
    setpoint_vel_w: wp.array(dtype=wp.vec3),
    kp_base: wp.array(dtype=wp.vec2),
    kd_base: wp.array(dtype=wp.vec2),
    virtual_mass_base: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    if not mask[tid]:
        return

    seed_ = wp.rand_init(seed, tid)
    lin_kp = wp.randf(seed_, 2.0, 30.0)
    lin_kd = 1.8 * wp.sqrt(lin_kp)
    ang_kp = lin_kp
    ang_kd = lin_kd
    
    virtual_mass = 2.0 ** wp.float32(wp.randi(seed_, 1, 4))

    pos_offset_xy = wp.sample_unit_ring(seed_) * wp.randf(seed_, 0.5, 1.5)
    setpoint_pos_w[tid] = root_link_pos_w[tid] + wp.vec3(pos_offset_xy.x, pos_offset_xy.y, 0.0)
    setpoint_rpy_w[tid] = wp.vec3(0., 0., 0.)
    
    if wp.randf(seed_, 0., 1.0) < 0.5:
        setpoint_vel_w[tid] = wp.vec3(wp.randf(seed_, 0.5, 1.5), 0., 0.)
    
    kp_base[tid] = wp.vec2(lin_kp, ang_kp)
    kd_base[tid] = wp.vec2(lin_kd, ang_kd)
    virtual_mass_base[tid] = virtual_mass


class ImpedanceCommandManager(Command):
    
    # training
    MODE0_PROB = 0.5
    FORCE_PROB = 0.5

    # evaluation
    # MODE0_PROB = 0.0
    # FORCE_PROB = 1.0

    def __init__(
        self,
        env,
        virtual_mass_range,
        eef_body_name: Optional[str]=None,
        arm_body_name: Optional[str]=None,
        ref_steps=[8, 16, 32],
        temporal_smoothing: int=32,
    ) -> None:
        super().__init__(env)
        
        if eef_body_name is not None:
            assert arm_body_name is not None, "`arm_body_name` is required if `eef_body_name` is provided"
            self.eef_body_id = self.asset.find_bodies(eef_body_name)[0][0]
            self.arm_body_id = self.asset.find_bodies(arm_body_name)[0][0]
            self.has_eef = True
        else:
            self.has_eef = False
        
        self.temporal_smoothing = temporal_smoothing
        self.base_kp_range = (4., 40.)
        self.eef_kp_range = (4., 40.)
        # self.base_kp_range = (24., 60.)
        # self.eef_kp_range = (24., 60.)
        
        self.cmd = ImpedanceCommand.zero(self.num_envs, self.device)
        
        with torch.device(self.device):
            self.ref_steps = torch.tensor(ref_steps)
            self.root_link_rpy_w = torch.zeros(self.num_envs, 3)
            self.virtual_mass_range = torch.tensor(virtual_mass_range).float()

            # states of the reference dynamics
            bshape = (self.num_envs, self.temporal_smoothing + 1)
            self.pos_base = State.zeros(bshape)
            self.rot_base = State.zeros(bshape)
            self.pos_eef = State.zeros(bshape)

            self.setpoint_w = torch.zeros(self.num_envs, 3 + 3) # translation and rotation
            # not `setvel`, just to move the setpoint to generate more diverse commands
            self.setpoint_vel_w = torch.zeros(self.num_envs, 3)
            self.setpoint_b = torch.zeros(self.num_envs, 3 + 3) # translation and rotation
            
            # we only control the translation of the eef
            self.setpoint_eef_w = torch.zeros(self.num_envs, 3) # translation only
            self.setpoint_eef_b = torch.zeros(self.num_envs, 3) # translation only

            # surrogate targets
            self.surr_pos_target = torch.zeros(self.num_envs, len(self.ref_steps), 3)
            self.surr_lin_vel_target = torch.zeros(self.num_envs, len(self.ref_steps), 3)
            self.surr_yaw_target = torch.zeros(self.num_envs, len(self.ref_steps), 1)
            self.surr_yaw_vel_target = torch.zeros(self.num_envs, len(self.ref_steps), 1)

            self.surr_eef_pos_target = torch.zeros(self.num_envs, len(self.ref_steps), 3)
            self.surr_eef_lin_vel_target = torch.zeros(self.num_envs, len(self.ref_steps), 3)

            # for gait control
            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=torch.bool)
        
        self.force_impulse = ImpulseForce.zeros(self.num_envs, device=self.device)
        # self.force_pull = ConstantForce.zeros(self.num_envs, device=self.device)
        
        if self.has_eef:
            path = Path(__file__).parent / "target_archive.pt"
            self.ee_target_archive = torch.load(path)["target_pos"].to(self.device)
            self.ee_target_archive += self.asset.data.body_pos_w[0, self.arm_body_id] - self.asset.data.root_link_pos_w[0]
        
        self.seed = wp.rand_init(0)

        if self.env.sim.has_gui() and self.env.backend == "isaac":
            self.ref_pos_marker = make_frame_marker("ref_pos")
            self.setpoint_marker = make_point_marker("setpoint", (0.8, 0.0, 0.0))
            # self.spring_marker = make_point_marker("spring", (0.0, 0.0, 0.8))
            

    def reference_dynamics(self, dt: float):
        base_pos_error = self.setpoint_w[:, None, :3] - self.pos_base.pos_w
        base_vel_error = 0.0 - self.pos_base.vel_w
        lin_kp, ang_kp = self.cmd.kp_base.unbind(1)
        lin_kd, ang_kd = self.cmd.kd_base.unbind(1)
        base_lin_force = (
            lin_kp.reshape(self.num_envs, 1, 1) * base_pos_error
            + lin_kd.reshape(self.num_envs, 1, 1) * base_vel_error
            # + self.force_pull.get_force(None, None).unsqueeze(1) # for state-dependent force, unused here
            + self.force_impulse.get_force(None, None).unsqueeze(1) # for state-dependent force, unused here
        )
        base_lin_acc = base_lin_force / self.cmd.virtual_mass_base.unsqueeze(1)
        
        base_rpy_error = wrap_to_pi(self.setpoint_w[:, None, 3:6] - self.rot_base.pos_w)
        base_ang_vel_error = 0.0 - self.rot_base.vel_w
        base_ang_force = (
            ang_kp.reshape(self.num_envs, 1, 1) * base_rpy_error
            + ang_kd.reshape(self.num_envs, 1, 1) * base_ang_vel_error
        )
        base_ang_acc = base_ang_force / 1.5

        self.pos_base.acc_w.copy_(base_lin_acc)
        self.rot_base.acc_w.copy_(base_ang_acc)
        
        mask_xy = torch.tensor([1., 1., 0.], device=self.device)
        mask_yaw = torch.tensor([0., 0., 1.], device=self.device)
        self.pos_base.integrate(dt, mask_xy, vel_clamp=2.0) # integrate only in xy plane
        self.rot_base.integrate(dt, mask_yaw, vel_clamp=2.0) # integrate only in yaw direction
        
        if self.has_eef:
            eef_kp, eef_kd = self.cmd.kp_eef, self.cmd.kd_eef
            eef_pos_error = self.setpoint_eef_w.unsqueeze(1) - self.pos_eef.pos_w
            eef_lin_vel_error = 0.0 - self.pos_eef.vel_w
            eef_lin_acc = (
                eef_kp.reshape(self.num_envs, 1, 1) * eef_pos_error
                + eef_kd.reshape(self.num_envs, 1, 1) * eef_lin_vel_error
            )
            eef_lin_acc = eef_lin_acc / self.cmd.virtual_mass_eef.unsqueeze(1)
            self.pos_eef.acc_w.copy_(eef_lin_acc)
            self.pos_eef.integrate(dt, torch.tensor([1., 1., 1.], device=self.device))

    @property
    def command(self):
        result = []
        result.append(self.setpoint_b)
        result.append(self.setpoint_b[:, 0:3] * self.cmd.kp_base[:, 0:1])
        result.append(self.setpoint_b[:, 3:6] * self.cmd.kp_base[:, 1:2])
        result = torch.cat(result, dim=1)
        return result
    
    def symmetry_transform(self):
        linear =  SymmetryTransform(
            perm=torch.arange(3),
            signs=torch.tensor([1., -1., 1.]), # flip y
        )
        angular = SymmetryTransform(
            perm=torch.arange(3),
            signs=torch.tensor([-1., 1., -1.]), # flip roll and yaw
        )
        return SymmetryTransform.cat([linear, angular]).repeat(2)


    @property
    def command_hidden(self):
        result =  torch.cat([
            self.w2b(self.pos_base.pos_w[:, self.ref_steps] - self.asset.data.root_link_pos_w.unsqueeze(1)).reshape(self.num_envs, -1),
            self.w2b(self.pos_base.vel_w[:, self.ref_steps]).reshape(self.num_envs, -1),
            wrap_to_pi(self.rot_base.pos_w[:, self.ref_steps] - self.root_link_rpy_w.unsqueeze(1)).reshape(self.num_envs, -1),
            self.w2b(self.rot_base.vel_w[:, self.ref_steps]).reshape(self.num_envs, -1),
            # self.w2b(self.pos_eef.vel[:, self.ref_steps]).reshape(self.num_envs, -1),
        ], dim=1)
        return result
    
    def w2b(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            quat = self.asset.data.root_quat_w
        elif x.ndim == 3:
            quat = self.asset.data.root_quat_w.unsqueeze(1)
        else:
            raise ValueError(f"Invalid input dimension: {x.ndim}")
        return quat_rotate_inverse(quat, x)
    
    def b2w(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            quat = self.asset.data.root_quat_w
        elif x.ndim == 3:
            quat = self.asset.data.root_quat_w.unsqueeze(1)
        else:
            raise ValueError(f"Invalid input dimension: {x.ndim}")
        return quat_rotate(quat, x)

    def step(self, substep: int):
        # self.asset._external_force_b[:, self.eef_body_id] += quat_rotate_inverse(
        #     self.asset.data.body_quat_w[:, self.eef_body_id],
        #     self.get_eef_force(self.asset.data.body_pos_w[:, self.eef_body_id], self.asset.data.body_lin_vel_w[:, self.eef_body_id])
        # )
        self.asset._external_force_b[:, 0] += quat_rotate_inverse(
            self.asset.data.root_link_quat_w,
            self.force_impulse.get_force(None, None)
        )
        self.asset.has_external_wrench = True

    def reset(self, env_ids: torch.Tensor):
        root_link_pos = self.asset.data.root_link_pos_w[env_ids]
        root_link_rpy = euler_from_quat(self.asset.data.root_link_quat_w[env_ids])
        self.pos_base.pos_w[env_ids] = root_link_pos.unsqueeze(1)
        self.pos_base.vel_w[env_ids] = self.asset.data.root_link_lin_vel_w[env_ids].unsqueeze(1)
        self.rot_base.pos_w[env_ids] = root_link_rpy.unsqueeze(1)
        self.rot_base.vel_w[env_ids] = self.asset.data.root_link_ang_vel_w[env_ids].unsqueeze(1)
        self.root_link_rpy_w[env_ids] = root_link_rpy

        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        mask[env_ids] = True
        wp.launch(
            sample_command_world,
            device=wp.get_device(str(self.device)),
            dim=[self.num_envs],
            inputs=[
                wp.from_torch(mask, dtype=wp.bool, return_ctype=True),
                wp.from_torch(self.asset.data.root_link_pos_w, dtype=wp.vec3, return_ctype=True),
                wp.from_torch(self.root_link_rpy_w, dtype=wp.vec3, return_ctype=True),
                self.seed,
            ],
            outputs=[
                wp.from_torch(self.setpoint_w[:, :3], dtype=wp.vec3, return_ctype=True),
                wp.from_torch(self.setpoint_w[:, 3:], dtype=wp.vec3, return_ctype=True),
                wp.from_torch(self.setpoint_vel_w, dtype=wp.vec3, return_ctype=True),
                wp.from_torch(self.cmd.kp_base, dtype=wp.vec2, return_ctype=True),
                wp.from_torch(self.cmd.kd_base, dtype=wp.vec2, return_ctype=True),
                wp.from_torch(self.cmd.virtual_mass_base, dtype=wp.float32, return_ctype=True),
            ],
        )
    
    def update(self):
        self.seed = wp.rand_init(self.seed)
        mask = self.env.episode_length_buf % 100 == 0
        root_link_pos = self.asset.data.root_link_pos_w
        self.root_link_rpy_w = euler_from_quat(self.asset.data.root_link_quat_w)
        wp.launch(
            sample_command_world,
            device=wp.get_device(str(self.device)),
            dim=[self.num_envs],
            inputs=[
                wp.from_torch(mask, dtype=wp.bool, return_ctype=True),
                wp.from_torch(root_link_pos, dtype=wp.vec3, return_ctype=True),
                wp.from_torch(self.root_link_rpy_w, dtype=wp.vec3, return_ctype=True),
                self.seed,
            ],
            outputs=[
                wp.from_torch(self.setpoint_w[:, :3], dtype=wp.vec3, return_ctype=True),
                wp.from_torch(self.setpoint_w[:, 3:], dtype=wp.vec3, return_ctype=True),
                wp.from_torch(self.setpoint_vel_w, dtype=wp.vec3, return_ctype=True),
                wp.from_torch(self.cmd.kp_base, dtype=wp.vec2, return_ctype=True),
                wp.from_torch(self.cmd.kd_base, dtype=wp.vec2, return_ctype=True),
                wp.from_torch(self.cmd.virtual_mass_base, dtype=wp.float32, return_ctype=True),
            ],
        )
        self.setpoint_w[:, :3] += self.setpoint_vel_w * self.env.step_dt
        

        self.pos_base = self.pos_base.roll(1, 1)
        self.pos_base.pos_w[:, 0] = self.asset.data.root_link_pos_w
        self.pos_base.vel_w[:, 0] = self.asset.data.root_link_lin_vel_w

        self.rot_base = self.rot_base.roll(1, 1)
        self.rot_base.pos_w[:, 0] = self.root_link_rpy_w
        self.rot_base.vel_w[:, 0] = self.asset.data.root_link_ang_vel_w

        if self.has_eef:
            self.pos_eef = self.pos_eef.roll(1, 1)
            self.pos_eef.pos_w[:, 0] = self.asset.data.body_pos_w[:, self.eef_body_id]
            self.pos_eef.vel_w[:, 0] = self.asset.data.body_lin_vel_w[:, self.eef_body_id]

        self.force_impulse.time.add_(self.env.step_dt)
        resample = self.force_impulse.expired & (torch.rand(self.num_envs, 1, device=self.device) < 0.003)
        impulse_force = ImpulseForce.sample(self.num_envs, self.device, (40., 200.), (40., 200.), (0., 20.))
        self.force_impulse = impulse_force.where(resample, self.force_impulse)

        self.reference_dynamics(self.env.step_dt)

        self.surr_pos_target = self.pos_base.pos_w[:, self.ref_steps]
        self.surr_lin_vel_target = self.pos_base.vel_w[:, self.ref_steps]
        self.surr_yaw_target = self.rot_base.pos_w[:, self.ref_steps, 2:3]
        self.surr_yaw_vel_target = self.rot_base.vel_w[:, self.ref_steps, 2:3]

        self.setpoint_b[:, :3] = quat_rotate_inverse(
            self.asset.data.root_quat_w,
            self.setpoint_w[:, :3] - self.asset.data.root_link_pos_w
        )
        self.setpoint_b[:, 3:] = wrap_to_pi(self.setpoint_w[:, 3:] - self.root_link_rpy_w)

    def debug_draw(self):
        self.env.debug_draw.vector(
            self.asset.data.root_link_pos_w,
            self.setpoint_w[:, :3] - self.asset.data.root_link_pos_w,
            color=(1.0, 0.0, 0.0, 1.0),
            size=5.0,
        )
        self.env.debug_draw.vector( # reference lin vel, green
            self.asset.data.root_link_pos_w,
            self.pos_base.vel_w[:, -2],
            color=(0.0, 1.0, 0.0, 1.0),
            size=5.0,
        )
        self.ref_pos_marker.visualize(
            translations=self.pos_base.pos_w[:, self.ref_steps].reshape(-1, 3),
            orientations=quat_from_euler_xyz(self.rot_base.pos_w[:, self.ref_steps].reshape(-1, 3)),
        )
        self.setpoint_marker.visualize(self.setpoint_w[:, :3])
        # external forces
        self.env.debug_draw.vector(
            self.asset.data.root_link_pos_w,
            self.force_impulse.get_force(None, None) / 9.81,
            color=(1.0, 0.6, 0.0, 1.0), # orange
            size=3.0,
        )

