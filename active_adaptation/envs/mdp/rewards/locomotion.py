import torch
from typing import TYPE_CHECKING, Optional, List, Union
from typing_extensions import override

from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse, yaw_quat
from .base import RewardV2
from active_adaptation.envs.mdp.commands.locomotion import Twist

if TYPE_CHECKING:
    from isaaclab.sensors import ContactSensor
    from isaaclab.assets import Articulation
    from active_adaptation.envs.env_base import EnvBase

Names = Union[str, List[str]]


class survival(RewardV2):
    @override
    def _compute(self):
        return torch.ones(self.num_envs, 1, device=self.device)


class linvel_z_l2(RewardV2):
    def __init__(self, weight: float, track_var: bool = False):
        super().__init__(weight, track_var=track_var)

    @override
    def _initialize(self, env: "EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.linvel_z = torch.zeros(self.num_envs, 1, device=self.device)

    @override
    def update(self):
        self.linvel_z = self.asset.data.root_com_lin_vel_b[:, 2].unsqueeze(1)

    @override
    def _compute(self) -> torch.Tensor:
        return -self.linvel_z.square()


class angvel_xy_l2(RewardV2):
    def __init__(
        self,
        weight: float,
        body_names: Optional[Names] = None,
        track_var: bool = False,
    ):
        super().__init__(weight, track_var=track_var)
        self.body_names = body_names

    @override
    def _initialize(self, env: "EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        if self.body_names is not None:
            self.body_ids, self.body_names = self.asset.find_bodies(self.body_names)
            self.body_ids = torch.tensor(self.body_ids, device=self.device)
        else:
            self.body_ids = None
            self.body_names = [self.asset.body_names[0]]

    def _compute(self) -> torch.Tensor:
        if self.body_ids is not None:
            angvel = quat_rotate_inverse(
                self.asset.data.body_link_quat_w[:, self.body_ids],
                self.asset.data.body_com_ang_vel_w[:, self.body_ids],
            )
            reward = -angvel[:, :, :2].square().sum((1, 2))
        else:
            angvel = self.asset.data.root_com_ang_vel_b
            reward = -angvel[:, :2].square().sum(1)
        return reward.reshape(self.num_envs, 1)


class undesired_contact(RewardV2):
    supported_backends = ("isaac", "mjlab", "motrix")

    def __init__(self, body_names: Names, weight: float, track_var: bool = False):
        super().__init__(weight, track_var=track_var)
        self.body_names = body_names

    @override
    def _initialize(self, env: "EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.contact_sensor: ContactSensor = self.env.scene.sensors["contact_forces"]

        self.articulation_body_ids = self.asset.find_bodies(self.body_names)[0]
        self.body_ids, self.body_names = self.contact_sensor.find_bodies(self.body_names)
        self.num_bodies = len(self.body_ids)

    def update(self):
        contact = self.contact_sensor.data.current_contact_time[:, self.body_ids] > 0.0
        self.undesired_contact = -contact.float().sum(1, keepdim=True)

    def _compute(self) -> torch.Tensor:
        return self.undesired_contact.reshape(self.num_envs, 1)


class linvel_exp(RewardV2[Twist]):
    def __init__(
        self,
        weight: float,
        sigma: float = 0.25,
        dim: int = 2,
        gamma: float = 0.0,
        track_var: bool = False,
    ):
        super().__init__(weight, track_var=track_var)
        self.sigma = sigma
        self.dim = dim
        self.gamma = gamma

    @override
    def _initialize(self, env: "EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.linvel_w_sum = torch.zeros(self.num_envs, 3, device=self.device)
        self.count = torch.zeros(self.num_envs, 1, device=self.device)

    def reset(self, env_ids):
        self.linvel_w_sum[env_ids] = 0.0
        self.count[env_ids] = 0.0

    def update(self):
        linvel_w = self.asset.data.root_com_lin_vel_w
        self.linvel_w_sum.mul_(self.gamma).add_(linvel_w)
        self.count.mul_(self.gamma).add_(1.0)

    def _compute(self) -> torch.Tensor:
        linvel_w = self.linvel_w_sum / self.count.clamp_min(1.0)
        cmd_linvel_w = self.command_manager.cmd_linvel_w[:, : self.dim]
        linvel_error = (linvel_w[:, : self.dim] - cmd_linvel_w).square().sum(-1, True)
        rew = torch.exp(-linvel_error / self.sigma)
        return rew.reshape(self.num_envs, 1)

    def debug_draw(self):
        if self.env.backend == "isaac":
            linvel_w = self.linvel_w_sum / self.count.clamp_min(1.0)
            self.env.debug_draw.vector(
                self.asset.data.root_link_pos_w
                + torch.tensor([0.0, 0.0, 0.2], device=self.device),
                linvel_w,
                color=(0.8, 0.1, 0.8, 1.0),
            )


class root_pos_exp(RewardV2):
    def __init__(self, weight: float, dim: int = 2, track_var: bool = False):
        super().__init__(weight, track_var=track_var)
        self.dim = dim

    @override
    def _initialize(self, env: "EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]

    def _compute(self) -> torch.Tensor:
        target_pos = self.command_manager.cmd_pos_w[:, : self.dim]
        pos_error = (
            self.asset.data.root_link_pos_w[:, : self.dim] - target_pos
        ).square().sum(-1, True)
        rew = torch.exp(-pos_error / 0.25)
        return rew.reshape(self.num_envs, 1)


class root_pos_l2(RewardV2):
    def __init__(self, weight: float, dim: int = 2, track_var: bool = False):
        super().__init__(weight, track_var=track_var)
        self.dim = dim

    @override
    def _initialize(self, env: "EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]

    def _compute(self) -> torch.Tensor:
        target_pos = self.command_manager.cmd_pos_w[:, : self.dim]
        pos_error = (
            self.asset.data.root_link_pos_w[:, : self.dim] - target_pos
        ).square().sum(-1, True)
        rew = -pos_error
        return rew.reshape(self.num_envs, 1)


class linvel_projection(RewardV2[Twist]):
    def __init__(
        self,
        weight: float,
        dim: int = 2,
        track_var: bool = False,
    ):
        super().__init__(weight, track_var=track_var)
        self.dim = dim

    @override
    def _initialize(self, env: "EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]

    @override
    def _compute(self) -> torch.Tensor:
        linvel_w = self.asset.data.root_com_lin_vel_w[:, : self.dim]
        cmd_linvel_w = self.command_manager.cmd_linvel_w[:, : self.dim]
        projection = (linvel_w * cmd_linvel_w).sum(dim=-1, keepdim=True)
        rew = projection.clamp_max(self.command_manager.command_speed)
        return rew.reshape(self.num_envs, 1)


class angvel_z_exp(RewardV2[Twist]):
    def __init__(
        self,
        weight: float,
        world_frame: bool = False,
        gamma: float = 0.0,
        track_var: bool = False,
    ):
        super().__init__(weight, track_var=track_var)
        self.world_frame = world_frame
        self.gamma = gamma

    @override
    def _initialize(self, env: "EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.count = torch.zeros(self.num_envs, 1, device=self.device)
        self.angvel_sum = torch.zeros(self.num_envs, 3, device=self.device)

    @override
    def update(self):
        if self.world_frame:
            angvel = self.asset.data.root_com_ang_vel_w
        else:
            angvel = self.asset.data.root_com_ang_vel_b
        self.angvel_sum.mul_(self.gamma).add_(angvel)
        self.count.mul_(self.gamma).add_(1)

    @override
    def _compute(self) -> torch.Tensor:
        angvel = self.angvel_sum / self.count.clamp_min(1.0)
        target_angvel = self.command_manager.cmd_yawvel_b
        angvel_error = (target_angvel - angvel[:, 2:3]).square()
        rew = torch.exp(-angvel_error / 0.25)
        return rew.reshape(self.num_envs, 1)


class tracking_yaw(RewardV2):
    def __init__(self, weight: float, track_var: bool = False):
        super().__init__(weight, track_var=track_var)

    @override
    def _initialize(self, env: "EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]

    @override
    def _compute(self):
        yaw_diff = self.command_manager.ref_yaw - self.asset.data.heading_w.unsqueeze(1)
        return torch.exp(-yaw_diff.square())


class body_upright(RewardV2):
    """Reward for keeping the specified body upright."""

    def __init__(self, body_name: str, weight: float, track_var: bool = False):
        super().__init__(weight, track_var=track_var)
        self.body_name = body_name

    @override
    def _initialize(self, env: "EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.body_ids, body_names = self.asset.find_bodies(self.body_name)
        self.body_ids = torch.tensor(self.body_ids, device=self.device)

    @override
    def _compute(self) -> torch.Tensor:
        down = torch.tensor([[0.0, 0.0, -1.0]], device=self.device)
        g = quat_rotate_inverse(
            self.asset.data.body_link_quat_w[:, self.body_ids],
            down.expand(self.num_envs, len(self.body_ids), 3),
        )
        rew = 1.0 - g[:, :, :2].square().sum(-1)
        return rew.mean(1, True)


class base_height_l1(RewardV2[Twist]):
    def __init__(
        self,
        weight: float,
        target_height: Optional[float] = None,
        track_var: bool = False,
    ):
        super().__init__(weight, track_var=track_var)
        self.target_height = target_height

    @override
    def _initialize(self, env: "EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]

    @override
    def _compute(self) -> torch.Tensor:
        if self.target_height is None:
            target_height = self.command_manager.cmd_base_height
        else:
            target_height = self.target_height
        root_link_pos_w = self.asset.data.root_link_pos_w
        height = root_link_pos_w[:, 2] - self.env.get_ground_height_at(root_link_pos_w)
        error_l1 = (height.unsqueeze(1) - target_height).abs()
        return -error_l1.reshape(self.num_envs, 1)


class base_height_exp(RewardV2[Twist]):
    def __init__(
        self,
        weight: float,
        target_height: Optional[float] = None,
        track_var: bool = False,
        sigma: float = 0.2,
    ):
        super().__init__(weight, track_var=track_var)
        self.target_height = target_height
        self.sigma = sigma

    @override
    def _initialize(self, env: "EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]

    @override
    def _compute(self) -> torch.Tensor:
        if self.target_height is None:
            target_height = self.command_manager.cmd_base_height
        else:
            target_height = self.target_height
        root_link_pos_w = self.asset.data.root_link_pos_w
        height = root_link_pos_w[:, 2] - self.env.get_ground_height_at(root_link_pos_w)
        error_l2 = (height.unsqueeze(1) - target_height).square()
        rew = torch.exp(-error_l2 / self.sigma)
        return rew.reshape(self.num_envs, 1)


class is_standing_env(RewardV2):
    def __init__(self, weight: float, track_var: bool = False):
        super().__init__(weight, track_var=track_var)

    @override
    def _compute(self) -> torch.Tensor:
        return self.command_manager.is_standing_env.reshape(self.num_envs, 1)


class joint_limits(RewardV2):
    def __init__(
        self,
        joint_names: str,
        offset: float,
        weight: float,
        track_var: bool = False,
    ):
        super().__init__(weight, track_var=track_var)
        self.joint_names = joint_names
        self.offset = offset

    @override
    def _initialize(self, env: "EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.joint_ids = self.asset.find_joints(self.joint_names)[0]
        self.joint_limits = self.asset.data.joint_limits[:, self.joint_ids].clone()
        self.joint_limits_max = self.joint_limits[:, :, 1] - self.offset
        self.joint_limits_min = self.joint_limits[:, :, 0] + self.offset

    def _compute(self) -> torch.Tensor:
        joint_pos = self.asset.data.joint_pos[:, self.joint_ids]
        violation_min = (joint_pos - self.joint_limits_min).clamp_max(0.0)
        violation_max = (self.joint_limits_max - joint_pos).clamp_max(0.0)
        return (violation_min + violation_max).sum(1, keepdim=True)


class oscillator(RewardV2):
    def __init__(
        self,
        feet_names: str = ".*_foot",
        omega_range=(2.0, 2.0),
        margin: float = 0.0,
        weight=1.0,
        track_var: bool = False,
    ):
        super().__init__(weight, track_var=track_var)
        self.feet_names = feet_names
        self.omega_range = omega_range
        self.margin = margin
        self.target_swing_height = 0.08

    @override
    def _initialize(self, env: "EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.art_feet_ids = self.asset.find_bodies(self.feet_names)[0]
        self.contact_sensor: ContactSensor = self.env.scene.sensors["contact_forces"]

        self.feet_ids, feet_names = self.contact_sensor.find_bodies(self.feet_names)
        self.mass = self.asset.data.default_mass[0].sum().to(self.device)
        self.gravity = self.mass * 9.81

        self.grf_substep = torch.zeros(
            self.num_envs,
            self.env.decimation,
            len(self.feet_ids),
            device=self.device,
        )
        self.omega = torch.zeros(self.num_envs, 1, device=self.device)
        self.omega.uniform_(*self.omega_range).mul_(torch.pi)

        self.rest_target = torch.pi * 3 / 2
        self.keep_steping = torch.zeros(
            self.num_envs, 1, dtype=bool, device=self.device
        )

    def post_step(self, substep):
        grf = self.contact_sensor.data.net_forces_w[:, self.feet_ids].norm(dim=-1)
        grf += self.asset._external_force_b[:, self.art_feet_ids].norm(dim=-1)
        self.grf_substep[:, substep] = grf

    def update(self):
        self.grf = self.grf_substep.mean(1) / self.gravity

    def _compute(self):
        phi_sin = self.asset.phi.sin()
        feet_height = self.asset.data.feet_height.clamp_max(self.target_swing_height)
        r = (
            (feet_height - self.grf.clamp_max(0.4))
            * phi_sin
            * (phi_sin.abs() > self.margin)
        )
        return r.sum(1, True)

    def stand(self, phi: torch.Tensor, phi_dot: torch.Tensor):
        two_pi = torch.pi * 2
        target = self.rest_target
        dt = self.env.step_dt
        a = ((phi % two_pi) < target - 1e-4) & (((phi + phi_dot * dt) % two_pi) > target + 1e-4)
        b = ((phi % two_pi) - target).abs() < 1e-4
        phi_dot = torch.where(a, (((target - phi) % two_pi) / dt), phi_dot)
        return phi_dot * (~b)

    def trot(self, phi: torch.Tensor, phi_dot: torch.Tensor):
        phi_dot = torch.zeros_like(phi)
        phi_dot[:, 0] = (phi[:, 3] - phi[:, 0]) + (phi[:, 1] + torch.pi - phi[:, 0])
        phi_dot[:, 1] = (phi[:, 2] - phi[:, 1]) + (phi[:, 0] - torch.pi - phi[:, 1])
        phi_dot[:, 2] = (phi[:, 1] - phi[:, 2]) + (phi[:, 0] - torch.pi - phi[:, 2])
        phi_dot[:, 3] = (phi[:, 0] - phi[:, 3]) + (phi[:, 1] + torch.pi - phi[:, 3])
        return phi_dot


class quadruped_stand(RewardV2):
    def __init__(self, feet_names: str, weight: float, track_var: bool = False):
        super().__init__(weight, track_var=track_var)
        self.feet_names = feet_names

    @override
    def _initialize(self, env: "EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.feet_ids = self.asset.find_bodies(self.feet_names)[0]
        if not hasattr(self.command_manager, "is_standing_env"):
            raise ValueError("is_standing_env is not defined in command_manager")

    def _compute(self):
        jpos_errors = (self.asset.data.joint_pos - self.asset.data.default_joint_pos).abs()
        feet_pos_w = self.asset.data.body_link_pos_w[:, self.feet_ids]
        feet_pos_b = quat_rotate_inverse(
            self.asset.data.root_link_quat_w.unsqueeze(1),
            feet_pos_w - self.asset.data.root_link_pos_w.unsqueeze(1),
        )
        front_symmetry = feet_pos_b[:, [0, 1], 1].sum(dim=1, keepdim=True).abs()
        back_symmetry = feet_pos_b[:, [2, 3], 1].sum(dim=1, keepdim=True).abs()
        cost = -(jpos_errors.sum(dim=1, keepdim=True) + front_symmetry + back_symmetry)

        return cost * self.command_manager.is_standing_env.reshape(self.num_envs, 1)


class lateral_swing_height(RewardV2):
    def __init__(
        self,
        weight: float,
        feet_names: str,
        target_height: float,
        track_var: bool = False,
    ):
        super().__init__(weight, track_var=track_var)
        self.feet_names = feet_names
        self.target_height = target_height

    @override
    def _initialize(self, env: "EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.feet_ids = self.asset.find_bodies(self.feet_names)[0]

    def _compute(self):
        feet_pos_w = self.asset.data.body_link_pos_w[:, self.feet_ids]
        feet_lin_vel_w = self.asset.data.body_lin_vel_w[:, self.feet_ids]
        feet_lin_vel_b = quat_rotate_inverse(
            self.asset.data.root_link_quat_w.unsqueeze(1),
            feet_lin_vel_w,
        )
        feet_height_w = feet_pos_w[:, :, 2] - self.env.get_ground_height_at(feet_pos_w)
        rew = torch.where(
            feet_lin_vel_b[:, :, 1].abs() > 0.1,
            1.0 + (feet_height_w / self.target_height - self.target_height).clamp_max(0.0),
            0.0,
        )
        return rew.sum(1, True) / len(self.feet_ids)


class action_rate_l2(RewardV2):
    """Penalize the rate of change of the action."""

    def __init__(
        self,
        weight: float,
        key: str = "action",
        enabled: bool = True,
        track_var: bool = False,
    ):
        super().__init__(weight, enabled=enabled, track_var=track_var)
        self.key = key

    @override
    def _initialize(self, env: "EnvBase"):
        super()._initialize(env)
        self.action_manager = self.env.input_managers[self.key]
        assert self.action_manager.action_buf.shape[-1] == self.action_manager.action_dim

    def _compute(self) -> torch.Tensor:
        action_buf = self.action_manager.action_buf
        action_diff = action_buf[:, 0] - action_buf[:, 1]
        rew = -action_diff.square().sum(dim=-1, keepdim=True)
        return rew


class action_rate2_l2(RewardV2):
    """Penalize the second order rate of change of the action."""

    def __init__(
        self,
        weight: float,
        key: str = "action",
        enabled: bool = True,
        track_var: bool = False,
    ):
        super().__init__(weight, enabled=enabled, track_var=track_var)
        self.key = key

    @override
    def _initialize(self, env: "EnvBase"):
        super()._initialize(env)
        self.action_manager = self.env.input_managers[self.key]
        assert self.action_manager.action_buf.shape[-1] == self.action_manager.action_dim

    def _compute(self) -> torch.Tensor:
        action_buf = self.action_manager.action_buf
        action_diff = action_buf[:, 0] - 2 * action_buf[:, 1] + action_buf[:, 2]
        rew = -action_diff.square().sum(dim=-1, keepdim=True)
        return rew
