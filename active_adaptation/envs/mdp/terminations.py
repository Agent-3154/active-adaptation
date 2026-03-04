import torch
import abc

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.sensors import ContactSensor

from .base import Termination


class crash(Termination):
    """
    Hard termination given by undesired contact forces on the specified body names.
    """

    def __init__(self, env, body_names_expr: str, t_thres: float = 0.0):
        super().__init__(env)
        self.t_thres = t_thres
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene.sensors["contact_forces"]
        self.body_indices, self.body_names = self.contact_sensor.find_bodies(
            body_names_expr
        )
        self.body_indices = torch.tensor(self.body_indices, device=self.env.device)

    def compute(self, termination: torch.Tensor):
        contact_time = self.contact_sensor.data.current_contact_time[
            :, self.body_indices
        ]
        return (contact_time > self.t_thres).any(1, True)


class undesired_contact(Termination):
    """
    Soft termination based on the contact forces on the specified body names.
    """

    supported_backends = ("isaac",)

    def __init__(
        self,
        env,
        body_names: str,
        thres: float = 1.0,
        lateral_only: bool = False,
    ):
        super().__init__(env)
        self.thres = thres
        if lateral_only:
            self.dim = 2
        else:
            self.dim = 3
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.body_indices, self.body_names = self.contact_sensor.find_bodies(body_names)
        self.body_indices = torch.tensor(self.body_indices, device=self.env.device)

    def compute(self, termination: torch.Tensor):
        terminated = torch.zeros(self.num_envs, 1, device=self.env.device, dtype=bool)
        forces = self.contact_sensor.data.net_forces_w[
            :, self.body_indices, : self.dim
        ].norm(dim=-1, keepdim=True)
        in_contact = (forces > self.thres).sum(dim=1)
        discount = 0.8**in_contact
        return terminated, discount.reshape(self.num_envs, 1)


class fall_over(Termination):
    def __init__(
        self,
        env,
        xy_thres: float = 0.8,
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.xy_thres = xy_thres

    def compute(self, termination: torch.Tensor):
        gravity_xy: torch.Tensor = self.asset.data.projected_gravity_b[:, :2]
        fall_over = gravity_xy.norm(dim=1, keepdim=True) >= self.xy_thres
        return fall_over


class root_pos_error(Termination):
    def __init__(self, env, threshold: float = 2.0, dim: int = 2):
        super().__init__(env)
        self.threshold = threshold
        self.asset: Articulation = self.env.scene["robot"]
        self.dim = dim

    def compute(self, termination: torch.Tensor) -> torch.Tensor:
        valid = (self.env.episode_length_buf > 10).unsqueeze(-1)
        target_pos = self.command_manager.cmd_pos_w[:, : self.dim]
        pos_error = (
            (self.asset.data.root_link_pos_w[:, : self.dim] - target_pos)
            .square()
            .sum(-1, True)
        )
        return (valid & (pos_error > self.threshold)).reshape(self.num_envs, 1)


class cum_error(Termination):
    def __init__(self, env, thres: float = 0.85, min_steps: int = 50):
        super().__init__(env)
        self.thres = torch.tensor(thres, device=self.env.device)
        self.command_manager = self.env.command_manager
        if not hasattr(self.command_manager, "cum_error"):
            raise ValueError("`cum_error` attribute not found in command manager")

    def compute(self, termination: torch.Tensor) -> torch.Tensor:
        return (self.command_manager.cum_error > self.thres).any(-1, True)


class ee_cum_error(Termination):
    def __init__(self, env, thres: float = 1.0, min_steps: int = 50):
        super().__init__(env)
        from .commands import CommandEEPose_Cont

        self.thres = torch.as_tensor(thres, device=self.env.device)
        self.min_steps = min_steps
        self.command_manager: CommandEEPose_Cont = self.env.command_manager

    def compute(self, termination: torch.Tensor) -> torch.Tensor:
        a = (self.command_manager._cum_error > self.thres).any(-1)
        b = self.env.episode_length_buf > self.min_steps
        return (a & b).reshape(-1, 1)


class joint_acc_exceeds(Termination):
    def __init__(self, env, thres: float):
        super().__init__(env)
        self.thres = thres
        self.asset: Articulation = self.env.scene["robot"]

    def compute(self, termination: torch.Tensor) -> torch.Tensor:
        valid = (self.env.episode_length_buf > 2).unsqueeze(-1)
        return valid & (self.asset.data.joint_acc.abs() > self.thres).any(1, True)


class impedance_pos_error(Termination):
    def __init__(self, env, thres: float = 0.3):
        super().__init__(env)
        self.thres = thres
        self.command_manger = self.env.command_manager
        self.asset: Articulation = self.env.scene["robot"]

    def compute(self, termination: torch.Tensor):
        error = (self.asset.data.root_pos_w - self.command_manger.des_pos_w)[
            :, :2
        ].norm(dim=-1, keepdim=True)
        return error > self.thres


class root_height_below(Termination):
    def __init__(self, env, thres: float):
        super().__init__(env)
        self.thres = thres
        self.asset: Articulation = self.env.scene["robot"]

    def compute(self, termination: torch.Tensor) -> torch.Tensor:
        ground_height = self.env.get_ground_height_at(self.asset.data.root_pos_w)
        height = self.asset.data.root_pos_w[:, 2] - ground_height
        return (height < self.thres).reshape(self.num_envs, 1)


class force_contact(Termination):
    def __init__(self, env, body_names: str, threshold: float):
        super().__init__(env)
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.body_indices, self.body_names = self.contact_sensor.find_bodies(body_names)
        self.threshold = threshold

    def compute(self, termination: torch.Tensor):
        forces = self.contact_sensor.data.net_forces_w[:, self.body_indices].norm(
            dim=-1
        )
        in_contact = forces.sum(dim=1, keepdim=True) > self.threshold
        return in_contact
