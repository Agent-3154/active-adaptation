from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from typing_extensions import override

from active_adaptation.utils.math import (
    clamp_norm,
    quat_rotate,
    quat_rotate_inverse,
    yaw_quat,
)

from .loco_manip_object import LocoManipObjectScripted

if TYPE_CHECKING:
    from active_adaptation.envs.env_base import EnvBase


class LocoManipBusketScripted(LocoManipObjectScripted):
    """Scripted basket-grasp command using the loco-manip EEF command layout.

    The grasp point is a fixed basket-local handle point instead of a sampled
    height along a vertical stand.
    """

    def __init__(
        self,
        eef_body_name: str,
        gripper_joint_names: str,
        gripper_body_names: str,
        object_name: str = "object",
        platform_name: str | None = None,
        platform_height: float = 0.2,
        grasp_offset_obj: Tuple[float, float, float] = (0.0, 0.0, 0.188),
        standoff_distance: float = 0.45,
        standoff_linvel_gain: float = 2.0,
        standoff_yaw_gain: float = 1.0,
        speed_limit: float = 0.8,
        yaw_rate_range: Tuple[float, float] = (-1.0, 1.0),
        approach_reach_limit: float = 0.75,
        grasp_close_radius: float = 0.035,
        contact_transition_time: float = 0.3,
        lift_height_range: Tuple[float, float] = (0.05, 0.15),
    ) -> None:
        super().__init__(
            eef_body_name=eef_body_name,
            gripper_joint_names=gripper_joint_names,
            gripper_body_names=gripper_body_names,
            object_name=object_name,
            grasp_height_range=(grasp_offset_obj[2], grasp_offset_obj[2]),
            standoff_distance=standoff_distance,
            standoff_linvel_gain=standoff_linvel_gain,
            standoff_yaw_gain=standoff_yaw_gain,
            speed_limit=speed_limit,
            yaw_rate_range=yaw_rate_range,
        )
        self.grasp_offset_obj = tuple(float(v) for v in grasp_offset_obj)
        self.platform_name = platform_name
        self.platform_height = float(platform_height)
        self.approach_reach_limit = approach_reach_limit
        self.grasp_close_radius = grasp_close_radius
        self.contact_transition_time = contact_transition_time
        self.lift_height_range = lift_height_range

    @override
    def _initialize(self, env: "EnvBase") -> None:
        super()._initialize(env)
        if self.platform_name is not None:
            self.platform = self.env.scene[self.platform_name]
            self.platform_init_root_state = self.platform.data.default_root_state.clone()
        else:
            self.platform = None
            self.platform_init_root_state = None
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        self._sample_lift_offsets(all_env_ids)

    @override
    def sample_init(self, env_ids: torch.Tensor) -> dict:
        init_state = super().sample_init(env_ids)
        if self.platform_name is None:
            return init_state

        object_init = init_state[self.object_name]
        platform_init = self.platform_init_root_state[env_ids].clone()
        platform_init[:, :2] = object_init[:, :2]
        platform_init[:, 2] = (
            self.env.get_ground_height_at(platform_init[:, :3])
            + 0.5 * self.platform_height
        )
        platform_init[:, 3:7] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0],
            device=self.device,
            dtype=platform_init.dtype,
        )
        platform_init[:, 7:] = 0.0
        object_init[:, 2] += self.platform_height
        init_state[self.platform_name] = platform_init
        return init_state

    @override
    def sample_commands(self, env_ids: torch.Tensor) -> None:
        self.grasp_height_per_env[env_ids] = self.grasp_offset_obj[2]

    @override
    def reset(self, env_ids: torch.Tensor) -> None:
        super().reset(env_ids)
        self._sample_lift_offsets(env_ids)

    def _sample_lift_offsets(self, env_ids: torch.Tensor) -> None:
        self._lift_offset[env_ids].zero_()
        self._lift_offset[env_ids, 2].uniform_(*self.lift_height_range)

    def _grasp_offset_tensor(self, num_envs: int) -> torch.Tensor:
        return torch.tensor(
            self.grasp_offset_obj,
            device=self.device,
            dtype=self.object_pos_w.dtype,
        ).reshape(1, 3).expand(num_envs, 3)

    @override
    def _read_robot_and_object_state(self) -> None:
        self.root_pos_w = self.asset.data.root_link_pos_w
        self.root_yaw_quat = yaw_quat(self.asset.data.root_link_quat_w)
        self.object_pos_w = self.object.data.root_pos_w
        self.object_quat_w = self.object.data.root_quat_w
        offset_obj = self._grasp_offset_tensor(self.num_envs)
        self.grasp_point_w = self.object_pos_w + quat_rotate(
            self.object_quat_w, offset_obj
        )

    @override
    def _phase_approach(self, env_ids: torch.Tensor) -> None:
        root_pos_w = self.asset.data.root_link_pos_w[env_ids]
        root_yaw_q = self.root_yaw_quat[env_ids]
        grasp_point = self.grasp_point_w[env_ids]

        delta_w = grasp_point - root_pos_w
        if self.approach_reach_limit > 0.0:
            delta_w = clamp_norm(delta_w, max=self.approach_reach_limit)
        self.cmd_eef_pos_w[env_ids] = root_pos_w + delta_w
        self.cmd_eef_pos_b[env_ids] = quat_rotate_inverse(
            root_yaw_q,
            self.cmd_eef_pos_w[env_ids]
            - root_pos_w * torch.tensor([1.0, 1.0, 0.0], device=self.device),
        )

        self.cmd_eef_rot_w[env_ids] = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]], device=self.device
        )
        self._drive_base(
            env_ids,
            self.approach_standoff_w[env_ids],
            torch.zeros(len(env_ids), device=self.device),
        )

        eef_pos_error = (grasp_point - self.eef_pos_w[env_ids]).norm(
            dim=-1, keepdim=True
        )
        should_grasp = (eef_pos_error < self.grasp_close_radius).reshape(-1, 1)
        self.should_grasp[env_ids] = should_grasp | self.should_grasp[env_ids]
        self.cmd_eef_status[env_ids] = torch.where(should_grasp, 1, 0)

        ct = self.contact_forces.data.current_contact_time[env_ids][
            :, self.gripper_body_ids
        ]
        ct = ct.amax(dim=-1, keepdim=True)
        next_phase = torch.where(
            should_grasp & (ct > self.contact_transition_time), 1, 0
        )
        self.phase_ids[env_ids] = next_phase.squeeze(-1)


__all__ = ["LocoManipBusketScripted"]
