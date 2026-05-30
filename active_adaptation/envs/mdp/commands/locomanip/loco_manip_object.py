from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from typing_extensions import override

from active_adaptation.utils.math import (
    clamp_norm,
    quat_rotate,
    quat_rotate_inverse,
    wrap_to_pi,
    yaw_quat,
)
from active_adaptation.utils.symmetry import SymmetryTransform
from ..base import Command

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject


class LocoManipObject(Command):
    """Scripted FSM command for picking up an object and transporting it to a target.

    Outputs the same ``command(key="dense"|"sparse")`` interface as
    ``SingleEEFLocoManip`` (17D dense / 14D sparse, body/yaw frame) so a
    pre-trained policy can execute the full manipulation sequence without
    retraining.

    FSM states (per env):
        APPROACH   – drive base to approach standoff; EEF pre-positions above
                     grasp point; gripper open.
        GRASP_POSE – base holds at standoff; EEF descends to exact grasp point;
                     gripper open.
        CLOSE      – hold position; command gripper closed; wait for closure.
        LIFT       – raise EEF to lift height above grasp point; gripper closed.
        MOVE       – drive base to target standoff; EEF tracks lifted goal;
                     gripper closed.
        RELEASE    – hold at target standoff; lower EEF; command gripper open;
                     wait for opening.
        BACKUP     – drive base back to approach standoff; gripper open.
                     Transitions back to APPROACH to loop within the episode.
    """

    APPROACH   = 0
    GRASP_POSE = 1
    CLOSE      = 2
    LIFT       = 3
    MOVE       = 4
    RELEASE    = 5
    BACKUP     = 6

    supported_backends = ("isaac",)

    def __init__(
        self,
        env,
        eef_body_name: str,
        gripper_joint_names: str,
        object_name: str = "object",
        object_distance_range: Tuple[float, float] = (2.0, 3.0),
        target_distance_range: Tuple[float, float] = (2.0, 3.0),
        grasp_height_range: Tuple[float, float] = (0.05, 0.5),
        pre_grasp_height_offset: float = 0.25,
        lift_height: float = 0.35,
        standoff_distance_range: Tuple[float, float] = (0.5, 0.8),
        standoff_angle_range: Tuple[float, float] = (-torch.pi / 3, torch.pi / 3),
        yaw_rate_range: Tuple[float, float] = (-1.0, 1.0),
        standoff_linvel_gain: float = 2.0,
        standoff_yaw_gain: float = 1.0,
        speed_limit: float = 0.8,
        eef_pos_threshold: float = 0.05,
        base_pos_threshold: float = 0.2,
        yaw_threshold: float = 0.15,
        gripper_close_threshold: float = 0.7,
        gripper_open_threshold: float = 0.3,
        teleop: bool = False,
    ) -> None:
        super().__init__(env, teleop)

        body_ids, _ = self.asset.find_bodies(eef_body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected exactly one body matching {eef_body_name!r}, got {len(body_ids)}"
            )
        self.eef_body_idx = body_ids[0]

        joint_ids, _ = self.asset.find_joints(gripper_joint_names)
        self.gripper_joint_ids = torch.tensor(joint_ids, device=self.device)
        limits = self.asset.data.soft_joint_pos_limits[0, self.gripper_joint_ids]
        self._gripper_max_open = limits.abs().amax(dim=-1).max().clamp_min(1e-6)

        self.object_name = object_name
        self.object: RigidObject = self.env.scene[object_name]
        self.object_init_root_state = self.object.data.default_root_state.clone()

        self.object_distance_range = object_distance_range
        self.target_distance_range = target_distance_range
        self.grasp_height_range = grasp_height_range
        self.pre_grasp_height_offset = pre_grasp_height_offset
        self.lift_height = lift_height
        self.standoff_distance_range = standoff_distance_range
        self.standoff_angle_range = standoff_angle_range
        self.yaw_rate_range = yaw_rate_range
        self.standoff_linvel_gain = standoff_linvel_gain
        self.standoff_yaw_gain = standoff_yaw_gain
        self.speed_limit = speed_limit
        self.eef_pos_threshold = eef_pos_threshold
        self.base_pos_threshold = base_pos_threshold
        self.yaw_threshold = yaw_threshold
        self.gripper_close_threshold = gripper_close_threshold
        self.gripper_open_threshold = gripper_open_threshold

        self.phase_approach_end = 200
        self.phase_grasp_end = 400
        self.phase_lift_end = 500

        with torch.device(self.device):
            # FSM state per env
            self.state = torch.zeros(self.num_envs, dtype=torch.long)

            # Sampled scene layout (set in sample_init / sample_commands)
            self.grasp_height_per_env = torch.zeros(self.num_envs)
            self.grasp_point_w = torch.zeros(self.num_envs, 3)
            self.target_pos_w = torch.zeros(self.num_envs, 3)
            self.robot_init_pos_w = torch.zeros(self.num_envs, 3)
            self.approach_standoff_w = torch.zeros(self.num_envs, 3)
            self.approach_yaw_w = torch.zeros(self.num_envs)
            self.target_standoff_w = torch.zeros(self.num_envs, 3)
            self.target_yaw_w = torch.zeros(self.num_envs)

            # Command tensors – match SingleEEFLocoManip field names exactly
            self.cmd_linvel_b = torch.zeros(self.num_envs, 3)
            self.cmd_linvel_w = torch.zeros(self.num_envs, 3)
            self.cmd_yawvel_b = torch.zeros(self.num_envs, 1)
            self.cmd_eef_pos_b = torch.zeros(self.num_envs, 3)
            self.cmd_eef_pos_w = torch.zeros(self.num_envs, 3)
            self.eef_pos_w = torch.zeros(self.num_envs, 3)

            self.pos_diff_w = torch.zeros(self.num_envs, 3)
            self.pos_diff_b = torch.zeros(self.num_envs, 3)
            self.pos_error_norm2 = torch.zeros(self.num_envs, 1)
            self.pos_error_norm = torch.zeros(self.num_envs, 1)
            self.eef_pos_reached = torch.zeros(self.num_envs, 1, dtype=torch.bool)
            self.eef_pos_reaching = torch.zeros(self.num_envs, 1, dtype=torch.bool)

            self.eef_forward_w = torch.zeros(self.num_envs, 3)
            self.eef_forward_b = torch.zeros(self.num_envs, 3)
            self.cmd_eef_forward_w = torch.zeros(self.num_envs, 3)
            self.cmd_eef_forward_b = torch.zeros(self.num_envs, 3)

            # Gripper: eef_status = continuous closedness [0,1]; cmd = {0,1}
            self.should_grasp = torch.zeros(self.num_envs, 1, dtype=torch.bool)
            self.eef_status = torch.zeros(self.num_envs, 1)
            self.cmd_eef_status = torch.zeros(self.num_envs, 1, dtype=torch.long)

            self.command_speed = torch.zeros(self.num_envs, 1)
            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=torch.bool)

            self.move_offset_w = torch.zeros(self.num_envs, 3)

            self._forward_axis_b = torch.tensor([[1.0, 0.0, 0.0]])
            # self._grasp_pre_offset = torch.tensor([[-0.15, 0.0, 0.0]])
            self._grasp_cmd_offset = torch.tensor([[-0.15, 0.0, 0.0]])
            self._lift_offset = torch.tensor([[0.0, 0.0, 0.15]])

        self.grasp_marker = None
        self.target_marker = None
        if self.env.backend == "isaac" and self.env.sim.has_gui():
            from active_adaptation.envs.backends.isaac import IsaacSceneAdapter

            self.scene: IsaacSceneAdapter = self.env.scene
            self.grasp_marker = self.scene.create_sphere_marker(
                "/Visuals/Command/object_grasp_point", (1.0, 0.4, 0.0), radius=0.03
            )
            self.target_marker = self.scene.create_sphere_marker(
                "/Visuals/Command/object_target_pos", (0.0, 0.4, 1.0), radius=0.05
            )

    # ------------------------------------------------------------------ #
    # Command / symmetry interface (matches SingleEEFLocoManip exactly)   #
    # ------------------------------------------------------------------ #

    def command(self, key: str = "dense") -> torch.Tensor:
        if key == "dense":
            cmd = torch.cat([
                self.cmd_linvel_b[:, :2],                    # 2
                self.cmd_yawvel_b,                           # 1
                self.cmd_eef_pos_b,                          # 3
                self.pos_diff_b,                             # 3
                self.cmd_eef_forward_b,                      # 3
                self.cmd_eef_forward_b - self.eef_forward_b, # 3
                self.cmd_eef_status.float(),                 # 1
                (1 - self.cmd_eef_status).float(),           # 1
            ], dim=-1) # [N, 17]
            assert cmd.shape == (self.num_envs, 17)
            return cmd
        elif key == "sparse":
            return torch.cat([
                self.cmd_eef_pos_b,                          # 3
                self.pos_diff_b,                             # 3
                self.cmd_eef_forward_b,                      # 3
                self.cmd_eef_forward_b - self.eef_forward_b, # 3
                self.cmd_eef_status.float(),                 # 1
                (1 - self.cmd_eef_status).float(),           # 1
            ], dim=-1)
        else:
            raise ValueError(f"Invalid key: {key}")

    @override
    def symmetry_transform(self, key: str = "dense"):
        if key == "dense":
            cmd_linvel_b     = SymmetryTransform(perm=[0, 1],    signs=[1, -1])
            cmd_yawvel_b     = SymmetryTransform(perm=[0],       signs=[-1])
            cmd_eef_pos_b    = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
            pos_diff_b       = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
            cmd_eef_forward_b = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
            fwd_diff_b       = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
            eef_status       = SymmetryTransform(perm=[0, 1],    signs=[1,  1])
            return SymmetryTransform.cat([
                cmd_linvel_b, cmd_yawvel_b, cmd_eef_pos_b, pos_diff_b,
                cmd_eef_forward_b, fwd_diff_b, eef_status,
            ])
        elif key == "sparse":
            cmd_eef_pos_b    = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
            pos_diff_b       = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
            cmd_eef_forward_b = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
            fwd_diff_b       = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
            eef_status       = SymmetryTransform(perm=[0, 1],    signs=[1,  1])
            return SymmetryTransform.cat([
                cmd_eef_pos_b, pos_diff_b, cmd_eef_forward_b, fwd_diff_b, eef_status,
            ])
        else:
            raise ValueError(f"Invalid key: {key}")

    # ------------------------------------------------------------------ #
    # Init / sampling                                                      #
    # ------------------------------------------------------------------ #

    def _sample_uniform(
        self, num_samples: int, value_range: Tuple[float, float]
    ) -> torch.Tensor:
        lo, hi = value_range
        return torch.rand(num_samples, device=self.device) * (hi - lo) + lo

    def _compute_standoff(
        self,
        ref_w: torch.Tensor,
        robot_w: torch.Tensor,
        distance: float,
    ) -> torch.Tensor:
        """Standoff on the segment from ``ref_w`` toward ``robot_w``."""
        diff = robot_w - ref_w
        direction = diff / diff.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        standoff = ref_w + direction * distance
        standoff[:, 2] = self.env.get_ground_height_at(standoff)
        return standoff

    def _update_standoffs(self, env_ids: torch.Tensor) -> None:
        robot_w = self.asset.data.root_link_pos_w[env_ids]
        object_w = self.object.data.root_pos_w[env_ids]
        target_w = self.target_pos_w[env_ids]
        heading_w = self.asset.data.heading_w[env_ids]

        standoff_dist = 0.7
        self.approach_standoff_w[env_ids] = self._compute_standoff(
            object_w, robot_w, standoff_dist
        )
        self.target_standoff_w[env_ids] = self._compute_standoff(
            target_w, robot_w, standoff_dist
        )
        # Match SingleEEFLocoManip world-goal behavior: standoff yaw = current heading.
        self.approach_yaw_w[env_ids] = heading_w
        self.target_yaw_w[env_ids] = heading_w

    @override
    def sample_init(self, env_ids: torch.Tensor) -> dict:
        origins = self.env.scene.get_spawn_origins(env_ids)
        n = len(env_ids)

        # Object at env-local [2.0, 0., 0.]
        object_init = self.object_init_root_state[env_ids].clone()
        default_obj_z = object_init[:, 2].clone()
        object_init[:, 0] = origins[:, 0] + 2.0
        object_init[:, 1] = origins[:, 1] + 0.0
        object_init[:, 2] = (
            self.env.get_ground_height_at(object_init[:, :3]) + default_obj_z
        )
        if object_init.shape[-1] > 7:
            object_init[:, 7:] = 0.0

        # Robot at env-local [0., y, z_default], default rotation
        robot_init = self.init_root_state[env_ids].clone()
        default_robot_z = robot_init[:, 2].clone()
        y = self._sample_uniform(n, (-0.5, 0.5))
        robot_init[:, 0] = origins[:, 0] + 0.0
        robot_init[:, 1] = origins[:, 1] + y
        robot_init[:, 2] = (
            self.env.get_ground_height_at(robot_init[:, :3]) + default_robot_z
        )

        robot_init_pos_w = robot_init[:, :3].clone()
        self.robot_init_pos_w[env_ids] = robot_init_pos_w

        target = origins.clone()
        target[:, 2] = self.env.get_ground_height_at(target)
        self.target_pos_w[env_ids] = target

        self.grasp_height_per_env[env_ids] = self._sample_uniform(n, self.grasp_height_range)

        return {"robot": robot_init, self.object_name: object_init}

    def sample_commands(self, env_ids: torch.Tensor) -> None:
        """Mid-episode resample: re-use existing object/target positions and restart."""
        self.grasp_height_per_env[env_ids] = self._sample_uniform(
            len(env_ids), self.grasp_height_range
        )

    # ------------------------------------------------------------------ #
    # Per-step helpers                                                     #
    # ------------------------------------------------------------------ #

    def _compute_grasp_point(self, env_ids: torch.Tensor) -> torch.Tensor:
        object_pos_w = self.object.data.root_pos_w[env_ids]
        object_quat_w = self.object.data.root_quat_w[env_ids]
        offset_obj = torch.zeros(len(env_ids), 3, device=self.device)
        offset_obj[:, 2] = self.grasp_height_per_env[env_ids]
        return object_pos_w + quat_rotate(object_quat_w, offset_obj)

    def _sync_eef_pose(self, env_ids: torch.Tensor) -> torch.Tensor:
        root_yaw_q = yaw_quat(self.asset.data.root_link_quat_w[env_ids])
        self.eef_pos_w[env_ids] = self.asset.data.body_link_pos_w[env_ids, self.eef_body_idx]
        eef_quat_w = self.asset.data.body_link_quat_w[env_ids, self.eef_body_idx]
        self.eef_forward_w[env_ids] = quat_rotate(eef_quat_w, self._forward_axis_b)
        self.eef_forward_b[env_ids] = quat_rotate_inverse(root_yaw_q, self.eef_forward_w[env_ids])
        return root_yaw_q

    def _set_eef_cmd_world(
        self,
        env_ids: torch.Tensor,
        target_w: torch.Tensor,
        height_ref_w: torch.Tensor,
        root_pos_w: torch.Tensor,
        root_yaw_q: torch.Tensor,
    ) -> None:
        self.cmd_eef_pos_w[env_ids] = target_w
        self.cmd_eef_pos_b[env_ids] = quat_rotate_inverse(root_yaw_q, target_w - root_pos_w)
        self.cmd_eef_pos_b[env_ids, 2] = (
            height_ref_w[:, 2] - self.env.get_ground_height_at(height_ref_w)
        )

    def _set_eef_cmd_from_body(self, env_ids: torch.Tensor) -> None:
        root_pos_w = self.asset.data.root_link_pos_w[env_ids]
        root_yaw_q = yaw_quat(self.asset.data.root_link_quat_w[env_ids])
        exy = torch.zeros(len(env_ids), 3, device=self.device)
        exy[:, :2] = self.cmd_eef_pos_b[env_ids, :2]
        delta_w = quat_rotate(root_yaw_q, exy)
        horiz_w = root_pos_w + delta_w
        ground_h = self.env.get_ground_height_at(horiz_w)
        self.cmd_eef_pos_w[env_ids, :2] = horiz_w[:, :2]
        self.cmd_eef_pos_w[env_ids, 2] = ground_h + self.cmd_eef_pos_b[env_ids, 2]

    def _sync_pos_error(self, env_ids: torch.Tensor, root_yaw_q: torch.Tensor) -> None:
        self.pos_diff_w[env_ids] = self.cmd_eef_pos_w[env_ids] - self.eef_pos_w[env_ids]
        self.pos_diff_b[env_ids] = quat_rotate_inverse(root_yaw_q, self.pos_diff_w[env_ids])
        self.pos_error_norm2[env_ids] = self.pos_diff_w[env_ids].square().sum(dim=-1, keepdim=True)
        self.pos_error_norm[env_ids] = self.pos_error_norm2[env_ids].sqrt()

    def _set_forward_cmd(self, env_ids: torch.Tensor, root_yaw_q: torch.Tensor) -> None:
        forward_w = self._forward_axis_b.expand(len(env_ids), -1)
        self.cmd_eef_forward_w[env_ids] = forward_w
        self.cmd_eef_forward_b[env_ids] = quat_rotate_inverse(root_yaw_q, forward_w)

    def _phase_ids(self, mask: torch.Tensor) -> torch.Tensor:
        return mask.nonzero(as_tuple=False).reshape(-1)

    def _phase_approach(self, env_ids: torch.Tensor) -> None:
        grasp_point = self._compute_grasp_point(env_ids)
        self.grasp_point_w[env_ids] = grasp_point

        root_pos_w = self.asset.data.root_link_pos_w[env_ids]
        root_yaw_q = self._sync_eef_pose(env_ids)

        self._set_eef_cmd_world(
            env_ids,
            grasp_point + self._grasp_cmd_offset,
            grasp_point,
            root_pos_w,
            root_yaw_q,
        )
        self._sync_pos_error(env_ids, root_yaw_q)
        self._set_forward_cmd(env_ids, root_yaw_q)
        self.cmd_eef_status[env_ids, 0] = 0

        self._drive_base(env_ids, self.approach_standoff_w[env_ids], torch.zeros(len(env_ids), device=self.device))

    def _phase_grasp(self, env_ids: torch.Tensor) -> None:
        grasp_point = self._compute_grasp_point(env_ids)
        self.grasp_point_w[env_ids] = grasp_point

        root_pos_w = self.asset.data.root_link_pos_w[env_ids]
        root_yaw_q = self._sync_eef_pose(env_ids)

        self.cmd_eef_pos_w[env_ids] = self.cmd_eef_pos_w[env_ids].lerp(grasp_point, 0.05)
        self._set_eef_cmd_world(env_ids, self.cmd_eef_pos_w[env_ids], grasp_point, root_pos_w, root_yaw_q)
        self._sync_pos_error(env_ids, root_yaw_q)
        self._set_forward_cmd(env_ids, root_yaw_q)

        pos_xy_error = self.pos_diff_w[env_ids, :2].norm(dim=-1, keepdim=True)
        pos_error = (self.cmd_eef_pos_w[env_ids] - grasp_point).norm(dim=-1, keepdim=True)
        should_grasp = (pos_xy_error < 0.02) & (pos_error < 0.02)
        self.should_grasp[env_ids] = should_grasp | self.should_grasp[env_ids]
        self.cmd_eef_status[env_ids, 0] = self.should_grasp[env_ids, 0].long()

        self._drive_base(env_ids, self.approach_standoff_w[env_ids], torch.zeros(len(env_ids), device=self.device))

    def _phase_lift(self, env_ids: torch.Tensor) -> None:
        grasp_point = self._compute_grasp_point(env_ids)
        self.grasp_point_w[env_ids] = grasp_point

        root_pos_w = self.asset.data.root_link_pos_w[env_ids]
        root_yaw_q = self._sync_eef_pose(env_ids)

        lift_target = grasp_point + self._lift_offset
        self._set_eef_cmd_world(env_ids, lift_target, grasp_point, root_pos_w, root_yaw_q)
        self._sync_pos_error(env_ids, root_yaw_q)
        self._set_forward_cmd(env_ids, root_yaw_q)
        self.cmd_eef_status[env_ids, 0] = 1

        self._drive_base(env_ids, self.approach_standoff_w[env_ids], torch.zeros(len(env_ids), device=self.device))

    def _phase_move(self, env_ids: torch.Tensor) -> None:
        grasp_point = self._compute_grasp_point(env_ids)
        self.grasp_point_w[env_ids] = grasp_point

        root_yaw_q = self._sync_eef_pose(env_ids)
        self._set_eef_cmd_from_body(env_ids)
        self._sync_pos_error(env_ids, root_yaw_q)
        self._set_forward_cmd(env_ids, root_yaw_q)
        self.cmd_eef_status[env_ids, 0] = 1

        move_yaw = torch.full((len(env_ids),), torch.pi / 3, device=self.device)
        self._drive_base(
            env_ids,
            self.approach_standoff_w[env_ids] + self.move_offset_w[env_ids],
            move_yaw,
        )

    def _drive_base(
        self,
        ids: torch.Tensor,
        standoff_w: torch.Tensor,
        yaw_w: torch.Tensor,
    ) -> None:
        root_pos = self.asset.data.root_link_pos_w[ids]
        root_yaw_q = yaw_quat(self.asset.data.root_link_quat_w[ids])
        delta_w = standoff_w - root_pos
        delta_w[:, 2] = 0.0
        linvel_w = clamp_norm(
            self.standoff_linvel_gain * delta_w, max=self.speed_limit
        )
        self.cmd_linvel_w[ids] = linvel_w
        self.cmd_linvel_b[ids] = quat_rotate_inverse(root_yaw_q, linvel_w)
        yaw_err = wrap_to_pi(yaw_w - self.asset.data.heading_w[ids])
        self.cmd_yawvel_b[ids, 0] = (
            self.standoff_yaw_gain * yaw_err
        ).clamp(*self.yaw_rate_range)

    # ------------------------------------------------------------------ #
    # Overrides                                                            #
    # ------------------------------------------------------------------ #

    @override
    def reset(self, env_ids: torch.Tensor) -> None:
        self.sample_commands(env_ids)
        self._update_standoffs(env_ids)
        self.eef_pos_reached[env_ids] = False
        self.eef_pos_reaching[env_ids] = False
        self.should_grasp[env_ids] = False
        move_offset = torch.zeros(len(env_ids), 3, device=self.device)
        move_offset[:, 0].uniform_(-1.0, 1.0)
        move_offset[:, 1].uniform_(-1.0, 1.0)
        self.move_offset_w[env_ids] = move_offset

    @override
    def update(self) -> None:
        step = self.env.episode_length_buf
        approach_ids = self._phase_ids(step < self.phase_approach_end)
        grasp_ids = self._phase_ids(
            (step >= self.phase_approach_end) & (step < self.phase_grasp_end)
        )
        lift_ids = self._phase_ids(
            (step >= self.phase_grasp_end) & (step < self.phase_lift_end)
        )
        move_ids = self._phase_ids(step >= self.phase_lift_end)

        if approach_ids.numel() > 0:
            self._phase_approach(approach_ids)
        if grasp_ids.numel() > 0:
            self._phase_grasp(grasp_ids)
        if lift_ids.numel() > 0:
            self._phase_lift(lift_ids)
        if move_ids.numel() > 0:
            self._phase_move(move_ids)

        self.command_speed = self.cmd_linvel_w.norm(dim=-1, keepdim=True)
        self.is_standing_env = self.command_speed < 0.1

    @override
    def debug_draw(self) -> None:
        self.env.debug_draw.vector(
            self.asset.data.root_link_pos_w,
            self.cmd_linvel_w,
            color=(1.0, 1.0, 1.0, 1.0),
        )
        self.env.debug_draw.vector(
            self.eef_pos_w, self.eef_forward_w, color=(1.0, 0.0, 0.0, 1.0)
        )
        self.env.debug_draw.vector(
            self.eef_pos_w, self.cmd_eef_forward_w, color=(0.0, 1.0, 0.0, 1.0)
        )
        self.env.debug_draw.vector(
            self.eef_pos_w,
            self.cmd_eef_pos_w - self.eef_pos_w,
            color=(0.0, 0.0, 1.0, 1.0),
        )
        self.grasp_marker.visualize(self.cmd_eef_pos_w)
        self.target_marker.visualize(self.approach_standoff_w)


__all__ = ["LocoManipObject"]
