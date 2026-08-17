"""Task-space action terms (end-effector pose via differential IK)."""

from __future__ import annotations

import torch

from typing import TYPE_CHECKING, Sequence
from typing_extensions import override

from tensordict import TensorDictBase

from active_adaptation.envs.utils import find_bodies, find_joints
from active_adaptation.utils.ik import DifferentialIKController
from active_adaptation.utils.math import (
    matrix_from_quat,
    quat_conjugate,
    quat_rotate,
    quat_rotate_inverse,
)

from .base import ActionV2

if TYPE_CHECKING:
    from active_adaptation.envs.env_base import _EnvBase


class EndEffectorPose(ActionV2):
    """Absolute end-effector position command in the articulation body frame.

    Policy actions are desired EE positions ``(x, y, z)`` expressed in the
    robot root / body frame. Each physics substep solves one damped
    least-squares (DLS) differential IK step and writes joint position
    targets to the articulation PD controller.

    Relative (delta) commands are intentionally out of scope; use a separate
    ``EndEffectorPoseDelta`` term for that.
    """

    supported_backends = ("isaac", "mjlab")

    def __init__(
        self,
        joint_names: str | Sequence[str],
        body_name: str,
        *,
        damping: float = 0.05,
        max_dq: float = 0.5,
    ) -> None:
        super().__init__()
        self.joint_names_expr = joint_names
        self.body_name = body_name
        self.damping = float(damping)
        self.max_dq = float(max_dq)

    @override
    def _initialize(self, env: "_EnvBase") -> None:
        super()._initialize(env)

        joint_ids, self.joint_names = find_joints(self.asset, self.joint_names_expr)
        body_ids, body_names = find_bodies(self.asset, self.body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected exactly one body matching {self.body_name!r}, "
                f"got {body_names}."
            )
        self.joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)
        self.body_idx = int(body_ids[0])
        self.body_name_resolved = body_names[0]
        self.num_joints = len(joint_ids)

        self.controller = DifferentialIKController(
            self.num_envs,
            self.device,
            damping=self.damping,
            max_dq=self.max_dq,
        )
        self.action_dim = self.controller.action_dim
        self.names = ["ee_x", "ee_y", "ee_z"]

        with torch.device(self.device):
            self.raw_actions = torch.zeros(self.num_envs, self.action_dim)

        self._setup_jacobian_indexing()

    def _setup_jacobian_indexing(self) -> None:
        """Cache backend-specific Jacobian / body indices."""
        if self.env.backend == "isaac":
            # PhysX omits the root body column for fixed-base articulations and
            # prepends 6 floating-base DoFs for mobile bases.
            if self.asset.is_fixed_base:
                self._jacobi_body_idx = self.body_idx - 1
                self._jacobi_joint_ids = self.joint_ids
            else:
                self._jacobi_body_idx = self.body_idx
                self._jacobi_joint_ids = self.joint_ids + 6
            self._mj_joint_dof_ids = None
            self._mj_body_id = None
        elif self.env.backend == "mjlab":
            indexing = self.asset.indexing
            self._mj_joint_dof_ids = indexing.joint_v_adr[self.joint_ids]
            self._mj_body_id = int(indexing.body_ids[self.body_idx].item())
            self._jacobi_body_idx = None
            self._jacobi_joint_ids = None

            import warp as wp

            sim = self.env.sim._sim  # underlying mjlab Simulation
            nworld = self.num_envs
            nv = sim.mj_model.nv
            with wp.ScopedDevice(sim.wp_device):
                self._jacp_wp = wp.zeros((nworld, 3, nv), dtype=float)
                self._jacr_wp = wp.zeros((nworld, 3, nv), dtype=float)
                self._point_wp = wp.zeros(nworld, dtype=wp.vec3)
                self._body_wp = wp.zeros(nworld, dtype=wp.int32)
                self._body_wp.fill_(self._mj_body_id)
            self._jacp_torch = wp.to_torch(self._jacp_wp)
            self._point_torch = wp.to_torch(self._point_wp).view(nworld, 3)
        else:
            raise RuntimeError(
                f"EndEffectorPose does not support backend {self.env.backend!r}."
            )

    def __repr__(self) -> str:
        return (
            f"EndEffectorPose(body={self.body_name_resolved!r}, "
            f"joints={self.joint_names}, damping={self.damping}, "
            f"max_dq={self.max_dq})"
        )

    @override
    def reset(self, env_ids: torch.Tensor, tensordict: TensorDictBase) -> None:
        self.raw_actions[env_ids] = 0.0
        self.controller.reset(env_ids)

    @override
    def process_action(self, action: torch.Tensor | None) -> None:
        if action is None:
            return
        self.raw_actions[:] = action
        # Absolute body-frame position target.
        self.controller.set_command(action)

    @override
    def apply_action(self, substep: int) -> None:
        ee_pos_b = self._ee_pos_body()
        jacobian_b = self._jacobian_pos_body()
        joint_pos = self.asset.data.joint_pos[:, self.joint_ids]
        joint_pos_des = self.controller.compute(ee_pos_b, jacobian_b, joint_pos)
        self.asset.set_joint_position_target(joint_pos_des, joint_ids=self.joint_ids)

    @override
    def symmetry_transform(self):
        raise NotImplementedError(
            "EndEffectorPose has no defined left/right symmetry transform."
        )

    @override
    def debug_draw(self) -> None:
        if not self.env.sim.has_gui():
            return
        # Desired and current EE positions in world for visualization.
        root_pos = self.asset.data.root_link_pos_w
        root_quat = self.asset.data.root_link_quat_w
        des_w = root_pos + quat_rotate(root_quat, self.controller.ee_pos_des)
        cur_w = self.asset.data.body_link_pos_w[:, self.body_idx]
        self.env.scene.draw_point(des_w, color=(0.1, 0.9, 0.2, 1.0), size=12.0)
        self.env.scene.draw_vector(
            cur_w, des_w - cur_w, color=(0.2, 0.6, 1.0, 1.0), size=2.0
        )

    # ------------------------------------------------------------------
    # Pose / Jacobian (body frame)
    # ------------------------------------------------------------------

    def _ee_pos_body(self) -> torch.Tensor:
        """Current EE position in the articulation root / body frame."""
        ee_pos_w = self.asset.data.body_link_pos_w[:, self.body_idx]
        root_pos_w = self.asset.data.root_link_pos_w
        root_quat_w = self.asset.data.root_link_quat_w
        return quat_rotate_inverse(root_quat_w, ee_pos_w - root_pos_w)

    def _jacobian_pos_body(self) -> torch.Tensor:
        """Geometric position Jacobian of the EE in the body frame.

        Shape ``(num_envs, 3, num_joints)``.
        """
        jacobian_w = self._jacobian_pos_world()
        root_quat_w = self.asset.data.root_link_quat_w
        # R_bw = R_wbᵀ maps world spatial vectors into the body frame.
        rot_bw = matrix_from_quat(quat_conjugate(root_quat_w))
        return torch.bmm(rot_bw, jacobian_w)

    def _jacobian_pos_world(self) -> torch.Tensor:
        """Geometric position Jacobian of the EE in the world frame."""
        if self.env.backend == "isaac":
            # (N, 6, n_dof_full) → translational rows for controlled joints.
            jac = self.asset.root_physx_view.get_jacobians()[
                :, self._jacobi_body_idx, :3, self._jacobi_joint_ids
            ]
            return jac

        # mjlab: mjwarp geometric Jacobian at the current EE position.
        import mujoco_warp as mjwarp
        import warp as wp

        sim = self.env.sim._sim
        ee_pos_w = self.asset.data.body_link_pos_w[:, self.body_idx]
        self._point_torch[:] = ee_pos_w
        with wp.ScopedDevice(sim.wp_device):
            mjwarp.jac(
                sim.wp_model,
                sim.wp_data,
                self._jacp_wp,
                self._jacr_wp,
                self._point_wp,
                self._body_wp,
            )
        return self._jacp_torch[:, :, self._mj_joint_dof_ids]


__all__ = ["EndEffectorPose"]
