from __future__ import annotations

import logging

import numpy as np
import torch
from typing import TYPE_CHECKING, Optional, Sequence, Tuple
from typing_extensions import override

from active_adaptation.envs.mdp.randomizations.base import RandomizationV2
from active_adaptation.envs.mdp.randomizations.common import (
    NestedRangeType,
    sample_uniform,
)
from active_adaptation.envs.utils.quat_layout import isaaclab_uses_xyzw


if TYPE_CHECKING:
    from active_adaptation.envs.env_base import _EnvBase


def _set_physx_material_properties(view, materials_cpu: torch.Tensor, env_ids: torch.Tensor) -> None:
    """Write rigid materials. Lab 3: unflattened ``wp`` arrays. Lab 2: flattened torch."""
    if isaaclab_uses_xyzw():
        import warp as wp

        view.set_material_properties(
            wp.from_torch(materials_cpu, dtype=wp.float32),
            wp.from_torch(env_ids, dtype=wp.int32),
        )
        return
    view.set_material_properties(materials_cpu.flatten(), env_ids)


class randomize_materials_isaac(RandomizationV2):

    supported_backends = ("isaac",)

    def __init__(
        self,
        body_names: str,
        static_friction_range: Optional[NestedRangeType] = None,
        dynamic_friction_range: Optional[NestedRangeType] = None,
        restitution_range: Optional[NestedRangeType] = None,
        homogeneous: bool = True,
    ):
        super().__init__()
        self.body_names = body_names
        self.static_friction_range = static_friction_range
        self.dynamic_friction_range = dynamic_friction_range
        self.restitution_range = restitution_range
        self.homogeneous = homogeneous

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.asset = self.env.scene.articulations["robot"]
        self.body_ids, self.body_names = self.asset.find_bodies(self.body_names)

        num_shapes_per_body = [0,]
        from active_adaptation.envs.backends.isaac.canonical_asset import isaac_root_view

        view = isaac_root_view(self.asset)
        for link_path in view.link_paths[0]:
            link_physx_view = self.asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore
            num_shapes_per_body.append(link_physx_view.max_shapes)
        cumsum = np.cumsum(num_shapes_per_body)
        self.shape_ids = torch.cat(
            [torch.arange(cumsum[i], cumsum[i + 1]) for i in self.body_ids]
        )
        self.num_buckets = 64
        if self.static_friction_range is not None:
            self.static_friction_buckets = torch.linspace(
                *self.static_friction_range, self.num_buckets
            )
        if self.dynamic_friction_range is not None:
            self.dynamic_friction_buckets = torch.linspace(
                *self.dynamic_friction_range, self.num_buckets
            )
        if self.restitution_range is not None:
            self.restitution_buckets = torch.linspace(
                *self.restitution_range, self.num_buckets
            )

    @override
    def startup(self):
        from active_adaptation.envs.backends.isaac.canonical_asset import (
            as_torch,
            isaac_root_view,
        )

        view = isaac_root_view(self.asset)
        materials = as_torch(view.get_material_properties()).clone()
        if self.homogeneous:
            shape = (self.num_envs, 1)
        else:
            shape = (self.num_envs, len(self.shape_ids))
        device = materials.device
        shape_ids = self.shape_ids.to(device)
        if self.static_friction_range is not None:
            materials[:, shape_ids, 0] = self.static_friction_buckets.to(device)[
                torch.randint(0, self.num_buckets, shape, device=device)
            ]
        if self.dynamic_friction_range is not None:
            materials[:, shape_ids, 1] = self.dynamic_friction_buckets.to(device)[
                torch.randint(0, self.num_buckets, shape, device=device)
            ]
        if self.restitution_range is not None:
            materials[:, shape_ids, 2] = self.restitution_buckets.to(device)[
                torch.randint(0, self.num_buckets, shape, device=device)
            ]

        # Lab 3 PhysX Warp frontend wants unflattened wp arrays + CPU int32
        # env ids. Lab 2 takes a flattened torch buffer.
        env_ids = torch.arange(self.asset.num_instances, device="cpu", dtype=torch.int32)
        materials_cpu = materials.cpu().contiguous()
        _set_physx_material_properties(view, materials_cpu, env_ids)
        self.asset.data.body_materials = materials.to(self.device)


class randomize_materials_mjlab(RandomizationV2):
    """Randomize MuJoCo ``geom_friction`` for geoms attached to selected bodies.

    ``geom_friction`` axes are ``(slide, spin, roll)`` — not Isaac static/dynamic/
    restitution. Sliding (axis 0) is the tangential µ used for ``condim >= 3``.
    Spin/roll only affect contacts with ``condim >= 4`` / ``6``.

    Declares ``mj_fields = ("geom_friction",)`` so per-env Warp arrays are expanded.
    No ``set_const`` recompute is required.
    """

    supported_backends = ("mjlab",)

    mj_fields = ("geom_friction",)

    def __init__(
        self,
        body_names: str | Sequence[str],
        sliding_friction_range: Optional[Tuple[float, float]] = None,
        torsional_friction_range: Optional[Tuple[float, float]] = None,
        rolling_friction_range: Optional[Tuple[float, float]] = None,
        homogeneous: bool = True,
    ):
        super().__init__()
        self.body_names_expr = body_names
        self.sliding_friction_range = tuple(map(float, sliding_friction_range))
        self.torsional_friction_range = tuple(map(float, torsional_friction_range))
        self.rolling_friction_range = tuple(map(float, rolling_friction_range))
        self.homogeneous = bool(homogeneous)
        if (
            self.sliding_friction_range is None
            and self.torsional_friction_range is None
            and self.rolling_friction_range is None
        ):
            raise ValueError(
                "randomize_materials_mjlab requires at least one of "
                "sliding_friction_range, torsional_friction_range, rolling_friction_range."
            )

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.asset = self.env.scene.entities["robot"]
        self.body_ids, self.body_names = self.asset.find_bodies(self.body_names_expr)
        if len(self.body_ids) == 0:
            raise ValueError(
                f"No bodies matched {self.body_names_expr!r} for material randomization."
            )

        local_body_ids = torch.as_tensor(
            self.body_ids, device=self.device, dtype=torch.long
        )
        global_body_ids = self.asset.indexing.body_ids[local_body_ids]
        selected_body_set = set(global_body_ids.cpu().tolist())

        geom_global_ids = self.asset.indexing.geom_ids.cpu().tolist()
        geom_names = self.asset.geom_names
        selected_geom_global: list[int] = []
        selected_geom_names: list[str] = []

        cpu_model = self.env.sim.mj_model
        for local_idx, global_idx in enumerate(geom_global_ids):
            body_id = int(cpu_model.geom_bodyid[global_idx])
            if body_id in selected_body_set:
                selected_geom_global.append(int(global_idx))
                selected_geom_names.append(geom_names[local_idx])

        if not selected_geom_global:
            raise ValueError(
                f"No geoms found under bodies {self.body_names} for material randomization."
            )

        self.geom_global_ids = torch.as_tensor(
            selected_geom_global, device=self.device, dtype=torch.long
        )
        self.geom_names = selected_geom_names
        logging.info(
            "randomize_materials_mjlab: %d geom(s) on bodies %s → %s",
            len(self.geom_names),
            self.body_names,
            self.geom_names,
        )

    def _sample_axis(self, axis: int, friction_range: Tuple[float, float]) -> None:
        num_geoms = self.geom_global_ids.numel()
        sample_cols = 1 if self.homogeneous else num_geoms
        values = sample_uniform(
            (self.num_envs, sample_cols),
            friction_range[0],
            friction_range[1],
            device=self.device,
        )
        if sample_cols == 1:
            values = values.expand(-1, num_geoms)
        self.env.sim.model.geom_friction[:, self.geom_global_ids, axis] = values

    @override
    def startup(self):
        logging.info(
            "Randomize geom_friction for %s upon startup (homogeneous=%s).",
            self.geom_names,
            self.homogeneous,
        )
        if self.sliding_friction_range is not None:
            self._sample_axis(0, self.sliding_friction_range)
        if self.torsional_friction_range is not None:
            self._sample_axis(1, self.torsional_friction_range)
        if self.rolling_friction_range is not None:
            self._sample_axis(2, self.rolling_friction_range)
