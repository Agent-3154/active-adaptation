from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING, Optional
from typing_extensions import override

from active_adaptation.envs.mdp.randomizations.base import RandomizationV2
from active_adaptation.envs.mdp.randomizations.common import NestedRangeType


if TYPE_CHECKING:
    from active_adaptation.envs.env_base import _EnvBase


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
        for link_path in self.asset.root_physx_view.link_paths[0]:
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
        materials = self.asset.root_physx_view.get_material_properties().clone()
        if self.homogeneous:
            shape = (self.num_envs, 1)
        else:
            shape = (self.num_envs, len(self.shape_ids))
        if self.static_friction_range is not None:
            materials[:, self.shape_ids, 0] = self.static_friction_buckets[
                torch.randint(0, self.num_buckets, shape)
            ]
        if self.dynamic_friction_range is not None:
            materials[:, self.shape_ids, 1] = self.dynamic_friction_buckets[
                torch.randint(0, self.num_buckets, shape)
            ]
        if self.restitution_range is not None:
            materials[:, self.shape_ids, 2] = self.restitution_buckets[
                torch.randint(0, self.num_buckets, shape)
            ]

        indices = torch.arange(self.asset.num_instances)
        self.asset.root_physx_view.set_material_properties(materials.flatten(), indices)
        self.asset.data.body_materials = materials.to(self.device)


class randomize_materials_mjlab(RandomizationV2):
    supported_backends = ("mjlab",)

    mj_fields = ("geom_friction",)

    def __init__(
        self,
        body_names: str,
        sliding_friction_range: Optional[NestedRangeType] = None,
        torsional_friction_range: Optional[NestedRangeType] = None,
        rolling_friction_range: Optional[NestedRangeType] = None,
        homogeneous: bool = True,
    ):
        self.body_names = body_names
        self.sliding_friction_range = sliding_friction_range
        self.torsional_friction_range = torsional_friction_range
        self.rolling_friction_range = rolling_friction_range

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.asset = self.env.scene.entities["robot"]
        self.body_ids, self.body_names = self.asset.find_bodies(self.body_names)
