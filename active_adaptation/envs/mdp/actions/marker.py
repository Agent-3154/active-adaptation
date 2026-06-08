from __future__ import annotations

import torch

from typing import TYPE_CHECKING, Tuple
from typing_extensions import override

from active_adaptation.utils.math import quat_rotate

from .base import ActionV2


if TYPE_CHECKING:
    from active_adaptation.envs.env_base import _EnvBase


class Marker(ActionV2):
    """
    This is a marker action that visualizes a set of markers in the world frame.
    It does not have a fixed action dimension, and the action is the position of the markers in the world frame.
    """

    def __init__(
        self,
        body_frame: bool = False,
        color: Tuple[float, float, float] = (0.0, 1.0, 0.0),
        radius: float = 0.05,
    ):
        super().__init__()
        self.body_frame = body_frame
        self.color = tuple(color)
        self.radius = radius

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.has_gui = self.env.sim.has_gui()
        self.action_dim = 3  # not actually limited to 3

        if self.has_gui and self.env.backend == "isaac":
            from isaaclab.markers import (
                VisualizationMarkers,
                VisualizationMarkersCfg,
                sim_utils,
            )
            # unique prim path per Marker instance (multiple envs / actions may coexist)
            name = f"marker_{id(self):x}"
            self.marker = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path=f"/Visuals/Input/{name}",
                    markers={
                        "marker": sim_utils.SphereCfg(
                            radius=self.radius,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=self.color
                            ),
                        ),
                    },
                )
            )
            self.marker.set_visibility(True)

    @override
    def process_action(self, action: torch.Tensor):
        if not self.has_gui or action is None:
            return

        assert action.shape[-1] == 3

        if self.body_frame:
            root_pos_w = self.asset.data.root_link_pos_w.reshape(self.num_envs, 1, 3)
            root_quat_w = self.asset.data.root_link_quat_w.reshape(self.num_envs, 1, 4)
            marker_pos = action.reshape(self.num_envs, -1, 3)
            translations = root_pos_w + quat_rotate(root_quat_w, marker_pos)
        else:
            translations = action.reshape(self.num_envs, -1, 3)
            translations += self.env.scene.env_origins.unsqueeze(1)
        translations = translations.reshape(-1, 3)
        self.marker.visualize(
            translations=translations,
            scales=torch.ones(3, device=self.device).expand_as(translations),
        )

    @override
    def apply_action(self, substep: int):
        pass


__all__ = ["Marker"]
