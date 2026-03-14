from __future__ import annotations

import torch

from typing_extensions import override

from active_adaptation.utils.math import quat_rotate

from .base import Action


class Marker(Action):
    def __init__(self, env, num_markers: int = 1, body_frame: bool = False):
        super().__init__(env)
        self.asset = self.env.scene.articulations["robot"]
        self.num_markers = num_markers
        self.body_frame = body_frame
        self.has_gui = self.env.sim.has_gui()
        self.action_dim = 3 * self.num_markers

        if self.has_gui and self.env.backend == "isaac":
            from isaaclab.markers import (
                VisualizationMarkers,
                VisualizationMarkersCfg,
                sim_utils,
            )

            self.marker = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/Input/Marker",
                    markers={
                        "marker": sim_utils.SphereCfg(
                            radius=0.05,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.0, 1.0, 0.0)
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

        if self.body_frame:
            pos = self.asset.data.root_link_pos_w.reshape(self.num_envs, 1, 3)
            quat = self.asset.data.root_link_quat_w.reshape(self.num_envs, 1, 4)
            translations = pos + quat_rotate(
                quat, action.reshape(self.num_envs, self.num_markers, 3)
            )
        else:
            translations = action.reshape(self.num_envs, self.num_markers, 3)
            translations += self.env.scene.env_origins.unsqueeze(1)
        translations = translations.reshape(self.num_envs * self.num_markers, 3)
        self.marker.visualize(
            translations=translations,
            scales=torch.ones(3, device=self.device).expand_as(translations),
        )

    @override
    def apply_action(self, substep: int):
        pass


__all__ = ["Marker"]
