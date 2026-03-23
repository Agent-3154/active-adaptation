from __future__ import annotations

import numpy as np
import torch
import warp as wp

from active_adaptation.utils.warp import raycast_mesh


class GroundQuery:
    def __init__(self, terrain_type: str | None, device: torch.device, mesh: wp.Mesh | None = None):
        self.terrain_type = terrain_type
        self.device = device
        self.mesh = mesh

    def height_at(self, pos: torch.Tensor) -> torch.Tensor:
        if self.terrain_type == "plane":
            return torch.zeros(pos.shape[:-1], device=pos.device, dtype=pos.dtype)

        bshape = pos.shape[:-1]
        ray_starts = pos.reshape(-1, 3)
        ray_directions = torch.tensor(
            [0.0, 0.0, -1.0], device=ray_starts.device, dtype=ray_starts.dtype
        ).expand(ray_starts.shape[0], 3)
        _, ray_distances = raycast_mesh(
            ray_starts=ray_starts,
            ray_directions=ray_directions,
            min_dist=0.0,
            max_dist=100.0,
            mesh=self.mesh,
        )
        ray_distance = ray_distances.reshape(-1).nan_to_num(posinf=100.0)
        return (ray_starts[:, 2] - ray_distance).reshape(*bshape)
