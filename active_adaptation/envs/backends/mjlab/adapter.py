from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from typing_extensions import override

from active_adaptation.envs.adapters import SimAdapter, SceneAdapter

if TYPE_CHECKING:
    from active_adaptation.envs.backends.mjlab.viewer import MjLabViewer
    from mjlab.scene import Scene
    from mjlab.sim import Simulation
    from mjlab.viewer.offscreen_renderer import OffscreenRenderer
    from mjlab.viewer.viewer_config import ViewerConfig


class MjlabSimAdapter(SimAdapter):
    def __init__(
        self,
        sim: "Simulation",
        viewer: "MjLabViewer" = None,
        viewer_cfg: "ViewerConfig" = None,
        scene: "Scene" = None,
    ):
        self._sim = sim
        self.viewer = viewer
        self._viewer_cfg = viewer_cfg
        self._scene = scene
        self._offscreen_renderer: "OffscreenRenderer | None" = None

    def get_physics_dt(self) -> float:
        return self._sim.cfg.mujoco.timestep

    def has_gui(self) -> bool:
        return self.viewer is not None

    def step(self, render: bool = False) -> None:
        self._sim.step()

    def render(self) -> None:
        if self.viewer is not None:
            self.viewer.update()

    def render_rgb_array(self) -> np.ndarray:
        renderer = self._get_offscreen_renderer()
        renderer.update(self._sim.data)
        return renderer.render()

    def set_camera_view(self, eye=None, target=None, **kwargs) -> None:
        if eye is None or target is None or self._viewer_cfg is None:
            return

        eye = np.asarray(eye, dtype=float)
        target = np.asarray(target, dtype=float)
        delta = eye - target
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-8:
            return

        planar = math.hypot(float(delta[0]), float(delta[1]))
        self._viewer_cfg.lookat = tuple(float(v) for v in target.tolist())
        self._viewer_cfg.distance = distance
        self._viewer_cfg.azimuth = math.degrees(math.atan2(delta[1], delta[0]))
        self._viewer_cfg.elevation = -math.degrees(math.atan2(delta[2], planar))

        if self._offscreen_renderer is not None:
            self._offscreen_renderer.close()
            self._offscreen_renderer = None

    def close(self) -> None:
        if self._offscreen_renderer is not None:
            self._offscreen_renderer.close()
            self._offscreen_renderer = None

    def _get_offscreen_renderer(self) -> "OffscreenRenderer":
        if self._offscreen_renderer is None:
            if self._viewer_cfg is None or self._scene is None:
                raise ValueError("MjLab offscreen renderer is not configured.")

            from mjlab.viewer.offscreen_renderer import OffscreenRenderer

            renderer = OffscreenRenderer(
                model=self._sim.mj_model,
                cfg=self._viewer_cfg,
                scene=self._scene,
            )
            renderer.initialize()
            self._offscreen_renderer = renderer
        return self._offscreen_renderer

    def __getattr__(self, name):
        return getattr(self._sim, name)


class MjlabSceneAdapter(SceneAdapter):
    def __init__(self, scene: Scene, sim: Simulation):
        self._scene = scene
        self._sim = sim

    @override
    def zero_external_wrenches(self) -> None:
        for asset in self._scene.entities.values():
            asset.data.data.xfrc_applied.zero_()

    @property
    def articulations(self):
        return self._scene.entities

    def __getattr__(self, name):
        return getattr(self._scene, name)

    @property
    def ground_mesh(self):
        """Warp mesh for the mjlab terrain body (name ``terrain``), for ray height queries."""
        if hasattr(self, "_ground_mesh"):
            return self._ground_mesh

        if self._scene.terrain is None:
            self._ground_mesh = None
            return self._ground_mesh

        import mujoco
        import numpy as np
        import trimesh
        import warp as wp
        from mujoco import mjtGeom
        from mjviser.conversions import create_primitive_mesh, mujoco_mesh_to_trimesh

        mj_model = self._sim.mj_model
        terrain_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "terrain")
        if terrain_bid < 0:
            self._ground_mesh = None
            return self._ground_mesh

        mj_data = mujoco.MjData(mj_model)
        mujoco.mj_forward(mj_model, mj_data)

        parts: list[trimesh.Trimesh] = []
        for gid in range(mj_model.ngeom):
            if mj_model.geom_bodyid[gid] != terrain_bid:
                continue
            gt = int(mj_model.geom_type[gid])
            if gt == int(mjtGeom.mjGEOM_MESH):
                mesh = mujoco_mesh_to_trimesh(mj_model, gid)
            else:
                mesh = create_primitive_mesh(mj_model, gid)
            if mesh is None and gt == int(mjtGeom.mjGEOM_PLANE):
                mesh = trimesh.creation.box(extents=[400.0, 400.0, 0.1])
                mesh.apply_translation([0.0, 0.0, -0.05])
            if mesh is None:
                continue
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = mj_data.geom_xmat[gid].reshape(3, 3)
            T[:3, 3] = mj_data.geom_xpos[gid]
            mesh.apply_transform(T)
            parts.append(mesh)

        if not parts:
            self._ground_mesh = None
            return self._ground_mesh

        combined = parts[0] if len(parts) == 1 else trimesh.util.concatenate(parts)
        device = wp.get_device(str(self._scene.device))
        self._ground_mesh = wp.Mesh(
            points=wp.array(
                np.asarray(combined.vertices, dtype=np.float32),
                dtype=wp.vec3,
                device=device,
            ),
            indices=wp.array(
                np.asarray(combined.faces, dtype=np.int32).flatten(),
                dtype=wp.int32,
                device=device,
            ),
        )
        return self._ground_mesh


__all__ = [
    "MjlabSimAdapter",
    "MjlabSceneAdapter",
]
