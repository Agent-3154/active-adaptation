from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

from typing_extensions import override

from active_adaptation.envs.adapters import SimAdapter, SceneAdapter

if TYPE_CHECKING:
    from active_adaptation.envs.backends.mjlab.viewer import MjLabViewer
    from mjlab.scene import Scene
    from mjlab.sim import Simulation
    from mjlab.viewer.offscreen_renderer import OffscreenRenderer
    from mjlab.viewer.viewer_config import ViewerConfig


class _NoopSphereMarker:
    """Fallback marker for headless mode or unavailable viewer."""

    def set_visibility(self, visible: bool) -> None:  # noqa: ARG002
        return

    def visualize(self, translations=None, positions=None) -> None:  # noqa: ANN001,ARG002
        return


class _MjlabSphereMarker:
    """Lightweight marker wrapper with an Isaac-like visualize API."""

    def __init__(
        self,
        viewer: "MjLabViewer",
        name: str,
        color: tuple[float, float, float],
        radius: float,
    ) -> None:
        self._viewer = viewer
        self._name = name
        self._radius = float(radius)
        rgb = np.asarray(color, dtype=np.float32).clip(0.0, 1.0)
        self._color = (rgb * 255.0).astype(np.uint8)
        self._opacity = 1.0
        self._visible = True
        self._handle = None
        self._mesh = None

    def set_visibility(self, visible: bool) -> None:
        self._visible = bool(visible)
        if self._handle is not None:
            self._handle.visible = self._visible

    def visualize(self, translations=None, positions=None) -> None:  # noqa: ANN001
        points = translations if translations is not None else positions
        if points is None:
            return

        points_np = self._to_numpy_points(points)
        if points_np.shape[0] == 0:
            if self._handle is not None:
                self._handle.visible = False
            return

        self._sync_handle(points_np)
        if self._handle is not None:
            self._handle.visible = self._visible

    def _to_numpy_points(self, points: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        return points

    def _sync_handle(self, points: np.ndarray) -> None:
        count = points.shape[0]
        if self._handle is None or len(self._handle.batched_positions) != count:
            self._recreate_handle(points)
            return

        self._handle.batched_positions = points

    def _recreate_handle(self, points: np.ndarray) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

        if self._mesh is None:
            import trimesh

            self._mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)

        count = points.shape[0]
        self._handle = self._viewer._server.scene.add_batched_meshes_simple(
            self._name,
            self._mesh.vertices,
            self._mesh.faces,
            batched_wxyzs=np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (count, 1)),
            batched_positions=points,
            batched_scales=np.full((count,), self._radius, dtype=np.float32),
            batched_colors=np.tile(self._color, (count, 1)),
            opacity=self._opacity,
            cast_shadow=False,
            receive_shadow=False,
            visible=self._visible,
        )


class _MjlabFrameMarker:
    """Frame marker wrapper backed by Viser batched axes."""

    def __init__(
        self,
        viewer: "MjLabViewer",
        name: str,
        scale: tuple[float, float, float],
    ) -> None:
        self._viewer = viewer
        self._name = name
        self._default_scale = np.asarray(scale, dtype=np.float32)
        self._visible = True
        self._handle = None

    def set_visibility(self, visible: bool) -> None:
        self._visible = bool(visible)
        if self._handle is not None:
            self._handle.visible = self._visible

    def visualize(
        self,
        translations=None,
        orientations=None,
        scales=None,
        positions=None,
    ) -> None:  # noqa: ANN001
        points = translations if translations is not None else positions
        if points is None:
            return

        points_np = self._to_numpy(points).reshape(-1, 3)
        if points_np.shape[0] == 0:
            if self._handle is not None:
                self._handle.visible = False
            return

        count = points_np.shape[0]
        rots_np = self._as_orientations(orientations, count)
        scales_np = self._as_scales(scales, count)
        self._sync_handle(points_np, rots_np, scales_np)
        if self._handle is not None:
            self._handle.visible = self._visible

    def _to_numpy(self, data: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        return np.asarray(data, dtype=np.float32)

    def _as_orientations(self, orientations, count: int) -> np.ndarray:  # noqa: ANN001
        if orientations is None:
            return np.tile(
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                (count, 1),
            )

        rots = self._to_numpy(orientations).reshape(-1, 4)
        if rots.shape[0] == 1 and count > 1:
            rots = np.tile(rots, (count, 1))
        if rots.shape[0] != count:
            raise ValueError(
                f"Frame marker orientations count ({rots.shape[0]}) does not match positions count ({count})."
            )
        return rots

    def _as_scales(self, scales, count: int) -> np.ndarray:  # noqa: ANN001
        if scales is None:
            return np.tile(self._default_scale, (count, 1))

        scales_np = self._to_numpy(scales)
        if scales_np.ndim == 1 and scales_np.shape[0] == 3:
            return np.tile(scales_np, (count, 1))
        if scales_np.ndim == 1 and scales_np.shape[0] == count:
            return scales_np
        if scales_np.ndim == 2 and scales_np.shape == (count, 3):
            return scales_np
        raise ValueError(
            f"Invalid frame marker scales shape {scales_np.shape}; expected (3,), ({count},), or ({count}, 3)."
        )

    def _sync_handle(
        self,
        positions: np.ndarray,
        wxyzs: np.ndarray,
        scales: np.ndarray,
    ) -> None:
        count = positions.shape[0]
        if self._handle is None or len(self._handle.batched_positions) != count:
            self._recreate_handle(positions, wxyzs, scales)
            return

        self._handle.batched_positions = positions
        self._handle.batched_wxyzs = wxyzs
        self._handle.batched_scales = scales

    def _recreate_handle(
        self,
        positions: np.ndarray,
        wxyzs: np.ndarray,
        scales: np.ndarray,
    ) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

        self._handle = self._viewer._server.scene.add_batched_axes(
            self._name,
            batched_wxyzs=wxyzs,
            batched_positions=positions,
            batched_scales=scales,
            visible=self._visible,
        )


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
    def __init__(self, scene: Scene, sim: Simulation, viewer: "MjLabViewer" = None):
        self._scene = scene
        self._sim = sim
        self._viewer = viewer

    @override
    def zero_external_wrenches(self) -> None:
        for asset in self._scene.entities.values():
            asset.data.data.xfrc_applied.zero_()

    @property
    def articulations(self):
        return self._scene.entities

    def __getattr__(self, name):
        return getattr(self._scene, name)
    
    def __getitem__(self, key):
        return self._scene.entities[key]

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
    
    def create_sphere_marker(
        self,
        name: str,
        color: tuple[float, float, float],
        radius: float = 0.05,
    ):
        if self._viewer is None:
            return _NoopSphereMarker()
        return _MjlabSphereMarker(self._viewer, name=name, color=color, radius=radius)
    
    def create_frame_marker(
        self,
        name: str,
        scale: tuple[float, float, float] = (0.5, 0.5, 0.5),
    ):
        if self._viewer is None:
            return _NoopSphereMarker()
        return _MjlabFrameMarker(self._viewer, name=name, scale=scale)


__all__ = [
    "MjlabSimAdapter",
    "MjlabSceneAdapter",
]
