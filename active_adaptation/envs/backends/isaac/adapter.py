from typing import TYPE_CHECKING

import torch
import trimesh
from typing_extensions import override

from active_adaptation.envs.adapters import SimAdapter, SceneAdapter, CameraFrustumHandle
from active_adaptation.envs.backends.isaac.viewer import IsaacViserViewer


if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationContext

class IsaacDebugDraw:
    def __init__(self):
        from isaacsim.util.debug_draw import _debug_draw
        self._draw = _debug_draw.acquire_debug_draw_interface()
    
    def clear(self):
        self._draw.clear_lines()
        self._draw.clear_points()

    def plot(self, x: torch.Tensor, size=2.0, color=(1., 1., 1., 1.)):
        if not (x.ndim == 2) and (x.shape[1] == 3):
            raise ValueError("x must be a tensor of shape (N, 3).")
        x = x.cpu()
        point_list_0 = x[:-1].tolist()
        point_list_1 = x[1:].tolist()
        sizes = [size] * len(point_list_0)
        colors = [color] * len(point_list_0)
        self._draw.draw_lines(point_list_0, point_list_1, colors, sizes)
        
    def vector(self, x: torch.Tensor, v: torch.Tensor, size=2.0, color=(0., 1., 1., 1.)):
        x = x.cpu().reshape(-1, 3)
        v = v.cpu().reshape(-1, 3)
        if not (x.shape == v.shape):
            raise ValueError("x and v must have the same shape, got {} and {}.".format(x.shape, v.shape))
        point_list_0 = x.tolist()
        point_list_1 = (x + v).tolist()
        sizes = [size] * len(point_list_0)
        colors = [color] * len(point_list_0)
        self._draw.draw_lines(point_list_0, point_list_1, colors, sizes)
    
    def point(self, x: torch.Tensor, color=(1., 0., 0., 1.), size=10.0):
        point_list = x.cpu().reshape(-1, 3).tolist()
        sizes = [size] * len(point_list)
        colors = [color] * len(point_list)
        self._draw.draw_points(point_list, colors, sizes)


class IsaacSimAdapter(SimAdapter):
    def __init__(self, sim: "SimulationContext", viser_viewer: IsaacViserViewer = None):
        self._sim = sim
        self._viser_viewer = viser_viewer
        self._kit_rendered_this_step = False

    def get_physics_dt(self) -> float:
        return self._sim.get_physics_dt()

    def has_gui(self) -> bool:
        # True for Omniverse Kit GUI *or* browser Viser (debug callbacks / mesh sync).
        return self._sim.has_gui() or self._viser_viewer is not None

    def step(self) -> None:
        self._kit_rendered_this_step = False
        self._sim.step(render=False)

    def render_sensors(self) -> None:
        # Isaac Lab camera products require a Kit render pass.
        self._sim.render()
        self._kit_rendered_this_step = True

    def render_gui(self) -> None:
        # Kit couples viewport and sensor render; skip a second Kit pass if
        # render_sensors() already ran this physics step.
        if not self._kit_rendered_this_step and self._sim.has_gui():
            self._sim.render()
            self._kit_rendered_this_step = True
        if self._viser_viewer is not None:
            self._viser_viewer.update()

    def render(self) -> None:
        self.render_gui()

    def set_camera_view(self, eye=None, target=None, **kwargs) -> None:
        if eye is not None and target is not None:
            self._sim.set_camera_view(eye=eye, target=target)

    def __getattr__(self, name):
        return getattr(self._sim, name)


class IsaacSceneAdapter(SceneAdapter):
    def __init__(
        self, scene: "InteractiveScene",
        viser_viewer: IsaacViserViewer = None,
        debug_draw: IsaacDebugDraw = None,
    ):
        self._scene: "InteractiveScene" = scene
        self._viser_viewer = viser_viewer
        self._debug_draw = debug_draw

    @override
    def zero_external_wrenches(self) -> None:
        for asset in self._scene.articulations.values():
            if hasattr(asset, "instantaneous_wrench_composer"):
                asset.instantaneous_wrench_composer.reset()
            if hasattr(asset, "permanent_wrench_composer"):
                asset.permanent_wrench_composer.reset()
            if hasattr(asset, "_external_force_b") and hasattr(asset, "_external_torque_b"):
                asset._external_force_b.zero_()
                asset._external_torque_b.zero_()
                asset.has_external_wrench = False

    @override
    def create_camera_frustum(
        self,
        name: str,
        *,
        fov_y: float,
        aspect: float,
        scale: float = 0.15,
    ) -> CameraFrustumHandle:
        if self._viser_viewer is None:
            raise RuntimeError("`create_camera_frustum` requires a Viser viewer.")
        handle = self._viser_viewer.register_camera(
            name,
            fov_y=fov_y,
            aspect=aspect,
            scale=scale,
        )
        return CameraFrustumHandle(handle)

    @property
    def ground_mesh(self):
        """Warp ground mesh for the Isaac ground plane or mesh.

        This mirrors the logic previously implemented at the environment
        level, but keeps the backend-specific USD and warp handling inside
        the Isaac scene adapter.
        """
        if hasattr(self, "_ground_mesh"):
            return self._ground_mesh

        # Local imports to avoid making IsaacLab a hard dependency when other
        # backends are used.
        import numpy as np
        import warp as wp
        from isaaclab.utils.warp import convert_to_warp_mesh
        from isaaclab.terrains.trimesh.utils import make_plane
        from pxr import UsdGeom
        import isaaclab.sim as sim_utils

        mesh_prim_path = "/World/ground"
        device = wp.get_device(str(self._scene.device))

        # Check if there is a PhysX plane; otherwise fall back to a mesh prim.
        mesh_prim = sim_utils.get_first_matching_child_prim(
            mesh_prim_path, lambda prim: prim.GetTypeName() == "Plane"
        )
        if mesh_prim is None:
            mesh_prim = sim_utils.get_first_matching_child_prim(
                mesh_prim_path, lambda prim: prim.GetTypeName() == "Mesh"
            )
            if mesh_prim is None or not mesh_prim.IsValid():
                raise RuntimeError(f"Invalid mesh prim path: {mesh_prim_path}")
            mesh_prim = UsdGeom.Mesh(mesh_prim)
            points = np.asarray(mesh_prim.GetPointsAttr().Get())
            indices = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get())
            wp_mesh = convert_to_warp_mesh(points, indices, device=device)
        else:
            mesh = make_plane(size=(2e6, 2e6), height=0.0, center_zero=True)
            wp_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=device)

        self._ground_mesh = wp_mesh
        return self._ground_mesh

    @property
    def articulations(self):
        return self._scene.articulations

    @property
    def rigid_objects(self):
        return self._scene.rigid_objects
    
    @property
    def entities(self):
        return {**self._scene.articulations, **self._scene.rigid_objects}

    def get_visual_meshes(self, name: str) -> list[trimesh.Trimesh]:
        """Body-local visual trimeshes for ``entities[name]`` (``body_names`` order).

        Uses ``{body}/visuals`` via ``simple_raycaster.utils_usd`` (same layout as
        raycasting / Viser). Multiply by ``body_link_pose_w`` at runtime.
        """
        from active_adaptation.envs.backends.isaac.meshes import load_entity_body_meshes

        entity = self.entities[name]
        return load_entity_body_meshes(
            entity, suffixes=("visuals",), require_all=True
        )

    def get_collision_meshes(self, name: str) -> list[trimesh.Trimesh]:
        """Body-local collision trimeshes for ``entities[name]`` (``body_names`` order).

        Tries ``{body}/collisions`` then ``{body}/collision``. Bodies without a
        collision prim get an empty trimesh so the list stays aligned with
        ``num_bodies`` / ``body_link_pose_w``.
        """
        from active_adaptation.envs.backends.isaac.meshes import load_entity_body_meshes

        entity = self.entities[name]
        return load_entity_body_meshes(
            entity,
            suffixes=("collisions", "collision"),
            require_all=False,
        )

    def __getattr__(self, name):
        return getattr(self._scene, name)
    
    def __getitem__(self, key):
        return self._scene[key]

    def create_sphere_marker(
        self,
        prim_path: str,
        color: tuple[float, float, float],
        radius: float = 0.05,
    ):
        """Create an Isaac Lab VisualizationMarkers with a single sphere (for GUI debug).

        Returns a VisualizationMarkers instance. Call .set_visibility(True) and
        .visualize(positions_tensor) to use it.
        """
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        import isaaclab.sim as sim_utils

        color = tuple(map(float, color))
        marker = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path=prim_path,
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=radius,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                    ),
                },
            )
        )
        marker.set_visibility(True)
        return marker
    
    def create_arrow_marker(
        self,
        prim_path: str,
        color: tuple[float, float, float] = (1.0, 0.0, 0.0),
        scale: tuple[float, float, float] = (1.0, 0.1, 0.1),
    ):
        """Create an Isaac Lab VisualizationMarkers with a single arrow (for GUI debug).

        Returns a VisualizationMarkers instance. Call .set_visibility(True) and
        .visualize(positions_tensor) to use it.
        """
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg, ISAAC_NUCLEUS_DIR
        import isaaclab.sim as sim_utils
        marker = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path=prim_path,
                markers={
                    "arrow": sim_utils.UsdFileCfg(
                        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                        scale=scale,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                    )
                },
            )
        )
        marker.set_visibility(True)
        return marker

    def create_frame_marker(
        self,
        prim_path: str,
        scale: tuple[float, float, float] = (0.1, 0.1, 0.1),
    ):
        """Create an Isaac Lab VisualizationMarkers with a single frame (for GUI debug).

        Returns a VisualizationMarkers instance. Call
        ``.visualize(translations=..., orientations=...)`` (WXYZ) to update poses.
        """
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg, ISAAC_NUCLEUS_DIR
        import isaaclab.sim as sim_utils
        marker = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path=prim_path,
                markers={
                    "frame": sim_utils.UsdFileCfg(
                        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                        scale=scale,
                    )
                },
            )
        )
        marker.set_visibility(True)
        return marker
    
    def clear_debug(self) -> None:
        if self._debug_draw is not None:
            self._debug_draw.clear()
        if self._viser_viewer is not None:
            self._viser_viewer.clear()

    def draw_vector(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        size: float = 2.0,
        color: tuple[float, ...] = (0.0, 1.0, 1.0, 1.0),
    ):
        if self._debug_draw is not None:
            self._debug_draw.vector(x, v, size, color)
        if self._viser_viewer is not None:
            self._viser_viewer.vector(x, v, size, color)

    def draw_point(
        self,
        x: torch.Tensor,
        color: tuple[float, ...] = (1.0, 0.0, 0.0, 1.0),
        size: float = 10.0,
    ):
        if self._debug_draw is not None:
            self._debug_draw.point(x, color=color, size=size)
        if self._viser_viewer is not None:
            self._viser_viewer.point(x, color=color, size=size)

    def draw_plot(
        self,
        x: torch.Tensor,
        size: float = 2.0,
        color: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0),
    ):
        if self._debug_draw is not None:
            self._debug_draw.plot(x, size=size, color=color)
        if self._viser_viewer is not None:
            self._viser_viewer.plot(x, size=size, color=color)

    @override
    def sample_spawn_origin_candidates(
        self, env_ids: torch.Tensor
    ) -> torch.Tensor:
        if self._scene.terrain.terrain_origins is None:
            return self.env_origins[env_ids]

        terrain_origins = self._scene.terrain.terrain_origins.reshape(-1, 3)
        idx = torch.randint(
            0,
            terrain_origins.shape[0],
            (len(env_ids),),
            device=env_ids.device,
        )
        return terrain_origins[idx]

    def reset_to(self, state: dict, env_ids: torch.Tensor):
        self._scene.reset_to(state, env_ids=env_ids)
    
    def get_state(self, env_ids: torch.Tensor) -> dict:
        return self._scene.get_state(env_ids=env_ids)


__all__ = [
    "IsaacSimAdapter",
    "IsaacSceneAdapter",
]
