from typing import TYPE_CHECKING

import torch
from typing_extensions import override

from active_adaptation.envs.adapters import SimAdapter, SceneAdapter

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationContext


class IsaacSimAdapter(SimAdapter):
    def __init__(self, sim: "SimulationContext"):
        self._sim = sim

    def get_physics_dt(self) -> float:
        return self._sim.get_physics_dt()

    def has_gui(self) -> bool:
        return self._sim.has_gui()

    def step(self, render: bool = False) -> None:
        self._sim.step(render=render)

    def render(self) -> None:
        self._sim.render()

    def set_camera_view(self, eye=None, target=None, **kwargs) -> None:
        if eye is not None and target is not None:
            self._sim.set_camera_view(eye=eye, target=target)

    def __getattr__(self, name):
        return getattr(self._sim, name)


class IsaacSceneAdapter(SceneAdapter):
    def __init__(self, scene: "InteractiveScene"):
        self._scene: "InteractiveScene" = scene

    @override
    def zero_external_wrenches(self) -> None:
        for asset in self._scene.articulations.values():
            if hasattr(asset, "instantaneous_wrench_composer"):
                asset.instantaneous_wrench_composer.reset()
            if hasattr(asset, "permanent_wrench_composer"):
                asset.permanent_wrench_composer.reset()
            if getattr(asset, "has_external_wrench", False):
                asset._external_force_b.zero_()
                asset._external_torque_b.zero_()
                asset.has_external_wrench = False

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
        from isaaclab.utils.warp import convert_to_warp_mesh
        from isaaclab.terrains.trimesh.utils import make_plane
        from pxr import UsdGeom
        import isaaclab.sim as sim_utils

        mesh_prim_path = "/World/ground"
        device = (
            self._scene.device.type
            if hasattr(self._scene, "device") and hasattr(self._scene.device, "type")
            else "cuda"
        )

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

    @override
    def get_spawn_origins(self, env_ids: torch.Tensor) -> torch.Tensor:
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


__all__ = [
    "IsaacSimAdapter",
    "IsaacSceneAdapter",
]
