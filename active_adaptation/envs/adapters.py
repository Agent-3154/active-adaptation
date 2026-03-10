"""Adapter classes to provide a unified API for different simulation backends."""

from typing import Dict, Protocol, TYPE_CHECKING, Union, Mapping
from typing_extensions import override
import torch
import warp as wp
import numpy as np

if TYPE_CHECKING:
    from isaaclab.sim import SimulationContext
    from isaaclab.scene import InteractiveScene
    from isaaclab.assets import Articulation
    from mjlab.sim import Simulation
    from mjlab.scene import Scene
    from mjlab.entity import Entity


class SimAdapter(Protocol):
    """Unified interface for simulation objects across backends.

    This Protocol is used only for type checking - it has zero runtime overhead.
    """

    def get_physics_dt(self) -> float:
        """Get the physics timestep."""
        ...

    def has_gui(self) -> bool:
        """Check if GUI is available."""
        ...

    def step(self, render: bool = False) -> None:
        """Step the simulation."""
        ...

    def render(self) -> None:
        """Render the simulation."""
        ...

    def set_camera_view(self, eye=None, target=None, **kwargs) -> None:
        """Set camera view (optional, backend-specific)."""
        ...


class SceneAdapter(Protocol):
    """Unified interface for scene objects across backends.

    This Protocol is used only for type checking - it has zero runtime overhead.
    """

    _scene: Union["InteractiveScene", "Scene"]

    @property
    def num_envs(self) -> int:
        """Number of environments."""
        return self._scene.num_envs

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset environments."""
        self._scene.reset(env_ids)

    def update(self, dt: float) -> None:
        """Update scene state."""
        self._scene.update(dt)

    def write_data_to_sim(self) -> None:
        """Write data to simulation."""
        self._scene.write_data_to_sim()

    def zero_external_wrenches(self) -> None:
        """Zero external wrenches."""
        raise NotImplementedError(
            f"Zero external wrenches is not implemented for {self.__class__.__name__}."
        )
    
    def get(self, name, default=None):
        raise NotImplementedError

    @property
    def articulations(self) -> Dict[str, Union["Articulation", "Entity"]]:
        """Dictionary of articulations (robots)."""
        ...

    @property
    def sensors(self) -> dict:
        """Dictionary of sensors."""
        return self._scene.sensors

    @property
    def env_origins(self) -> torch.Tensor:
        """Origins of the environments."""
        return self._scene.env_origins

    @property
    def ground_mesh(self):
        """Warp ground mesh used for ray-based height queries.

        Backends that support ground raycasting must provide a warp-compatible
        mesh here. Backends without a concept of a shared ground can raise
        ``NotImplementedError``.
        """
        raise NotImplementedError


class IsaacSimAdapter:
    """Adapter for IsaacLab SimulationContext."""

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
        # Forward any other attributes to the underlying sim
        return getattr(self._sim, name)


class MujocoSimAdapter:
    """Adapter for MuJoCo MJSim."""

    def __init__(self, sim):
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
        # MuJoCo doesn't have set_camera_view, but we can add it if needed
        pass

    def __getattr__(self, name):
        return getattr(self._sim, name)


class MjlabSimAdapter:
    """Adapter for mjlab Simulation."""

    def __init__(self, sim: "Simulation", viewer=None):
        self._sim = sim
        self.viewer = viewer

    def get_physics_dt(self) -> float:
        return self._sim.cfg.mujoco.timestep

    def has_gui(self) -> bool:
        # mjlab doesn't have GUI support yet
        return self.viewer is not None

    def step(self, render: bool = False) -> None:
        # mjlab's step doesn't take render parameter
        self._sim.step()

    def render(self) -> None:
        # mjlab doesn't have render method yet
        pass

    def set_camera_view(self, eye=None, target=None, **kwargs) -> None:
        # mjlab doesn't have set_camera_view
        pass

    def __getattr__(self, name):
        return getattr(self._sim, name)


class IsaacSceneAdapter(SceneAdapter):
    """Adapter for IsaacLab InteractiveScene."""

    def __init__(self, scene: "InteractiveScene"):
        self._scene = scene

    @override
    def zero_external_wrenches(self) -> None:
        for asset in self._scene.articulations.values():
            # IsaacLab >= 2.3 uses wrench composers instead of has_external_wrench buffers.
            if hasattr(asset, "instantaneous_wrench_composer"):
                asset.instantaneous_wrench_composer.reset()
            if hasattr(asset, "permanent_wrench_composer"):
                asset.permanent_wrench_composer.reset()
            # Backward compatibility for older APIs.
            if getattr(asset, "has_external_wrench", False):
                asset._external_force_b.zero_()
                asset._external_torque_b.zero_()
                asset.has_external_wrench = False

    @property
    def articulations(self):
        return self._scene.articulations

    @property
    def rigid_objects(self):
        return self._scene.rigid_objects

    @property
    def terrain(self):
        return self._scene.terrain

    def __getitem__(self, name):
        return self._scene[name]

    def get(self, name, default=None):
        """Dict-like access with a default value."""
        try:
            return self[name]
        except KeyError:
            return default

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


class MujocoSceneAdapter(SceneAdapter):
    """Adapter for MuJoCo MJScene."""

    def __init__(self, scene):
        self._scene = scene

    @override
    def zero_external_wrenches(self) -> None:
        for asset in self._scene.articulations.values():
            if asset.has_external_wrench:
                asset._external_force_b.zero_()
                asset._external_torque_b.zero_()
                asset.has_external_wrench = False

    @property
    def num_envs(self) -> int:
        return self._scene.num_envs

    def reset(self, env_ids: torch.Tensor) -> None:
        self._scene.reset(env_ids)

    def update(self, dt: float) -> None:
        self._scene.update(dt)

    def write_data_to_sim(self) -> None:
        self._scene.write_data_to_sim()

    @property
    def articulations(self) -> dict:
        return self._scene.articulations

    def __getitem__(self, name):
        return self._scene[name]

    def get(self, name, default=None):
        """Dict-like access with a default value."""
        try:
            return self[name]
        except KeyError:
            return default

    @property
    def ground_mesh(self):
        """Warp ground mesh for MuJoCo scenes.

        Converts the underlying scene's ground mesh (with vertices/faces) into
        a warp mesh suitable for raycasting.
        """
        if hasattr(self, "_ground_mesh"):
            return self._ground_mesh

        device = (
            self._scene.device.type
            if hasattr(self._scene, "device") and hasattr(self._scene.device, "type")
            else "cuda"
        )
        gm = self._scene.ground_mesh
        self._ground_mesh = wp.Mesh(
            points=wp.array(gm.vertices, dtype=wp.vec3, device=device),
            indices=wp.array(gm.faces.flatten(), dtype=wp.int32, device=device),
        )
        return self._ground_mesh


class MjlabSceneAdapter(SceneAdapter):
    """Adapter for mjlab Scene."""

    def __init__(self, scene: "Scene"):
        self._scene: "Scene" = scene

    @override
    def zero_external_wrenches(self) -> None:
        for asset in self._scene.entities.values():
            asset.data.data.xfrc_applied.zero_()

    @property
    def articulations(self) -> Mapping[str, Union["Entity", "Articulation"]]:
        # mjlab uses 'entities' instead of 'articulations'
        # Return entities dict for compatibility
        return self._scene.entities

    def __getitem__(self, name):
        return self._scene.entities[name]

    def get(self, name, default=None):
        """Dict-like access with a default value."""
        try:
            return self[name]
        except KeyError:
            return default


def wrap_sim(sim, backend: str) -> SimAdapter:
    """Wrap a simulation object with the appropriate adapter."""
    if backend == "isaac":
        return IsaacSimAdapter(sim)
    elif backend == "mujoco":
        return MujocoSimAdapter(sim)
    elif backend == "mjlab":
        return MjlabSimAdapter(sim)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def wrap_scene(scene, backend: str) -> SceneAdapter:
    """Wrap a scene object with the appropriate adapter."""
    if backend == "isaac":
        return IsaacSceneAdapter(scene)
    elif backend == "mujoco":
        return MujocoSceneAdapter(scene)
    elif backend == "mjlab":
        return MjlabSceneAdapter(scene)
    else:
        raise ValueError(f"Unknown backend: {backend}")
