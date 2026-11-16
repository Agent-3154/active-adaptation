"""Static PLY terrain for the Isaac backend, with optional AABB tiling.

One combined triangle mesh is spawned at ``/World/ground/terrain/mesh``. The
grid of tile AABB centers is

    ((r - (rows-1)/2) * (extent_x + spacing_x),
     (c - (cols-1)/2) * (extent_y + spacing_y),
     0)

so the gap between AABB edges is ``spacing`` and the union is xy-centered on
the global origin. Each copy is translated by that center minus the source
AABB center (xy only; z is unchanged). ``center: true`` shifts the PLY first
so that AABB center is the source origin; spawn points written in the PLY
frame move with it. Environment ``i`` is assigned tile
``i % (rows*cols)``; that translation is ``env_origins`` and therefore
``episode_origin`` for a shared 3DGS frame.

``MeshTerrainCfg`` subclasses Isaac's ``TerrainImporterCfg`` so
``InteractiveScene`` stores it as ``scene.terrain`` and reads ``env_origins``
from it. Plane and generator terrains stay on Isaac's importer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import trimesh

MESH_TERRAIN_COLLISION_PRIM = "/World/ground/terrain/mesh"


def center_terrain_mesh(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Return a copy whose XY AABB center is at the origin, and that shift.

    Z is unchanged. The shift is ``(-center_x, -center_y, 0)`` applied to
    vertices. Spawn points written in the original PLY frame need the same
    shift to stay on the mesh.
    """
    centered = mesh.copy()
    bounds = np.asarray(centered.bounds, dtype=np.float64)
    shift = np.array(
        [
            -0.5 * float(bounds[0, 0] + bounds[1, 0]),
            -0.5 * float(bounds[0, 1] + bounds[1, 1]),
            0.0,
        ],
        dtype=np.float64,
    )
    centered.apply_translation(shift)
    return centered, shift


def load_terrain_ply(path: str | Path) -> trimesh.Trimesh:
    """Load a triangle mesh from a PLY file."""
    ply_path = Path(path)
    if not ply_path.is_file():
        raise FileNotFoundError(f"Terrain PLY not found: {ply_path}")
    loaded = trimesh.load(ply_path, force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"Expected a triangle mesh in {ply_path}, got {type(loaded)}")
    if len(loaded.vertices) == 0 or len(loaded.faces) == 0:
        raise ValueError(f"Terrain PLY has no triangles: {ply_path}")
    loaded.vertices = np.asarray(loaded.vertices, dtype=np.float64)
    loaded.faces = np.asarray(loaded.faces, dtype=np.int64)
    return loaded


def tile_terrain_mesh(
    mesh: trimesh.Trimesh,
    tiles: tuple[int, int],
    spacing: tuple[float, float],
) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Tile ``mesh`` so the union's AABB is xy-centered on the origin.

    Stride is the XY AABB size plus ``spacing``. The returned origins are the
    translation of each copy, shape ``(rows, cols, 3)``: the centered grid
    offset minus the source AABB center in xy. Z is not recentered.
    """
    rows, cols = int(tiles[0]), int(tiles[1])
    if rows < 1 or cols < 1:
        raise ValueError(f"tiles must be positive, got {(rows, cols)}")
    gap_x, gap_y = float(spacing[0]), float(spacing[1])

    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    extent = bounds[1] - bounds[0]
    center_xy = (bounds[0, :2] + bounds[1, :2]) * 0.5
    stride_x = float(extent[0]) + gap_x
    stride_y = float(extent[1]) + gap_y
    if rows > 1 and stride_x <= 0.0:
        raise ValueError(
            f"X tile stride must be positive (AABB extent {extent[0]} + spacing {gap_x})"
        )
    if cols > 1 and stride_y <= 0.0:
        raise ValueError(
            f"Y tile stride must be positive (AABB extent {extent[1]} + spacing {gap_y})"
        )

    origins = np.zeros((rows, cols, 3), dtype=np.float64)
    parts: list[trimesh.Trimesh] = []
    for r in range(rows):
        for c in range(cols):
            translation = (
                (r - (rows - 1) / 2.0) * stride_x - float(center_xy[0]),
                (c - (cols - 1) / 2.0) * stride_y - float(center_xy[1]),
                0.0,
            )
            origins[r, c] = translation
            part = mesh.copy()
            part.apply_translation(translation)
            parts.append(part)

    combined = parts[0] if len(parts) == 1 else trimesh.util.concatenate(parts)
    return combined, origins


def terrain_spawn_origins(
    tile_translations: np.ndarray,
    spawn_points: np.ndarray | None,
) -> np.ndarray:
    """Build ``terrain_origins`` for a tiled mesh.

    ``tile_translations`` is ``(rows, cols, 3)``. With no spawn list the result
    is that same array: one origin per tile, the copy's translation.

    ``spawn_points`` is ``(n_origins, 3)`` in the source mesh frame. Each point
    is shifted by every tile translation, giving
    ``(rows, cols, n_origins, 3)``.
    """
    if spawn_points is None:
        return tile_translations
    points = np.asarray(spawn_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 1:
        raise ValueError(
            f"spawn origins must have shape (n_origins, 3), got {points.shape}"
        )
    return tile_translations[:, :, None, :] + points[None, None, :, :]


def _as_pair(value: Any, name: str) -> tuple[float, float] | tuple[int, int]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (value, value)
    if isinstance(value, str):
        raise ValueError(f"{name} must be a number or a pair, got {value!r}")
    try:
        pair = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a number or a pair, got {value!r}") from exc
    if len(pair) != 2:
        raise ValueError(f"{name} must have length 2, got {pair}")
    return pair[0], pair[1]


def _as_origins(value: Any) -> tuple[tuple[float, float, float], ...] | None:
    """Parse a user spawn list. ``None`` means one origin per tile translation."""
    if value is None:
        return None
    points: list[tuple[float, float, float]] = []
    for point in value:
        xyz = tuple(point)
        if len(xyz) != 3:
            raise ValueError(f"each origin must be xyz, got {point!r}")
        points.append((float(xyz[0]), float(xyz[1]), float(xyz[2])))
    if not points:
        raise ValueError("origins must contain at least one xyz point")
    return tuple(points)


def apply_mesh_terrain_contact_spec(sensor_spec: dict) -> None:
    """Point a terrain-only contact sensor at the spawned mesh prim.

    A single ``secondary_pattern`` applies to every partner, so this only fills
    the pattern when every partner is ``terrain`` and the YAML left it unset.
    """
    if sensor_spec.get("secondary_pattern") is not None:
        return
    secondary = sensor_spec.get("secondary", None)
    if secondary is None:
        return
    partners = [secondary] if isinstance(secondary, str) else list(secondary)
    if partners and all(partner == "terrain" for partner in partners):
        sensor_spec["secondary_pattern"] = MESH_TERRAIN_COLLISION_PRIM


class MeshTerrain:
    """Spawn one static collision mesh and assign each env a tile origin."""

    def __init__(self, cfg) -> None:
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.terrains.utils import create_prim_from_mesh

        sim = sim_utils.SimulationContext.instance()
        if sim is None:
            raise RuntimeError("MeshTerrain requires a SimulationContext.")
        self.cfg = cfg
        self.device = sim.device
        self.terrain_prim_paths = [f"{cfg.prim_path}/terrain"]
        self._terrain_flat_patches: dict = {}

        source = load_terrain_ply(cfg.ply_path)
        spawn_points = None if not cfg.origins else np.asarray(cfg.origins, dtype=np.float64)
        if cfg.center:
            source, shift = center_terrain_mesh(source)
            if spawn_points is not None:
                spawn_points = spawn_points + shift
        combined, tile_translations = tile_terrain_mesh(source, cfg.tiles, cfg.spacing)
        origins = terrain_spawn_origins(tile_translations, spawn_points)
        extent = np.asarray(source.bounds[1] - source.bounds[0], dtype=np.float64)
        stride = (
            float(extent[0]) + float(cfg.spacing[0]),
            float(extent[1]) + float(cfg.spacing[1]),
        )
        print(
            f"[MeshTerrain] {cfg.ply_path}: {len(source.vertices)} verts, "
            f"{len(source.faces)} faces, tiles={cfg.tiles[0]}x{cfg.tiles[1]}, "
            f"aabb_xy=({extent[0]:.3f}, {extent[1]:.3f}), "
            f"spacing=({cfg.spacing[0]:.3f}, {cfg.spacing[1]:.3f}), "
            f"center={cfg.center}, "
            f"stride=({stride[0]:.3f}, {stride[1]:.3f}), "
            f"spawned {len(combined.faces)} faces at {MESH_TERRAIN_COLLISION_PRIM}, "
            f"terrain_origins {tuple(origins.shape)}"
        )

        create_prim_from_mesh(
            f"{cfg.prim_path}/terrain",
            combined,
            visual_material=cfg.visual_material,
            physics_material=cfg.physics_material,
        )

        self.terrain_origins = torch.as_tensor(origins, device=self.device, dtype=torch.float32)
        flat = self.terrain_origins.reshape(-1, 3)
        tile_ids = torch.arange(cfg.num_envs, device=self.device) % flat.shape[0]
        self.env_origins = flat[tile_ids]

    @property
    def flat_patches(self) -> dict:
        return self._terrain_flat_patches

    @property
    def has_debug_vis_implementation(self) -> bool:
        """Isaac's env window reads this when building the debug-vis checkbox."""
        return False

    def set_debug_vis(self, debug_vis: bool) -> bool:
        return False

    def update_env_origins(self, env_ids, move_up, move_down) -> None:
        """Mesh tiles are fixed. Curriculum updates apply to generator terrains."""
        return None


def _mesh_terrain_cfg_cls():
    import isaaclab.sim as sim_utils
    from isaaclab.terrains import TerrainImporterCfg
    from isaaclab.utils import configclass

    @configclass
    class MeshTerrainCfg(TerrainImporterCfg):
        """Isaac scene terrain that loads a PLY and optionally tiles it.

        YAML::

            terrain:
              _target_: mesh
              ply_path: /path/to/collision.ply
              tiles: [2, 2]     # rows along x, cols along y
              spacing: [1.0, 1.0]  # extra gap beyond the mesh AABB; a scalar applies to both
              center: false        # shift the PLY so its XY AABB center is at the origin
              # Optional spawn points in the PLY frame. Each point is
              # copied onto every tile, so terrain_origins is
              # (rows, cols, n_origins, 3). Omit to spawn at the tile translation.
              origins: [[1.0, 0.5, 0.0], [2.0, 0.5, 0.0]]
        """

        class_type: type = MeshTerrain
        prim_path: str = "/World/ground"
        terrain_type: str = "mesh"
        ply_path: str = ""
        tiles: tuple[int, int] = (1, 1)
        spacing: tuple[float, float] = (1.0, 1.0)
        origins: tuple = ()
        center: bool = False
        physics_material: sim_utils.RigidBodyMaterialCfg = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=1.0,
        )

    return MeshTerrainCfg


_MeshTerrainCfg = None


def mesh_terrain_cfg_type():
    """Isaac ``TerrainImporterCfg`` subclass. Imports ``pxr``; call after the app starts."""
    global _MeshTerrainCfg
    if _MeshTerrainCfg is None:
        _MeshTerrainCfg = _mesh_terrain_cfg_cls()
    return _MeshTerrainCfg


def resolve_terrain_cfg(spec: Any, registry) -> Any:
    """Resolve ``task.terrain`` to a scene terrain cfg.

    A string is a registry name (``plane``, ``medium``, …). A mapping with
    ``_target_: mesh`` builds :class:`MeshTerrainCfg`.
    """
    if spec is None or isinstance(spec, str):
        name = "plane" if spec is None else spec
        if name == "mesh":
            raise ValueError(
                "terrain 'mesh' needs a mapping with ply_path, for example:\n"
                "  terrain:\n"
                "    _target_: mesh\n"
                "    ply_path: /path/to/collision.ply\n"
                "    tiles: [1, 1]\n"
                "    spacing: [0.0, 0.0]"
            )
        return registry.get("terrain", name)

    if hasattr(spec, "items"):
        cfg = {k: spec[k] for k in spec}
    else:
        raise ValueError(f"terrain must be a name or a mapping, got {type(spec)}")

    target = str(cfg.pop("_target_", "mesh"))
    if target not in ("mesh", "MeshTerrain"):
        raise ValueError(f"Unknown terrain target {target!r}. Expected 'mesh'.")
    if "ply_path" not in cfg:
        raise ValueError("mesh terrain requires ply_path.")
    ply_path = cfg.pop("ply_path")
    tiles = _as_pair(cfg.pop("tiles", (1, 1)), "tiles")
    spacing = _as_pair(cfg.pop("spacing", (0.0, 0.0)), "spacing")
    origins = _as_origins(cfg.pop("origins", None))
    center = bool(cfg.pop("center", False))
    if cfg:
        unknown = ", ".join(sorted(cfg))
        raise ValueError(f"Unknown mesh terrain keys: {unknown}")
    return mesh_terrain_cfg_type()(
        ply_path=str(ply_path),
        tiles=(int(tiles[0]), int(tiles[1])),
        spacing=(float(spacing[0]), float(spacing[1])),
        origins=() if origins is None else origins,
        center=center,
    )
